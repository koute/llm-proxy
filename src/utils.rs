use std::future::Future;
use std::io;
use std::ops::Add;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::{Buf, BytesMut};
use log::{info, warn};
use nix::fcntl::{fcntl, FcntlArg, FdFlag};
use nix::time::{clock_gettime, ClockId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpStream, UnixStream};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

const READ_RESERVE: usize = 16 * 1024;
const APPEND_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const MAX_PENDING_BYTES: usize = 64 * 1024 * 1024;

pub trait Socket: Sized {
    fn readable(&self) -> impl Future<Output = io::Result<()>> + Send;
    fn writable(&self) -> impl Future<Output = io::Result<()>> + Send;
    fn try_read_into(&self, buffer: &mut BytesMut) -> io::Result<usize>;
    fn try_write_from(&self, data: &[u8]) -> io::Result<usize>;
    fn into_owned_fd(self) -> io::Result<OwnedFd>;
    fn from_owned_fd(fd: OwnedFd) -> io::Result<Self>;
}

macro_rules! socket_implementation {
    ($stream:ty, $standard_stream:ty) => {
        impl Socket for $stream {
            fn readable(&self) -> impl Future<Output = io::Result<()>> + Send {
                <$stream>::readable(self)
            }

            fn writable(&self) -> impl Future<Output = io::Result<()>> + Send {
                <$stream>::writable(self)
            }

            fn try_read_into(&self, buffer: &mut BytesMut) -> io::Result<usize> {
                self.try_read_buf(buffer)
            }

            fn try_write_from(&self, data: &[u8]) -> io::Result<usize> {
                self.try_write(data)
            }

            fn into_owned_fd(self) -> io::Result<OwnedFd> {
                Ok(OwnedFd::from(self.into_std()?))
            }

            fn from_owned_fd(fd: OwnedFd) -> io::Result<$stream> {
                let stream = <$standard_stream>::from(fd);
                stream.set_nonblocking(true)?;
                <$stream>::from_std(stream)
            }
        }
    };
}

socket_implementation!(TcpStream, std::net::TcpStream);
socket_implementation!(UnixStream, std::os::unix::net::UnixStream);

#[derive(Debug, Serialize, Deserialize)]
pub struct ConnectionSnapshot {
    fd: HandoverFd,
    #[serde(with = "base64_bytes")]
    read_buffer: BytesMut,
    #[serde(with = "base64_bytes")]
    write_buffer: BytesMut,
}

pub struct BufferedConnection<Stream> {
    stream: Stream,
    pub read_buffer: BytesMut,
    pub write_buffer: BytesMut,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ReadResult {
    Data,
    EndOfStream,
    NotReady,
}

impl<Stream: Socket> BufferedConnection<Stream> {
    pub fn new(stream: Stream) -> BufferedConnection<Stream> {
        BufferedConnection {
            stream,
            read_buffer: BytesMut::new(),
            write_buffer: BytesMut::new(),
        }
    }

    pub async fn readable(&self) -> io::Result<()> {
        self.stream.readable().await
    }

    pub async fn writable(&self) -> io::Result<()> {
        self.stream.writable().await
    }

    pub fn read_available(&mut self) -> io::Result<ReadResult> {
        self.read_buffer.reserve(READ_RESERVE);
        match self.stream.try_read_into(&mut self.read_buffer) {
            Ok(0) => Ok(ReadResult::EndOfStream),
            Ok(_) => Ok(ReadResult::Data),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(ReadResult::NotReady),
            Err(error) => Err(error),
        }
    }

    pub fn write_available(&mut self) -> io::Result<()> {
        match self.stream.try_write_from(&self.write_buffer) {
            Ok(0) => Err(io::Error::from(io::ErrorKind::WriteZero)),
            Ok(length) => {
                self.write_buffer.advance(length);
                Ok(())
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(()),
            Err(error) => Err(error),
        }
    }

    pub async fn fill(&mut self) -> io::Result<ReadResult> {
        loop {
            self.readable().await?;
            match self.read_available()? {
                ReadResult::NotReady => continue,
                result => return Ok(result),
            }
        }
    }

    pub fn has_pending_writes(&self) -> bool {
        !self.write_buffer.is_empty()
    }

    pub async fn flush(&mut self) -> io::Result<()> {
        while self.has_pending_writes() {
            self.writable().await?;
            self.write_available()?;
        }
        Ok(())
    }

    pub fn snapshot(self) -> io::Result<ConnectionSnapshot> {
        Ok(ConnectionSnapshot {
            fd: HandoverFd::new(self.stream.into_owned_fd()?)?,
            read_buffer: self.read_buffer,
            write_buffer: self.write_buffer,
        })
    }

    pub fn restore(snapshot: ConnectionSnapshot) -> io::Result<BufferedConnection<Stream>> {
        Ok(BufferedConnection {
            stream: Stream::from_owned_fd(snapshot.fd.adopt()?)?,
            read_buffer: snapshot.read_buffer,
            write_buffer: snapshot.write_buffer,
        })
    }
}

#[derive(Debug)]
pub struct HandoverFd(OwnedFd);

impl HandoverFd {
    pub fn new(fd: OwnedFd) -> io::Result<HandoverFd> {
        fcntl(&fd, FcntlArg::F_SETFD(FdFlag::empty()))?;
        Ok(HandoverFd(fd))
    }

    pub fn adopt(self) -> io::Result<OwnedFd> {
        fcntl(&self.0, FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC))?;
        Ok(self.0)
    }
}

impl Serialize for HandoverFd {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_i32(self.0.as_raw_fd())
    }
}

impl<'de> Deserialize<'de> for HandoverFd {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<HandoverFd, D::Error> {
        let fd = RawFd::deserialize(deserializer)?;
        if fd <= 2 {
            return Err(serde::de::Error::custom(format!(
                "invalid inherited file descriptor {fd}"
            )));
        }
        Ok(HandoverFd(unsafe { OwnedFd::from_raw_fd(fd) }))
    }
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Timestamp {
    monotonic_nanoseconds: u64,
}

impl Timestamp {
    pub fn now() -> Timestamp {
        let time = clock_gettime(ClockId::CLOCK_MONOTONIC).expect("unavailable monotonic clock");
        let duration = Duration::from(time);
        Timestamp {
            monotonic_nanoseconds: duration.as_nanos() as u64,
        }
    }

    pub fn instant(self) -> tokio::time::Instant {
        tokio::time::Instant::now() + self.saturating_duration_since(Timestamp::now())
    }

    pub fn saturating_duration_since(self, earlier: Timestamp) -> Duration {
        Duration::from_nanos(
            self.monotonic_nanoseconds
                .saturating_sub(earlier.monotonic_nanoseconds),
        )
    }
}

impl Add<Duration> for Timestamp {
    type Output = Timestamp;

    fn add(self, duration: Duration) -> Timestamp {
        Timestamp {
            monotonic_nanoseconds: self.monotonic_nanoseconds + duration.as_nanos() as u64,
        }
    }
}

#[derive(Clone)]
pub struct AppendFile {
    messages: mpsc::UnboundedSender<AppendMessage>,
    pending: Arc<Pending>,
}

#[derive(Default)]
struct Pending {
    bytes: AtomicUsize,
    dropped: AtomicUsize,
}

enum AppendMessage {
    Record(Vec<u8>),
    Close,
}

impl AppendFile {
    pub fn open(
        path: &Path,
        purpose: &'static str,
    ) -> Result<(AppendFile, JoinHandle<()>), String> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| {
                format!("failed to open the {purpose} '{}': {error}", path.display())
            })?;
        let mut file = Some(tokio::fs::File::from_std(file));
        let path = path.to_path_buf();
        let (messages, mut receiver) = mpsc::unbounded_channel();
        let pending = Arc::new(Pending::default());
        let writer_pending = pending.clone();
        let writer = tokio::spawn(async move {
            let mut is_open = true;
            let mut is_failing = false;
            while is_open {
                let Some(message) = receiver.recv().await else {
                    return;
                };
                let mut batch = Vec::new();
                for message in
                    std::iter::once(message).chain(std::iter::from_fn(|| receiver.try_recv().ok()))
                {
                    match message {
                        AppendMessage::Record(record) => batch.extend_from_slice(&record),
                        AppendMessage::Close => {
                            is_open = false;
                            break;
                        }
                    }
                }
                while let Err(error) = write_batch(&mut file, &path, &batch).await {
                    if !is_failing {
                        warn!("failed to write the {purpose} ({error}); retrying every {APPEND_RETRY_INTERVAL:?}");
                        is_failing = true;
                    }
                    tokio::time::sleep(APPEND_RETRY_INTERVAL).await;
                }
                writer_pending
                    .bytes
                    .fetch_sub(batch.len(), Ordering::Relaxed);
                if is_failing {
                    info!("writing the {purpose} works again");
                    is_failing = false;
                }
                let dropped = writer_pending.dropped.swap(0, Ordering::Relaxed);
                if dropped > 0 {
                    warn!("dropped {dropped} record(s) of the {purpose} while it lagged behind");
                }
            }
        });
        Ok((AppendFile { messages, pending }, writer))
    }

    pub fn append(&self, record: Vec<u8>) {
        let length = record.len();
        if self.pending.bytes.fetch_add(length, Ordering::Relaxed) + length > MAX_PENDING_BYTES {
            self.pending.bytes.fetch_sub(length, Ordering::Relaxed);
            self.pending.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let _ = self.messages.send(AppendMessage::Record(record));
    }

    pub fn close(&self) {
        let _ = self.messages.send(AppendMessage::Close);
    }
}

async fn write_batch(
    file: &mut Option<tokio::fs::File>,
    path: &Path,
    batch: &[u8],
) -> io::Result<()> {
    if file.is_none() {
        let path = path.to_path_buf();
        let reopened = tokio::task::spawn_blocking(move || reopen_for_append(&path))
            .await
            .map_err(io::Error::other)??;
        *file = Some(tokio::fs::File::from_std(reopened));
    }
    let output = file.as_mut().expect("the file was just opened");
    let result = async {
        output.write_all(batch).await?;
        output.flush().await
    }
    .await;
    if result.is_err() {
        *file = None;
    }
    result
}

fn reopen_for_append(path: &Path) -> io::Result<std::fs::File> {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)?;
    let length = file.metadata()?.len();
    if length > 0 {
        let mut last = [0u8];
        file.seek(SeekFrom::Start(length - 1))?;
        file.read_exact(&mut last)?;
        if last != *b"\n" {
            file.write_all(b"\n")?;
        }
    }
    Ok(file)
}

pub mod base64_bytes {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;
    use bytes::{Bytes, BytesMut};
    use serde::{Deserialize, Deserializer, Serializer};

    pub trait ByteBuffer: AsRef<[u8]> + Sized {
        fn from_vector(bytes: Vec<u8>) -> Self;
    }

    impl ByteBuffer for Bytes {
        fn from_vector(bytes: Vec<u8>) -> Bytes {
            Bytes::from(bytes)
        }
    }

    impl ByteBuffer for BytesMut {
        fn from_vector(bytes: Vec<u8>) -> BytesMut {
            BytesMut::from(bytes.as_slice())
        }
    }

    impl ByteBuffer for Vec<u8> {
        fn from_vector(bytes: Vec<u8>) -> Vec<u8> {
            bytes
        }
    }

    pub fn serialize<S: Serializer, Buffer: ByteBuffer>(
        buffer: &Buffer,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&STANDARD.encode(buffer))
    }

    pub fn deserialize<'de, D: Deserializer<'de>, Buffer: ByteBuffer>(
        deserializer: D,
    ) -> Result<Buffer, D::Error> {
        let text = String::deserialize(deserializer)?;
        STANDARD
            .decode(text)
            .map(Buffer::from_vector)
            .map_err(serde::de::Error::custom)
    }
}

pub mod request_head_serde {
    use serde::{Deserializer, Serializer};

    use crate::http1::RequestHead;

    pub fn serialize<S: Serializer>(head: &RequestHead, serializer: S) -> Result<S::Ok, S::Error> {
        super::base64_bytes::serialize(&head.encode(), serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<RequestHead, D::Error> {
        let bytes: Vec<u8> = super::base64_bytes::deserialize(deserializer)?;
        RequestHead::parse(&bytes)
            .map_err(serde::de::Error::custom)?
            .map(|(head, _)| head)
            .ok_or_else(|| serde::de::Error::custom("incomplete request head"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reopen_ends_a_partial_line() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("log.jsonl");
        std::fs::write(&path, "{\"complete\":1}\n{\"parti").unwrap();
        drop(reopen_for_append(&path).unwrap());
        drop(reopen_for_append(&path).unwrap());
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, "{\"complete\":1}\n{\"parti\n");
    }

    #[tokio::test]
    async fn append_drops_records_over_the_limit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("log.jsonl");
        let (file, writer) = AppendFile::open(&path, "test log").unwrap();
        file.pending
            .bytes
            .store(MAX_PENDING_BYTES, Ordering::Relaxed);
        file.append(b"dropped\n".to_vec());
        assert_eq!(file.pending.dropped.load(Ordering::Relaxed), 1);
        file.pending.bytes.store(0, Ordering::Relaxed);
        file.append(b"kept\n".to_vec());
        file.close();
        writer.await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "kept\n");
        assert_eq!(file.pending.bytes.load(Ordering::Relaxed), 0);
    }
}
