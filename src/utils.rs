use std::future::Future;
use std::io;
use std::ops::Add;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::{Buf, BytesMut};
use log::warn;
use nix::fcntl::{fcntl, FcntlArg, FdFlag};
use nix::time::{clock_gettime, ClockId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tokio::net::{TcpStream, UnixStream};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

const READ_RESERVE: usize = 16 * 1024;
const APPEND_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const MAX_PENDING_BYTES: usize = 64 * 1024 * 1024;
/// How long a write or an open may take before the file counts as failing.
const APPEND_STALL_TIMEOUT: Duration = Duration::from_secs(5);
/// A write or an open without progress for this long at a stop counts as stuck, and its
/// records are handed back; a slow write which makes progress is waited for, as what it
/// writes can't be taken back.
const STUCK_AT_STOP: Duration = Duration::from_millis(500);
const WRITE_PIECE_BYTES: usize = 4096;
/// How far back from the end the opening of a file looks for its last complete line.
const MAX_PARTIAL_LINE: u64 = 16 * 1024 * 1024;

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
    dropped_total: AtomicUsize,
    is_failing: AtomicBool,
    /// Tells the writer to stop trying and return what it couldn't write.
    give_up: tokio::sync::Notify,
    /// When the current write or open started or last made progress, or 0 while idle.
    busy_since: AtomicU64,
    /// How much of the current batch the writing thread has written.
    position: AtomicUsize,
    /// Set when a stop gives up: the writing thread starts no other write.
    is_stopping: AtomicBool,
}

/// Monotonic nanoseconds (never 0), for `busy_since`.
fn now() -> u64 {
    Timestamp::now().monotonic_nanoseconds
}

/// The writer task of an [`AppendFile`], which returns the records it couldn't write.
pub type AppendWriter = JoinHandle<Vec<u8>>;

/// Whether an [`AppendFile`] keeps up.
#[derive(Clone, Copy, Debug)]
pub struct AppendHealth {
    pub is_failing: bool,
    pub dropped: usize,
}

enum AppendMessage {
    Record(Vec<u8>),
    /// Records a previous process couldn't (or didn't know whether it could) write.
    HandedOver(Vec<u8>),
    Close,
}

impl AppendFile {
    pub fn open(path: &Path, purpose: &'static str) -> Result<(AppendFile, AppendWriter), String> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| {
                format!("failed to open the {purpose} '{}': {error}", path.display())
            })?;
        Ok(AppendFile::start(path, purpose, Some(file)))
    }

    /// Like `open`, but the writer opens the file, so that a file which blocks (e.g. on
    /// a hung mount or a FIFO) can't block the caller; until the file can be opened and
    /// written, the writer keeps the records and retries.
    pub fn open_lazily(path: &Path, purpose: &'static str) -> (AppendFile, AppendWriter) {
        AppendFile::start(path, purpose, None)
    }

    pub fn health(&self) -> AppendHealth {
        AppendHealth {
            is_failing: self.pending.is_failing.load(Ordering::Relaxed),
            dropped: self.pending.dropped_total.load(Ordering::Relaxed),
        }
    }

    fn start(
        path: &Path,
        purpose: &'static str,
        file: Option<std::fs::File>,
    ) -> (AppendFile, AppendWriter) {
        let mut output = Output {
            file,
            path: path.to_path_buf(),
            purpose,
            ..Output::default()
        };
        let pending = output.pending.clone();
        let (messages, mut receiver) = mpsc::unbounded_channel();
        let writer = tokio::spawn(async move {
            let mut is_open = true;
            while is_open {
                let Some(message) = receiver.recv().await else {
                    return Vec::new();
                };
                let mut batch = Vec::new();
                let mut handed_over = 0;
                for message in
                    std::iter::once(message).chain(std::iter::from_fn(|| receiver.try_recv().ok()))
                {
                    match message {
                        AppendMessage::Record(record) => batch.extend_from_slice(&record),
                        AppendMessage::HandedOver(records) => {
                            // Only handed-over records at the start may be in the file.
                            if batch.len() == handed_over {
                                handed_over += records.len();
                            }
                            batch.extend_from_slice(&records);
                        }
                        AppendMessage::Close => {
                            is_open = false;
                            break;
                        }
                    }
                }
                if let Err(offset) = output.write(&batch, handed_over).await {
                    // Given up: hand back the rest from the start of a record written in
                    // part (the next opening of the file removes that part), and the queue.
                    let mut unwritten = batch.split_off(record_start(&batch, offset));
                    while let Ok(message) = receiver.try_recv() {
                        if let AppendMessage::Record(record) | AppendMessage::HandedOver(record) =
                            message
                        {
                            unwritten.extend_from_slice(&record);
                        }
                    }
                    return unwritten;
                }
            }
            Vec::new()
        });
        (AppendFile { messages, pending }, writer)
    }

    /// Closes the file and waits for the writer for up to `timeout`; if it can't finish
    /// (e.g. the file can't be written), it gives up and returns the unwritten records.
    pub async fn stop(&self, mut writer: AppendWriter, timeout: Duration) -> Vec<u8> {
        self.close();
        // A file which fails, or whose write hangs, won't work within the timeout: don't
        // hold up a restart, which waits with all requests paused.
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let busy_since = self.pending.busy_since.load(Ordering::Relaxed);
            let is_stuck = busy_since != 0
                && now().saturating_sub(busy_since) >= STUCK_AT_STOP.as_nanos() as u64;
            if self.health().is_failing || is_stuck {
                break;
            }
            let check = (tokio::time::Instant::now() + Duration::from_millis(50)).min(deadline);
            match tokio::time::timeout_at(check, &mut writer).await {
                Ok(result) => return result.unwrap_or_default(),
                Err(_) if check >= deadline => break,
                Err(_) => {}
            }
        }
        self.pending.is_stopping.store(true, Ordering::SeqCst);
        self.pending.give_up.notify_one();
        match tokio::time::timeout(Duration::from_secs(1), writer).await {
            Ok(Ok(unwritten)) => unwritten,
            _ => Vec::new(),
        }
    }

    pub fn append(&self, record: Vec<u8>) {
        let length = record.len();
        if self.pending.bytes.fetch_add(length, Ordering::Relaxed) + length > MAX_PENDING_BYTES {
            self.pending.bytes.fetch_sub(length, Ordering::Relaxed);
            self.pending.dropped.fetch_add(1, Ordering::Relaxed);
            self.pending.dropped_total.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let _ = self.messages.send(AppendMessage::Record(record));
    }

    /// Appends records which a previous process handed over: some of them may be in the
    /// file already, as that process may have been writing them when it stopped, and
    /// those aren't written again (see `written_already`).
    pub fn append_handed_over(&self, records: Vec<u8>) {
        let length = records.len();
        self.pending.bytes.fetch_add(length, Ordering::Relaxed);
        let _ = self.messages.send(AppendMessage::HandedOver(records));
    }

    pub fn close(&self) {
        let _ = self.messages.send(AppendMessage::Close);
    }
}

/// The writer task's file, and how far it got with the current batch.
#[derive(Default)]
struct Output {
    file: Option<std::fs::File>,
    path: std::path::PathBuf,
    purpose: &'static str,
    pending: Arc<Pending>,
    /// Handed-over bytes at the start of the batch, until checked against the file.
    handed_over: usize,
    /// How much of the batch is written.
    offset: usize,
    /// The length the last attempt left the file at.
    left_at: Option<u64>,
}

impl Output {
    /// Writes a batch, retrying until it works, each record exactly once: a slow write or
    /// open marks the file as failing but is awaited, as a cancelled one might still
    /// complete. When told to give up, returns how much is written.
    async fn write(&mut self, batch: &[u8], handed_over: usize) -> Result<(), usize> {
        self.handed_over = handed_over;
        self.offset = 0;
        self.left_at = None;
        let pending = self.pending.clone();
        let purpose = self.purpose;
        let mut has_warned = false;
        let retrying = async {
            loop {
                pending.busy_since.store(now(), Ordering::Relaxed);
                let write = self.write_batch(batch);
                tokio::pin!(write);
                let result = match tokio::time::timeout(APPEND_STALL_TIMEOUT, &mut write).await {
                    Ok(result) => result,
                    Err(_) => {
                        pending.is_failing.store(true, Ordering::Relaxed);
                        if !has_warned {
                            has_warned = true;
                            warn!(
                                "writing the {purpose} takes longer than {APPEND_STALL_TIMEOUT:?}; still trying"
                            );
                        }
                        write.await
                    }
                };
                pending.busy_since.store(0, Ordering::Relaxed);
                let Err(error) = result else {
                    return;
                };
                pending.is_failing.store(true, Ordering::Relaxed);
                if !has_warned {
                    has_warned = true;
                    warn!(
                        "failed to write the {purpose} ({error}); retrying every {APPEND_RETRY_INTERVAL:?}"
                    );
                }
                tokio::time::sleep(APPEND_RETRY_INTERVAL).await;
            }
        };
        // A stop which gives up ends the retries wherever they are.
        tokio::select! {
            () = retrying => {}
            () = pending.give_up.notified() => {
                // The writing thread may have got further than `offset` says.
                return Err(self.offset.max(pending.position.load(Ordering::SeqCst)));
            }
        }
        pending.bytes.fetch_sub(batch.len(), Ordering::Relaxed);
        if pending.is_failing.swap(false, Ordering::Relaxed) {
            log::info!("writing the {purpose} works again");
        }
        let dropped = pending.dropped.swap(0, Ordering::Relaxed);
        if dropped > 0 {
            warn!("dropped {dropped} record(s) of the {purpose} while it lagged behind");
        }
        Ok(())
    }

    /// Writes the rest of the batch from `offset`. After a full disk the file stays open
    /// and the next attempt goes on mid-record, unless the file got shorter (e.g. it was
    /// truncated to free space); after other errors the file is opened again, which
    /// removes a record written in part, and the next attempt starts at that record.
    async fn write_batch(&mut self, batch: &[u8]) -> io::Result<()> {
        let file = match self.file.take() {
            Some(file) => file,
            None => {
                self.offset = record_start(batch, self.offset);
                self.left_at = None;
                let path = self.path.clone();
                tokio::task::spawn_blocking(move || reopen_for_append(&path))
                    .await
                    .map_err(io::Error::other)??
            }
        };
        // Now that the file is open (partial line removed): skip records written after all.
        if self.handed_over > 0 && self.offset == 0 {
            let (path, records) = (self.path.clone(), batch[..self.handed_over].to_vec());
            let task = tokio::task::spawn_blocking(move || written_already(&path, &records));
            let written = tokio::time::timeout(APPEND_STALL_TIMEOUT, task).await;
            self.offset = written.ok().and_then(Result::ok).unwrap_or(0);
            self.handed_over = 0;
        }
        let (data, start, left_at) = (batch.to_vec(), self.offset, self.left_at);
        self.pending.position.store(start, Ordering::SeqCst);
        let pending = self.pending.clone();
        let (file, end, length, result) = tokio::task::spawn_blocking(move || {
            use std::io::Write;
            let mut file = file;
            let mut position = start;
            let is_mid_record = record_start(&data, start) != start;
            if let (true, Some(left_at)) = (is_mid_record, left_at) {
                let metadata = file.metadata();
                if metadata.is_ok_and(|metadata| metadata.is_file() && metadata.len() < left_at) {
                    position = record_start(&data, start);
                }
            }
            // In whole-record pieces: a slow write shows progress, and none starts after a stop.
            let result = loop {
                if position == data.len() {
                    break Ok(());
                }
                if pending.is_stopping.load(Ordering::SeqCst) {
                    break Err(io::Error::other("stopped"));
                }
                match file.write(&data[position..piece_end(&data, position)]) {
                    Ok(0) => break Err(io::ErrorKind::WriteZero.into()),
                    Ok(length) => {
                        position += length;
                        pending.position.store(position, Ordering::SeqCst);
                        pending.busy_since.store(now(), Ordering::Relaxed);
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) => break Err(error),
                }
            };
            let length = file.metadata().ok().map(|metadata| metadata.len());
            (file, position, length, result)
        })
        .await
        .map_err(io::Error::other)?;
        self.offset = end;
        self.left_at = length;
        let is_disk_full = result.as_ref().is_err_and(|error| {
            matches!(
                error.raw_os_error(),
                Some(nix::libc::ENOSPC | nix::libc::EDQUOT)
            )
        });
        if result.is_ok() || is_disk_full {
            self.file = Some(file);
        }
        result
    }
}

/// The end of the next piece to write from `position`: the rest of the record it is in
/// and then as many whole records as fit in `WRITE_PIECE_BYTES`.
fn piece_end(data: &[u8], position: usize) -> usize {
    let record_end = |from: usize| {
        data[from..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(data.len(), |end| from + end + 1)
    };
    let mut end = record_end(position);
    while end < data.len() && record_end(end) - position <= WRITE_PIECE_BYTES {
        end = record_end(end);
    }
    end
}

/// Where the record which `offset` is in starts.
fn record_start(batch: &[u8], offset: usize) -> usize {
    batch[..offset]
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |end| end + 1)
}

/// The last `max` bytes (or fewer) of a file of `length` bytes.
fn read_tail(file: &std::fs::File, length: u64, max: u64) -> io::Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let window = length.min(max);
    let mut tail = vec![0; window as usize];
    file.read_exact_at(&mut tail, length - window)?;
    Ok(tail)
}

/// How much of `handed_over` the file ends with already: the previous process's last
/// write (one piece) may have gone through after all. Records carry their time to the
/// microsecond, so equal bytes are the same records.
fn written_already(path: &Path, handed_over: &[u8]) -> usize {
    let Ok(file) = std::fs::File::open(path) else {
        return 0;
    };
    let length = match file.metadata() {
        Ok(metadata) if metadata.is_file() => metadata.len(),
        _ => return 0,
    };
    let Ok(tail) = read_tail(&file, length, piece_end(handed_over, 0) as u64) else {
        return 0;
    };
    (1..=tail.len())
        .rev()
        .filter(|end| handed_over[end - 1] == b'\n')
        .find(|end| tail.ends_with(&handed_over[..*end]))
        .unwrap_or(0)
}

/// Opens a file of JSON lines for appending. A regular file loses an incomplete last line
/// (a record written in part, which is written again in full). Anything else (e.g. a
/// FIFO) is opened for writing only, so that it blocks until there's a reader.
fn reopen_for_append(path: &Path) -> io::Result<std::fs::File> {
    use std::io::Write;
    if std::fs::metadata(path).is_ok_and(|metadata| !metadata.is_file()) {
        return std::fs::OpenOptions::new().append(true).open(path);
    }
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)?;
    let length = file.metadata()?.len();
    let tail = read_tail(&file, length, MAX_PARTIAL_LINE)?;
    if length == 0 || tail.last() == Some(&b'\n') {
        return Ok(file);
    }
    let window = tail.len() as u64;
    match tail.iter().rposition(|byte| *byte == b'\n') {
        Some(end) => file.set_len(length - window + end as u64 + 1)?,
        None if window == length => file.set_len(0)?,
        // A line longer than the window: end it, rather than cut a lot.
        None => file.write_all(b"\n")?,
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
        // The partial record is removed; whoever wrote it writes it again in full.
        assert_eq!(text, "{\"complete\":1}\n");
    }

    #[tokio::test]
    async fn append_drops_records_over_the_limit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("log.jsonl");
        let (file, writer) = AppendFile::open(&path, "test log").unwrap();
        let bytes = &file.pending.bytes;
        bytes.store(MAX_PENDING_BYTES, Ordering::Relaxed);
        file.append(b"dropped\n".to_vec());
        assert_eq!(file.pending.dropped.load(Ordering::Relaxed), 1);
        bytes.store(0, Ordering::Relaxed);
        file.append(b"kept\n".to_vec());
        file.close();
        writer.await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "kept\n");
        assert_eq!(bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn pieces_are_whole_records() {
        let small = b"a\nbb\nccc\n";
        assert_eq!(piece_end(small, 0), small.len());
        assert_eq!(piece_end(small, 3), small.len());
        let big = format!("{}\n{}\n", "x".repeat(3000), "y".repeat(3000));
        assert_eq!(piece_end(big.as_bytes(), 0), 3001);
        assert_eq!(piece_end(big.as_bytes(), 10), 3001);
        let huge = format!("{}\n", "z".repeat(10_000));
        assert_eq!(piece_end(huge.as_bytes(), 0), huge.len());
    }

    /// Records handed over by a previous process which it wrote after all, in its last
    /// write, are written once.
    #[tokio::test]
    async fn handed_over_records_are_written_once() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("log.jsonl");
        // The previous process wrote "b" (and half of "c") although it handed both over.
        std::fs::write(&path, "{\"a\":1}\n{\"b\":2}\n{\"c\"").unwrap();
        assert_eq!(written_already(&path, b"{\"b\":2}\n{\"c\":3}\n"), 0);
        let (file, writer) = AppendFile::open_lazily(&path, "test log");
        file.append_handed_over(b"{\"b\":2}\n{\"c\":3}\n".to_vec());
        file.append(b"{\"d\":4}\n".to_vec());
        assert!(file.stop(writer, Duration::from_secs(5)).await.is_empty());
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, "{\"a\":1}\n{\"b\":2}\n{\"c\":3}\n{\"d\":4}\n");
        // Nothing of it written yet: all of it.
        std::fs::write(&path, "{\"a\":1}\n").unwrap();
        assert_eq!(written_already(&path, b"{\"b\":2}\n"), 0);
        assert_eq!(written_already(&path, b"{\"a\":1}\n{\"b\":2}\n"), 8);
    }

    #[test]
    fn retries_start_at_the_record_written_in_part() {
        let batch = b"{\"a\":1}\n{\"b\":2}\n";
        for (offset, start) in [(0, 0), (4, 0), (8, 8), (11, 8), (batch.len(), batch.len())] {
            assert_eq!(record_start(batch, offset), start);
        }
    }

    #[tokio::test]
    async fn a_stuck_file_does_not_hold_up_a_stop() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("log.fifo");
        nix::unistd::mkfifo(&path, nix::sys::stat::Mode::S_IRWXU).unwrap();
        // Without a reader, the writer's open of the FIFO blocks.
        let (file, writer) = AppendFile::open_lazily(&path, "test log");
        file.append(b"kept\n".to_vec());
        tokio::time::sleep(Duration::from_millis(100)).await;
        let started = std::time::Instant::now();
        let unwritten = file.stop(writer, Duration::from_secs(5)).await;
        let elapsed = started.elapsed();
        // A reader lets the stuck open finish, so that the runtime can stop.
        drop(std::fs::File::open(&path).unwrap());
        assert!(elapsed < Duration::from_secs(2));
        assert_eq!(unwritten, b"kept\n");
    }

    /// A slow write which makes progress isn't taken for stuck at a stop: handing its
    /// records over would write them twice.
    #[tokio::test(flavor = "multi_thread")]
    async fn slow_writes_which_make_progress_are_waited_for() {
        use std::io::Read;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("log.fifo");
        nix::unistd::mkfifo(&path, nix::sys::stat::Mode::S_IRWXU).unwrap();
        let reader_path = path.clone();
        let reader = std::thread::spawn(move || {
            let mut fifo = std::fs::File::open(reader_path).unwrap();
            let (mut received, mut buffer) = (Vec::new(), [0; 2048]);
            loop {
                std::thread::sleep(Duration::from_millis(50));
                match fifo.read(&mut buffer).unwrap() {
                    0 => return received,
                    length => received.extend_from_slice(&buffer[..length]),
                }
            }
        });
        let (file, writer) = AppendFile::open_lazily(&path, "test log");
        let record = format!("{}\n", "r".repeat(1023)).into_bytes();
        for _ in 0..60 {
            file.append(record.clone());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        let unwritten = file.stop(writer, Duration::from_secs(10)).await;
        assert_eq!(unwritten.len(), 0, "bytes handed back");
        assert_eq!(reader.join().unwrap(), record.repeat(60));
    }

    #[tokio::test]
    async fn unwritable_records_are_handed_back() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("missing/log.jsonl");
        let (file, writer) = AppendFile::open_lazily(&path, "test log");
        file.append(b"first\n".to_vec());
        while !file.health().is_failing {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        file.append(b"second\n".to_vec());
        let unwritten = file.stop(writer, Duration::from_millis(50)).await;
        assert_eq!(unwritten, b"first\nsecond\n");
    }

    #[tokio::test]
    async fn records_are_written_once_the_file_works() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("later/log.jsonl");
        let (file, writer) = AppendFile::open_lazily(&path, "test log");
        file.append(b"kept\n".to_vec());
        while !file.health().is_failing {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        std::fs::create_dir(path.parent().unwrap()).unwrap();
        while file.health().is_failing {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let unwritten = file.stop(writer, Duration::from_secs(5)).await;
        assert!(unwritten.is_empty());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "kept\n");
    }
}
