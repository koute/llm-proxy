use std::future::Future;
use std::io;
use std::ops::Add;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::{Buf, BytesMut};
use log::{info, warn};
use nix::fcntl::{fcntl, FcntlArg, FdFlag};
use nix::time::{clock_gettime, ClockId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
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
    dropped_total: AtomicUsize,
    is_failing: std::sync::atomic::AtomicBool,
    /// Tells the writer to stop trying and return what it couldn't write.
    give_up: tokio::sync::Notify,
    /// When the writer's current write or open started or last made progress
    /// (milliseconds since `process_start`), or 0 while it waits for records.
    busy_since: AtomicU64,
    /// How much of the current batch is written, as the writing thread goes along.
    position: AtomicUsize,
    /// Set when a stop gives up: the writing thread doesn't start another write.
    is_stopping: std::sync::atomic::AtomicBool,
}

fn process_start() -> std::time::Instant {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    *START.get_or_init(std::time::Instant::now)
}

fn milliseconds_since_start() -> u64 {
    process_start().elapsed().as_millis() as u64 + 1
}

/// A write or open without progress for this long at a stop counts as stuck: the stop
/// doesn't wait for it (the records are handed back), so a restart isn't held up. A
/// slow write which makes progress is waited for (up to the stop's timeout), as what
/// it writes can't be taken back.
const STUCK_AT_STOP: Duration = Duration::from_millis(500);

/// The writer task of an [`AppendFile`], which returns the records it couldn't write.
pub type AppendWriter = JoinHandle<Vec<u8>>;

/// How long a write or an open may take before the file counts as failing.
const APPEND_STALL_TIMEOUT: Duration = Duration::from_secs(5);

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
        let mut file = file;
        let path = path.to_path_buf();
        let (messages, mut receiver) = mpsc::unbounded_channel();
        let pending = Arc::new(Pending::default());
        let writer_pending = pending.clone();
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
                        AppendMessage::HandedOver(records) if batch.len() == handed_over => {
                            handed_over += records.len();
                            batch.extend_from_slice(&records);
                        }
                        AppendMessage::Record(record) | AppendMessage::HandedOver(record) => {
                            batch.extend_from_slice(&record)
                        }
                        AppendMessage::Close => {
                            is_open = false;
                            break;
                        }
                    }
                }
                let mut offset = 0;
                let written = write_watched(
                    &mut file,
                    &path,
                    (&batch, handed_over),
                    &mut offset,
                    &writer_pending,
                    purpose,
                )
                .await;
                if !written {
                    // Given up: hand back what isn't written, from the start of a record
                    // which is written in part (the next opening of the file removes
                    // that part), and everything queued after it.
                    // The writing thread may have got further than `offset` says.
                    let offset = offset.max(writer_pending.position.load(Ordering::SeqCst));
                    let mut batch = batch.split_off(record_start(&batch, offset));
                    while let Ok(message) = receiver.try_recv() {
                        if let AppendMessage::Record(record) | AppendMessage::HandedOver(record) =
                            message
                        {
                            batch.extend_from_slice(&record);
                        }
                    }
                    return batch;
                }
                writer_pending
                    .bytes
                    .fetch_sub(batch.len(), Ordering::Relaxed);
                if writer_pending.is_failing.swap(false, Ordering::Relaxed) {
                    info!("writing the {purpose} works again");
                }
                let dropped = writer_pending.dropped.swap(0, Ordering::Relaxed);
                if dropped > 0 {
                    warn!("dropped {dropped} record(s) of the {purpose} while it lagged behind");
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
            let is_stuck = || {
                let busy_since = self.pending.busy_since.load(Ordering::Relaxed);
                let busy_for = milliseconds_since_start().saturating_sub(busy_since);
                busy_since != 0 && busy_for >= STUCK_AT_STOP.as_millis() as u64
            };
            if self.health().is_failing || is_stuck() {
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
        self.pending
            .bytes
            .fetch_add(records.len(), Ordering::Relaxed);
        let _ = self.messages.send(AppendMessage::HandedOver(records));
    }

    pub fn close(&self) {
        let _ = self.messages.send(AppendMessage::Close);
    }
}

/// Writes a batch, retrying until it works, each record exactly once: `offset` is how
/// much of the batch is written. A write or an open which takes long marks the file as
/// failing, but is awaited all the same: cancelled, it might still complete and the
/// retry would duplicate it. Returns false if told to give up.
async fn write_watched(
    file: &mut Option<std::fs::File>,
    path: &Path,
    (batch, mut handed_over): (&[u8], usize),
    offset: &mut usize,
    pending: &Arc<Pending>,
    purpose: &'static str,
) -> bool {
    let mut has_warned = false;
    let mut left_at = None;
    loop {
        pending
            .busy_since
            .store(milliseconds_since_start(), Ordering::Relaxed);
        let write = write_batch(
            file,
            path,
            (batch, &mut handed_over),
            offset,
            &mut left_at,
            pending,
        );
        tokio::pin!(write);
        let stall = tokio::time::sleep(APPEND_STALL_TIMEOUT);
        tokio::pin!(stall);
        let mut is_stalled = false;
        let result = loop {
            tokio::select! {
                result = &mut write => break result,
                () = &mut stall, if !is_stalled => {
                    is_stalled = true;
                    pending.is_failing.store(true, Ordering::Relaxed);
                    if !has_warned {
                        warn!("writing the {purpose} takes longer than {APPEND_STALL_TIMEOUT:?}; still trying");
                        has_warned = true;
                    }
                }
                () = pending.give_up.notified() => return false,
            }
        };
        pending.busy_since.store(0, Ordering::Relaxed);
        let Err(error) = result else {
            return true;
        };
        pending.is_failing.store(true, Ordering::Relaxed);
        if !has_warned {
            warn!(
                "failed to write the {purpose} ({error}); retrying every {APPEND_RETRY_INTERVAL:?}"
            );
            has_warned = true;
        }
        tokio::select! {
            () = tokio::time::sleep(APPEND_RETRY_INTERVAL) => {}
            () = pending.give_up.notified() => return false,
        }
    }
}

const WRITE_PIECE_BYTES: usize = 4096;

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

/// How much of `handed_over` (records which a previous process handed over, in order)
/// the file ends with already: that process may have been writing them when it
/// stopped, and its last write may have gone through after all. Only the records of
/// that one write can be in question. Records carry their time to the microsecond, so
/// equal bytes are the same records.
fn written_already(path: &Path, handed_over: &[u8]) -> usize {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut file) = std::fs::File::open(path) else {
        return 0;
    };
    let length = match file.metadata() {
        Ok(metadata) if metadata.is_file() => metadata.len(),
        _ => return 0,
    };
    let in_question = piece_end(handed_over, 0) as u64;
    let window = in_question.min(length);
    let mut tail = vec![0; window as usize];
    let read = file
        .seek(SeekFrom::Start(length - window))
        .and_then(|_| file.read_exact(&mut tail));
    if read.is_err() {
        return 0;
    }
    (1..=tail.len())
        .rev()
        .filter(|end| handed_over[end - 1] == b'\n')
        .find(|end| tail.ends_with(&handed_over[..*end]))
        .unwrap_or(0)
}

/// Where the record which `offset` is in starts.
fn record_start(batch: &[u8], offset: usize) -> usize {
    batch[..offset]
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |end| end + 1)
}

/// Writes the rest of the batch from `offset` and moves `offset` along. After an error
/// which may pass (a full disk) the file stays open and the next attempt goes on where
/// this one stopped, unless the file got shorter meanwhile (e.g. it was truncated to
/// free space): then the record written in part is gone, and the attempt writes it
/// again. After other errors the file is opened again, which removes a record written
/// in part, so the next attempt starts at that record. `left_at` is the length the last
/// attempt left the file at.
async fn write_batch(
    file: &mut Option<std::fs::File>,
    path: &Path,
    (batch, handed_over): (&[u8], &mut usize),
    offset: &mut usize,
    left_at: &mut Option<u64>,
    pending: &Arc<Pending>,
) -> io::Result<()> {
    let output = match file.take() {
        Some(output) => output,
        None => {
            *offset = record_start(batch, *offset);
            *left_at = None;
            let path = path.to_path_buf();
            tokio::task::spawn_blocking(move || reopen_for_append(&path))
                .await
                .map_err(io::Error::other)??
        }
    };
    // Once the file is open (and its partial last line removed): skip handed-over
    // records which the previous process wrote after all.
    if *handed_over > 0 && *offset == 0 {
        let (path, records) = (path.to_path_buf(), batch[..*handed_over].to_vec());
        let task = tokio::task::spawn_blocking(move || written_already(&path, &records));
        *offset = tokio::time::timeout(APPEND_STALL_TIMEOUT, task)
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or(0);
        *handed_over = 0;
    }
    let (data, start, expected_length) = (batch.to_vec(), *offset, *left_at);
    pending.position.store(start, Ordering::SeqCst);
    let progress = pending.clone();
    let (output, end, length, result) = tokio::task::spawn_blocking(move || {
        use std::io::Write;
        let mut output = output;
        let mut position = start;
        let is_mid_record = record_start(&data, position) != position;
        if let (true, Some(expected_length)) = (is_mid_record, expected_length) {
            let metadata = output.metadata();
            let shrank = metadata
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() < expected_length);
            if shrank {
                position = record_start(&data, position);
            }
        }
        let result = loop {
            if position == data.len() {
                break Ok(());
            }
            // In pieces of whole records, so that a slow write shows its progress, and
            // none after a stop gave up on this write.
            if progress.is_stopping.load(Ordering::SeqCst) {
                break Err(io::Error::other("stopped"));
            }
            let end = piece_end(&data, position);
            match output.write(&data[position..end]) {
                Ok(0) => break Err(io::Error::from(io::ErrorKind::WriteZero)),
                Ok(length) => {
                    position += length;
                    progress.position.store(position, Ordering::SeqCst);
                    let now = milliseconds_since_start();
                    progress.busy_since.store(now, Ordering::Relaxed);
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => break Err(error),
            }
        };
        let length = output.metadata().ok().map(|metadata| metadata.len());
        (output, position, length, result)
    })
    .await
    .map_err(io::Error::other)?;
    *offset = end;
    *left_at = length;
    let is_transient = |error: &io::Error| {
        let errno = error.raw_os_error().map(nix::errno::Errno::from_raw);
        matches!(
            errno,
            Some(nix::errno::Errno::ENOSPC | nix::errno::Errno::EDQUOT)
        )
    };
    match result {
        Ok(()) => {
            *file = Some(output);
            Ok(())
        }
        Err(error) => {
            if is_transient(&error) {
                *file = Some(output);
            }
            Err(error)
        }
    }
}

/// How far back from the end the opening of a file looks for the end of its last
/// complete line.
const MAX_PARTIAL_LINE: u64 = 16 * 1024 * 1024;

/// Opens a file of JSON lines for appending. A regular file loses a last line which
/// isn't complete (a record written in part, e.g. when the disk was full), so that
/// records stay parseable; it's written again in full. Anything else (e.g. a FIFO) is
/// opened for writing only, so that it blocks until there's a reader.
fn reopen_for_append(path: &Path) -> io::Result<std::fs::File> {
    use std::io::{Read, Seek, SeekFrom, Write};
    let is_regular = std::fs::metadata(path).map_or(true, |metadata| metadata.is_file());
    if !is_regular {
        return std::fs::OpenOptions::new().append(true).open(path);
    }
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)?;
    let length = file.metadata()?.len();
    if length == 0 {
        return Ok(file);
    }
    let window = length.min(MAX_PARTIAL_LINE);
    let mut tail = vec![0; window as usize];
    file.seek(SeekFrom::Start(length - window))?;
    file.read_exact(&mut tail)?;
    if tail.last() == Some(&b'\n') {
        return Ok(file);
    }
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

    #[test]
    fn pieces_are_whole_records() {
        let small = b"a\nbb\nccc\n";
        assert_eq!(piece_end(small, 0), small.len());
        assert_eq!(piece_end(small, 3), small.len());
        let big = [
            vec![b'x'; 3000],
            b"\n".to_vec(),
            vec![b'y'; 3000],
            b"\n".to_vec(),
        ]
        .concat();
        assert_eq!(piece_end(&big, 0), 3001);
        assert_eq!(piece_end(&big, 10), 3001);
        let huge = [vec![b'z'; 10_000], b"\n".to_vec()].concat();
        assert_eq!(piece_end(&huge, 0), huge.len());
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
        assert_eq!(record_start(batch, 0), 0);
        assert_eq!(record_start(batch, 4), 0);
        assert_eq!(record_start(batch, 8), 8);
        assert_eq!(record_start(batch, 11), 8);
        assert_eq!(record_start(batch, batch.len()), batch.len());
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
        assert!(
            unwritten.is_empty(),
            "{} bytes handed back",
            unwritten.len()
        );
        assert_eq!(reader.join().unwrap(), record.repeat(60));
    }

    #[tokio::test]
    async fn unwritable_records_are_handed_back() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("missing").join("log.jsonl");
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
        let path = directory.path().join("later").join("log.jsonl");
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
