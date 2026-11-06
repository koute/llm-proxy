pub mod admin;
pub mod audit;
pub mod keys;
pub mod limits;
pub mod metrics;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use http::header::HeaderMap;
use http::StatusCode;
use log::{info, warn};
use nix::errno::Errno;
use nix::fcntl::{Flock, FlockArg};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{AccountsConfig, Config, GatewayConfig};
use crate::middle_end::routing::{truncated, PassthroughRecord, Rejection};
use crate::middle_end::scheduling_log::RequestSummary;
use crate::utils::Timestamp;
use crate::wire::PEER_ADDRESS_HEADER;
use audit::{AuditEvent, AuditLog, AuditWriter};
use keys::{read_keys_file, write_keys_file, KeyEntry, KeyRecord, KeyStore, KeysFile};
use limits::{unix_now, Limiter, SharedLimiter};
use metrics::Metrics;

const FILE_TIMEOUT: Duration = Duration::from_secs(10);

pub struct Gateway {
    pub config: GatewayConfig,
    accounts: Arc<AccountsConfig>,
    keys: RwLock<Arc<KeyStore>>,
    file: tokio::sync::Mutex<FileState>,
    pub limiter: SharedLimiter,
    pub metrics: Metrics,
    pub audit: AuditLog,
}

struct FileState {
    hash: Option<[u8; 32]>,
    last_error: Option<String>,
    /// Accesses which timed out and still block a thread (they can't be cancelled).
    hanging: Vec<tokio::task::AbortHandle>,
    /// When the last access was let through although others still hang.
    last_probe: Option<std::time::Instant>,
    /// Whether the last access completed: then the file works (again), and accesses
    /// don't wait for the ones which still hang.
    is_responsive: bool,
}

/// With this many accesses which still hang, further ones fail at once, except for one
/// probe a minute: a hung mount ties up a bounded number of threads, but a file which
/// works again (while the old accesses never return) is used again within a minute.
const MAX_HANGING_ACCESSES: usize = 4;
const HANGING_PROBE_INTERVAL: Duration = Duration::from_secs(60);
/// Never more hanging threads than this, probes and working phases included.
const MAX_HANGING_THREADS: usize = 64;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GatewaySnapshot {
    limiter: Limiter,
    /// The keys the old process had and the hash of the file they came from: the new
    /// process starts with them, not with the file the pre-flight check read, which an
    /// admin API change may have replaced since; it then checks the file at once.
    #[serde(default)]
    keys: Option<Vec<KeyRecord>>,
    #[serde(default)]
    keys_hash: Option<[u8; 32]>,
    /// Audit events the old process couldn't write, e.g. while the file was unwritable.
    #[serde(default)]
    unwritten_audit: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Attribution {
    pub key_id: Option<String>,
    pub client: Option<String>,
    pub path: String,
    pub model: Option<String>,
    pub stream: bool,
    #[serde(default)]
    pub arrival: Option<Timestamp>,
    /// The class the request was admitted with, which a restart hands over by name.
    #[serde(default)]
    pub class: Option<String>,
    /// Set when a request is handed over to a restarted middle end after it started.
    #[serde(default)]
    pub queued_seconds: Option<f64>,
    #[serde(default)]
    pub user_agent: Option<String>,
    /// The configured path pattern, the `path` label of metrics.
    #[serde(default)]
    pub route: Option<String>,
    /// The request's token estimate, reserved against its key's token limits.
    #[serde(default)]
    pub reserved_tokens: u64,
    /// Generating time before the last restart.
    #[serde(default)]
    pub generating_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Completed,
    /// The client went away.
    Aborted,
    /// The engine failed after the response had started, so it was cut off.
    Failed,
    Unknown,
}

impl Outcome {
    fn name(self) -> &'static str {
        match self {
            Outcome::Completed => "completed",
            Outcome::Aborted => "aborted",
            Outcome::Failed => "failed",
            Outcome::Unknown => "unknown",
        }
    }

    /// The `status` label: the HTTP status the client got, or what happened instead.
    fn status_label(self, status: Option<u16>) -> String {
        match (self, status) {
            (Outcome::Completed, Some(status)) => status.to_string(),
            (Outcome::Completed, None) => "error".to_string(),
            (outcome, _) => outcome.name().to_string(),
        }
    }
}

pub struct FinishedRequest<'a> {
    pub summary: &'a RequestSummary,
    pub attribution: &'a Attribution,
    pub outcome: Outcome,
    /// The status of the response head the client got, if it got one.
    pub status: Option<u16>,
    pub time_to_first_token: Option<f64>,
    /// From the arrival of the request until it ended, across restarts.
    pub total_seconds: f64,
    /// From the arrival of the request until it started, across restarts.
    pub queued_seconds: Option<f64>,
}

impl FinishedRequest<'_> {
    /// The class the request ran in: its own, or `handed over (<priority>)` if a
    /// restart couldn't find it any more (like the gauges).
    pub fn class(&self) -> &str {
        &self.summary.class
    }
}

#[derive(Debug)]
pub struct ChangeError {
    pub status: StatusCode,
    pub message: String,
}

impl ChangeError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> ChangeError {
        ChangeError {
            status,
            message: message.into(),
        }
    }
}

/// The IDs of the keys which were added, removed or changed between `old` and `new`.
fn changed_key_ids(old: &[KeyRecord], new: &KeyStore) -> Vec<String> {
    let old: BTreeMap<&String, &KeyRecord> =
        old.iter().map(|record| (&record.id, record)).collect();
    let new: BTreeMap<&String, &KeyRecord> = new
        .entries()
        .iter()
        .map(|entry| (&entry.record.id, &entry.record))
        .collect();
    let ids: BTreeSet<&String> = old.keys().chain(new.keys()).copied().collect();
    ids.into_iter()
        .filter(|id| old.get(id) != new.get(id))
        .cloned()
        .collect()
}

fn text_hash(text: &str) -> [u8; 32] {
    Sha256::digest(text).into()
}

/// The result of [`change_keys_file`].
pub struct KeysFileChange<T> {
    pub value: T,
    pub store: KeyStore,
    pub text: String,
    /// The file as it was read before the change, with the keys it had.
    pub previous: Option<(String, KeyStore)>,
}

/// Reads the keys file, applies `change` and replaces the file, under an exclusive lock
/// which `llm-proxy create-key` and the middle end share. A set `abandoned` flag (the
/// caller gave up waiting) stops the change before the file is written or replaced.
pub fn change_keys_file<T>(
    path: &Path,
    accounts: &AccountsConfig,
    create_missing: bool,
    abandonment: &Abandonment,
    change: impl FnOnce(&mut KeysFile) -> Result<T, ChangeError>,
) -> Result<KeysFileChange<T>, ChangeError> {
    let unavailable = |message: String| ChangeError::new(StatusCode::SERVICE_UNAVAILABLE, message);
    let name = path.display();
    // Replace the target of a symbolic link, not the link; never replace a link whose
    // target is missing (e.g. on a mount which is down) with a file.
    let is_link = std::fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_symlink());
    let path = match std::fs::canonicalize(path) {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !is_link => {
            path.to_path_buf()
        }
        Err(error) => {
            return Err(unavailable(format!(
                "failed to resolve the keys file '{name}': {error}"
            )))
        }
    };
    let lock = lock_keys_file(&path)
        .map_err(|error| unavailable(format!("failed to lock the keys file '{name}': {error}")))?;
    abandonment.hold(lock);
    // Released when the change ends, or at once when its caller gives up on it.
    struct Release<'a>(&'a Abandonment);
    impl Drop for Release<'_> {
        fn drop(&mut self) {
            self.0.release();
        }
    }
    let _release = Release(abandonment);
    let text = read_keys_file(&path)
        .map_err(|error| unavailable(format!("failed to read the keys file '{name}': {error}")))?;
    let (mut file, previous) = match text {
        Some(text) => {
            let invalid =
                |error: String| unavailable(format!("the keys file '{name}' is invalid: {error}"));
            let file: KeysFile =
                serde_json::from_str(&text).map_err(|error| invalid(error.to_string()))?;
            let store = KeyStore::build(&file, accounts).map_err(invalid)?;
            (file, Some((text, store)))
        }
        None if create_missing => (KeysFile::default(), None),
        None => return Err(unavailable(format!("the keys file '{name}' is missing"))),
    };
    let value = change(&mut file)?;
    let store = KeyStore::build(&file, accounts)
        .map_err(|error| ChangeError::new(StatusCode::BAD_REQUEST, error))?;
    let text = write_keys_file(&path, &file, &abandonment.is_abandoned)
        .map_err(|error| unavailable(format!("failed to write the keys file '{name}': {error}")))?;
    Ok(KeysFileChange {
        value,
        store,
        text,
        previous,
    })
}

/// A change of the keys file which its caller may give up on: then it doesn't write the
/// file any more, and its lock is released at once, although the thread may still hang
/// (e.g. reading the file on a hung mount), so that later changes can take the lock.
#[derive(Default)]
pub struct Abandonment {
    is_abandoned: AtomicBool,
    lock: std::sync::Mutex<Option<Flock<std::fs::File>>>,
}

impl Abandonment {
    #[cfg(test)]
    fn abandoned() -> Abandonment {
        let abandonment = Abandonment::default();
        abandonment.abandon();
        abandonment
    }

    fn abandon(&self) {
        self.is_abandoned.store(true, Ordering::SeqCst);
        self.release();
    }

    fn hold(&self, lock: Option<Flock<std::fs::File>>) {
        let mut slot = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Given up already: release the lock at once.
        *slot = lock.filter(|_| !self.is_abandoned.load(Ordering::SeqCst));
    }

    fn release(&self) {
        let lock = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        drop(lock);
    }
}

// Shorter than FILE_TIMEOUT, so that waiting for another change says so.
const LOCK_TIMEOUT: Duration = Duration::from_secs(8);
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(20);

/// An exclusive lock on `.<keys file name>.lock` next to the keys file, waiting up to
/// `LOCK_TIMEOUT` for it. File systems without locks (some FUSE mounts) and lock files
/// which can't be opened (e.g. created by `sudo create-key` with a strict umask) only get
/// the middle end's own serialization.
fn lock_keys_file(path: &Path) -> std::io::Result<Option<Flock<std::fs::File>>> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("keys.json");
    let lock_path = path.with_file_name(format!(".{file_name}.lock"));
    // Never through a symbolic link: `create-key` often runs as root, and a link planted
    // next to the keys file could make it create or change any file.
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
    let no_follow = nix::fcntl::OFlag::O_NOFOLLOW.bits();
    let opened = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .custom_flags(no_follow)
        .open(&lock_path)
        // flock works on a read-only file too, e.g. one which `sudo create-key` created.
        .or_else(|_| {
            let mut options = std::fs::OpenOptions::new();
            options.read(true).custom_flags(no_follow).open(&lock_path)
        });
    let mut file = match opened {
        Ok(file) => {
            // Readable by everyone, which is all flock needs: e.g. the lock file of a
            // `sudo create-key` with a strict umask must work for the middle end too.
            // Only a regular file of our own (checked on the open descriptor).
            if let Ok(metadata) = file.metadata() {
                let mode = metadata.permissions().mode();
                let is_own = metadata.uid() == nix::unistd::geteuid().as_raw();
                if metadata.is_file() && is_own && mode & 0o444 != 0o444 {
                    let readable = std::fs::Permissions::from_mode(mode | 0o444);
                    let _ = file.set_permissions(readable);
                }
            }
            file
        }
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
            warn!(
                "can't open the lock file '{}' ({error}); changing the keys file without it",
                lock_path.display()
            );
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let deadline = std::time::Instant::now() + LOCK_TIMEOUT;
    loop {
        match Flock::lock(file, FlockArg::LockExclusiveNonblock) {
            Ok(lock) => return Ok(Some(lock)),
            Err((_, Errno::ENOLCK | Errno::EOPNOTSUPP | Errno::ENOSYS)) => return Ok(None),
            Err((unlocked, Errno::EWOULDBLOCK)) if std::time::Instant::now() < deadline => {
                file = unlocked;
                std::thread::sleep(LOCK_RETRY_INTERVAL);
            }
            Err((_, Errno::EWOULDBLOCK)) => {
                return Err(std::io::Error::other(format!(
                    "another change holds the lock '{}'",
                    lock_path.display()
                )))
            }
            Err((_, errno)) => return Err(errno.into()),
        }
    }
}

async fn blocking<T: Send + 'static>(
    file_state: &mut FileState,
    task: impl FnOnce(&Abandonment) -> T + Send + 'static,
) -> Result<T, String> {
    let timeouts = (FILE_TIMEOUT, HANGING_PROBE_INTERVAL);
    blocking_with(file_state, timeouts, task).await
}

/// `blocking` with its timeout and probe interval as arguments, for tests.
async fn blocking_with<T: Send + 'static>(
    file_state: &mut FileState,
    (timeout, probe_interval): (Duration, Duration),
    task: impl FnOnce(&Abandonment) -> T + Send + 'static,
) -> Result<T, String> {
    file_state.hanging.retain(|task| !task.is_finished());
    let hanging = file_state.hanging.len();
    if hanging >= MAX_HANGING_THREADS {
        return Err(format!(
            "{hanging} previous accesses to the keys file still hang"
        ));
    }
    if hanging >= MAX_HANGING_ACCESSES && !file_state.is_responsive {
        let is_probe_due = file_state
            .last_probe
            .is_none_or(|time| time.elapsed() >= probe_interval);
        if !is_probe_due {
            return Err(format!(
                "{hanging} previous accesses to the keys file still hang"
            ));
        }
        file_state.last_probe = Some(std::time::Instant::now());
    }
    let abandonment = Arc::new(Abandonment::default());
    let task_abandonment = abandonment.clone();
    let mut task = tokio::task::spawn_blocking(move || task(&task_abandonment));
    match tokio::time::timeout(timeout, &mut task).await {
        Ok(result) => {
            file_state.is_responsive = true;
            result.map_err(|error| error.to_string())
        }
        Err(_) => {
            file_state.is_responsive = false;
            abandonment.abandon();
            file_state.hanging.push(task.abort_handle());
            Err(format!("the keys file did not respond within {timeout:?}"))
        }
    }
}

impl Gateway {
    pub fn new(
        config: &Config,
        snapshot: Option<GatewaySnapshot>,
    ) -> Option<(Arc<Gateway>, AuditWriter)> {
        let gateway_config = config.general.gateway.clone()?;
        let (audit, writer) = AuditLog::open(gateway_config.audit_log.as_deref());
        let snapshot = snapshot.unwrap_or_default();
        audit.append_unwritten(snapshot.unwritten_audit.clone().into_bytes());
        // The handed-over keys, unless they don't fit the new accounts.json.
        let handed_over = snapshot.keys.and_then(|keys| {
            let store = KeyStore::build(&KeysFile { keys }, &config.accounts).ok()?;
            Some((store, snapshot.keys_hash))
        });
        let (keys, hash) = handed_over.unwrap_or_else(|| {
            let keys = config.keys.clone().unwrap_or_default();
            (keys, config.texts.keys.as_deref().map(text_hash))
        });
        let path = gateway_config.keys_file.display();
        info!("gateway: {} API key(s) from '{path}'", keys.entries().len());
        let gateway = Gateway {
            accounts: Arc::new(config.accounts.clone()),
            keys: RwLock::new(Arc::new(keys)),
            file: tokio::sync::Mutex::new(FileState {
                hash,
                last_error: None,
                hanging: Vec::new(),
                last_probe: None,
                is_responsive: true,
            }),
            limiter: SharedLimiter::new(snapshot.limiter),
            metrics: Metrics::default(),
            audit,
            config: gateway_config,
        };
        Some((Arc::new(gateway), writer))
    }

    pub fn keys(&self) -> Arc<KeyStore> {
        let keys = self.keys.read();
        keys.unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn replace_keys(&self, store: KeyStore) {
        let ids: HashSet<&str> = store
            .entries()
            .iter()
            .map(|entry| entry.record.id.as_str())
            .collect();
        self.limiter.lock().retain(&ids);
        *self
            .keys
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Arc::new(store);
    }

    pub fn authenticate(&self, api_key: &str) -> Option<Arc<KeyEntry>> {
        if api_key.is_empty() {
            return None;
        }
        self.keys().lookup(api_key).cloned()
    }

    pub fn accounts(&self) -> &AccountsConfig {
        &self.accounts
    }

    pub fn client_address(&self, headers: &HeaderMap) -> Option<String> {
        let configured = self.config.client_ip_header().and_then(|name| {
            let value = headers.get(&name)?.to_str().ok()?;
            let last = value.rsplit(',').next().unwrap_or_default().trim();
            Some(truncated(last)).filter(|last| !last.is_empty())
        });
        configured.or_else(|| Some(truncated(headers.get(PEER_ADDRESS_HEADER)?.to_str().ok()?)))
    }

    pub async fn change_keys<T: Send + 'static>(
        &self,
        change: impl FnOnce(&mut KeysFile) -> Result<T, ChangeError> + Send + 'static,
    ) -> Result<T, ChangeError> {
        let unavailable =
            |message: String| ChangeError::new(StatusCode::SERVICE_UNAVAILABLE, message);
        let mut file_state = tokio::time::timeout(FILE_TIMEOUT, self.file.lock())
            .await
            .map_err(|_| unavailable("the keys file is busy; try again later".to_string()))?;
        let path = self.config.keys_file.clone();
        let accounts = self.accounts.clone();
        let change = move |abandonment: &Abandonment| {
            change_keys_file(&path, &accounts, false, abandonment, change)
        };
        let result = blocking(&mut file_state, change).await.map_err(|error| {
            unavailable(format!(
                "{error}; the change wasn't applied, unless the file system was already replacing the file"
            ))
        });
        match result.and_then(|result| result) {
            Ok(change) => {
                // Hand edits since the last reload are part of what's replaced now.
                if let Some((text, store)) = change.previous {
                    if file_state.hash != Some(text_hash(&text)) {
                        self.record_reload(&store);
                        self.metrics.keys_file_reloaded();
                    }
                }
                file_state.hash = Some(text_hash(&change.text));
                file_state.last_error = None;
                self.replace_keys(change.store);
                Ok(change.value)
            }
            Err(error) => {
                if error.status == StatusCode::SERVICE_UNAVAILABLE {
                    warn!("{}", error.message);
                    self.metrics.keys_file_failed();
                }
                Err(error)
            }
        }
    }

    pub async fn watch_keys_file(self: Arc<Gateway>) {
        let mut interval = tokio::time::interval(self.config.keys_file_poll_seconds.duration());
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first check is at once: e.g. after a restart, which started with the keys
        // of the old process.
        loop {
            interval.tick().await;
            self.reload_keys_file().await;
        }
    }

    async fn reload_keys_file(&self) {
        let Ok(mut file_state) = self.file.try_lock() else {
            return;
        };
        let path = self.config.keys_file.clone();
        let text = match blocking(&mut file_state, move |_: &Abandonment| {
            read_keys_file(&path)
        })
        .await
        {
            Ok(Ok(Some(text))) => Ok(text),
            Ok(Ok(None)) => Err("the file is missing".to_string()),
            Ok(Err(error)) => Err(error.to_string()),
            Err(error) => Err(error),
        };
        let hash = text.as_deref().map(text_hash).ok();
        if hash.is_some() && hash == file_state.hash {
            file_state.last_error = None;
            return;
        }
        match text.and_then(|text| KeyStore::parse(&text, &self.accounts)) {
            Ok(store) => {
                file_state.hash = hash;
                file_state.last_error = None;
                self.record_reload(&store);
                self.replace_keys(store);
                self.metrics.keys_file_reloaded();
            }
            Err(error) => {
                self.metrics.keys_file_failed();
                if file_state.last_error.as_ref() != Some(&error) {
                    let path = self.config.keys_file.display();
                    warn!("failed to reload the keys file '{path}': {error}; keeping the previous keys");
                    file_state.last_error = Some(error);
                }
            }
        }
    }

    fn record_reload(&self, store: &KeyStore) {
        let old: Vec<KeyRecord> = self
            .keys()
            .entries()
            .iter()
            .map(|entry| entry.record.clone())
            .collect();
        let changed = changed_key_ids(&old, store);
        let path = self.config.keys_file.display();
        info!(
            "reloaded the keys file '{path}': {} key(s) changed",
            changed.len()
        );
        let changed = changed.iter().map(String::as_str).collect();
        self.audit.record(AuditEvent::KeysReloaded { changed });
    }

    pub fn record_rejection(&self, rejection: &Rejection) {
        let status = rejection.status.as_u16();
        self.metrics.observe_rejection(status, rejection.reason);
        if [401, 403, 429].contains(&status) {
            self.audit.record(AuditEvent::Rejected {
                status,
                reason: rejection.reason,
                message: &rejection.message,
                method: &rejection.method,
                path: &rejection.path,
                user: rejection.account.as_deref(),
                key_id: rejection.key_id.as_deref(),
                api_key: rejection.masked_key.as_deref(),
                client: rejection.client_address.as_deref(),
                user_agent: rejection.user_agent.as_deref(),
            });
        }
    }

    pub fn request_finished(&self, finished: FinishedRequest) {
        let (summary, attribution) = (finished.summary, finished.attribution);
        let status = finished.outcome.status_label(finished.status);
        let key_id = attribution.key_id.as_deref();
        self.metrics.observe_request(&finished, &status);
        if let (Some(key_id), Some(tokens)) = (key_id, summary.tokens) {
            let tokens = tokens.prompt + tokens.output;
            self.limiter
                .lock()
                .record_tokens(key_id, tokens, unix_now());
        }
        self.audit.record(AuditEvent::Request {
            request: Some(summary.request),
            key_id,
            user: &summary.user,
            class: Some(finished.class()),
            method: None,
            path: &attribution.path,
            model: attribution.model.as_deref(),
            stream: attribution.stream,
            status: finished.status,
            outcome: finished.outcome.name(),
            client: attribution.client.as_deref(),
            user_agent: attribution.user_agent.as_deref(),
            total_seconds: finished.total_seconds,
            queued_seconds: finished.queued_seconds,
            time_to_first_token_seconds: finished.time_to_first_token,
            generating_seconds: summary.generating_seconds + attribution.generating_seconds,
            suspensions: summary.suspensions,
            tokens: summary.tokens,
        });
    }

    /// Records a finished passthrough request (e.g. /v1/models, /tokenize).
    pub fn passthrough_finished(
        &self,
        record: PassthroughRecord,
        status: Option<u16>,
        outcome: Outcome,
    ) {
        let key_id = record.key_id.as_deref();
        let label = outcome.status_label(status);
        self.metrics
            .observe_passthrough(&record.user, key_id, &record.route, &label);
        self.audit.record(AuditEvent::Request {
            request: None,
            key_id,
            user: &record.user,
            class: None,
            method: Some(&record.method),
            path: &record.path,
            model: None,
            stream: false,
            status,
            outcome: outcome.name(),
            client: record.client.as_deref(),
            user_agent: record.user_agent.as_deref(),
            total_seconds: Timestamp::now()
                .saturating_duration_since(record.arrival)
                .as_secs_f64(),
            queued_seconds: None,
            time_to_first_token_seconds: None,
            generating_seconds: 0.0,
            suspensions: 0,
            tokens: None,
        });
    }

    pub fn snapshot(&self, unwritten_audit: Vec<u8>) -> GatewaySnapshot {
        let keys = self
            .keys()
            .entries()
            .iter()
            .map(|entry| entry.record.clone())
            .collect();
        // No change can be in progress after the restart flush; if one is, the new
        // process just reads the file again.
        let keys_hash = self
            .file
            .try_lock()
            .ok()
            .and_then(|file_state| file_state.hash);
        GatewaySnapshot {
            limiter: self.limiter.lock().clone(),
            keys: Some(keys),
            keys_hash,
            unwritten_audit: String::from_utf8_lossy(&unwritten_audit).into_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accounts() -> AccountsConfig {
        AccountsConfig::parse(r#"{"classes": [{"name": "normal", "priority": 0}], "accounts": []}"#)
            .unwrap()
    }

    fn add_key(path: &Path, abandoned: &Abandonment) -> Result<KeysFileChange<()>, ChangeError> {
        let settings = serde_json::json!({"user": "u", "default_class": "normal"});
        let serde_json::Value::Object(settings) = settings else {
            unreachable!()
        };
        change_keys_file(path, &accounts(), true, abandoned, |file| {
            let (record, _) = admin::new_record(file, settings, None)?;
            file.keys.push(record);
            Ok(())
        })
    }

    #[test]
    fn concurrent_changes_keep_every_key() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("keys.json");
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || {
                    add_key(&path, &Abandonment::default()).map(|_| ()).unwrap()
                })
            })
            .collect();
        threads
            .into_iter()
            .for_each(|thread| thread.join().unwrap());
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            KeyStore::parse(&text, &accounts()).unwrap().entries().len(),
            16
        );
    }

    #[test]
    fn abandoned_changes_release_the_lock() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("keys.json");
        add_key(&path, &Abandonment::default()).map(|_| ()).unwrap();
        // A change which hangs while it holds the lock (like a read on a hung mount),
        // until the second change below is done.
        let abandonment = Arc::new(Abandonment::default());
        let (hung, is_hung) = std::sync::mpsc::channel::<()>();
        let (proceed, may_proceed) = std::sync::mpsc::channel::<()>();
        let (task_path, task_abandonment) = (path.clone(), abandonment.clone());
        let hanging = std::thread::spawn(move || {
            change_keys_file(&task_path, &accounts(), false, &task_abandonment, |_| {
                hung.send(()).unwrap();
                may_proceed.recv().unwrap();
                Ok(())
            })
            .map(|_| ())
        });
        is_hung.recv().unwrap();
        abandonment.abandon();
        // Its caller gave up: another change gets the lock while it still hangs.
        let started = std::time::Instant::now();
        add_key(&path, &Abandonment::default()).map(|_| ()).unwrap();
        assert!(started.elapsed() < Duration::from_secs(1));
        proceed.send(()).unwrap();
        let error = hanging.join().unwrap().unwrap_err();
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            KeyStore::parse(&text, &accounts()).unwrap().entries().len(),
            2
        );
    }

    /// A symbolic link planted as the lock file is never followed (`create-key` may run
    /// as root): its target is neither created nor changed.
    #[test]
    fn lock_files_are_not_followed() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("keys.json");
        let secret = directory.path().join("secret");
        std::fs::write(&secret, "secret").unwrap();
        std::fs::set_permissions(&secret, std::fs::Permissions::from_mode(0o600)).unwrap();
        let lock_path = directory.path().join(".keys.json.lock");
        std::os::unix::fs::symlink(&secret, &lock_path).unwrap();
        assert!(add_key(&path, &Abandonment::default()).is_err());
        let mode = std::fs::metadata(&secret).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        std::fs::remove_file(&lock_path).unwrap();
        let missing = directory.path().join("missing");
        std::os::unix::fs::symlink(&missing, &lock_path).unwrap();
        assert!(add_key(&path, &Abandonment::default()).is_err());
        assert!(!missing.exists());
    }

    #[test]
    fn lock_files_are_readable_for_everyone() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("keys.json");
        let lock_path = directory.path().join(".keys.json.lock");
        std::fs::write(&lock_path, "").unwrap();
        std::fs::set_permissions(&lock_path, std::fs::Permissions::from_mode(0o600)).unwrap();
        add_key(&path, &Abandonment::default()).map(|_| ()).unwrap();
        let mode = std::fs::metadata(&lock_path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o644);
    }

    /// Hung accesses are bounded, a probe a minute is let through, and one which works
    /// brings back normal access although the hung ones never return.
    #[tokio::test]
    async fn hung_accesses_and_recovery() {
        let mut file_state = FileState {
            hash: None,
            last_error: None,
            hanging: Vec::new(),
            last_probe: None,
            is_responsive: true,
        };
        let timeouts = (Duration::from_millis(20), Duration::from_millis(300));
        let (release, released) = std::sync::mpsc::channel::<()>();
        let released = Arc::new(std::sync::Mutex::new(released));
        let hang = || {
            let released = released.clone();
            move |_: &Abandonment| {
                let _ = released.lock().unwrap().recv();
            }
        };
        for _ in 0..MAX_HANGING_ACCESSES {
            let error = blocking_with(&mut file_state, timeouts, hang())
                .await
                .unwrap_err();
            assert!(error.contains("did not respond"));
        }
        // With that many hanging, one access is let through as a probe; it hangs too.
        let error = blocking_with(&mut file_state, timeouts, hang())
            .await
            .unwrap_err();
        assert!(error.contains("did not respond"));
        // Until the next probe is due, accesses fail at once.
        let error = blocking_with(&mut file_state, timeouts, |_| ())
            .await
            .unwrap_err();
        assert!(error.contains("still hang"), "{error}");
        // The next probe works: access is normal again, though the others still hang.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(blocking_with(&mut file_state, timeouts, |_| ())
            .await
            .is_ok());
        assert!(blocking_with(&mut file_state, timeouts, |_| ())
            .await
            .is_ok());
        for _ in 0..=MAX_HANGING_ACCESSES {
            release.send(()).unwrap();
        }
    }

    #[test]
    fn abandoned_changes_are_not_written() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("keys.json");
        add_key(&path, &Abandonment::default()).map(|_| ()).unwrap();
        let before = std::fs::read_to_string(&path).unwrap();
        let error = add_key(&path, &Abandonment::abandoned())
            .map(|_| ())
            .unwrap_err();
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
        let leftovers = std::fs::read_dir(directory.path()).unwrap().count();
        assert_eq!(leftovers, 2, "keys.json and its lock file only");
    }

    #[test]
    fn changes_follow_symbolic_links() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.json");
        add_key(&target, &Abandonment::default())
            .map(|_| ())
            .unwrap();
        let link = directory.path().join("keys.json");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let change = add_key(&link, &Abandonment::default()).unwrap();
        assert!(std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(std::fs::read_to_string(&target).unwrap(), change.text);
        assert_eq!(change.previous.unwrap().1.entries().len(), 1);
    }

    #[test]
    fn an_invalid_file_is_a_file_failure() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("keys.json");
        add_key(&path, &Abandonment::default()).map(|_| ()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let mut file: serde_json::Value = serde_json::from_str(&text).unwrap();
        let duplicate = file["keys"][0].clone();
        file["keys"].as_array_mut().unwrap().push(duplicate);
        std::fs::write(&path, file.to_string()).unwrap();
        let error = add_key(&path, &Abandonment::default())
            .map(|_| ())
            .unwrap_err();
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
    }
}
