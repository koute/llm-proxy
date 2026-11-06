pub mod admin;
pub mod audit;
pub mod keys;
pub mod limits;
pub mod metrics;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use http::header::HeaderMap;
use http::StatusCode;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{AccountsConfig, Config, GatewayConfig};
use crate::middle_end::routing::Rejection;
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
    hanging: Option<tokio::task::AbortHandle>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GatewaySnapshot {
    limiter: Limiter,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Aborted,
    Unknown,
}

pub struct FinishedRequest<'a> {
    pub summary: &'a RequestSummary,
    pub attribution: &'a Attribution,
    pub outcome: Outcome,
    pub status: Option<u16>,
    pub time_to_first_token: Option<f64>,
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

fn text_hash(text: &str) -> [u8; 32] {
    Sha256::digest(text).into()
}

pub fn change_keys_file<T>(
    path: &Path,
    accounts: &AccountsConfig,
    create_missing: bool,
    change: impl FnOnce(&mut KeysFile) -> Result<T, ChangeError>,
) -> Result<(T, KeyStore, String), ChangeError> {
    let unavailable = |message: String| ChangeError::new(StatusCode::SERVICE_UNAVAILABLE, message);
    let name = path.display();
    let text = read_keys_file(path)
        .map_err(|error| unavailable(format!("failed to read the keys file '{name}': {error}")))?;
    let mut file = match text {
        Some(text) => serde_json::from_str(&text)
            .map_err(|error| unavailable(format!("the keys file '{name}' is invalid: {error}")))?,
        None if create_missing => KeysFile::default(),
        None => return Err(unavailable(format!("the keys file '{name}' is missing"))),
    };
    let value = change(&mut file)?;
    let store = KeyStore::build(&file, accounts)
        .map_err(|error| ChangeError::new(StatusCode::BAD_REQUEST, error))?;
    let text = write_keys_file(path, &file)
        .map_err(|error| unavailable(format!("failed to write the keys file '{name}': {error}")))?;
    Ok((value, store, text))
}

async fn blocking<T: Send + 'static>(
    file_state: &mut FileState,
    task: impl FnOnce() -> T + Send + 'static,
) -> Result<T, String> {
    if file_state
        .hanging
        .as_ref()
        .is_some_and(|task| !task.is_finished())
    {
        return Err("a previous access to the keys file still hangs".to_string());
    }
    let mut task = tokio::task::spawn_blocking(task);
    match tokio::time::timeout(FILE_TIMEOUT, &mut task).await {
        Ok(result) => result.map_err(|error| error.to_string()),
        Err(_) => {
            file_state.hanging = Some(task.abort_handle());
            Err(format!(
                "the keys file did not respond within {FILE_TIMEOUT:?}"
            ))
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
        let keys = config.keys.clone().unwrap_or_default();
        let path = gateway_config.keys_file.display();
        info!("gateway: {} API key(s) in '{path}'", keys.entries().len());
        let limiter = snapshot
            .map(|snapshot| snapshot.limiter)
            .unwrap_or_default();
        let gateway = Gateway {
            accounts: Arc::new(config.accounts.clone()),
            keys: RwLock::new(Arc::new(keys)),
            file: tokio::sync::Mutex::new(FileState {
                hash: config.texts.keys.as_deref().map(text_hash),
                last_error: None,
                hanging: None,
            }),
            limiter: SharedLimiter::new(limiter),
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
            Some(last.to_string()).filter(|last| !last.is_empty())
        });
        configured.or_else(|| Some(headers.get(PEER_ADDRESS_HEADER)?.to_str().ok()?.to_string()))
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
        let change = move || change_keys_file(&path, &accounts, false, change);
        let result = blocking(&mut file_state, change).await;
        match result.map_err(unavailable).and_then(|result| result) {
            Ok((value, store, text)) => {
                file_state.hash = Some(text_hash(&text));
                self.replace_keys(store);
                Ok(value)
            }
            Err(error) => {
                if error.status != StatusCode::BAD_REQUEST {
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
        interval.tick().await;
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
        let text = match blocking(&mut file_state, move || read_keys_file(&path)).await {
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
        let records = |store: &KeyStore| -> BTreeMap<String, KeyRecord> {
            let entries = store.entries().iter();
            entries
                .map(|entry| (entry.record.id.clone(), entry.record.clone()))
                .collect()
        };
        let (old, new) = (records(&self.keys()), records(store));
        let ids: BTreeSet<&String> = old.keys().chain(new.keys()).collect();
        let changed: Vec<&str> = ids
            .into_iter()
            .filter(|id| old.get(*id) != new.get(*id))
            .map(String::as_str)
            .collect();
        let path = self.config.keys_file.display();
        info!(
            "reloaded the keys file '{path}': {} key(s) changed",
            changed.len()
        );
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
        let (status, outcome) = match (finished.outcome, finished.status) {
            (Outcome::Aborted, _) => ("aborted".to_string(), "aborted"),
            (Outcome::Unknown, _) => ("unknown".to_string(), "unknown"),
            (Outcome::Completed, Some(status)) => (status.to_string(), "completed"),
            (Outcome::Completed, None) => ("error".to_string(), "completed"),
        };
        let key_id = attribution.key_id.as_deref();
        self.metrics.observe_request(&finished, &status);
        if let (Some(key_id), Some(tokens)) = (key_id, summary.tokens) {
            let tokens = tokens.prompt + tokens.output;
            self.limiter
                .lock()
                .record_tokens(key_id, tokens, unix_now());
        }
        self.audit.record(AuditEvent::Request {
            request: summary.request,
            key_id,
            user: &summary.user,
            class: &summary.class,
            path: &attribution.path,
            model: attribution.model.as_deref(),
            stream: attribution.stream,
            status: finished.status,
            outcome,
            client: attribution.client.as_deref(),
            total_seconds: summary.total_seconds,
            queued_seconds: summary.queued_seconds,
            time_to_first_token_seconds: finished.time_to_first_token,
            generating_seconds: summary.generating_seconds,
            suspensions: summary.suspensions,
            tokens: summary.tokens,
        });
    }

    pub fn snapshot(&self) -> GatewaySnapshot {
        GatewaySnapshot {
            limiter: self.limiter.lock().clone(),
        }
    }
}
