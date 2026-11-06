pub mod admin;
pub mod audit;
pub mod keys;
pub mod limits;
pub mod metrics;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};

use http::header::HeaderMap;
use http::StatusCode;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{AccountsConfig, Config, GatewayConfig};
use crate::middle_end::scheduling_log::RequestSummary;
use crate::wire::PEER_ADDRESS_HEADER;
use audit::{AuditEvent, AuditLog, AuditWriter, RequestRecord};
use keys::{read_keys_file, write_keys_file, KeyEntry, KeyStore, KeysFile};
use limits::{unix_now, Limiter, SharedLimiter};
use metrics::{Metrics, RequestMetric};

const FORWARDED_FOR_HEADER: &str = "x-forwarded-for";

pub struct Gateway {
    pub config: GatewayConfig,
    accounts: Arc<AccountsConfig>,
    keys: RwLock<Arc<KeyStore>>,
    file: tokio::sync::Mutex<FileState>,
    pub limiter: SharedLimiter,
    pub metrics: Metrics,
    pub audit: AuditLog,
}

#[derive(Default)]
struct FileState {
    hash: Option<[u8; 32]>,
    last_error: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GatewaySnapshot {
    limiter: Limiter,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Attribution {
    pub key_id: Option<String>,
    pub client: Option<String>,
    pub method: Option<String>,
    pub path: String,
    #[serde(default)]
    pub route: String,
    pub model: Option<String>,
    pub stream: bool,
}

impl Attribution {
    fn metric_path(&self) -> &str {
        if self.route.is_empty() {
            &self.path
        } else {
            &self.route
        }
    }
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

pub struct FinishedPassthrough<'a> {
    pub attribution: &'a Attribution,
    pub user: &'a str,
    pub outcome: Outcome,
    pub status: Option<u16>,
    pub total_seconds: f64,
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
    let text = read_keys_file(path).map_err(|error| {
        unavailable(format!(
            "failed to read the keys file '{}': {error}",
            path.display()
        ))
    })?;
    let mut file = match text {
        Some(text) => serde_json::from_str(&text).map_err(|error| {
            unavailable(format!(
                "the keys file '{}' is invalid: {error}",
                path.display()
            ))
        })?,
        None if create_missing => KeysFile::default(),
        None => {
            return Err(unavailable(format!(
                "the keys file '{}' is missing",
                path.display()
            )))
        }
    };
    let value = change(&mut file)?;
    let store = KeyStore::build(&file, accounts)
        .map_err(|error| ChangeError::new(StatusCode::BAD_REQUEST, error))?;
    let text = write_keys_file(path, &file).map_err(|error| {
        unavailable(format!(
            "failed to write the keys file '{}': {error}",
            path.display()
        ))
    })?;
    Ok((value, store, text))
}

impl Gateway {
    pub fn new(
        config: &Config,
        snapshot: Option<GatewaySnapshot>,
    ) -> Option<(Arc<Gateway>, AuditWriter)> {
        let gateway_config = config.general.gateway.clone()?;
        let (audit, writer) = AuditLog::open(gateway_config.audit_log.as_deref());
        let gateway = Gateway {
            config: gateway_config,
            accounts: Arc::new(config.accounts.clone()),
            keys: RwLock::new(Arc::new(config.keys.clone().unwrap_or_default())),
            file: tokio::sync::Mutex::new(FileState {
                hash: config.texts.keys.as_deref().map(text_hash),
                last_error: None,
            }),
            limiter: SharedLimiter::new(
                snapshot
                    .map(|snapshot| snapshot.limiter)
                    .unwrap_or_default(),
            ),
            metrics: Metrics::default(),
            audit,
        };
        info!(
            "gateway: {} API key(s) in '{}'",
            gateway.keys().entries().len(),
            gateway.config.keys_file.display()
        );
        Some((Arc::new(gateway), writer))
    }

    pub fn keys(&self) -> Arc<KeyStore> {
        self.keys
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
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
            let value = if name == FORWARDED_FOR_HEADER {
                value.split(',').next().unwrap_or_default()
            } else {
                value
            };
            Some(value.trim().to_string()).filter(|value| !value.is_empty())
        });
        configured.or_else(|| {
            headers
                .get(PEER_ADDRESS_HEADER)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string)
        })
    }

    pub async fn change_keys<T: Send + 'static>(
        &self,
        change: impl FnOnce(&mut KeysFile) -> Result<T, ChangeError> + Send + 'static,
    ) -> Result<T, ChangeError> {
        let timeout = self.config.keys_file_timeout_seconds.duration();
        let busy = || {
            ChangeError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "the keys file is busy or unreachable; try again later",
            )
        };
        let mut file_state = tokio::time::timeout(timeout, self.file.lock())
            .await
            .map_err(|_| busy())?;
        let path = self.config.keys_file.clone();
        let accounts = self.accounts.clone();
        let task =
            tokio::task::spawn_blocking(move || change_keys_file(&path, &accounts, false, change));
        let result = match tokio::time::timeout(timeout, task).await {
            Ok(Ok(result)) => result,
            Ok(Err(error)) => Err(ChangeError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("the keys file change failed: {error}"),
            )),
            Err(_) => Err(ChangeError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("the keys file did not respond within {timeout:?}"),
            )),
        };
        match result {
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

    pub async fn reload_keys_file(&self) {
        let Ok(mut file_state) = self.file.try_lock() else {
            return;
        };
        let timeout = self.config.keys_file_timeout_seconds.duration();
        let path = self.config.keys_file.clone();
        let task = tokio::task::spawn_blocking(move || read_keys_file(&path));
        let result = match tokio::time::timeout(timeout, task).await {
            Ok(Ok(Ok(Some(text)))) => Ok(text),
            Ok(Ok(Ok(None))) => Err("the file is missing".to_string()),
            Ok(Ok(Err(error))) => Err(error.to_string()),
            Ok(Err(error)) => Err(error.to_string()),
            Err(_) => Err(format!("no response within {timeout:?}")),
        };
        let result = result.and_then(|text| {
            let hash = text_hash(&text);
            if file_state.hash == Some(hash) {
                return Ok(None);
            }
            KeyStore::parse(&text, &self.accounts).map(|store| Some((store, hash)))
        });
        match result {
            Ok(None) => file_state.last_error = None,
            Ok(Some((store, hash))) => {
                file_state.hash = Some(hash);
                file_state.last_error = None;
                self.log_reload(&store);
                self.replace_keys(store);
                self.metrics.keys_file_reloaded();
            }
            Err(error) => {
                self.metrics.keys_file_failed();
                if file_state.last_error.as_ref() != Some(&error) {
                    warn!(
                        "failed to reload the keys file '{}': {error}; keeping the previous keys",
                        self.config.keys_file.display()
                    );
                    file_state.last_error = Some(error);
                }
            }
        }
    }

    fn log_reload(&self, store: &KeyStore) {
        let previous = self.keys();
        let records = |store: &KeyStore| -> HashMap<String, _> {
            store
                .entries()
                .iter()
                .map(|entry| (entry.record.id.clone(), entry.record.clone()))
                .collect()
        };
        let (old, new) = (records(&previous), records(store));
        let mut added: Vec<&str> = new
            .keys()
            .filter(|id| !old.contains_key(*id))
            .map(String::as_str)
            .collect();
        let mut removed: Vec<&str> = old
            .keys()
            .filter(|id| !new.contains_key(*id))
            .map(String::as_str)
            .collect();
        let mut changed: Vec<&str> = new
            .iter()
            .filter(|(id, record)| old.get(*id).is_some_and(|old| old != *record))
            .map(|(id, _)| id.as_str())
            .collect();
        added.sort_unstable();
        removed.sort_unstable();
        changed.sort_unstable();
        info!(
            "reloaded the keys file '{}': {} added, {} removed, {} changed",
            self.config.keys_file.display(),
            added.len(),
            removed.len(),
            changed.len()
        );
        self.audit.record(AuditEvent::KeysReloaded {
            added,
            removed,
            changed,
        });
    }

    pub fn request_finished(&self, finished: FinishedRequest) {
        let summary = finished.summary;
        let attribution = finished.attribution;
        let status = status_label(finished.outcome, finished.status);
        let key = attribution.key_id.as_deref().unwrap_or_default();
        self.metrics.observe_request(RequestMetric {
            user: &summary.user,
            key,
            class: &summary.class,
            path: attribution.metric_path(),
            status,
            tokens: summary.tokens,
            suspensions: summary.suspensions,
            generating_seconds: summary.generating_seconds,
            total_seconds: summary.total_seconds,
            queued_seconds: summary.queued_seconds,
            time_to_first_token: finished.time_to_first_token,
        });
        if let (Some(key_id), Some(tokens)) = (&attribution.key_id, summary.tokens) {
            self.limiter
                .lock()
                .record_tokens(key_id, tokens.prompt, tokens.output, unix_now());
        }
        self.audit.record(AuditEvent::Request(&RequestRecord {
            request: Some(summary.request),
            key_id: attribution.key_id.clone(),
            user: summary.user.clone(),
            class: Some(summary.class.clone()),
            method: attribution.method.clone(),
            path: attribution.path.clone(),
            model: attribution.model.clone(),
            stream: attribution.stream,
            status: finished.status,
            outcome: outcome_name(finished.outcome),
            client: attribution.client.clone(),
            total_seconds: summary.total_seconds,
            queued_seconds: summary.queued_seconds,
            time_to_first_token_seconds: finished.time_to_first_token,
            generating_seconds: summary.generating_seconds,
            suspensions: summary.suspensions,
            tokens: summary.tokens,
        }));
    }

    pub fn passthrough_finished(&self, finished: FinishedPassthrough) {
        let attribution = finished.attribution;
        let key = attribution.key_id.as_deref().unwrap_or_default();
        self.metrics.observe_request(RequestMetric {
            user: finished.user,
            key,
            class: "",
            path: attribution.metric_path(),
            status: status_label(finished.outcome, finished.status),
            tokens: None,
            suspensions: 0,
            generating_seconds: 0.0,
            total_seconds: finished.total_seconds,
            queued_seconds: None,
            time_to_first_token: None,
        });
        self.audit.record(AuditEvent::Request(&RequestRecord {
            request: None,
            key_id: attribution.key_id.clone(),
            user: finished.user.to_string(),
            class: None,
            method: attribution.method.clone(),
            path: attribution.path.clone(),
            model: None,
            stream: false,
            status: finished.status,
            outcome: outcome_name(finished.outcome),
            client: attribution.client.clone(),
            total_seconds: finished.total_seconds,
            queued_seconds: None,
            time_to_first_token_seconds: finished.time_to_first_token,
            generating_seconds: 0.0,
            suspensions: 0,
            tokens: None,
        }));
    }

    pub fn snapshot(&self) -> GatewaySnapshot {
        GatewaySnapshot {
            limiter: self.limiter.lock().clone(),
        }
    }
}

fn status_label(outcome: Outcome, status: Option<u16>) -> String {
    match (outcome, status) {
        (Outcome::Aborted, _) => "aborted".to_string(),
        (Outcome::Unknown, _) => "unknown".to_string(),
        (Outcome::Completed, Some(status)) => status.to_string(),
        (Outcome::Completed, None) => "error".to_string(),
    }
}

fn outcome_name(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Completed => "completed",
        Outcome::Aborted => "aborted",
        Outcome::Unknown => "unknown",
    }
}
