use std::collections::VecDeque;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use log::{error, warn};
use serde::Serialize;
use serde_json::Value;
use tokio::task::JoinHandle;

use super::keys::KeyRecord;
use crate::engine_protocol::{ProxyId, TokenCounts};
use crate::utils::AppendFile;

const RECENT_EVENTS: usize = 200;

#[derive(Default)]
pub struct AuditLog {
    file: Option<AppendFile>,
    recent: Mutex<VecDeque<Value>>,
}

pub struct AuditWriter(Option<JoinHandle<()>>);

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Actor<'a> {
    pub source: &'a str,
    pub user: Option<&'a str>,
    pub key_id: Option<&'a str>,
    pub client: Option<&'a str>,
}

#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum AuditEvent<'a> {
    KeyCreated {
        actor: Actor<'a>,
        key: Value,
    },
    KeyUpdated {
        actor: Actor<'a>,
        key_id: &'a str,
        before: Value,
        after: Value,
    },
    KeyRotated {
        actor: Actor<'a>,
        key: Value,
    },
    KeyDeleted {
        actor: Actor<'a>,
        key: Value,
    },
    KeysReloaded {
        added: Vec<&'a str>,
        removed: Vec<&'a str>,
        changed: Vec<&'a str>,
    },
    Rejected {
        status: u16,
        reason: &'a str,
        message: &'a str,
        method: &'a str,
        path: &'a str,
        user: Option<&'a str>,
        key_id: Option<&'a str>,
        api_key: Option<&'a str>,
        client: Option<&'a str>,
        user_agent: Option<&'a str>,
    },
    Request(&'a RequestRecord),
}

#[derive(Debug, Serialize)]
pub struct RequestRecord {
    pub request: Option<ProxyId>,
    pub key_id: Option<String>,
    pub user: String,
    pub class: Option<String>,
    pub method: Option<String>,
    pub path: String,
    pub model: Option<String>,
    pub stream: bool,
    pub status: Option<u16>,
    pub outcome: &'static str,
    pub client: Option<String>,
    pub total_seconds: f64,
    pub queued_seconds: Option<f64>,
    pub time_to_first_token_seconds: Option<f64>,
    pub generating_seconds: f64,
    pub suspensions: u32,
    pub tokens: Option<TokenCounts>,
}

#[derive(Serialize)]
struct Line<'a> {
    time: String,
    #[serde(flatten)]
    event: &'a AuditEvent<'a>,
}

pub fn key_view(record: &KeyRecord) -> Value {
    let mut value = serde_json::to_value(record).expect("unserializable key record");
    if let Value::Object(fields) = &mut value {
        fields.remove("key_hash");
    }
    value
}

fn line(event: &AuditEvent) -> Value {
    serde_json::to_value(Line {
        time: format!("{:.6}", jiff::Timestamp::now()),
        event,
    })
    .expect("unserializable audit event")
}

impl AuditLog {
    pub fn open(path: Option<&Path>) -> (AuditLog, AuditWriter) {
        let Some(path) = path else {
            return (AuditLog::default(), AuditWriter(None));
        };
        match AppendFile::open(path, "audit log") {
            Ok((file, writer)) => (
                AuditLog {
                    file: Some(file),
                    recent: Mutex::default(),
                },
                AuditWriter(Some(writer)),
            ),
            Err(message) => {
                error!("{message}; the audit log is off");
                (AuditLog::default(), AuditWriter(None))
            }
        }
    }

    pub fn record(&self, event: AuditEvent) {
        let line = line(&event);
        if let Some(file) = &self.file {
            let mut bytes = serde_json::to_vec(&line).expect("unserializable audit event");
            bytes.push(b'\n');
            file.append(bytes);
        }
        let mut recent = self
            .recent
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if recent.len() == RECENT_EVENTS {
            recent.pop_front();
        }
        recent.push_back(line);
    }

    pub fn recent(&self) -> Vec<Value> {
        self.recent
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .rev()
            .cloned()
            .collect()
    }

    pub async fn stop(&self, writer: &mut AuditWriter, timeout: Duration) {
        let (Some(file), Some(writer)) = (&self.file, writer.0.take()) else {
            return;
        };
        file.close();
        if tokio::time::timeout(timeout, writer).await.is_err() {
            warn!("the audit log was not complete after {timeout:?}");
        }
    }
}

pub fn append_blocking(path: &Path, event: &AuditEvent) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec(&line(event)).expect("unserializable audit event");
    bytes.push(b'\n');
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(&bytes)
}
