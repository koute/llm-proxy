use std::io::Write;
use std::path::Path;
use std::time::Duration;

use log::{error, warn};
use serde::Serialize;
use serde_json::Value;
use tokio::task::JoinHandle;

use super::keys::KeyRecord;
use crate::engine_protocol::{ProxyId, TokenCounts};
use crate::utils::AppendFile;

#[derive(Default)]
pub struct AuditLog {
    file: Option<AppendFile>,
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
    Request {
        request: ProxyId,
        key_id: Option<&'a str>,
        user: &'a str,
        class: &'a str,
        path: &'a str,
        model: Option<&'a str>,
        stream: bool,
        status: Option<u16>,
        outcome: &'a str,
        client: Option<&'a str>,
        total_seconds: f64,
        queued_seconds: Option<f64>,
        time_to_first_token_seconds: Option<f64>,
        generating_seconds: f64,
        suspensions: u32,
        tokens: Option<TokenCounts>,
    },
}

#[derive(Serialize)]
struct Line<'a> {
    time: String,
    #[serde(flatten)]
    event: &'a AuditEvent<'a>,
}

pub fn key_view(record: &KeyRecord) -> Value {
    let mut value = serde_json::to_value(record).expect("unserializable key record");
    value
        .as_object_mut()
        .map(|fields| fields.remove("key_hash"));
    value
}

fn line(event: &AuditEvent) -> Vec<u8> {
    let time = format!("{:.6}", jiff::Timestamp::now());
    let mut bytes = serde_json::to_vec(&Line { time, event }).expect("unserializable audit event");
    bytes.push(b'\n');
    bytes
}

impl AuditLog {
    pub fn open(path: Option<&Path>) -> (AuditLog, AuditWriter) {
        let Some(path) = path else {
            return (AuditLog::default(), AuditWriter(None));
        };
        match AppendFile::open(path, "audit log") {
            Ok((file, writer)) => (AuditLog { file: Some(file) }, AuditWriter(Some(writer))),
            Err(message) => {
                error!("{message}; the audit log is off");
                (AuditLog::default(), AuditWriter(None))
            }
        }
    }

    pub fn record(&self, event: AuditEvent) {
        if let Some(file) = &self.file {
            file.append(line(&event));
        }
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
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(&line(event))
}
