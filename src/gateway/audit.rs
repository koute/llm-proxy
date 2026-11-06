use std::io::Write;
use std::path::Path;
use std::time::Duration;

use log::warn;
use serde::Serialize;
use serde_json::Value;

use super::keys::KeyRecord;
use crate::engine_protocol::{ProxyId, TokenCounts};
use crate::utils::{AppendFile, AppendHealth, AppendWriter};

#[derive(Default)]
pub struct AuditLog {
    file: Option<AppendFile>,
}

pub struct AuditWriter(Option<AppendWriter>);

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
        /// None for passthrough requests, which the scheduler doesn't see.
        request: Option<ProxyId>,
        key_id: Option<&'a str>,
        user: &'a str,
        class: Option<&'a str>,
        /// Only for passthrough requests; generating requests are POSTs.
        #[serde(skip_serializing_if = "Option::is_none")]
        method: Option<&'a str>,
        path: &'a str,
        model: Option<&'a str>,
        stream: bool,
        status: Option<u16>,
        outcome: &'a str,
        client: Option<&'a str>,
        user_agent: Option<&'a str>,
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
        // Never off: if the file can't be opened or written, keep the events and retry.
        let (file, writer) = AppendFile::open_lazily(path, "audit log");
        (AuditLog { file: Some(file) }, AuditWriter(Some(writer)))
    }

    pub fn health(&self) -> Option<AppendHealth> {
        self.file.as_ref().map(AppendFile::health)
    }

    pub fn record(&self, event: AuditEvent) {
        if let Some(file) = &self.file {
            file.append(line(&event));
        }
    }

    /// Writes what's pending, for up to `timeout`, and returns what couldn't be written,
    /// which a restart hands over to the new process.
    pub async fn stop(&self, writer: &mut AuditWriter, timeout: Duration) -> Vec<u8> {
        let (Some(file), Some(writer)) = (&self.file, writer.0.take()) else {
            return Vec::new();
        };
        let unwritten = file.stop(writer, timeout).await;
        if !unwritten.is_empty() {
            warn!(
                "the audit log can't be written; {} bytes of events are left",
                unwritten.len()
            );
        }
        unwritten
    }

    /// Appends events which a previous process couldn't write.
    pub fn append_unwritten(&self, events: Vec<u8>) {
        if let (Some(file), false) = (&self.file, events.is_empty()) {
            file.append_handed_over(events);
        }
    }
}

/// Appends an event without the middle end (e.g. from `create-key`). Never through a
/// symbolic link: `create-key` often runs as root, and a link planted in place of the
/// audit log could make it append to any file.
pub fn append_blocking(path: &Path, event: &AuditEvent) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .custom_flags(nix::fcntl::OFlag::O_NOFOLLOW.bits())
        .open(path)?;
    file.write_all(&line(event))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_never_follow_links() {
        let directory = tempfile::tempdir().unwrap();
        let victim = directory.path().join("victim.conf");
        std::fs::write(&victim, "intact\n").unwrap();
        let path = directory.path().join("audit.jsonl");
        std::os::unix::fs::symlink(&victim, &path).unwrap();
        let event = AuditEvent::KeysReloaded {
            changed: Vec::new(),
        };
        assert!(append_blocking(&path, &event).is_err());
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "intact\n");
    }
}
