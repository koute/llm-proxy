use std::collections::{HashMap, HashSet};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::config::{matches_any, Account, AccountsConfig, ApiKeyHash, PathPattern};
use crate::quality_of_service::USER_ID_SEPARATOR;

const KEY_SECRET_BYTES: usize = 24;
const KEY_ID_BYTES: usize = 6;
const KEY_PREFIX_LENGTH: usize = 8;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeysFile {
    #[serde(default)]
    pub keys: Vec<KeyRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyRecord {
    pub id: String,
    pub user: String,
    #[serde(default)]
    pub description: String,
    pub key_hash: String,
    #[serde(default)]
    pub key_prefix: String,
    #[serde(default)]
    pub role: Role,
    pub default_class: String,
    #[serde(default)]
    pub allowed_classes: Vec<String>,
    #[serde(default)]
    pub blocked: bool,
    pub expires_at: Option<String>,
    #[serde(default)]
    pub limits: KeyLimits,
    pub allowed_routes: Option<Vec<String>>,
    pub created_at: Option<String>,
    pub created_by: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    #[default]
    User,
    Admin,
    Metrics,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyLimits {
    // 0 denies every request, like in LiteLLM.
    pub max_parallel_requests: Option<u64>,
    pub rpm_limit: Option<u64>,
    pub tpm_limit: Option<u64>,
    pub tpd_limit: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyState {
    Active,
    Blocked,
    Expired,
}

#[derive(Debug)]
pub struct KeyEntry {
    pub record: KeyRecord,
    pub account: Account,
    pub expires_at: Option<jiff::Timestamp>,
    pub allowed_routes: Option<Vec<PathPattern>>,
}

#[derive(Clone, Debug, Default)]
pub struct KeyStore {
    entries: Vec<Arc<KeyEntry>>,
    by_hash: HashMap<ApiKeyHash, usize>,
}

pub type Denial = (StatusCode, &'static str, &'static str);

impl KeyEntry {
    pub fn check(&self) -> Result<(), Denial> {
        // 401 like LiteLLM: clients treat a blocked key like a revoked one.
        let (reason, message) = match self.state(jiff::Timestamp::now()) {
            KeyState::Active => return Ok(()),
            KeyState::Blocked => ("blocked_api_key", "API key is blocked"),
            KeyState::Expired => ("expired_api_key", "API key expired"),
        };
        Err((StatusCode::UNAUTHORIZED, reason, message))
    }

    pub fn state(&self, now: jiff::Timestamp) -> KeyState {
        if self.record.blocked {
            KeyState::Blocked
        } else if self.expires_at.is_some_and(|expires_at| expires_at <= now) {
            KeyState::Expired
        } else {
            KeyState::Active
        }
    }

    pub fn allows_route(&self, path: &str) -> bool {
        self.allowed_routes
            .as_ref()
            .is_none_or(|patterns| matches_any(patterns, path))
    }

    fn build(record: &KeyRecord, accounts: &AccountsConfig) -> Result<KeyEntry, String> {
        let is_url_safe =
            |character: char| character.is_ascii_alphanumeric() || "-._".contains(character);
        if record.id.is_empty() || !record.id.chars().all(is_url_safe) {
            return Err("the id must consist of letters, digits, '-', '.' and '_'".to_string());
        }
        if record.user.is_empty() || record.user.contains(USER_ID_SEPARATOR) {
            return Err(format!(
                "the user must not be empty or contain '{USER_ID_SEPARATOR}'"
            ));
        }
        let expires_at = match &record.expires_at {
            Some(text) => Some(
                text.parse()
                    .map_err(|error| format!("invalid expires_at '{text}': {error}"))?,
            ),
            None => None,
        };
        // Limits a browser (the admin panel) can hold exactly.
        const MAX_LIMIT: u64 = (1 << 53) - 1;
        let limits = &record.limits;
        let values = [
            limits.max_parallel_requests,
            limits.rpm_limit,
            limits.tpm_limit,
            limits.tpd_limit,
        ];
        if values.into_iter().flatten().any(|value| value > MAX_LIMIT) {
            return Err(format!("limits can be at most {MAX_LIMIT}"));
        }
        let routes = record.allowed_routes.as_ref();
        for route in routes.into_iter().flatten() {
            let body = route.strip_suffix("/*").unwrap_or(route);
            if !route.starts_with('/') || body.contains('*') {
                return Err(format!(
                    "invalid route '{route}': routes are paths like '/v1/models', or '/prefix/*'"
                ));
            }
        }
        Ok(KeyEntry {
            account: Account {
                name: record.user.clone(),
                default_class: accounts.class_index(&record.default_class)?,
                allowed_classes: record
                    .allowed_classes
                    .iter()
                    .map(|name| accounts.class_index(name))
                    .collect::<Result<_, _>>()?,
                virtual_user_http_header: None,
            },
            expires_at,
            allowed_routes: routes
                .map(|routes| routes.iter().cloned().map(PathPattern::from).collect()),
            record: record.clone(),
        })
    }
}

impl KeyStore {
    pub fn parse(text: &str, accounts: &AccountsConfig) -> Result<KeyStore, String> {
        let file: KeysFile = serde_json::from_str(text).map_err(|error| error.to_string())?;
        KeyStore::build(&file, accounts)
    }

    pub fn build(file: &KeysFile, accounts: &AccountsConfig) -> Result<KeyStore, String> {
        let mut ids = HashSet::new();
        let mut store = KeyStore::default();
        for record in &file.keys {
            let id = &record.id;
            let in_key = |error: String| format!("key '{id}': {error}");
            let entry = KeyEntry::build(record, accounts).map_err(in_key)?;
            if !ids.insert(id.as_str()) {
                return Err(format!("duplicate key id '{id}'"));
            }
            let hash = ApiKeyHash::try_from(record.key_hash.clone()).map_err(in_key)?;
            if accounts.has_key_hash(&hash) {
                return Err(format!(
                    "key '{id}' uses the same API key as an account in the accounts file"
                ));
            }
            if store.by_hash.insert(hash, store.entries.len()).is_some() {
                return Err(format!("key '{id}' uses the same API key as another key"));
            }
            store.entries.push(Arc::new(entry));
        }
        Ok(store)
    }

    pub fn lookup(&self, api_key: &str) -> Option<&Arc<KeyEntry>> {
        self.lookup_hash(&ApiKeyHash::from_api_key(api_key))
    }

    pub fn lookup_hash(&self, hash: &ApiKeyHash) -> Option<&Arc<KeyEntry>> {
        let index = self.by_hash.get(hash)?;
        Some(&self.entries[*index])
    }

    pub fn entries(&self) -> &[Arc<KeyEntry>] {
        &self.entries
    }
}

pub struct NewKey {
    pub api_key: String,
    pub hash: String,
    pub prefix: String,
}

pub fn generate_key() -> io::Result<NewKey> {
    let api_key = format!("sk-{}", random_hex(KEY_SECRET_BYTES)?);
    Ok(NewKey {
        hash: ApiKeyHash::from_api_key(&api_key).to_hex(),
        prefix: api_key[..KEY_PREFIX_LENGTH].to_string(),
        api_key,
    })
}

pub fn generate_key_id() -> io::Result<String> {
    Ok(format!("key-{}", random_hex(KEY_ID_BYTES)?))
}

fn random_hex(length: usize) -> io::Result<String> {
    let mut bytes = vec![0u8; length];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

pub fn read_keys_file(path: &Path) -> io::Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// `.<keys file name>.<suffix>` next to the keys file.
pub fn hidden_sibling(path: &Path, suffix: &str) -> PathBuf {
    let file_name = path.file_name().and_then(|name| name.to_str());
    path.with_file_name(format!(".{}.{suffix}", file_name.unwrap_or("keys.json")))
}

/// Replaces the keys file through a temporary file and `rename`, unless `abandoned` is set
/// before the rename.
pub fn write_keys_file(path: &Path, file: &KeysFile, abandoned: &AtomicBool) -> io::Result<String> {
    let abandoned_error = || io::Error::other("abandoned after the timeout");
    if abandoned.load(Ordering::SeqCst) {
        return Err(abandoned_error());
    }
    let mut text = serde_json::to_string_pretty(file).expect("unserializable keys file");
    text.push('\n');
    let nanoseconds = jiff::Timestamp::now().as_nanosecond();
    let temporary = hidden_sibling(path, &format!("{}.{nanoseconds}.tmp", std::process::id()));
    let result = (|| {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        output.write_all(text.as_bytes())?;
        if let Ok(metadata) = std::fs::metadata(path) {
            output.set_permissions(metadata.permissions())?;
            // Also its owner, e.g. when `sudo create-key` replaces the middle end's file
            // (only root can, so others leave it to the file system).
            use std::os::unix::fs::MetadataExt;
            let _ = std::os::unix::fs::fchown(&output, Some(metadata.uid()), Some(metadata.gid()));
        }
        output.sync_all()?;
        if abandoned.load(Ordering::SeqCst) {
            return Err(abandoned_error());
        }
        std::fs::rename(&temporary, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.map(|()| text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(id: &str, api_key: &str, settings: serde_json::Value) -> KeyRecord {
        let mut record = json!({
            "id": id, "user": "alice", "key_hash": ApiKeyHash::from_api_key(api_key).to_hex(),
            "default_class": "normal", "allowed_classes": ["idle"],
        });
        for (field, value) in settings.as_object().unwrap() {
            record[field] = value.clone();
        }
        serde_json::from_value(record).unwrap()
    }

    fn build(keys: Vec<KeyRecord>) -> Result<KeyStore, String> {
        let accounts = AccountsConfig::parse(
            r#"{"classes": [{"name": "normal", "priority": 0}, {"name": "idle", "priority": -1}],
                "accounts": [{"name": "static", "api_key": "sk-static", "default_class": "normal"}]}"#,
        );
        KeyStore::build(&KeysFile { keys }, &accounts.unwrap())
    }

    #[test]
    fn lookup_and_validation() {
        let key = |id, api_key| record(id, api_key, json!({}));
        let store = build(vec![key("a", "sk-a"), key("b", "sk-b")]).unwrap();
        assert_eq!(store.lookup("sk-b").unwrap().record.id, "b");
        let allowed_classes = &store.lookup("sk-a").unwrap().account.allowed_classes;
        assert_eq!(allowed_classes, &vec![1]);
        assert!(store.lookup("sk-c").is_none());
        let error = |keys| build(keys).unwrap_err();
        assert!(error(vec![key("a", "sk-a"), key("a", "sk-b")]).contains("duplicate key id"));
        assert!(error(vec![key("a", "sk-a"), key("b", "sk-a")]).contains("another key"));
        assert!(error(vec![key("a", "sk-static")]).contains("accounts file"));
        for (settings, message) in [
            (json!({"default_class": "gold"}), "unknown class 'gold'"),
            (
                json!({"user": "litellm/bob"}),
                "must not be empty or contain",
            ),
            (json!({"id": "a b"}), "the id must consist"),
            (json!({"expires_at": "tomorrow"}), "invalid expires_at"),
            (json!({"key_hash": "abc"}), "invalid api_key_hash"),
        ] {
            let error = error(vec![record("a", "sk-a", settings)]);
            assert!(error.contains(message), "{message}");
        }
    }

    #[test]
    fn states_and_routes() {
        let routes = json!({"allowed_routes": ["/v1/chat/completions", "/v1/models/*"]});
        let store = build(vec![
            record("a", "sk-a", json!({"expires_at": "2020-01-01T00:00:00Z"})),
            record("b", "sk-b", json!({"blocked": true})),
            record("c", "sk-c", routes),
        ])
        .unwrap();
        let now = jiff::Timestamp::now();
        assert_eq!(store.lookup("sk-a").unwrap().state(now), KeyState::Expired);
        assert_eq!(store.lookup("sk-b").unwrap().state(now), KeyState::Blocked);
        let limited = store.lookup("sk-c").unwrap();
        assert_eq!(limited.state(now), KeyState::Active);
        assert!(limited.allows_route("/v1/models/foo") && !limited.allows_route("/v1/completions"));
        assert!(store.lookup("sk-a").unwrap().allows_route("/anything"));
    }

    #[test]
    fn generated_keys_and_atomic_writes() {
        use std::os::unix::fs::PermissionsExt;
        let key = generate_key().unwrap();
        assert!(key.api_key.starts_with(&key.prefix) && key.api_key.len() == 51);
        assert_eq!(key.hash, ApiKeyHash::from_api_key(&key.api_key).to_hex());
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("keys.json");
        assert_eq!(read_keys_file(&path).unwrap(), None);
        let file = KeysFile {
            keys: vec![record("a", "sk-a", json!({}))],
        };
        let text = write_keys_file(&path, &file, &AtomicBool::new(false)).unwrap();
        assert_eq!(serde_json::from_str::<KeysFile>(&text).unwrap(), file);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        write_keys_file(&path, &file, &AtomicBool::new(false)).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
