use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::num::NonZeroU64;
use std::path::Path;
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
    pub max_parallel_requests: Option<NonZeroU64>,
    pub rpm_limit: Option<NonZeroU64>,
    pub tpm_limit: Option<NonZeroU64>,
    pub tpd_limit: Option<NonZeroU64>,
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
        match self.state(jiff::Timestamp::now()) {
            KeyState::Active => Ok(()),
            KeyState::Blocked => Err((
                StatusCode::FORBIDDEN,
                "blocked_api_key",
                "API key is blocked",
            )),
            KeyState::Expired => Err((
                StatusCode::UNAUTHORIZED,
                "expired_api_key",
                "API key expired",
            )),
        }
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
}

impl KeyStore {
    pub fn parse(text: &str, accounts: &AccountsConfig) -> Result<KeyStore, String> {
        let file: KeysFile = serde_json::from_str(text).map_err(|error| error.to_string())?;
        KeyStore::build(&file, accounts)
    }

    pub fn build(file: &KeysFile, accounts: &AccountsConfig) -> Result<KeyStore, String> {
        let mut ids = HashSet::new();
        let mut entries = Vec::with_capacity(file.keys.len());
        let mut by_hash = HashMap::with_capacity(file.keys.len());
        for record in &file.keys {
            let entry = KeyEntry::build(record, accounts)
                .map_err(|error| format!("key '{}': {error}", record.id))?;
            if !ids.insert(record.id.as_str()) {
                return Err(format!("duplicate key id '{}'", record.id));
            }
            let hash = ApiKeyHash::try_from(record.key_hash.clone())
                .map_err(|error| format!("key '{}': {error}", record.id))?;
            if accounts.has_key_hash(&hash) {
                return Err(format!(
                    "key '{}' uses the same API key as an account in the accounts file",
                    record.id
                ));
            }
            if by_hash.insert(hash, entries.len()).is_some() {
                return Err(format!(
                    "key '{}' uses the same API key as another key",
                    record.id
                ));
            }
            entries.push(Arc::new(entry));
        }
        Ok(KeyStore { entries, by_hash })
    }

    pub fn lookup(&self, api_key: &str) -> Option<&Arc<KeyEntry>> {
        let index = self.by_hash.get(&ApiKeyHash::from_api_key(api_key))?;
        Some(&self.entries[*index])
    }

    pub fn entries(&self) -> &[Arc<KeyEntry>] {
        &self.entries
    }
}

impl KeyEntry {
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
        let class_index = |name: &String| {
            accounts
                .class_index(name)
                .ok_or_else(|| format!("unknown class '{name}'"))
        };
        let expires_at = match &record.expires_at {
            Some(text) => Some(
                text.parse()
                    .map_err(|error| format!("invalid expires_at '{text}': {error}"))?,
            ),
            None => None,
        };
        let routes = record.allowed_routes.as_ref();
        Ok(KeyEntry {
            account: Account {
                name: record.user.clone(),
                default_class: class_index(&record.default_class)?,
                allowed_classes: record
                    .allowed_classes
                    .iter()
                    .map(class_index)
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

pub struct NewKey {
    pub api_key: String,
    pub hash: String,
    pub prefix: String,
}

pub fn generate_key() -> std::io::Result<NewKey> {
    let api_key = format!("sk-{}", random_hex(KEY_SECRET_BYTES)?);
    Ok(NewKey {
        hash: ApiKeyHash::from_api_key(&api_key).to_hex(),
        prefix: api_key[..KEY_PREFIX_LENGTH].to_string(),
        api_key,
    })
}

pub fn generate_key_id() -> std::io::Result<String> {
    Ok(format!("key-{}", random_hex(KEY_ID_BYTES)?))
}

fn random_hex(length: usize) -> std::io::Result<String> {
    let mut bytes = vec![0u8; length];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

pub fn read_keys_file(path: &Path) -> std::io::Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

pub fn write_keys_file(path: &Path, file: &KeysFile) -> std::io::Result<String> {
    let mut text = serde_json::to_string_pretty(file).expect("unserializable keys file");
    text.push('\n');
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("keys.json");
    let temporary = path.with_file_name(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        jiff::Timestamp::now().as_nanosecond()
    ));
    let result = (|| {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        output.write_all(text.as_bytes())?;
        if let Ok(metadata) = std::fs::metadata(path) {
            output.set_permissions(metadata.permissions())?;
        }
        output.sync_all()?;
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

    fn record(id: &str, api_key: &str, settings: serde_json::Value) -> KeyRecord {
        let mut record = serde_json::json!({
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
        let none = serde_json::json!({});
        let store = build(vec![
            record("a", "sk-a", none.clone()),
            record("b", "sk-b", none.clone()),
        ])
        .unwrap();
        assert_eq!(store.lookup("sk-b").unwrap().record.id, "b");
        assert_eq!(
            store.lookup("sk-a").unwrap().account.allowed_classes,
            vec![1]
        );
        assert!(store.lookup("sk-c").is_none());
        let error = |keys| build(keys).unwrap_err();
        assert!(error(vec![
            record("a", "sk-a", none.clone()),
            record("a", "sk-b", none.clone())
        ])
        .contains("duplicate key id"));
        assert!(error(vec![
            record("a", "sk-a", none.clone()),
            record("b", "sk-a", none.clone())
        ])
        .contains("another key"));
        assert!(error(vec![record("a", "sk-static", none.clone())]).contains("accounts file"));
        for (settings, message) in [
            (
                serde_json::json!({"default_class": "gold"}),
                "unknown class 'gold'",
            ),
            (
                serde_json::json!({"user": "litellm/bob"}),
                "must not be empty or contain",
            ),
            (serde_json::json!({"id": "a b"}), "the id must consist"),
            (
                serde_json::json!({"expires_at": "tomorrow"}),
                "invalid expires_at",
            ),
            (
                serde_json::json!({"key_hash": "abc"}),
                "invalid api_key_hash",
            ),
        ] {
            assert!(
                error(vec![record("a", "sk-a", settings)]).contains(message),
                "{message}"
            );
        }
    }

    #[test]
    fn states_and_routes() {
        let store = build(vec![
            record(
                "a",
                "sk-a",
                serde_json::json!({"expires_at": "2020-01-01T00:00:00Z"}),
            ),
            record("b", "sk-b", serde_json::json!({"blocked": true})),
            record(
                "c",
                "sk-c",
                serde_json::json!({"allowed_routes": ["/v1/chat/completions", "/v1/models/*"]}),
            ),
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
        let key = generate_key().unwrap();
        assert!(key.api_key.starts_with(&key.prefix) && key.api_key.len() == 51);
        assert_eq!(key.hash, ApiKeyHash::from_api_key(&key.api_key).to_hex());
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("keys.json");
        assert_eq!(read_keys_file(&path).unwrap(), None);
        let file = KeysFile {
            keys: vec![record("a", "sk-a", serde_json::json!({}))],
        };
        let text = write_keys_file(&path, &file).unwrap();
        assert_eq!(serde_json::from_str::<KeysFile>(&text).unwrap(), file);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        write_keys_file(&path, &file).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
