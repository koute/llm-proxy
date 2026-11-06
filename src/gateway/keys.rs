use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::{matches_any, Account, AccountsConfig, ApiKeyHash, PathPattern};
use crate::quality_of_service::USER_ID_SEPARATOR;

const KEY_SECRET_BYTES: usize = 24;
const KEY_ID_BYTES: usize = 6;
const KEY_PREFIX_LENGTH: usize = 8;
const MAX_NAME_LENGTH: usize = 128;
const MAX_DESCRIPTION_LENGTH: usize = 500;

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
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub limits: KeyLimits,
    #[serde(default)]
    pub allowed_routes: Option<Vec<String>>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
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
    #[serde(default)]
    pub max_parallel_requests: Option<NonZeroU64>,
    #[serde(default)]
    pub rpm_limit: Option<NonZeroU64>,
    #[serde(default)]
    pub tpm_limit: Option<NonZeroU64>,
    #[serde(default)]
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

impl KeyEntry {
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

    pub fn file(&self) -> KeysFile {
        KeysFile {
            keys: self
                .entries
                .iter()
                .map(|entry| entry.record.clone())
                .collect(),
        }
    }
}

impl KeyEntry {
    fn build(record: &KeyRecord, accounts: &AccountsConfig) -> Result<KeyEntry, String> {
        validate_id(&record.id)?;
        if record.user.is_empty() || record.user.len() > MAX_NAME_LENGTH {
            return Err(format!(
                "the user must have between 1 and {MAX_NAME_LENGTH} characters"
            ));
        }
        if record.user.contains(USER_ID_SEPARATOR) || record.user.chars().any(char::is_control) {
            return Err(format!(
                "the user must not contain '{USER_ID_SEPARATOR}' or control characters"
            ));
        }
        if record.description.len() > MAX_DESCRIPTION_LENGTH {
            return Err(format!(
                "the description must not be longer than {MAX_DESCRIPTION_LENGTH} characters"
            ));
        }
        let class_index = |name: &str| {
            accounts
                .class_index(name)
                .ok_or_else(|| format!("unknown class '{name}'"))
        };
        let account = Account {
            name: record.user.clone(),
            default_class: class_index(&record.default_class)?,
            allowed_classes: record
                .allowed_classes
                .iter()
                .map(|name| class_index(name))
                .collect::<Result<_, _>>()?,
            virtual_user_http_header: None,
        };
        let expires_at = record
            .expires_at
            .as_deref()
            .map(|text| {
                text.parse::<jiff::Timestamp>()
                    .map_err(|error| format!("invalid expires_at '{text}': {error}"))
            })
            .transpose()?;
        let allowed_routes = record.allowed_routes.as_ref().map(|routes| {
            routes
                .iter()
                .map(|route| PathPattern::from(route.clone()))
                .collect()
        });
        if let Some(routes) = &record.allowed_routes {
            if let Some(route) = routes.iter().find(|route| !route.starts_with('/')) {
                return Err(format!("allowed route '{route}' must start with '/'"));
            }
        }
        Ok(KeyEntry {
            record: record.clone(),
            account,
            expires_at,
            allowed_routes,
        })
    }
}

fn validate_id(id: &str) -> Result<(), String> {
    let is_valid = !id.is_empty()
        && id.len() <= MAX_NAME_LENGTH
        && id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "-_.".contains(character));
    if !is_valid {
        return Err(
            "the id must consist of letters, digits, '-', '_' and '.' (at most 128)".to_string(),
        );
    }
    Ok(())
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

pub fn now_text() -> String {
    format!("{:.0}", jiff::Timestamp::now())
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

    fn accounts() -> AccountsConfig {
        AccountsConfig::parse(
            r#"{"classes": [{"name": "normal", "priority": 0}, {"name": "idle", "priority": -1}],
                "accounts": [{"name": "static", "api_key": "sk-static", "default_class": "normal"}]}"#,
        )
        .unwrap()
    }

    fn record(id: &str, api_key: &str) -> KeyRecord {
        KeyRecord {
            id: id.to_string(),
            user: "alice".to_string(),
            description: String::new(),
            key_hash: ApiKeyHash::from_api_key(api_key).to_hex(),
            key_prefix: String::new(),
            role: Role::User,
            default_class: "normal".to_string(),
            allowed_classes: vec!["idle".to_string()],
            blocked: false,
            expires_at: None,
            limits: KeyLimits::default(),
            allowed_routes: None,
            created_at: None,
            created_by: None,
        }
    }

    fn build(keys: Vec<KeyRecord>) -> Result<KeyStore, String> {
        KeyStore::build(&KeysFile { keys }, &accounts())
    }

    #[test]
    fn lookup_by_api_key() {
        let store = build(vec![record("a", "sk-a"), record("b", "sk-b")]).unwrap();
        assert_eq!(store.lookup("sk-b").unwrap().record.id, "b");
        assert!(store.lookup("sk-c").is_none());
        let entry = store.lookup("sk-a").unwrap();
        assert_eq!(entry.account.name, "alice");
        assert_eq!(entry.account.allowed_classes, vec![1]);
    }

    #[test]
    fn validation_errors() {
        let error = |keys: Vec<KeyRecord>| build(keys).unwrap_err();
        assert!(error(vec![record("a", "sk-a"), record("a", "sk-b")]).contains("duplicate key id"));
        assert!(error(vec![record("a", "sk-a"), record("b", "sk-a")]).contains("another key"));
        assert!(error(vec![record("a", "sk-static")]).contains("accounts file"));
        let mut unknown_class = record("a", "sk-a");
        unknown_class.default_class = "gold".to_string();
        assert!(error(vec![unknown_class]).contains("unknown class 'gold'"));
        let mut bad_user = record("a", "sk-a");
        bad_user.user = "litellm/bob".to_string();
        assert!(error(vec![bad_user]).contains("must not contain"));
        let mut bad_expiry = record("a", "sk-a");
        bad_expiry.expires_at = Some("tomorrow".to_string());
        assert!(error(vec![bad_expiry]).contains("invalid expires_at"));
        assert!(error(vec![record("a b", "sk-a")]).contains("the id must"));
        let mut bad_hash = record("a", "sk-a");
        bad_hash.key_hash = "abc".to_string();
        assert!(error(vec![bad_hash]).contains("invalid api_key_hash"));
        let mut bad_route = record("a", "sk-a");
        bad_route.allowed_routes = Some(vec!["v1/models".to_string()]);
        assert!(error(vec![bad_route]).contains("must start with '/'"));
    }

    #[test]
    fn states_and_routes() {
        let mut expired = record("a", "sk-a");
        expired.expires_at = Some("2020-01-01T00:00:00Z".to_string());
        let mut blocked = record("b", "sk-b");
        blocked.blocked = true;
        let mut limited = record("c", "sk-c");
        limited.allowed_routes = Some(vec![
            "/v1/chat/completions".to_string(),
            "/v1/models/*".to_string(),
        ]);
        let store = build(vec![expired, blocked, limited]).unwrap();
        let now = jiff::Timestamp::now();
        assert_eq!(store.lookup("sk-a").unwrap().state(now), KeyState::Expired);
        assert_eq!(store.lookup("sk-b").unwrap().state(now), KeyState::Blocked);
        let limited = store.lookup("sk-c").unwrap();
        assert_eq!(limited.state(now), KeyState::Active);
        assert!(limited.allows_route("/v1/chat/completions"));
        assert!(limited.allows_route("/v1/models/foo"));
        assert!(!limited.allows_route("/v1/completions"));
        assert!(store.lookup("sk-a").unwrap().allows_route("/anything"));
    }

    #[test]
    fn generated_keys() {
        let key = generate_key().unwrap();
        assert!(key.api_key.starts_with("sk-"));
        assert_eq!(key.api_key.len(), 3 + 2 * KEY_SECRET_BYTES);
        assert_eq!(key.hash, ApiKeyHash::from_api_key(&key.api_key).to_hex());
        assert!(key.api_key.starts_with(&key.prefix));
        assert_ne!(generate_key().unwrap().api_key, key.api_key);
        assert!(validate_id(&generate_key_id().unwrap()).is_ok());
    }

    #[test]
    fn atomic_write_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("keys.json");
        assert_eq!(read_keys_file(&path).unwrap(), None);
        let file = KeysFile {
            keys: vec![record("a", "sk-a")],
        };
        let text = write_keys_file(&path, &file).unwrap();
        assert_eq!(
            read_keys_file(&path).unwrap().as_deref(),
            Some(text.as_str())
        );
        let parsed: KeysFile = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed, file);
        let names: Vec<_> = std::fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(names, vec!["keys.json"]);
    }
}
