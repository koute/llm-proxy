use std::collections::{HashMap, HashSet};
use std::fmt;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use http::uri::Authority;
use http::HeaderName;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::engine_protocol::EngineKind;
use crate::quality_of_service::USER_ID_SEPARATOR;

const MAX_SECONDS: f64 = 365.0 * 24.0 * 3600.0;
const MAX_USAGE_HALF_LIFE_MINUTES: u32 = 365 * 24 * 60;

const DEFAULT_GENERATING_PATHS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/messages",
    "/generate",
];

const DEFAULT_WHITELIST: &[&str] = &[
    "/v1/messages/count_tokens",
    "/v1/models",
    "/v1/models/*",
    "/v1/embeddings",
    "/tokenize",
    "/detokenize",
    "/health",
    "/version",
];

#[derive(Clone, Debug, PartialEq, Eq, clap::Args)]
pub struct ConfigPaths {
    #[arg(long, default_value = "config/general.json")]
    pub general: PathBuf,
    #[arg(long, default_value = "config/accounts.json")]
    pub accounts: PathBuf,
    #[arg(long, default_value = "config/backend.json")]
    pub backend: PathBuf,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub general: GeneralConfig,
    pub accounts: AccountsConfig,
    pub backend: Option<BackendConfig>,
    pub rank_capacities: Vec<usize>,
    pub texts: ConfigTexts,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigTexts {
    pub general: String,
    pub accounts: String,
    pub backend: Option<String>,
}

impl ConfigTexts {
    pub fn read(paths: &ConfigPaths) -> Result<ConfigTexts, String> {
        let backend = match std::fs::read_to_string(&paths.backend) {
            Ok(text) => Some(text),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(read_error(&paths.backend, &error)),
        };
        Ok(ConfigTexts {
            general: std::fs::read_to_string(&paths.general)
                .map_err(|error| read_error(&paths.general, &error))?,
            accounts: std::fs::read_to_string(&paths.accounts)
                .map_err(|error| read_error(&paths.accounts, &error))?,
            backend,
        })
    }
}

impl Config {
    pub fn load(paths: &ConfigPaths) -> Result<Config, String> {
        Config::parse(&ConfigTexts::read(paths)?, paths)
    }

    pub fn parse(texts: &ConfigTexts, paths: &ConfigPaths) -> Result<Config, String> {
        let general = GeneralConfig::parse(&texts.general)
            .map_err(|error| invalid_file_error(&paths.general, &error))?;
        let accounts = AccountsConfig::parse(&texts.accounts)
            .map_err(|error| invalid_file_error(&paths.accounts, &error))?;
        let backend = texts
            .backend
            .as_deref()
            .map(BackendConfig::parse)
            .transpose()
            .map_err(|error| invalid_file_error(&paths.backend, &error))?;
        let rank_capacities = match &backend {
            Some(backend) => {
                rank_capacities(&general.quality_of_service, backend).map_err(|error| {
                    format!(
                        "invalid combination of '{}' and '{}': {error}",
                        paths.general.display(),
                        paths.backend.display()
                    )
                })?
            }
            None => Vec::new(),
        };
        Ok(Config {
            general,
            accounts,
            backend,
            rank_capacities,
            texts: texts.clone(),
        })
    }
}

fn read_error(path: &Path, error: &std::io::Error) -> String {
    format!("failed to read '{}': {error}", path.display())
}

fn invalid_file_error(path: &Path, error: &str) -> String {
    format!("invalid '{}': {error}", path.display())
}

fn parse_json<T: DeserializeOwned>(text: &str) -> Result<T, String> {
    serde_json::from_str(text).map_err(|error| error.to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Seconds(Duration);

impl Seconds {
    pub const fn new(seconds: u64) -> Seconds {
        Seconds(Duration::from_secs(seconds))
    }

    pub fn duration(self) -> Duration {
        self.0
    }
}

impl TryFrom<f64> for Seconds {
    type Error = String;

    fn try_from(seconds: f64) -> Result<Seconds, String> {
        if !(0.0..=MAX_SECONDS).contains(&seconds) {
            return Err(format!(
                "seconds must be between 0 and {MAX_SECONDS}, not {seconds}"
            ));
        }
        Ok(Seconds(Duration::from_secs_f64(seconds)))
    }
}

impl FromStr for Seconds {
    type Err = String;

    fn from_str(text: &str) -> Result<Seconds, String> {
        let seconds: f64 = text
            .parse()
            .map_err(|_| format!("invalid number of seconds '{text}'"))?;
        Seconds::try_from(seconds)
    }
}

impl From<Seconds> for f64 {
    fn from(seconds: Seconds) -> f64 {
        seconds.0.as_secs_f64()
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralConfig {
    pub frontend: FrontendConfig,
    pub middle_end_socket: PathBuf,
    #[serde(default)]
    pub paths: PathsConfig,
    #[serde(default)]
    pub limits: LimitsConfig,
    #[serde(default)]
    pub quality_of_service: QualityOfServiceConfig,
    #[serde(default)]
    pub process: ProcessConfig,
    #[serde(default)]
    pub scheduling_log: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendConfig {
    pub listen: SocketAddr,
    #[serde(default)]
    pub tls: Option<TlsConfig>,
}

#[cfg(feature = "tls")]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum TlsConfig {
    Files {
        certificate: PathBuf,
        private_key: PathBuf,
    },
    SelfSigned(SelfSignedMarker),
}

#[cfg(not(feature = "tls"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TlsConfig {}

#[cfg(not(feature = "tls"))]
impl<'de> Deserialize<'de> for TlsConfig {
    fn deserialize<D: serde::Deserializer<'de>>(_: D) -> Result<TlsConfig, D::Error> {
        Err(serde::de::Error::custom(
            "frontend.tls requires a build with the 'tls' cargo feature",
        ))
    }
}

#[cfg(feature = "tls")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum SelfSignedMarker {
    #[serde(rename = "self_signed")]
    SelfSigned,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PathsConfig {
    generating: Vec<PathPattern>,
    whitelist: Vec<PathPattern>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathKind {
    Generating,
    Passthrough,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(from = "String")]
pub enum PathPattern {
    Exact(String),
    Prefix(String),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LimitsConfig {
    pub max_request_body_bytes: usize,
    pub request_body_timeout_seconds: Seconds,
    pub max_queued_requests: usize,
    pub keepalive_interval_seconds: Seconds,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct QualityOfServiceConfig {
    pub minimum_peak_throughput: f64,
    pub peak_throughput_tolerance: f64,
    pub maximum_parallel_requests: Option<usize>,
    pub swap_hysteresis_seconds: Seconds,
    pub usage_half_life_minutes: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ProcessConfig {
    pub restart_flush_timeout_seconds: Seconds,
    pub shutdown_drain_timeout_seconds: Seconds,
}

impl GeneralConfig {
    pub fn parse(text: &str) -> Result<GeneralConfig, String> {
        let config: GeneralConfig = parse_json(text)?;
        config.quality_of_service.validate()?;
        if config
            .limits
            .keepalive_interval_seconds
            .duration()
            .is_zero()
        {
            return Err("limits.keepalive_interval_seconds must be greater than zero".to_string());
        }
        Ok(config)
    }
}

impl QualityOfServiceConfig {
    fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.minimum_peak_throughput) {
            return Err(format!(
                "quality_of_service.minimum_peak_throughput must be between 0.0 and 1.0, not {}",
                self.minimum_peak_throughput
            ));
        }
        if !(0.0..1.0).contains(&self.peak_throughput_tolerance) {
            return Err(format!(
                "quality_of_service.peak_throughput_tolerance must be at least 0.0 and below 1.0, not {}",
                self.peak_throughput_tolerance
            ));
        }
        if self.maximum_parallel_requests == Some(0) {
            return Err(
                "quality_of_service.maximum_parallel_requests must be greater than zero"
                    .to_string(),
            );
        }
        if !(1..=MAX_USAGE_HALF_LIFE_MINUTES).contains(&self.usage_half_life_minutes) {
            return Err(format!(
                "quality_of_service.usage_half_life_minutes must be between 1 and {MAX_USAGE_HALF_LIFE_MINUTES}, not {}",
                self.usage_half_life_minutes
            ));
        }
        Ok(())
    }
}

impl PathsConfig {
    pub fn classify(&self, path: &str) -> Option<PathKind> {
        let canonical_path = normalize_path(path);
        let matches = |patterns: &[PathPattern]| {
            patterns
                .iter()
                .any(|pattern| pattern.matches(&canonical_path))
        };
        if matches(&self.generating) {
            Some(PathKind::Generating)
        } else if matches(&self.whitelist) {
            Some(PathKind::Passthrough)
        } else {
            None
        }
    }
}

impl PathPattern {
    fn matches(&self, canonical_path: &str) -> bool {
        match self {
            PathPattern::Exact(path) => path == canonical_path,
            PathPattern::Prefix(prefix) => canonical_path
                .strip_prefix(prefix.as_str())
                .is_some_and(|rest| rest.starts_with('/')),
        }
    }
}

impl From<String> for PathPattern {
    fn from(pattern: String) -> PathPattern {
        match pattern.strip_suffix("/*") {
            Some(prefix) => PathPattern::Prefix(normalize_path(prefix)),
            None => PathPattern::Exact(normalize_path(&pattern)),
        }
    }
}

impl Default for PathsConfig {
    fn default() -> PathsConfig {
        let patterns = |texts: &[&str]| {
            texts
                .iter()
                .map(|text| PathPattern::from(text.to_string()))
                .collect()
        };
        PathsConfig {
            generating: patterns(DEFAULT_GENERATING_PATHS),
            whitelist: patterns(DEFAULT_WHITELIST),
        }
    }
}

impl Default for LimitsConfig {
    fn default() -> LimitsConfig {
        LimitsConfig {
            max_request_body_bytes: 64 * 1024 * 1024,
            request_body_timeout_seconds: Seconds::new(30),
            max_queued_requests: 1024,
            keepalive_interval_seconds: Seconds::new(15),
        }
    }
}

impl Default for QualityOfServiceConfig {
    fn default() -> QualityOfServiceConfig {
        QualityOfServiceConfig {
            minimum_peak_throughput: 1.0,
            peak_throughput_tolerance: 0.05,
            maximum_parallel_requests: None,
            swap_hysteresis_seconds: Seconds::new(30),
            usage_half_life_minutes: 60,
        }
    }
}

impl Default for ProcessConfig {
    fn default() -> ProcessConfig {
        ProcessConfig {
            restart_flush_timeout_seconds: Seconds::new(5),
            shutdown_drain_timeout_seconds: Seconds::new(30),
        }
    }
}

fn normalize_path(path: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    let mut result = String::with_capacity(path.len());
    for segment in segments {
        result.push('/');
        result.push_str(segment);
    }
    result
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassConfig {
    pub name: String,
    pub priority: i32,
    #[serde(default)]
    pub idle_only: bool,
    #[serde(default = "default_cost")]
    pub cost: f32,
    #[serde(default)]
    pub soft_cap: Option<NonZeroUsize>,
    #[serde(default)]
    pub max_parallel_per_user: Option<NonZeroUsize>,
}

#[derive(Clone, Debug)]
pub struct Account {
    pub name: String,
    pub default_class: usize,
    pub allowed_classes: Vec<usize>,
    pub virtual_user_http_header: Option<HeaderName>,
}

#[derive(Clone, Debug)]
pub struct AccountsConfig {
    pub classes: Vec<ClassConfig>,
    accounts_by_key: HashMap<String, Account>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ClassError {
    Unknown(String),
    Forbidden(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AccountsFile {
    classes: Vec<ClassConfig>,
    accounts: Vec<AccountEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AccountEntry {
    name: String,
    api_key: String,
    default_class: String,
    #[serde(default)]
    allowed_classes: Vec<String>,
    #[serde(default)]
    virtual_user_http_header: Option<String>,
}

fn default_cost() -> f32 {
    1.0
}

impl AccountsConfig {
    pub fn parse(text: &str) -> Result<AccountsConfig, String> {
        let file: AccountsFile = parse_json(text)?;
        validate_classes(&file.classes)?;
        let class_index = |name: &str| {
            file.classes
                .iter()
                .position(|class| class.name == name)
                .ok_or_else(|| format!("unknown class '{name}'"))
        };
        let mut account_names = HashSet::new();
        let mut accounts_by_key = HashMap::new();
        for entry in &file.accounts {
            if !account_names.insert(entry.name.as_str()) {
                return Err(format!("duplicate account name '{}'", entry.name));
            }
            let account_error = |error: String| format!("account '{}': {error}", entry.name);
            if entry.name.contains(USER_ID_SEPARATOR) {
                return Err(account_error(format!(
                    "the name must not contain '{USER_ID_SEPARATOR}'"
                )));
            }
            let virtual_user_http_header = entry
                .virtual_user_http_header
                .as_deref()
                .map(|name| {
                    HeaderName::try_from(name).map_err(|error| {
                        account_error(format!(
                            "invalid virtual_user_http_header '{name}': {error}"
                        ))
                    })
                })
                .transpose()?;
            let account = Account {
                name: entry.name.clone(),
                default_class: class_index(&entry.default_class).map_err(account_error)?,
                allowed_classes: entry
                    .allowed_classes
                    .iter()
                    .map(|name| class_index(name))
                    .collect::<Result<_, _>>()
                    .map_err(account_error)?,
                virtual_user_http_header,
            };
            if accounts_by_key
                .insert(entry.api_key.clone(), account)
                .is_some()
            {
                return Err(format!(
                    "account '{}' uses the same api_key as another account",
                    entry.name
                ));
            }
        }
        Ok(AccountsConfig {
            classes: file.classes,
            accounts_by_key,
        })
    }

    pub fn account(&self, api_key: &str) -> Option<&Account> {
        self.accounts_by_key.get(api_key)
    }

    pub fn class(
        &self,
        account: &Account,
        requested_name: Option<&str>,
    ) -> Result<&ClassConfig, ClassError> {
        let Some(name) = requested_name else {
            return Ok(&self.classes[account.default_class]);
        };
        let index = self
            .classes
            .iter()
            .position(|class| class.name == name)
            .ok_or_else(|| ClassError::Unknown(name.to_string()))?;
        if index != account.default_class && !account.allowed_classes.contains(&index) {
            return Err(ClassError::Forbidden(name.to_string()));
        }
        Ok(&self.classes[index])
    }
}

fn validate_classes(classes: &[ClassConfig]) -> Result<(), String> {
    if classes.is_empty() {
        return Err("at least one class is required".to_string());
    }
    let mut class_names = HashSet::new();
    for class in classes {
        if !class_names.insert(class.name.as_str()) {
            return Err(format!("duplicate class name '{}'", class.name));
        }
        if !(class.cost.is_finite() && class.cost >= 0.0) {
            return Err(format!(
                "class '{}': cost must be a finite number of at least 0.0",
                class.name
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    pub engine: EngineKind,
    pub model: String,
    pub engine_max_parallel_requests_per_rank: usize,
    #[serde(default)]
    pub priority_scheduling: bool,
    pub ranks: Vec<RankConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark: Option<BenchmarkResult>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankConfig {
    pub upstream: UpstreamAddress,
    pub control_socket: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_parallel_rank: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct UpstreamAddress {
    authority: Authority,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkResult {
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub single_stream_tokens_per_second: f64,
    pub measurements: Vec<BenchmarkMeasurement>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkMeasurement {
    pub parallel_requests_per_rank: usize,
    pub tokens_per_second_per_request: f64,
    pub tokens_per_second_per_rank: f64,
}

impl BackendConfig {
    pub fn parse(text: &str) -> Result<BackendConfig, String> {
        let config: BackendConfig = parse_json(text)?;
        if config.ranks.is_empty() {
            return Err("at least one rank is required".to_string());
        }
        if config.engine_max_parallel_requests_per_rank == 0 {
            return Err(
                "engine_max_parallel_requests_per_rank must be greater than zero".to_string(),
            );
        }
        let mut data_parallel_ranks = HashSet::new();
        for rank in &config.ranks {
            if let Some(data_parallel_rank) = rank.data_parallel_rank {
                if !data_parallel_ranks.insert(data_parallel_rank) {
                    return Err(format!("duplicate data_parallel_rank {data_parallel_rank}"));
                }
            }
        }
        if let Some(benchmark) = &config.benchmark {
            benchmark.validate()?;
        }
        Ok(config)
    }

    pub fn to_json(&self) -> String {
        let mut text = serde_json::to_string_pretty(self).expect("unserializable backend config");
        text.push('\n');
        text
    }
}

impl BenchmarkResult {
    fn validate(&self) -> Result<(), String> {
        if self.measurements.is_empty() {
            return Err("benchmark.measurements must not be empty".to_string());
        }
        let mut previous_parallel_requests = 0;
        for measurement in &self.measurements {
            if measurement.parallel_requests_per_rank <= previous_parallel_requests {
                return Err(
                    "benchmark.measurements must be sorted by strictly increasing parallel_requests_per_rank, starting at 1 or more"
                        .to_string(),
                );
            }
            previous_parallel_requests = measurement.parallel_requests_per_rank;
            let rates = [
                measurement.tokens_per_second_per_request,
                measurement.tokens_per_second_per_rank,
            ];
            if !rates.iter().all(|rate| rate.is_finite() && *rate >= 0.0) {
                return Err(format!(
                    "benchmark measurement for {} parallel requests contains an invalid rate",
                    measurement.parallel_requests_per_rank
                ));
            }
        }
        Ok(())
    }
}

impl UpstreamAddress {
    pub fn authority(&self) -> &str {
        self.authority.as_str()
    }
}

impl TryFrom<String> for UpstreamAddress {
    type Error = String;

    fn try_from(url: String) -> Result<UpstreamAddress, String> {
        let invalid = || {
            format!(
                "invalid upstream '{url}': use http://host:port, for example http://127.0.0.1:8000"
            )
        };
        let authority: Authority = url
            .strip_prefix("http://")
            .map(|rest| rest.trim_end_matches('/'))
            .and_then(|authority| authority.parse().ok())
            .filter(|authority: &Authority| !authority.as_str().contains('@'))
            .ok_or_else(invalid)?;
        Ok(UpstreamAddress { authority })
    }
}

impl From<UpstreamAddress> for String {
    fn from(address: UpstreamAddress) -> String {
        address.to_string()
    }
}

impl fmt::Display for UpstreamAddress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "http://{}", self.authority)
    }
}

pub fn rank_capacities(
    settings: &QualityOfServiceConfig,
    backend: &BackendConfig,
) -> Result<Vec<usize>, String> {
    let rank_count = backend.ranks.len();
    let throughput_cap = backend.benchmark.as_ref().map_or(usize::MAX, |benchmark| {
        throughput_limit(
            &benchmark.measurements,
            settings.minimum_peak_throughput,
            settings.peak_throughput_tolerance,
        )
    });
    let rank_cap = backend
        .engine_max_parallel_requests_per_rank
        .min(throughput_cap);
    let Some(maximum) = settings.maximum_parallel_requests else {
        return Ok(vec![rank_cap; rank_count]);
    };
    if maximum < rank_count {
        return Err(format!(
            "quality_of_service.maximum_parallel_requests ({maximum}) is lower than the number of ranks ({rank_count})"
        ));
    }
    Ok((0..rank_count)
        .map(|rank| rank_cap.min(rank_share(maximum, rank_count, rank)))
        .collect())
}

pub fn rank_share(total: usize, rank_count: usize, rank: usize) -> usize {
    total / rank_count + usize::from(rank < total % rank_count)
}

pub(crate) fn throughput_limit(
    measurements: &[BenchmarkMeasurement],
    minimum_peak_throughput: f64,
    tolerance: f64,
) -> usize {
    let peak = measurements
        .iter()
        .map(|measurement| measurement.tokens_per_second_per_rank)
        .fold(0.0, f64::max);
    let threshold = peak * minimum_peak_throughput * (1.0 - tolerance);
    let largest_measured = measurements
        .last()
        .map_or(1, |measurement| measurement.parallel_requests_per_rank);
    (1..=largest_measured)
        .rev()
        .find(|&parallel_requests| interpolated_rate(measurements, parallel_requests) >= threshold)
        .unwrap_or(1)
}

fn interpolated_rate(measurements: &[BenchmarkMeasurement], parallel_requests: usize) -> f64 {
    let upper_index = measurements
        .iter()
        .position(|measurement| measurement.parallel_requests_per_rank >= parallel_requests)
        .expect("parallel request count above the measured range");
    let upper = &measurements[upper_index];
    let (lower_requests, lower_rate) = match upper_index {
        0 => (0, 0.0),
        _ => {
            let lower = &measurements[upper_index - 1];
            (
                lower.parallel_requests_per_rank,
                lower.tokens_per_second_per_rank,
            )
        }
    };
    let fraction = (parallel_requests - lower_requests) as f64
        / (upper.parallel_requests_per_rank - lower_requests) as f64;
    lower_rate + fraction * (upper.tokens_per_second_per_rank - lower_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_paths() -> ConfigPaths {
        ConfigPaths {
            general: "config/general.example.json".into(),
            accounts: "config/accounts.example.json".into(),
            backend: "config/backend.example.json".into(),
        }
    }

    fn accounts(accounts_json: &str) -> Result<AccountsConfig, String> {
        AccountsConfig::parse(&format!(
            r#"{{"classes": [{{"name": "normal", "priority": 0}}, {{"name": "idle", "priority": -1, "idle_only": true}}, {{"name": "urgent", "priority": 5}}],
                "accounts": {accounts_json}}}"#
        ))
    }

    fn general(quality_of_service_json: &str) -> Result<GeneralConfig, String> {
        GeneralConfig::parse(&format!(
            r#"{{"frontend": {{"listen": "127.0.0.1:8080"}}, "middle_end_socket": "/tmp/middle.sock",
                "quality_of_service": {quality_of_service_json}}}"#
        ))
    }

    fn backend(ranks_json: &str, measurements_json: &str) -> Result<BackendConfig, String> {
        BackendConfig::parse(&format!(
            r#"{{"engine": "sglang", "model": "m", "engine_max_parallel_requests_per_rank": 8, "ranks": {ranks_json},
                "benchmark": {{"prompt_tokens": 1, "output_tokens": 1, "single_stream_tokens_per_second": 1.0,
                              "measurements": {measurements_json}}}}}"#
        ))
    }

    #[test]
    fn example_config_files() {
        let config = Config::load(&example_paths()).unwrap();
        assert_eq!(config.rank_capacities, vec![64]);
        let backend = config.backend.unwrap();
        assert_eq!(backend.ranks[0].upstream.authority(), "127.0.0.1:8000");
        assert_eq!(config.accounts.classes.len(), 3);
    }

    #[test]
    fn missing_backend_file() {
        let paths = ConfigPaths {
            backend: "config/missing-backend.json".into(),
            ..example_paths()
        };
        assert!(Config::load(&paths).unwrap().backend.is_none());
    }

    #[test]
    fn account_class_selection() {
        let config = accounts(
            r#"[{"name": "alice", "api_key": "sk-a", "default_class": "normal", "allowed_classes": ["idle"]},
                {"name": "guest", "api_key": "", "default_class": "idle"}]"#,
        )
        .unwrap();
        let alice = config.account("sk-a").unwrap();
        assert_eq!(config.class(alice, None).unwrap().name, "normal");
        assert_eq!(config.class(alice, Some("idle")).unwrap().name, "idle");
        assert_eq!(
            config.class(alice, Some("urgent")),
            Err(ClassError::Forbidden("urgent".to_string()))
        );
        assert_eq!(
            config.class(alice, Some("missing")),
            Err(ClassError::Unknown("missing".to_string()))
        );
        let guest = config.account("").unwrap();
        assert_eq!(config.class(guest, Some("idle")).unwrap().name, "idle");
        assert!(config.account("sk-unknown").is_none());
    }

    #[test]
    fn account_validation_errors() {
        let duplicate_key = accounts(
            r#"[{"name": "a", "api_key": "k", "default_class": "normal"},
                {"name": "b", "api_key": "k", "default_class": "normal"}]"#,
        );
        assert!(duplicate_key.unwrap_err().contains("same api_key"));
        let unknown_class = accounts(r#"[{"name": "a", "api_key": "k", "default_class": "gold"}]"#);
        assert!(unknown_class.unwrap_err().contains("unknown class 'gold'"));
        let invalid_header = accounts(
            r#"[{"name": "a", "api_key": "k", "default_class": "normal", "virtual_user_http_header": "x user"}]"#,
        );
        assert!(invalid_header
            .unwrap_err()
            .contains("invalid virtual_user_http_header"));
        let duplicate_class = AccountsConfig::parse(
            r#"{"classes": [{"name": "a", "priority": 0}, {"name": "a", "priority": 1}], "accounts": []}"#,
        );
        assert!(duplicate_class
            .unwrap_err()
            .contains("duplicate class name 'a'"));
        let separator_in_name =
            accounts(r#"[{"name": "litellm/bob", "api_key": "k", "default_class": "normal"}]"#);
        assert!(separator_in_name.unwrap_err().contains("'/'"));
        let zero_soft_cap = AccountsConfig::parse(
            r#"{"classes": [{"name": "a", "priority": 0, "soft_cap": 0}], "accounts": []}"#,
        );
        assert!(zero_soft_cap.unwrap_err().contains("nonzero"));
        let zero_max_parallel_per_user = AccountsConfig::parse(
            r#"{"classes": [{"name": "a", "priority": 0, "max_parallel_per_user": 0}], "accounts": []}"#,
        );
        assert!(zero_max_parallel_per_user.unwrap_err().contains("nonzero"));
        let max_parallel_per_user = AccountsConfig::parse(
            r#"{"classes": [{"name": "a", "priority": 0, "max_parallel_per_user": 8}], "accounts": []}"#,
        );
        assert_eq!(
            max_parallel_per_user.unwrap().classes[0].max_parallel_per_user,
            NonZeroUsize::new(8)
        );
    }

    #[test]
    fn quality_of_service_validation() {
        assert!(general("{}").is_ok());
        assert!(general(r#"{"minimum_peak_throughput": 1.5}"#).is_err());
        assert!(general(r#"{"peak_throughput_tolerance": 1.0}"#).is_err());
        assert!(general(r#"{"maximum_parallel_requests": 0}"#).is_err());
        assert!(general(r#"{"fuzz_factor": 0.1}"#).is_err());
    }

    #[test]
    fn frontend_tls_setting() {
        let tls = |tls_json: &str| {
            GeneralConfig::parse(&format!(
                r#"{{"frontend": {{"listen": "127.0.0.1:8080", "tls": {tls_json}}}, "middle_end_socket": "/tmp/middle.sock"}}"#
            ))
            .map(|config| config.frontend.tls)
        };
        assert_eq!(tls("null"), Ok(None));
        let self_signed = tls(r#""self_signed""#);
        #[cfg(feature = "tls")]
        assert_eq!(
            self_signed,
            Ok(Some(TlsConfig::SelfSigned(SelfSignedMarker::SelfSigned)))
        );
        #[cfg(not(feature = "tls"))]
        assert!(self_signed.unwrap_err().contains("'tls' cargo feature"));
    }

    #[test]
    fn backend_validation() {
        let rank = r#"[{"upstream": "http://127.0.0.1:8000", "control_socket": "/run/qos.sock"}]"#;
        let curve = r#"[{"parallel_requests_per_rank": 1, "tokens_per_second_per_request": 9, "tokens_per_second_per_rank": 9},
                        {"parallel_requests_per_rank": 2, "tokens_per_second_per_request": 8, "tokens_per_second_per_rank": 16}]"#;
        assert!(backend(rank, curve).is_ok());
        let unsorted = r#"[{"parallel_requests_per_rank": 2, "tokens_per_second_per_request": 8, "tokens_per_second_per_rank": 16},
                           {"parallel_requests_per_rank": 1, "tokens_per_second_per_request": 9, "tokens_per_second_per_rank": 9}]"#;
        assert!(backend(rank, unsorted).is_err());
        let https =
            r#"[{"upstream": "https://127.0.0.1:8000", "control_socket": "/run/qos.sock"}]"#;
        assert!(backend(https, curve)
            .unwrap_err()
            .contains("invalid upstream"));
        assert!(backend("[]", curve).is_err());
    }

    #[test]
    fn backend_json_round_trip() {
        let config = Config::load(&example_paths()).unwrap();
        let backend = config.backend.unwrap();
        let reparsed = BackendConfig::parse(&backend.to_json()).unwrap();
        assert_eq!(reparsed.ranks, backend.ranks);
        assert_eq!(reparsed.benchmark, backend.benchmark);
    }

    #[test]
    fn path_whitelist() {
        let paths = PathsConfig::default();
        assert_eq!(
            paths.classify("/v1//chat/completions/"),
            Some(PathKind::Generating)
        );
        assert_eq!(
            paths.classify("/v1/models/some-model"),
            Some(PathKind::Passthrough)
        );
        assert_eq!(paths.classify("/v1/models-extra"), None);
        assert_eq!(paths.classify("/flush_cache"), None);
        assert_eq!(paths.classify("/v1/models/../../flush_cache"), None);
    }

    fn measurements(points: &[(usize, f64)]) -> Vec<BenchmarkMeasurement> {
        points
            .iter()
            .map(|&(parallel_requests, total)| BenchmarkMeasurement {
                parallel_requests_per_rank: parallel_requests,
                tokens_per_second_per_request: total / parallel_requests as f64,
                tokens_per_second_per_rank: total,
            })
            .collect()
    }

    fn feedback_example() -> Vec<BenchmarkMeasurement> {
        measurements(&[(1, 400.0), (2, 600.0), (3, 750.0), (4, 800.0), (5, 750.0)])
    }

    #[test]
    fn throughput_limit_at_full_peak() {
        assert_eq!(throughput_limit(&feedback_example(), 1.0, 0.05), 4);
    }

    #[test]
    fn throughput_limit_at_ninety_percent_of_peak() {
        assert_eq!(throughput_limit(&feedback_example(), 0.9, 0.05), 5);
    }

    #[test]
    fn throughput_limit_with_noise_near_peak() {
        let noisy = measurements(&[(1, 400.0), (2, 600.0), (4, 790.0), (8, 800.0), (16, 600.0)]);
        assert_eq!(throughput_limit(&noisy, 1.0, 0.05), 9);
        assert_eq!(throughput_limit(&noisy, 1.0, 0.0), 8);
    }

    #[test]
    fn throughput_limit_between_sparse_measurements() {
        let sparse = measurements(&[(1, 100.0), (8, 800.0), (16, 1600.0), (32, 1000.0)]);
        assert_eq!(throughput_limit(&sparse, 0.9, 0.0), 20);
    }

    fn measured_backend(rank_count: usize, engine_max: usize) -> BackendConfig {
        BackendConfig {
            engine: EngineKind::Vllm,
            model: "model".to_string(),
            engine_max_parallel_requests_per_rank: engine_max,
            priority_scheduling: false,
            ranks: (0..rank_count)
                .map(|index| RankConfig {
                    upstream: "http://127.0.0.1:8000".to_string().try_into().unwrap(),
                    control_socket: format!("/run/qos-{index}.sock").into(),
                    data_parallel_rank: Some(index),
                })
                .collect(),
            benchmark: Some(BenchmarkResult {
                prompt_tokens: 1000,
                output_tokens: 256,
                single_stream_tokens_per_second: 400.0,
                measurements: feedback_example(),
            }),
        }
    }

    #[test]
    fn rank_capacity_lowest_limit() {
        let settings = QualityOfServiceConfig::default();
        assert_eq!(
            rank_capacities(&settings, &measured_backend(1, 256)),
            Ok(vec![4])
        );
        assert_eq!(
            rank_capacities(&settings, &measured_backend(1, 3)),
            Ok(vec![3])
        );
        let limited = QualityOfServiceConfig {
            maximum_parallel_requests: Some(2),
            ..QualityOfServiceConfig::default()
        };
        assert_eq!(
            rank_capacities(&limited, &measured_backend(1, 256)),
            Ok(vec![2])
        );
    }

    #[test]
    fn maximum_parallel_requests_across_ranks() {
        let settings = QualityOfServiceConfig {
            maximum_parallel_requests: Some(7),
            ..QualityOfServiceConfig::default()
        };
        assert_eq!(
            rank_capacities(&settings, &measured_backend(3, 256)),
            Ok(vec![3, 2, 2])
        );
        let too_low = QualityOfServiceConfig {
            maximum_parallel_requests: Some(2),
            ..QualityOfServiceConfig::default()
        };
        assert!(rank_capacities(&too_low, &measured_backend(3, 256)).is_err());
    }

    #[test]
    fn engine_limit_without_benchmark() {
        let mut unmeasured = measured_backend(2, 64);
        unmeasured.benchmark = None;
        assert_eq!(
            rank_capacities(&QualityOfServiceConfig::default(), &unmeasured),
            Ok(vec![64, 64])
        );
    }
}
