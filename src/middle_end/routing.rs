use std::sync::Arc;

use bytes::Bytes;
use http::header::{
    HeaderMap, HeaderName, HeaderValue, ACCEPT_ENCODING, ALLOW, AUTHORIZATION, CONNECTION,
    CONTENT_LENGTH, CONTENT_TYPE, HOST, RETRY_AFTER, USER_AGENT,
};
use http::{Method, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{
    normalize_path, Account, ApiKeyHash, BackendConfig, ClassConfig, ClassError, Config, PathKind,
};
use crate::engine_protocol::ProxyId;
use crate::gateway::admin::{is_local_path, LocalRequest};
use crate::gateway::keys::{Denial, KeyEntry, KeyLimits, Role};
use crate::gateway::limits::{unix_now, InFlightGuard, LimitExceeded};
use crate::gateway::{Attribution, Gateway};
use crate::http1::{error_json, RequestHead, ResponseHead};
use crate::quality_of_service::{Rank, UserId};
use crate::utils::Timestamp;
use crate::wire::PEER_ADDRESS_HEADER;

const CLASS_HEADER: &str = "x-priority";
const PRIORITY_FIELD: &str = "priority";
const API_KEY_HEADER: &str = "x-api-key";
const AZURE_API_KEY_HEADER: &str = "api-key";
const STREAM_FIELD: &str = "stream";
const METRICS_ONLY: Denial = (
    StatusCode::FORBIDDEN,
    "forbidden_role",
    "this API key can only read metrics",
);
pub const ROUTE_NOT_ALLOWED: Denial = (
    StatusCode::FORBIDDEN,
    "route_not_allowed",
    "path not allowed for this API key",
);
const MASKED_API_KEY_MINIMUM_LENGTH: usize = 16;
const TRUE_TEXTS: &[&str] = &["1", "on", "t", "true", "y", "yes"];
const CLIENT_CONTROL_HEADERS: &[&str] = &[
    "authorization",
    CLASS_HEADER,
    API_KEY_HEADER,
    AZURE_API_KEY_HEADER,
    // Credentials of other gateways, which LiteLLM strips too: never to the engine.
    "x-litellm-api-key",
    "x-goog-api-key",
    "ocp-apim-subscription-key",
    "x-vllm-priority",
    "x-override-priority",
    "x-override-rid",
    "x-override-routed-dp-rank",
    "x-request-id",
    "x-data-parallel-rank",
    PEER_ADDRESS_HEADER,
];
const CLIENT_CONTROL_FIELDS: &[&str] = &[
    PRIORITY_FIELD,
    "request_id",
    "rid",
    "routed_dp_rank",
    "data_parallel_rank",
];

#[allow(clippy::large_enum_variant)]
pub enum Route {
    Rejection(Rejection),
    Passthrough(PassthroughRequest),
    Generating(GeneratingRequest),
    Local(LocalRequest),
}

pub struct PassthroughRequest {
    pub upstream: UpstreamRequest,
    pub in_flight: Option<InFlightGuard>,
    /// With a gateway: what to record when the request finishes.
    pub record: Option<PassthroughRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PassthroughRecord {
    pub user: String,
    pub key_id: Option<String>,
    pub client: Option<String>,
    pub method: String,
    pub path: String,
    /// The configured path pattern, the `path` label of metrics.
    pub route: String,
    pub user_agent: Option<String>,
    pub arrival: Timestamp,
    /// The token estimate of a request which counts against its key's limits, if it does.
    pub reserved_tokens: Option<u64>,
}

pub struct Rejection {
    pub status: StatusCode,
    pub reason: &'static str,
    pub message: String,
    pub method: String,
    pub path: String,
    pub account: Option<String>,
    pub key_id: Option<String>,
    pub masked_key: Option<String>,
    pub client: String,
    pub client_address: Option<String>,
    pub user_agent: Option<String>,
    pub headers: Vec<(HeaderName, HeaderValue)>,
}

#[derive(Debug)]
pub struct GeneratingRequest {
    pub head: RequestHead,
    pub body: Map<String, Value>,
    pub user: UserId,
    pub class: ClassConfig,
    pub attribution: Attribution,
    pub in_flight: Option<InFlightGuard>,
}

enum Principal<'a> {
    Account(&'a Account),
    Key(Arc<KeyEntry>),
}

impl Principal<'_> {
    fn account(&self) -> &Account {
        match self {
            Principal::Account(account) => account,
            Principal::Key(entry) => &entry.account,
        }
    }

    fn key(&self) -> Option<&KeyEntry> {
        match self {
            Principal::Account(_) => None,
            Principal::Key(entry) => Some(entry),
        }
    }
}

fn principal<'a>(
    config: &'a Config,
    gateway: Option<&Gateway>,
    headers: &HeaderMap,
    key: &str,
) -> Option<Principal<'a>> {
    let Some(account) = config.accounts.account(key) else {
        return gateway?.authenticate(key).map(Principal::Key);
    };
    // LiteLLM sends its keys' hashes as virtual users: a hash of one of our keys
    // makes the request that key's, as if it came directly.
    let forwarded = account
        .virtual_user_http_header
        .as_ref()
        .and_then(|name| headers.get(name)?.to_str().ok())
        .and_then(|hash| ApiKeyHash::try_from(hash.to_string()).ok())
        .and_then(|hash| gateway?.keys().lookup_hash(&hash).cloned());
    Some(forwarded.map_or(Principal::Account(account), Principal::Key))
}

const MAX_LOGGED_HEADER_CHARACTERS: usize = 256;

/// Header values and paths in the audit log are cut, so that clients can't make
/// every rejected request write kilobytes.
pub fn truncated(text: &str) -> String {
    match text.char_indices().nth(MAX_LOGGED_HEADER_CHARACTERS) {
        Some((end, _)) => format!("{}...", &text[..end]),
        None => text.to_string(),
    }
}

fn user_agent(head: &RequestHead) -> Option<String> {
    let value = head.headers.get(USER_AGENT)?.to_str().ok()?;
    Some(truncated(value))
}

/// Virtual users are IDs like LiteLLM's key hashes (64 characters): longer header
/// values are cut, as they become metric labels and log fields.
const MAX_VIRTUAL_USER_CHARACTERS: usize = 128;

fn user_id(account: &Account, head: &RequestHead) -> UserId {
    let virtual_user = account
        .virtual_user_http_header
        .as_ref()
        .and_then(|name| head.headers.get(name))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(
            |value| match value.char_indices().nth(MAX_VIRTUAL_USER_CHARACTERS) {
                Some((end, _)) => &value[..end],
                None => value,
            },
        );
    UserId::new(&account.name, virtual_user)
}

const BYTES_PER_TOKEN: u64 = 4;
/// The most a request reserves: beyond any context, small enough that sums can't overflow.
const MAX_ESTIMATED_TOKENS: u64 = 1 << 32;
/// The fields whose prompt counts, like in LiteLLM's estimate: not tool definitions or
/// Anthropic's and the Responses API's separate instructions.
const PROMPT_FIELDS: &[&str] = &["messages", "prompt", "input"];
/// What LiteLLM counts for an image (OpenAI's low detail).
const IMAGE_TOKENS: u64 = 85;
/// What LiteLLM reserves for the output of a request without a maximum: a quarter of
/// the limit, up to this.
const DEFAULT_OUTPUT_TOKENS: u64 = 1024;

/// A request's tokens as far as they're known before it runs, like LiteLLM's estimate:
/// about one per four characters of its prompt, plus the maximum output of every sequence.
fn estimated_tokens(fields: &Map<String, Value>, limits: Option<&KeyLimits>) -> u64 {
    let prompt: u64 = PROMPT_FIELDS
        .iter()
        .filter_map(|field| Some((*field, fields.get(*field)?)))
        .map(|(field, value)| match (field, value) {
            ("messages", Value::Array(messages)) => messages.iter().map(message_tokens).sum(),
            _ => text_tokens(value),
        })
        .sum();
    // Engines accept numbers as integers, floats or strings.
    let number = |value: Option<&Value>| {
        let number = match value? {
            Value::Number(number) => number.as_f64()?,
            Value::String(text) => text.trim().parse().ok()?,
            _ => return None,
        };
        (number >= 0.0).then_some(number as u64)
    };
    // Like vLLM: the first one which is set and not 0.
    let output = [
        fields.get("max_completion_tokens"),
        fields.get("max_tokens"),
        fields.get("max_output_tokens"),
        fields
            .get("sampling_params")
            .and_then(|params| params.get("max_new_tokens")),
    ]
    .into_iter()
    .filter_map(number)
    .find(|tokens| *tokens > 0)
    .unwrap_or_else(|| {
        // The smallest token limit: a quarter of a big tpm could be over a small tpd.
        let limit =
            limits.and_then(|limits| limits.tpm_limit.into_iter().chain(limits.tpd_limit).min());
        limit.map_or(DEFAULT_OUTPUT_TOKENS, |limit| {
            (limit / 4).min(DEFAULT_OUTPUT_TOKENS)
        })
    });
    // A completion of several prompts generates for each of them.
    let prompts = match fields.get("prompt") {
        Some(Value::Array(items))
            if items.iter().all(|item| item.is_string() || item.is_array()) =>
        {
            items.len() as u64
        }
        _ => 1,
    };
    let sequences = number(fields.get("n"))
        .unwrap_or(1)
        .max(number(fields.get("best_of")).unwrap_or(1))
        .max(1);
    let total = output
        .saturating_mul(sequences)
        .saturating_mul(prompts.max(1));
    prompt.saturating_add(total).min(MAX_ESTIMATED_TOKENS)
}

/// A chat message's tokens. Like LiteLLM, an assistant's tool calls don't count, in
/// OpenAI's form (`tool_calls`, `function_call`) or Anthropic's (`tool_use` blocks).
fn message_tokens(message: &Value) -> u64 {
    let fields = match message.as_object() {
        Some(fields) if message["role"] == "assistant" => fields,
        _ => return text_tokens(message),
    };
    fields
        .iter()
        .map(|(key, value)| match (key.as_str(), value) {
            ("tool_calls" | "function_call", _) => 0,
            ("content", Value::Array(blocks)) => blocks
                .iter()
                .filter(|block| block["type"] != "tool_use")
                .map(text_tokens)
                .sum(),
            _ => field_tokens(key, value),
        })
        .sum()
}

/// About one token per four characters of text and one per token ID. Text which only
/// looks like media (e.g. a data URL in a message) counts as text, as the engine
/// tokenizes it.
fn text_tokens(value: &Value) -> u64 {
    match value {
        Value::String(text) => (text.chars().count() as u64).div_ceil(BYTES_PER_TOKEN),
        Value::Number(_) => 1,
        Value::Array(items) => items.iter().map(text_tokens).sum(),
        Value::Object(object) => object
            .iter()
            .map(|(key, value)| field_tokens(key, value))
            .sum(),
        _ => 0,
    }
}

/// An object field's tokens: media where the engine takes media (`image_url`,
/// `input_image`, `input_audio`, `file` and Anthropic's base64 or URL `source`s) count
/// as LiteLLM counts an image, whatever their size.
fn field_tokens(key: &str, value: &Value) -> u64 {
    match key {
        "image_url" | "input_image" | "input_audio" | "file" => IMAGE_TOKENS,
        "source" if matches!(value["type"].as_str(), Some("base64" | "url")) => IMAGE_TOKENS,
        _ => text_tokens(value),
    }
}

pub struct UpstreamRequest {
    pub authority: String,
    pub is_head_request: bool,
    pub bytes: Bytes,
}

#[allow(clippy::result_large_err)]
pub fn route(
    config: &Config,
    gateway: Option<&Gateway>,
    mut head: RequestHead,
    body: Bytes,
) -> Route {
    let client_address = gateway.and_then(|gateway| gateway.client_address(&head.headers));
    if gateway.is_some() && is_local_path(head.path()) {
        return Route::Local(LocalRequest {
            api_key: api_key(&head.headers).map(str::to_string),
            client: client_address,
            head,
            body,
        });
    }
    let paths = &config.general.paths;
    let original_path = head.path().to_string();
    let is_canonical = normalize_path(head.path()) == head.path();
    // OpenAI clients whose base URL lacks the /v1, as LiteLLM accepts them.
    let is_short_path =
        is_canonical && paths.classify(head.path()).is_none() && !head.path().starts_with("/v1/");
    if is_short_path && paths.classify(&format!("/v1{}", head.path())).is_some() {
        head.target = format!("/v1{}", head.target);
    }
    let rejected = |status, reason, message: &str| {
        let client = client_address.clone();
        rejection(config, gateway, &head, status, reason, message, client)
    };
    let reject =
        |status, reason, message: &str| Route::Rejection(rejected(status, reason, message));
    let path = head.path();
    // Only canonical paths: e.g. `//v1/chat/completions` or a trailing slash would be
    // classified like the canonical path but reach the engine as they are.
    let Some(path_kind) = paths.classify(path).filter(|_| is_canonical) else {
        return reject(StatusCode::NOT_FOUND, "not_found", "path not allowed");
    };
    if path_kind == PathKind::Generating && head.method != Method::POST {
        let mut rejection = rejected(
            StatusCode::METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "method not allowed; use POST",
        );
        rejection.headers = vec![(ALLOW, HeaderValue::from_static("POST"))];
        return Route::Rejection(rejection);
    }
    let route_label = paths.pattern(path);
    let Some(backend) = &config.backend else {
        return reject(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_backend",
            "no backend configured",
        );
    };
    let key = match api_key(&head.headers) {
        Ok(key) => key,
        Err(message) => return reject(StatusCode::UNAUTHORIZED, "invalid_api_key_header", message),
    };
    let Some(principal) = principal(config, gateway, &head.headers, key) else {
        let (reason, message) = match key {
            "" => ("missing_api_key", "missing API key"),
            _ => ("unknown_api_key", "unknown API key"),
        };
        return reject(StatusCode::UNAUTHORIZED, reason, message);
    };
    if let Some(entry) = principal.key() {
        let denial = entry.check().err().or(match entry.record.role {
            Role::Metrics => Some(METRICS_ONLY),
            // A key's routes may name the path as the client sent it, without /v1.
            _ => (!entry.allows_route(path) && !entry.allows_route(&original_path))
                .then_some(ROUTE_NOT_ALLOWED),
        });
        if let Some((status, reason, message)) = denial {
            return reject(status, reason, message);
        }
    }
    let account = principal.account();
    // Counts the request against the limits of its gateway key, if it has one.
    let admit = |tokens| {
        let (Some(gateway), Some(entry)) = (gateway, principal.key()) else {
            return Ok(None);
        };
        let (id, limits) = (&entry.record.id, &entry.record.limits);
        let admitted = gateway.limiter.lock().admit(id, limits, tokens, unix_now());
        let Err(exceeded) = admitted else {
            return Ok(Some(gateway.limiter.guard(id, tokens)));
        };
        let (reason, _, _) = exceeded.kind.names();
        let mut rejection = rejected(StatusCode::TOO_MANY_REQUESTS, reason, &exceeded.to_string());
        rejection.headers = rate_limit_headers(&exceeded);
        Err(Route::Rejection(rejection))
    };
    if path_kind == PathKind::Passthrough {
        // Like LiteLLM's LLM calls, requests with a body (embeddings, tokenization) count
        // against the key's limits, a token per four bytes; reads like GET /v1/models don't.
        let reserved_tokens = (body.len() as u64 / BYTES_PER_TOKEN).min(MAX_ESTIMATED_TOKENS);
        let in_flight = match head.method == Method::POST {
            true => match admit(reserved_tokens) {
                Ok(in_flight) => in_flight,
                Err(rejection) => return rejection,
            },
            false => None,
        };
        let record = gateway.map(|_| PassthroughRecord {
            user: user_id(account, &head).to_string(),
            key_id: principal.key().map(|entry| entry.record.id.clone()),
            client: client_address.clone(),
            method: head.method.to_string(),
            path: truncated(path),
            route: route_label.clone().unwrap_or_default(),
            user_agent: user_agent(&head),
            arrival: Timestamp::now(),
            reserved_tokens: in_flight.is_some().then_some(reserved_tokens),
        });
        remove_client_control_headers(&mut head.headers);
        let authority = backend.ranks[0].upstream.authority();
        return Route::Passthrough(PassthroughRequest {
            upstream: upstream_request(authority, head, body),
            in_flight,
            record,
        });
    }
    let is_pinned = backend
        .ranks
        .iter()
        .any(|rank| rank.data_parallel_rank.is_some());
    if is_pinned && !backend.engine.accepts_rank_pin(path) {
        return reject(
            StatusCode::BAD_REQUEST,
            "bad_request",
            &format!(
                "{path} does not support data parallel routing on {}; use /v1/chat/completions",
                backend.engine
            ),
        );
    }
    let Ok(Value::Object(mut fields)) = serde_json::from_slice::<Value>(&body) else {
        return reject(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "request body must be a JSON object",
        );
    };
    let requested_class = match head.headers.get(CLASS_HEADER).map(HeaderValue::to_str) {
        None => None,
        Some(Ok(name)) => Some(name.trim()),
        Some(Err(_)) => {
            return reject(
                StatusCode::BAD_REQUEST,
                "bad_request",
                "x-priority must be a class name",
            )
        }
    };
    let class = match config.accounts.class(account, requested_class) {
        Ok(class) => class.clone(),
        Err(ClassError::Unknown(name)) => {
            return reject(
                StatusCode::BAD_REQUEST,
                "unknown_class",
                &format!("unknown priority class '{name}'"),
            )
        }
        Err(ClassError::Forbidden(name)) => {
            return reject(
                StatusCode::FORBIDDEN,
                "class_not_allowed",
                &format!("priority class '{name}' is not allowed for this API key"),
            )
        }
    };
    let limits = principal.key().map(|entry| &entry.record.limits);
    let reserved_tokens = estimated_tokens(&fields, limits);
    let in_flight = match admit(reserved_tokens) {
        Ok(in_flight) => in_flight,
        Err(rejection) => return rejection,
    };
    let user = user_id(account, &head);
    remove_client_control_headers(&mut head.headers);
    head.headers.remove(ACCEPT_ENCODING);
    for field in CLIENT_CONTROL_FIELDS {
        fields.remove(*field);
    }
    if let Some(stream) = fields.get_mut(STREAM_FIELD) {
        *stream = Value::Bool(is_true(stream));
    }
    let attribution = Attribution {
        key_id: principal.key().map(|entry| entry.record.id.clone()),
        client: client_address,
        path: head.path().to_string(),
        model: fields.get("model").and_then(Value::as_str).map(truncated),
        stream: fields.get(STREAM_FIELD) == Some(&Value::Bool(true)),
        arrival: Some(Timestamp::now()),
        class: Some(class.name.clone()),
        user_agent: user_agent(&head),
        route: route_label,
        reserved_tokens,
        ..Attribution::default()
    };
    Route::Generating(GeneratingRequest {
        head,
        body: fields,
        user,
        class,
        attribution,
        in_flight,
    })
}

fn rate_limit_headers(exceeded: &LimitExceeded) -> Vec<(HeaderName, HeaderValue)> {
    let retry_after = exceeded.retry_after_seconds;
    let mut headers = vec![(RETRY_AFTER, HeaderValue::from(retry_after))];
    let (_, _, Some(unit)) = exceeded.kind.names() else {
        return headers;
    };
    let values = [
        exceeded.limit.to_string(),
        exceeded.remaining.to_string(),
        format!("{retry_after}s"),
    ];
    for (name, value) in ["limit", "remaining", "reset"].into_iter().zip(values) {
        let name = HeaderName::try_from(format!("x-ratelimit-{name}-{unit}"));
        headers.push((
            name.expect("invalid header"),
            HeaderValue::try_from(value).expect("invalid header"),
        ));
    }
    headers
}

impl GeneratingRequest {
    pub fn is_streaming(&self) -> bool {
        self.body.get(STREAM_FIELD) == Some(&Value::Bool(true))
    }
}

fn is_true(value: &Value) -> bool {
    match value {
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64() == Some(1.0),
        Value::String(text) => TRUE_TEXTS.contains(&text.to_ascii_lowercase().as_str()),
        _ => false,
    }
}

pub fn engine_request(
    backend: &BackendConfig,
    rank: Rank,
    request: &GeneratingRequest,
    proxy_id: ProxyId,
) -> UpstreamRequest {
    let rank_config = &backend.ranks[rank.0];
    let mut body = request.body.clone();
    let mut head = request.head.clone();
    if backend.priority_scheduling {
        let priority = backend.engine.priority(request.class.priority);
        body.insert(PRIORITY_FIELD.to_string(), Value::from(priority));
    }
    backend.engine.label_request(
        &mut head.headers,
        &mut body,
        proxy_id,
        rank_config.data_parallel_rank,
    );
    let body = serde_json::to_vec(&body).expect("unserializable request body");
    upstream_request(rank_config.upstream.authority(), head, Bytes::from(body))
}

fn remove_client_control_headers(headers: &mut HeaderMap) {
    for name in CLIENT_CONTROL_HEADERS {
        headers.remove(*name);
    }
}

fn upstream_request(authority: &str, mut head: RequestHead, body: Bytes) -> UpstreamRequest {
    let headers = &mut head.headers;
    headers.insert(
        HOST,
        HeaderValue::from_str(authority).expect("invalid authority"),
    );
    headers.insert(CONNECTION, HeaderValue::from_static("close"));
    headers.insert(CONTENT_LENGTH, HeaderValue::from(body.len()));
    let mut bytes = head.encode();
    bytes.extend_from_slice(&body);
    UpstreamRequest {
        authority: authority.to_string(),
        is_head_request: head.method == Method::HEAD,
        bytes: Bytes::from(bytes),
    }
}

fn api_key(headers: &HeaderMap) -> Result<&str, &'static str> {
    // An empty Authorization header counts as absent, like in LiteLLM, so that
    // clients which always send one can still use x-api-key or api-key.
    let authorization = headers
        .get(AUTHORIZATION)
        .filter(|value| !value.as_bytes().iter().all(u8::is_ascii_whitespace));
    if let Some(value) = authorization {
        let value = value.to_str().map_err(|_| "invalid authorization header")?;
        let value = value.trim();
        let (scheme, key) = value.split_once(' ').unwrap_or((value, ""));
        if !scheme.eq_ignore_ascii_case("bearer") {
            return Err("unsupported authorization scheme; use Bearer");
        }
        return Ok(key.trim());
    }
    // Like LiteLLM: a non-empty `api-key` (the Azure OpenAI SDKs') before `x-api-key`.
    let azure_key = headers
        .get(AZURE_API_KEY_HEADER)
        .and_then(|value| value.to_str().ok());
    if let Some(key) = azure_key.map(str::trim).filter(|key| !key.is_empty()) {
        return Ok(key);
    }
    match headers.get(API_KEY_HEADER) {
        Some(value) => value
            .to_str()
            .map(str::trim)
            .map_err(|_| "invalid x-api-key header"),
        None => Ok(""),
    }
}

pub fn error_response(status: StatusCode, message: &str) -> (ResponseHead, Bytes) {
    let body = Bytes::from(error_json(message));
    let mut head = ResponseHead::new(status);
    head.headers
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    head.headers
        .insert(CONTENT_LENGTH, HeaderValue::from(body.len()));
    (head, body)
}

pub fn rejection(
    config: &Config,
    gateway: Option<&Gateway>,
    head: &RequestHead,
    status: StatusCode,
    reason: &'static str,
    message: &str,
    client_address: Option<String>,
) -> Rejection {
    let headers = &head.headers;
    let key = api_key(headers);
    let principal = key
        .ok()
        .and_then(|key| principal(config, gateway, headers, key));
    let account = principal.as_ref().map(Principal::account);
    let key_id = principal
        .as_ref()
        .and_then(Principal::key)
        .map(|entry| entry.record.id.clone());
    let key_label = key_id
        .as_ref()
        .map(|key_id| format!(" (key {key_id})"))
        .unwrap_or_default();
    let masked_key = match (key, account) {
        (Ok(key), None) if !key.is_empty() => Some(masked_api_key(key)),
        _ => None,
    };
    let key = match key {
        Err(message) => message.to_string(),
        Ok(key) => match account {
            None if key.is_empty() => "no API key".to_string(),
            None => format!("unknown API key {}", masked_api_key(key)),
            Some(account) if key.is_empty() => format!("account \"{}\", no API key", account.name),
            Some(account) => format!(
                "account \"{}\"{key_label}, API key {}",
                account.name,
                masked_api_key(key)
            ),
        },
    };
    let user_agent = user_agent(head);
    let header_names: Vec<&str> = headers.keys().map(HeaderName::as_str).collect();
    Rejection {
        status,
        reason,
        message: message.to_string(),
        method: truncated(head.method.as_str()),
        path: truncated(head.path()),
        account: account.map(|account| account.name.clone()),
        key_id,
        masked_key,
        client: format!(
            "{key}, user-agent \"{}\", headers: {}",
            user_agent.as_deref().unwrap_or("-"),
            header_names.join(", ")
        ),
        client_address,
        user_agent,
        headers: Vec::new(),
    }
}

fn masked_api_key(key: &str) -> String {
    let characters: Vec<char> = key.chars().collect();
    let length = characters.len();
    if length < MASKED_API_KEY_MINIMUM_LENGTH {
        return format!("of {length} characters");
    }
    let start: String = characters[..3].iter().collect();
    let end: String = characters[length - 4..].iter().collect();
    format!("\"{start}...{end}\" ({length} characters)")
}

#[cfg(test)]
mod estimate_tests {
    use super::*;
    use serde_json::json;

    fn estimate(body: &Value) -> u64 {
        let limits = KeyLimits {
            tpm_limit: Some(2000),
            ..KeyLimits::default()
        };
        estimated_tokens(body.as_object().unwrap(), Some(&limits))
    }

    fn one_message(message: &Value) -> u64 {
        estimate(&json!({"messages": [message], "max_tokens": 1}))
    }

    #[test]
    fn estimated_tokens_like_litellm() {
        let message = json!([{"role": "user", "content": "12345678"}]);
        let prompts = json!({"prompt": ["abcd", "efgh"], "max_tokens": 10, "best_of": 2});
        let tools = json!([{"type": "function", "function": {"description": "d".repeat(4000)}}]);
        let extra = json!({"tools": tools, "system": "s".repeat(4000), "instructions": "i".repeat(4000), "max_tokens": 1});
        for (body, expected) in [
            // The prompt's text, not the body's JSON: "12345678" and "user".
            (json!({"messages": message, "max_tokens": 100}), 103),
            (json!({"max_completion_tokens": 50, "max_tokens": 100}), 50),
            // Like vLLM, a maximum of 0 counts as none.
            (json!({"max_completion_tokens": 0, "max_tokens": 100}), 100),
            (json!({"max_tokens": "100", "n": 2.0}), 200),
            (json!({"max_output_tokens": 30}), 30),
            (json!({"sampling_params": {"max_new_tokens": 40}}), 40),
            (prompts, 2 + 40),
            (json!({"prompt": [1, 2, 3], "max_tokens": 10}), 13),
            // Like LiteLLM, tool definitions and separate instructions don't count.
            (extra, 1),
            // Without a maximum, a quarter of the limit (up to 1024), like LiteLLM.
            (json!({}), 500),
        ] {
            assert_eq!(estimate(&body), expected, "{body}");
        }
        // Like LiteLLM, an assistant's tool calls don't count, also Anthropic's, and media
        // count like an image, not by their size.
        let tool_use = json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f", "input": {"q": "a".repeat(4000)}}]});
        let calls = json!({"role": "assistant", "content": null, "tool_calls": [{"function": {"arguments": "a".repeat(4000)}}]});
        let anthropic = json!([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A".repeat(100_000)}}]);
        let image = json!([{"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", "A".repeat(100_000))}}]);
        for (message, bound) in [
            (tool_use, 10),
            (calls, 10),
            (json!({"role": "user", "content": anthropic}), 100),
            (json!({"role": "user", "content": image}), 100),
        ] {
            assert!(one_message(&message) < bound, "{message}");
        }
        // ... but only an assistant's: whatever else is called a tool call counts, and text
        // which only looks a bit like a data URL counts as text.
        let text = "w ".repeat(2000);
        let data_url = format!("data:a;base64,{text}");
        for message in [
            json!({"role": "user", "type": "tool_use", "content": text}),
            json!({"role": "system", "type": "tool_use", "content": text}),
            json!({"role": "assistant", "type": "tool_use", "content": text}),
            json!({"role": "user", "content": [{"type": "tool_use", "text": text}]}),
            json!({"role": "user", "content": "x", "tool_calls": [{"function": {"arguments": text}}]}),
            json!({"role": "tool", "content": "x", "function_call": {"arguments": text}}),
            json!({"role": "assistant", "content": [{"type": "text", "text": text, "x": {"type": "tool_use"}}]}),
            json!({"role": "user", "content": data_url}),
            json!({"role": "user", "content": [{"type": "text", "text": data_url}]}),
        ] {
            assert!(one_message(&message) > 1000, "{message}");
        }
        // Without a maximum, a quarter of the smallest token limit.
        let limits = KeyLimits {
            tpm_limit: Some(100_000),
            tpd_limit: Some(1000),
            ..KeyLimits::default()
        };
        assert_eq!(estimated_tokens(&Map::new(), Some(&limits)), 250);
        let huge = estimate(&json!({"max_tokens": u64::MAX, "n": 4}));
        assert!(huge <= MAX_ESTIMATED_TOKENS);
    }
}
