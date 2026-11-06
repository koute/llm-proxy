use std::sync::Arc;

use bytes::Bytes;
use http::header::{
    HeaderMap, HeaderName, HeaderValue, ACCEPT_ENCODING, AUTHORIZATION, CONNECTION, CONTENT_LENGTH,
    CONTENT_TYPE, HOST, RETRY_AFTER, USER_AGENT,
};
use http::{Method, StatusCode};
use serde_json::{Map, Value};

use crate::config::{Account, BackendConfig, ClassConfig, ClassError, Config, PathKind};
use crate::engine_protocol::ProxyId;
use crate::gateway::admin::{is_local_path, LocalRequest};
use crate::gateway::keys::{Denial, KeyEntry, Role};
use crate::gateway::limits::{unix_now, InFlightGuard, LimitExceeded, LimitKind};
use crate::gateway::{Attribution, Gateway};
use crate::http1::{error_json, RequestHead, ResponseHead};
use crate::quality_of_service::{Rank, UserId};
use crate::wire::PEER_ADDRESS_HEADER;

const CLASS_HEADER: &str = "x-priority";
const PRIORITY_FIELD: &str = "priority";
const API_KEY_HEADER: &str = "x-api-key";
const STREAM_FIELD: &str = "stream";
const METRICS_ONLY: Denial = (
    StatusCode::FORBIDDEN,
    "forbidden_role",
    "this API key can only read metrics",
);
const ROUTE_NOT_ALLOWED: Denial = (
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
    Passthrough(UpstreamRequest),
    Generating(GeneratingRequest),
    Local(LocalRequest),
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
    key: &str,
) -> Option<Principal<'a>> {
    if let Some(account) = config.accounts.account(key) {
        return Some(Principal::Account(account));
    }
    gateway?.authenticate(key).map(Principal::Key)
}

pub struct UpstreamRequest {
    pub authority: String,
    pub is_head_request: bool,
    pub bytes: Bytes,
}

pub fn route(
    config: &Config,
    gateway: Option<&Gateway>,
    mut head: RequestHead,
    body: Bytes,
) -> Route {
    let client_address = gateway.and_then(|gateway| gateway.client_address(&head.headers));
    if gateway.is_some() && is_local_path(head.path()) {
        let api_key = api_key(&head.headers).map(str::to_string);
        let client = client_address;
        return Route::Local(LocalRequest {
            head,
            body,
            api_key,
            client,
        });
    }
    let reject = |status: StatusCode, reason: &'static str, message: &str| {
        Route::Rejection(rejection(
            config,
            gateway,
            &head,
            status,
            reason,
            message,
            client_address.clone(),
        ))
    };
    let path = head.path().to_string();
    let Some(path_kind) = config.general.paths.classify(&path) else {
        return reject(StatusCode::NOT_FOUND, "not_found", "path not allowed");
    };
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
    let Some(principal) = principal(config, gateway, key) else {
        return match key {
            "" => reject(
                StatusCode::UNAUTHORIZED,
                "missing_api_key",
                "missing API key",
            ),
            _ => reject(
                StatusCode::UNAUTHORIZED,
                "unknown_api_key",
                "unknown API key",
            ),
        };
    };
    if let Some(entry) = principal.key() {
        let denial = entry.check().err().or(match entry.record.role {
            Role::Metrics => Some(METRICS_ONLY),
            _ => (!entry.allows_route(&path)).then_some(ROUTE_NOT_ALLOWED),
        });
        if let Some((status, reason, message)) = denial {
            return reject(status, reason, message);
        }
    }
    let account = principal.account();
    if path_kind == PathKind::Passthrough {
        remove_client_control_headers(&mut head.headers);
        let authority = backend.ranks[0].upstream.authority();
        return Route::Passthrough(upstream_request(authority, head, body));
    }
    let is_pinned = backend
        .ranks
        .iter()
        .any(|rank| rank.data_parallel_rank.is_some());
    if is_pinned && !backend.engine.accepts_rank_pin(&path) {
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
    let in_flight = match (gateway, principal.key()) {
        (Some(gateway), Some(entry)) => {
            let admitted =
                gateway
                    .limiter
                    .lock()
                    .admit(&entry.record.id, &entry.record.limits, unix_now());
            if let Err(exceeded) = admitted {
                let mut rejection = rejection(
                    config,
                    Some(gateway),
                    &head,
                    StatusCode::TOO_MANY_REQUESTS,
                    exceeded.kind.reason(),
                    &exceeded.to_string(),
                    client_address.clone(),
                );
                rejection.headers = rate_limit_headers(&exceeded);
                return Route::Rejection(rejection);
            }
            Some(gateway.limiter.guard(&entry.record.id))
        }
        _ => None,
    };
    let virtual_user = account
        .virtual_user_http_header
        .as_ref()
        .and_then(|name| head.headers.get(name))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty());
    let user = UserId::new(&account.name, virtual_user);
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
        path,
        model: fields
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        stream: fields.get(STREAM_FIELD) == Some(&Value::Bool(true)),
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
    let name = |kind: &'static str| HeaderName::from_static(kind);
    let limit = HeaderValue::from(exceeded.limit);
    let reset = HeaderValue::try_from(format!("{}s", exceeded.retry_after_seconds))
        .expect("invalid reset header");
    let zero = HeaderValue::from_static("0");
    let mut headers = vec![(RETRY_AFTER, HeaderValue::from(exceeded.retry_after_seconds))];
    match exceeded.kind {
        LimitKind::RequestsPerMinute => headers.extend([
            (name("x-ratelimit-limit-requests"), limit),
            (name("x-ratelimit-remaining-requests"), zero),
            (name("x-ratelimit-reset-requests"), reset),
        ]),
        LimitKind::TokensPerMinute | LimitKind::TokensPerDay => headers.extend([
            (name("x-ratelimit-limit-tokens"), limit),
            (name("x-ratelimit-remaining-tokens"), zero),
            (name("x-ratelimit-reset-tokens"), reset),
        ]),
        LimitKind::ParallelRequests => {}
    }
    headers
}

fn header_text(headers: &HeaderMap, name: HeaderName) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
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

pub fn api_key(headers: &HeaderMap) -> Result<&str, &'static str> {
    if let Some(value) = headers.get(AUTHORIZATION) {
        let value = value.to_str().map_err(|_| "invalid authorization header")?;
        let (scheme, key) = value.split_once(' ').unwrap_or((value, ""));
        if !scheme.eq_ignore_ascii_case("bearer") {
            return Err("unsupported authorization scheme; use Bearer");
        }
        return Ok(key.trim());
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
    let principal = key.ok().and_then(|key| principal(config, gateway, key));
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
            None => format!("unknown API key \"{key}\""),
            Some(account) if key.is_empty() => format!("account \"{}\", no API key", account.name),
            Some(account) => format!(
                "account \"{}\"{key_label}, API key {}",
                account.name,
                masked_api_key(key)
            ),
        },
    };
    let user_agent = header_text(headers, USER_AGENT);
    let header_names: Vec<&str> = headers.keys().map(HeaderName::as_str).collect();
    Rejection {
        status,
        reason,
        message: message.to_string(),
        method: head.method.to_string(),
        path: head.path().to_string(),
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

pub fn masked_api_key(key: &str) -> String {
    let characters: Vec<char> = key.chars().collect();
    let length = characters.len();
    if length < MASKED_API_KEY_MINIMUM_LENGTH {
        return format!("of {length} characters");
    }
    let start: String = characters[..3].iter().collect();
    let end: String = characters[length - 4..].iter().collect();
    format!("\"{start}...{end}\" ({length} characters)")
}
