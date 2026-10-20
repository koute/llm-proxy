use bytes::Bytes;
use http::header::{
    HeaderMap, HeaderName, HeaderValue, ACCEPT_ENCODING, AUTHORIZATION, CONNECTION, CONTENT_LENGTH,
    CONTENT_TYPE, HOST, USER_AGENT,
};
use http::{Method, StatusCode};
use serde_json::{Map, Value};

use crate::config::{AccountsConfig, BackendConfig, ClassConfig, ClassError, Config, PathKind};
use crate::engine_protocol::ProxyId;
use crate::http1::{error_json, RequestHead, ResponseHead};
use crate::quality_of_service::{Rank, UserId};

const CLASS_HEADER: &str = "x-priority";
const PRIORITY_FIELD: &str = "priority";
const API_KEY_HEADER: &str = "x-api-key";
const STREAM_FIELD: &str = "stream";
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
}

pub struct Rejection {
    pub status: StatusCode,
    pub message: String,
    pub method: String,
    pub path: String,
    pub account: Option<String>,
    pub client: String,
}

#[derive(Debug)]
pub struct GeneratingRequest {
    pub head: RequestHead,
    pub body: Map<String, Value>,
    pub user: UserId,
    pub class: ClassConfig,
}

pub struct UpstreamRequest {
    pub authority: String,
    pub is_head_request: bool,
    pub bytes: Bytes,
}

pub fn route(config: &Config, mut head: RequestHead, body: Bytes) -> Route {
    let reject = |status: StatusCode, message: &str| {
        Route::Rejection(rejection(&config.accounts, &head, status, message))
    };
    let path = head.path();
    let Some(path_kind) = config.general.paths.classify(path) else {
        return reject(StatusCode::NOT_FOUND, "path not allowed");
    };
    let Some(backend) = &config.backend else {
        return reject(StatusCode::SERVICE_UNAVAILABLE, "no backend configured");
    };
    let key = match api_key(&head.headers) {
        Ok(key) => key,
        Err(message) => return reject(StatusCode::UNAUTHORIZED, message),
    };
    let Some(account) = config.accounts.account(key) else {
        let message = match key {
            "" => "missing API key",
            _ => "unknown API key",
        };
        return reject(StatusCode::UNAUTHORIZED, message);
    };
    if path_kind == PathKind::Passthrough {
        remove_client_control_headers(&mut head.headers);
        let authority = backend.ranks[0].upstream.authority();
        return Route::Passthrough(upstream_request(authority, head, body));
    }
    let is_pinned = backend
        .ranks
        .iter()
        .any(|rank| rank.data_parallel_rank.is_some());
    if is_pinned && !backend.engine.accepts_rank_pin(path) {
        return reject(
            StatusCode::BAD_REQUEST,
            &format!(
                "{path} does not support data parallel routing on {}; use /v1/chat/completions",
                backend.engine
            ),
        );
    }
    let Ok(Value::Object(mut fields)) = serde_json::from_slice::<Value>(&body) else {
        return reject(
            StatusCode::BAD_REQUEST,
            "request body must be a JSON object",
        );
    };
    let requested_class = match head.headers.get(CLASS_HEADER).map(HeaderValue::to_str) {
        None => None,
        Some(Ok(name)) => Some(name.trim()),
        Some(Err(_)) => return reject(StatusCode::BAD_REQUEST, "x-priority must be a class name"),
    };
    let class = match config.accounts.class(account, requested_class) {
        Ok(class) => class.clone(),
        Err(ClassError::Unknown(name)) => {
            return reject(
                StatusCode::BAD_REQUEST,
                &format!("unknown priority class '{name}'"),
            )
        }
        Err(ClassError::Forbidden(name)) => {
            return reject(
                StatusCode::FORBIDDEN,
                &format!("priority class '{name}' is not allowed for this API key"),
            )
        }
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
    Route::Generating(GeneratingRequest {
        head,
        body: fields,
        user,
        class,
    })
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
    accounts: &AccountsConfig,
    head: &RequestHead,
    status: StatusCode,
    message: &str,
) -> Rejection {
    let headers = &head.headers;
    let key = api_key(headers);
    let account = key.ok().and_then(|key| accounts.account(key));
    let key = match key {
        Err(message) => message.to_string(),
        Ok(key) => match account {
            None if key.is_empty() => "no API key".to_string(),
            None => format!("unknown API key \"{key}\""),
            Some(account) if key.is_empty() => format!("account \"{}\", no API key", account.name),
            Some(account) => format!(
                "account \"{}\", API key {}",
                account.name,
                masked_api_key(key)
            ),
        },
    };
    let user_agent = headers
        .get(USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("-");
    let header_names: Vec<&str> = headers.keys().map(HeaderName::as_str).collect();
    Rejection {
        status,
        message: message.to_string(),
        method: head.method.to_string(),
        path: head.path().to_string(),
        account: account.map(|account| account.name.clone()),
        client: format!(
            "{key}, user-agent \"{user_agent}\", headers: {}",
            header_names.join(", ")
        ),
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
