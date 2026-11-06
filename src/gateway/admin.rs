use std::sync::Arc;

use bytes::Bytes;
use http::header::{HeaderName, HeaderValue, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE};
use http::{Method, StatusCode};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::audit::{key_view, Actor, AuditEvent};
use super::keys::{
    generate_key, generate_key_id, now_text, KeyEntry, KeyLimits, KeyRecord, KeyState, KeysFile,
    Role,
};
use super::limits::unix_now;
use super::metrics::{Gauges, KeyGauge};
use super::{ChangeError, Gateway};
use crate::config::Config;
use crate::http1::{error_json, RequestHead, ResponseHead};
use crate::middle_end::scheduler::Scheduler;
use crate::quality_of_service::RequestCounts;
use crate::utils::Timestamp;

const PANEL: &str = include_str!("admin.html");
const ADMIN_PATH: &str = "/admin";
const METRICS_PATH: &str = "/metrics";
const API_PREFIX: &str = "/admin/api/";
const UPDATABLE_FIELDS: &[&str] = &[
    "user",
    "description",
    "role",
    "default_class",
    "allowed_classes",
    "blocked",
    "expires_at",
    "limits",
    "allowed_routes",
];
const PANEL_SECURITY_POLICY: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

pub fn is_local_path(path: &str) -> bool {
    path == ADMIN_PATH || path.starts_with("/admin/") || path == METRICS_PATH
}

pub struct LocalRequest {
    pub head: RequestHead,
    pub body: Bytes,
    pub api_key: Result<String, &'static str>,
    pub client: Option<String>,
    pub user_agent: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeySettings {
    pub user: String,
    #[serde(default)]
    pub description: String,
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
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Admin,
    Metrics,
}

type ApiResult = Result<(StatusCode, Value), ChangeError>;

pub fn new_record(
    file: &KeysFile,
    settings: KeySettings,
    created_by: Option<String>,
) -> Result<(KeyRecord, String), ChangeError> {
    let internal = |error: std::io::Error| {
        ChangeError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to generate a key: {error}"),
        )
    };
    let key = generate_key().map_err(internal)?;
    let id = loop {
        let id = generate_key_id().map_err(internal)?;
        if !file.keys.iter().any(|record| record.id == id) {
            break id;
        }
    };
    let record = KeyRecord {
        id,
        user: settings.user,
        description: settings.description,
        key_hash: key.hash,
        key_prefix: key.prefix,
        role: settings.role,
        default_class: settings.default_class,
        allowed_classes: settings.allowed_classes,
        blocked: settings.blocked,
        expires_at: normalized_expiry(settings.expires_at)?,
        limits: settings.limits,
        allowed_routes: settings.allowed_routes,
        created_at: Some(now_text()),
        created_by,
    };
    Ok((record, key.api_key))
}

fn normalized_expiry(expires_at: Option<String>) -> Result<Option<String>, ChangeError> {
    expires_at
        .filter(|text| !text.trim().is_empty())
        .map(|text| {
            text.trim()
                .parse::<jiff::Timestamp>()
                .map(|time| format!("{time:.0}"))
                .map_err(|error| {
                    ChangeError::new(
                        StatusCode::BAD_REQUEST,
                        format!("invalid expires_at '{text}': {error}"),
                    )
                })
        })
        .transpose()
}

fn require_admin_key(file: &KeysFile) -> Result<(), ChangeError> {
    let now = jiff::Timestamp::now();
    let is_active_admin = |record: &KeyRecord| {
        record.role == Role::Admin
            && !record.blocked
            && record
                .expires_at
                .as_deref()
                .and_then(|text| text.parse::<jiff::Timestamp>().ok())
                .is_none_or(|expires_at| expires_at > now)
    };
    if file.keys.iter().any(is_active_admin) {
        Ok(())
    } else {
        Err(ChangeError::new(
            StatusCode::CONFLICT,
            "at least one active admin key must remain",
        ))
    }
}

pub async fn handle(
    gateway: Arc<Gateway>,
    config: Arc<Config>,
    scheduler: Scheduler,
    request: LocalRequest,
) -> (ResponseHead, Bytes) {
    let path = request.head.path().to_string();
    let method = request.head.method.clone();
    if path == ADMIN_PATH || path == "/admin/" {
        return match method {
            Method::GET | Method::HEAD => panel(method == Method::HEAD),
            _ => error(StatusCode::METHOD_NOT_ALLOWED, "use GET"),
        };
    }
    let access = if path == METRICS_PATH {
        Access::Metrics
    } else {
        Access::Admin
    };
    let principal = match authorize(&gateway, &request, access) {
        Ok(principal) => principal,
        Err((status, reason, message)) => {
            record_rejection(&gateway, &request, status, reason, message);
            return error(status, message);
        }
    };
    if path == METRICS_PATH {
        return match method {
            Method::GET => metrics(&gateway, &config, &scheduler),
            _ => error(StatusCode::METHOD_NOT_ALLOWED, "use GET"),
        };
    }
    let Some(route) = path.strip_prefix(API_PREFIX) else {
        return error(StatusCode::NOT_FOUND, "unknown admin path");
    };
    let record = &principal.record;
    let actor = Actor {
        source: "admin_api",
        user: Some(&record.user),
        key_id: Some(&record.id),
        client: request.client.as_deref(),
    };
    let segments: Vec<&str> = route
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    let body = &request.body;
    let result = match (&method, segments.as_slice()) {
        (&Method::GET, ["session"]) => Ok((
            StatusCode::OK,
            json!({"user": record.user, "key_id": record.id, "role": record.role}),
        )),
        (&Method::GET, ["overview"]) => {
            Ok((StatusCode::OK, overview(&gateway, &config, &scheduler)))
        }
        (&Method::GET, ["keys"]) => Ok((StatusCode::OK, json!({"keys": key_list(&gateway)}))),
        (&Method::POST, ["keys"]) => create(&gateway, body, actor).await,
        (&Method::PATCH, ["keys", id]) => update(&gateway, id, body, actor).await,
        (&Method::POST, ["keys", id, "rotate"]) => rotate(&gateway, id, actor).await,
        (&Method::DELETE, ["keys", id]) => delete(&gateway, id, actor).await,
        _ => Err(ChangeError::new(
            StatusCode::NOT_FOUND,
            "unknown admin API path or method",
        )),
    };
    match result {
        Ok((status, value)) => json_response(status, &value),
        Err(failure) => error(failure.status, &failure.message),
    }
}

fn authorize(
    gateway: &Gateway,
    request: &LocalRequest,
    access: Access,
) -> Result<Arc<KeyEntry>, (StatusCode, &'static str, &'static str)> {
    let api_key = match &request.api_key {
        Ok(api_key) => api_key,
        Err(message) => return Err((StatusCode::UNAUTHORIZED, "invalid_api_key_header", message)),
    };
    if api_key.is_empty() {
        return Err((
            StatusCode::UNAUTHORIZED,
            "missing_api_key",
            "missing API key",
        ));
    }
    let Some(entry) = gateway.authenticate(api_key) else {
        return Err((
            StatusCode::UNAUTHORIZED,
            "unknown_api_key",
            "unknown API key",
        ));
    };
    match entry.state(jiff::Timestamp::now()) {
        KeyState::Active => {}
        KeyState::Blocked => {
            return Err((
                StatusCode::FORBIDDEN,
                "blocked_api_key",
                "API key is blocked",
            ))
        }
        KeyState::Expired => {
            return Err((
                StatusCode::UNAUTHORIZED,
                "expired_api_key",
                "API key expired",
            ))
        }
    }
    let allowed = match access {
        Access::Admin => entry.record.role == Role::Admin,
        Access::Metrics => matches!(entry.record.role, Role::Admin | Role::Metrics),
    };
    if !allowed {
        let message = match access {
            Access::Admin => "this API key is not an admin key",
            Access::Metrics => "this API key can't read metrics",
        };
        return Err((StatusCode::FORBIDDEN, "forbidden_role", message));
    }
    Ok(entry)
}

fn record_rejection(
    gateway: &Gateway,
    request: &LocalRequest,
    status: StatusCode,
    reason: &str,
    message: &str,
) {
    gateway.metrics.observe_rejection(status.as_u16(), reason);
    let entry = request
        .api_key
        .as_deref()
        .ok()
        .and_then(|api_key| gateway.authenticate(api_key));
    let masked = match (&request.api_key, &entry) {
        (Ok(api_key), None) if !api_key.is_empty() => {
            Some(crate::middle_end::routing::masked_api_key(api_key))
        }
        _ => None,
    };
    gateway.audit.record(AuditEvent::Rejected {
        status: status.as_u16(),
        reason,
        message,
        method: request.head.method.as_str(),
        path: request.head.path(),
        user: entry.as_ref().map(|entry| entry.record.user.as_str()),
        key_id: entry.as_ref().map(|entry| entry.record.id.as_str()),
        api_key: masked.as_deref(),
        client: request.client.as_deref(),
        user_agent: request.user_agent.as_deref(),
    });
}

fn parse_body<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, ChangeError> {
    serde_json::from_slice(body).map_err(|error| {
        ChangeError::new(
            StatusCode::BAD_REQUEST,
            format!("invalid request body: {error}"),
        )
    })
}

fn not_found(id: &str) -> ChangeError {
    ChangeError::new(StatusCode::NOT_FOUND, format!("no key with id '{id}'"))
}

async fn create(gateway: &Gateway, body: &[u8], actor: Actor<'_>) -> ApiResult {
    let settings: KeySettings = parse_body(body)?;
    let created_by = actor.user.map(str::to_string);
    let (record, api_key) = gateway
        .change_keys(move |file| {
            let (record, api_key) = new_record(file, settings, created_by)?;
            file.keys.push(record.clone());
            Ok((record, api_key))
        })
        .await?;
    gateway.audit.record(AuditEvent::KeyCreated {
        actor,
        key: key_view(&record),
    });
    Ok((
        StatusCode::CREATED,
        json!({"key": listed_key(gateway, &record), "api_key": api_key}),
    ))
}

async fn update(gateway: &Gateway, id: &str, body: &[u8], actor: Actor<'_>) -> ApiResult {
    let patch: Map<String, Value> = parse_body(body)?;
    if let Some(field) = patch
        .keys()
        .find(|field| !UPDATABLE_FIELDS.contains(&field.as_str()))
    {
        return Err(ChangeError::new(
            StatusCode::BAD_REQUEST,
            format!("the field '{field}' can't be changed"),
        ));
    }
    let key_id = id.to_string();
    let (before, after) = gateway
        .change_keys(move |file| {
            let record = file
                .keys
                .iter_mut()
                .find(|record| record.id == key_id)
                .ok_or_else(|| not_found(&key_id))?;
            let before = record.clone();
            let Value::Object(mut fields) =
                serde_json::to_value(&*record).expect("unserializable key record")
            else {
                unreachable!("a key record is a JSON object");
            };
            for (field, value) in patch {
                match (fields.get_mut(&field), value) {
                    (Some(Value::Object(current)), Value::Object(changes)) if field == "limits" => {
                        current.extend(changes);
                    }
                    (_, value) => {
                        fields.insert(field, value);
                    }
                }
            }
            let mut updated: KeyRecord =
                serde_json::from_value(Value::Object(fields)).map_err(|error| {
                    ChangeError::new(StatusCode::BAD_REQUEST, format!("invalid key: {error}"))
                })?;
            updated.expires_at = normalized_expiry(updated.expires_at)?;
            *record = updated.clone();
            require_admin_key(file)?;
            Ok((before, updated))
        })
        .await?;
    if before != after {
        gateway.audit.record(AuditEvent::KeyUpdated {
            actor,
            key_id: &after.id,
            before: key_view(&before),
            after: key_view(&after),
        });
    }
    Ok((StatusCode::OK, json!({"key": listed_key(gateway, &after)})))
}

async fn rotate(gateway: &Gateway, id: &str, actor: Actor<'_>) -> ApiResult {
    let key_id = id.to_string();
    let (record, api_key) = gateway
        .change_keys(move |file| {
            let record = file
                .keys
                .iter_mut()
                .find(|record| record.id == key_id)
                .ok_or_else(|| not_found(&key_id))?;
            let key = generate_key().map_err(|error| {
                ChangeError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to generate a key: {error}"),
                )
            })?;
            record.key_hash = key.hash;
            record.key_prefix = key.prefix;
            Ok((record.clone(), key.api_key))
        })
        .await?;
    gateway.audit.record(AuditEvent::KeyRotated {
        actor,
        key: key_view(&record),
    });
    Ok((
        StatusCode::OK,
        json!({"key": listed_key(gateway, &record), "api_key": api_key}),
    ))
}

async fn delete(gateway: &Gateway, id: &str, actor: Actor<'_>) -> ApiResult {
    let key_id = id.to_string();
    let record = gateway
        .change_keys(move |file| {
            let index = file
                .keys
                .iter()
                .position(|record| record.id == key_id)
                .ok_or_else(|| not_found(&key_id))?;
            let record = file.keys.remove(index);
            require_admin_key(file)?;
            Ok(record)
        })
        .await?;
    gateway.audit.record(AuditEvent::KeyDeleted {
        actor,
        key: key_view(&record),
    });
    Ok((StatusCode::OK, json!({"deleted": key_view(&record)})))
}

fn listed_key(gateway: &Gateway, record: &KeyRecord) -> Value {
    let state = gateway
        .keys()
        .entries()
        .iter()
        .find(|entry| entry.record.id == record.id)
        .map(|entry| entry.state(jiff::Timestamp::now()));
    let usage = gateway.limiter.lock().view(&record.id, unix_now());
    let mut value = key_view(record);
    value["state"] = json!(state);
    value["usage"] = json!(usage);
    value
}

fn key_list(gateway: &Gateway) -> Vec<Value> {
    let store = gateway.keys();
    let now = jiff::Timestamp::now();
    let unix = unix_now();
    let mut limiter = gateway.limiter.lock();
    store
        .entries()
        .iter()
        .map(|entry| {
            let mut value = key_view(&entry.record);
            value["state"] = json!(entry.state(now));
            value["usage"] = json!(limiter.view(&entry.record.id, unix));
            value
        })
        .collect()
}

fn counts_json(counts: &RequestCounts) -> Value {
    json!({
        "requests": counts.requests,
        "users": counts.users,
        "running": counts.running,
        "generating": counts.generating,
        "suspended": counts.suspended,
        "queued": counts.queued,
        "abandoned": counts.abandoned,
    })
}

fn overview(gateway: &Gateway, config: &Config, scheduler: &Scheduler) -> Value {
    let status = scheduler.lock().status(Timestamp::now());
    let classes: Vec<Value> = gateway
        .accounts()
        .classes
        .iter()
        .map(|class| {
            json!({
                "name": class.name,
                "priority": class.priority,
                "idle_only": class.idle_only,
                "cost": class.cost,
                "soft_cap": class.soft_cap,
                "max_parallel_per_user": class.max_parallel_per_user,
            })
        })
        .collect();
    let users: Vec<Value> = status
        .users
        .iter()
        .map(|user| {
            json!({
                "user": user.id.0,
                "counts": counts_json(&user.counts),
                "usage_seconds": user.usage_seconds,
                "engine_tokens": user.engine_tokens,
            })
        })
        .collect();
    let status_classes: Vec<Value> = status
        .classes
        .iter()
        .map(|class| {
            json!({"name": class.name, "priority": class.priority, "counts": counts_json(&class.counts)})
        })
        .collect();
    let backend = config.backend.as_ref().map(|backend| {
        json!({
            "engine": backend.engine,
            "model": backend.model,
            "ranks": backend.ranks.len(),
            "rank_caps": config.rank_capacities,
        })
    });
    json!({
        "classes": classes,
        "backend": backend,
        "status": {
            "total": counts_json(&status.total),
            "classes": status_classes,
            "users": users,
        },
        "recent_events": gateway.audit.recent(),
    })
}

fn metrics(gateway: &Gateway, config: &Config, scheduler: &Scheduler) -> (ResponseHead, Bytes) {
    let status = scheduler.lock().status(Timestamp::now());
    let store = gateway.keys();
    let now = jiff::Timestamp::now();
    let unix = unix_now();
    let keys = {
        let mut limiter = gateway.limiter.lock();
        store
            .entries()
            .iter()
            .map(|entry| KeyGauge {
                id: &entry.record.id,
                user: &entry.record.user,
                state: entry.state(now),
                limits: entry.record.limits,
                usage: limiter.view(&entry.record.id, unix),
            })
            .collect()
    };
    let text = gateway.metrics.render(&Gauges {
        status,
        rank_caps: &config.rank_capacities,
        keys,
    });
    response(
        StatusCode::OK,
        "text/plain; version=0.0.4; charset=utf-8",
        Bytes::from(text),
        &[],
    )
}

fn panel(is_head_request: bool) -> (ResponseHead, Bytes) {
    let (head, body) = response(
        StatusCode::OK,
        "text/html; charset=utf-8",
        Bytes::from_static(PANEL.as_bytes()),
        &[
            ("content-security-policy", PANEL_SECURITY_POLICY),
            ("referrer-policy", "no-referrer"),
            ("x-content-type-options", "nosniff"),
        ],
    );
    if is_head_request {
        (head, Bytes::new())
    } else {
        (head, body)
    }
}

fn json_response(status: StatusCode, value: &Value) -> (ResponseHead, Bytes) {
    response(
        status,
        "application/json",
        Bytes::from(value.to_string()),
        &[],
    )
}

fn error(status: StatusCode, message: &str) -> (ResponseHead, Bytes) {
    response(
        status,
        "application/json",
        Bytes::from(error_json(message)),
        &[],
    )
}

fn response(
    status: StatusCode,
    content_type: &'static str,
    body: Bytes,
    headers: &[(&'static str, &'static str)],
) -> (ResponseHead, Bytes) {
    let mut head = ResponseHead::new(status);
    head.headers
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    head.headers
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    head.headers
        .insert(CONTENT_LENGTH, HeaderValue::from(body.len()));
    for (name, value) in headers {
        head.headers.insert(
            HeaderName::from_static(name),
            HeaderValue::from_static(value),
        );
    }
    (head, body)
}
