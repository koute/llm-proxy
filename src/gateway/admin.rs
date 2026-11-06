use std::sync::Arc;

use bytes::Bytes;
use http::header::{HeaderValue, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE};
use http::{Method, StatusCode};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::audit::{key_view, Actor, AuditEvent};
use super::keys::{
    generate_key, generate_key_id, now_text, Denial, KeyEntry, KeyLimits, KeyRecord, KeysFile, Role,
};
use super::limits::unix_now;
use super::metrics::{Gauges, KeyGauge};
use super::{ChangeError, Gateway};
use crate::config::Config;
use crate::http1::{error_json, RequestHead, ResponseHead};
use crate::middle_end::routing::rejection;
use crate::middle_end::scheduler::Scheduler;
use crate::utils::Timestamp;

const PANEL: &str = include_str!("admin.html");
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

pub fn is_local_path(path: &str) -> bool {
    path == "/admin" || path.starts_with("/admin/") || path == METRICS_PATH
}

pub struct LocalRequest {
    pub head: RequestHead,
    pub body: Bytes,
    pub api_key: Result<String, &'static str>,
    pub client: Option<String>,
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

type ApiResult = Result<(StatusCode, Value), ChangeError>;

fn bad_request(message: String) -> ChangeError {
    ChangeError::new(StatusCode::BAD_REQUEST, message)
}

fn generation_failed(error: std::io::Error) -> ChangeError {
    ChangeError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("failed to generate a key: {error}"),
    )
}

pub fn new_record(
    file: &KeysFile,
    settings: KeySettings,
    created_by: Option<String>,
) -> Result<(KeyRecord, String), ChangeError> {
    let id = loop {
        let id = generate_key_id().map_err(generation_failed)?;
        if !file.keys.iter().any(|record| record.id == id) {
            break id;
        }
    };
    let key = generate_key().map_err(generation_failed)?;
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
    let expires_at = expires_at.filter(|text| !text.trim().is_empty());
    let parse = |text: String| match text.trim().parse::<jiff::Timestamp>() {
        Ok(time) => Ok(format!("{time:.0}")),
        Err(error) => Err(bad_request(format!("invalid expires_at '{text}': {error}"))),
    };
    expires_at.map(parse).transpose()
}

fn require_admin_key(file: &KeysFile) -> Result<(), ChangeError> {
    let now = jiff::Timestamp::now();
    let is_active_admin = |record: &KeyRecord| {
        let expires_at = record
            .expires_at
            .as_deref()
            .and_then(|text| text.parse().ok());
        record.role == Role::Admin
            && !record.blocked
            && expires_at.is_none_or(|time: jiff::Timestamp| time > now)
    };
    match file.keys.iter().any(is_active_admin) {
        true => Ok(()),
        false => Err(ChangeError::new(
            StatusCode::CONFLICT,
            "at least one active admin key must remain",
        )),
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
    if path == "/admin" || path == "/admin/" {
        return response(
            StatusCode::OK,
            "text/html; charset=utf-8",
            Bytes::from_static(PANEL.as_bytes()),
        );
    }
    let principal = match authorize(&gateway, &request, path == METRICS_PATH) {
        Ok(principal) => principal,
        Err((status, reason, message)) => {
            let head = &request.head;
            let client = request.client.clone();
            let rejection = rejection(
                &config,
                Some(&gateway),
                head,
                status,
                reason,
                message,
                client,
            );
            gateway.record_rejection(&rejection);
            return error(status, message);
        }
    };
    if path == METRICS_PATH {
        return metrics(&gateway, &config, &scheduler);
    }
    let record = &principal.record;
    let actor = Actor {
        source: "admin_api",
        user: Some(&record.user),
        key_id: Some(&record.id),
        client: request.client.as_deref(),
    };
    let route = path.strip_prefix(API_PREFIX).unwrap_or_default();
    let segments: Vec<&str> = route
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    let body = &request.body;
    let result = match (method, segments.as_slice()) {
        (Method::GET, ["session"]) => Ok((
            StatusCode::OK,
            json!({"user": record.user, "key_id": record.id, "role": record.role}),
        )),
        (Method::GET, ["keys"]) => Ok((StatusCode::OK, key_list(&gateway))),
        (Method::POST, ["keys"]) => create(&gateway, body, actor).await,
        (Method::PATCH, ["keys", id]) => update(&gateway, id, body, actor).await,
        (Method::POST, ["keys", id, "rotate"]) => rotate(&gateway, id, actor).await,
        (Method::DELETE, ["keys", id]) => delete(&gateway, id, actor).await,
        _ => Err(ChangeError::new(
            StatusCode::NOT_FOUND,
            "unknown admin API path or method",
        )),
    };
    match result {
        Ok((status, value)) => response(status, "application/json", Bytes::from(value.to_string())),
        Err(failure) => error(failure.status, &failure.message),
    }
}

fn authorize(
    gateway: &Gateway,
    request: &LocalRequest,
    is_metrics: bool,
) -> Result<Arc<KeyEntry>, Denial> {
    let unauthorized =
        |reason, message: &'static str| Err((StatusCode::UNAUTHORIZED, reason, message));
    let api_key = match &request.api_key {
        Err(message) => return unauthorized("invalid_api_key_header", message),
        Ok(api_key) if api_key.is_empty() => {
            return unauthorized("missing_api_key", "missing API key")
        }
        Ok(api_key) => api_key,
    };
    let Some(entry) = gateway.authenticate(api_key) else {
        return unauthorized("unknown_api_key", "unknown API key");
    };
    entry.check()?;
    match (entry.record.role, is_metrics) {
        (Role::Admin, _) | (Role::Metrics, true) => Ok(entry),
        _ => Err((
            StatusCode::FORBIDDEN,
            "forbidden_role",
            "this API key has no access here",
        )),
    }
}

fn find<'a>(file: &'a mut KeysFile, id: &str) -> Result<&'a mut KeyRecord, ChangeError> {
    let record = file.keys.iter_mut().find(|record| record.id == id);
    record.ok_or_else(|| ChangeError::new(StatusCode::NOT_FOUND, format!("no key with id '{id}'")))
}

async fn create(gateway: &Gateway, body: &[u8], actor: Actor<'_>) -> ApiResult {
    let settings: KeySettings = serde_json::from_slice(body)
        .map_err(|error| bad_request(format!("invalid request body: {error}")))?;
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
        json!({"key": key_view(&record), "api_key": api_key}),
    ))
}

async fn update(gateway: &Gateway, id: &str, body: &[u8], actor: Actor<'_>) -> ApiResult {
    let patch: Map<String, Value> = serde_json::from_slice(body)
        .map_err(|error| bad_request(format!("invalid request body: {error}")))?;
    if let Some(field) = patch
        .keys()
        .find(|field| !UPDATABLE_FIELDS.contains(&field.as_str()))
    {
        return Err(bad_request(format!("the field '{field}' can't be changed")));
    }
    let id = id.to_string();
    let (before, after) = gateway
        .change_keys(move |file| {
            let record = find(file, &id)?;
            let before = record.clone();
            let mut fields = serde_json::to_value(&*record).expect("unserializable key record");
            for (field, value) in patch {
                match (&mut fields[&field], value) {
                    (Value::Object(limits), Value::Object(changes)) => limits.extend(changes),
                    (current, value) => *current = value,
                }
            }
            let mut updated: KeyRecord = serde_json::from_value(fields)
                .map_err(|error| bad_request(format!("invalid key: {error}")))?;
            updated.expires_at = normalized_expiry(updated.expires_at)?;
            *record = updated.clone();
            require_admin_key(file)?;
            Ok((before, updated))
        })
        .await?;
    if before != after {
        gateway.audit.record(AuditEvent::KeyUpdated {
            actor,
            before: key_view(&before),
            after: key_view(&after),
        });
    }
    Ok((StatusCode::OK, json!({"key": key_view(&after)})))
}

async fn rotate(gateway: &Gateway, id: &str, actor: Actor<'_>) -> ApiResult {
    let id = id.to_string();
    let (record, api_key) = gateway
        .change_keys(move |file| {
            let key = generate_key().map_err(generation_failed)?;
            let record = find(file, &id)?;
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
        json!({"key": key_view(&record), "api_key": api_key}),
    ))
}

async fn delete(gateway: &Gateway, id: &str, actor: Actor<'_>) -> ApiResult {
    let id = id.to_string();
    let record = gateway
        .change_keys(move |file| {
            let record = find(file, &id)?.clone();
            file.keys.retain(|key| key.id != id);
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

fn key_list(gateway: &Gateway) -> Value {
    let store = gateway.keys();
    let (now, unix) = (jiff::Timestamp::now(), unix_now());
    let mut limiter = gateway.limiter.lock();
    let keys: Vec<Value> = store
        .entries()
        .iter()
        .map(|entry| {
            let mut value = key_view(&entry.record);
            value["state"] = json!(entry.state(now));
            value["usage"] = json!(limiter.view(&entry.record.id, unix));
            value
        })
        .collect();
    let classes: Vec<&str> = gateway
        .accounts()
        .classes
        .iter()
        .map(|class| class.name.as_str())
        .collect();
    json!({"keys": keys, "classes": classes})
}

fn metrics(gateway: &Gateway, config: &Config, scheduler: &Scheduler) -> (ResponseHead, Bytes) {
    let status = scheduler.lock().status(Timestamp::now());
    let store = gateway.keys();
    let (now, unix) = (jiff::Timestamp::now(), unix_now());
    let mut limiter = gateway.limiter.lock();
    let keys = store
        .entries()
        .iter()
        .map(|entry| KeyGauge {
            id: &entry.record.id,
            user: &entry.record.user,
            state: entry.state(now),
            limits: entry.record.limits,
            usage: limiter.view(&entry.record.id, unix),
        })
        .collect();
    drop(limiter);
    let text = gateway.metrics.render(&Gauges {
        status,
        rank_caps: &config.rank_capacities,
        keys,
    });
    response(
        StatusCode::OK,
        "text/plain; version=0.0.4",
        Bytes::from(text),
    )
}

fn error(status: StatusCode, message: &str) -> (ResponseHead, Bytes) {
    response(status, "application/json", Bytes::from(error_json(message)))
}

fn response(status: StatusCode, content_type: &'static str, body: Bytes) -> (ResponseHead, Bytes) {
    let mut head = ResponseHead::new(status);
    head.headers
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    head.headers
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    head.headers
        .insert(CONTENT_LENGTH, HeaderValue::from(body.len()));
    (head, body)
}
