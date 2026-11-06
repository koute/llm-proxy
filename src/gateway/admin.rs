use std::sync::Arc;

use bytes::Bytes;
use http::header::HeaderValue;
use http::{Method, StatusCode};
use serde_json::{json, Map, Value};

use super::audit::{key_view, Actor, AuditEvent};
use super::keys::{
    generate_key, generate_key_id, Denial, KeyEntry, KeyRecord, KeyStore, KeysFile, Role,
};
use super::limits::unix_now;
use super::metrics::{Gauges, KeyGauge};
use super::{ChangeError, Gateway};
use crate::config::{matches_any, AccountsConfig, Config, PathPattern};
use crate::http1::{error_json, RequestHead, ResponseHead};
use crate::middle_end::routing::{rejection, ROUTE_NOT_ALLOWED};
use crate::middle_end::scheduler::Scheduler;
use crate::utils::Timestamp;

const PANEL: &str = include_str!("admin.html");
const METRICS_PATH: &str = "/metrics";
const API_PREFIX: &str = "/admin/api/";
const FIXED_FIELDS: &[&str] = &["id", "key_hash", "key_prefix", "created_at", "created_by"];
const READINESS_PATH: &str = "/health/readiness";
/// LiteLLM's probes, without an API key: for uptime checks and container health checks.
const PROBE_PATHS: &[&str] = &["/health/liveliness", "/health/liveness", READINESS_PATH];
const JSON: &str = "application/json";

type Response = (ResponseHead, Bytes);

pub fn is_local_path(path: &str) -> bool {
    let is_admin = path == "/admin" || path.starts_with("/admin/");
    is_admin || path == METRICS_PATH || PROBE_PATHS.contains(&path)
}

pub struct LocalRequest {
    pub head: RequestHead,
    pub body: Bytes,
    pub api_key: Result<String, &'static str>,
    pub client: Option<String>,
}

type ApiResult = Result<(StatusCode, Value), ChangeError>;

fn bad_request(message: String) -> ChangeError {
    ChangeError::new(StatusCode::BAD_REQUEST, message)
}

fn generation_failed(error: std::io::Error) -> ChangeError {
    let message = format!("failed to generate a key: {error}");
    ChangeError::new(StatusCode::INTERNAL_SERVER_ERROR, message)
}

pub fn new_record(
    file: &KeysFile,
    settings: Map<String, Value>,
    created_by: Option<String>,
) -> Result<(KeyRecord, String), ChangeError> {
    let id = loop {
        let id = generate_key_id().map_err(generation_failed)?;
        if !file.keys.iter().any(|record| record.id == id) {
            break id;
        }
    };
    let key = generate_key().map_err(generation_failed)?;
    let fields = json!({
        "id": id,
        "key_hash": key.hash,
        "key_prefix": key.prefix,
        "created_at": format!("{:.0}", jiff::Timestamp::now()),
        "created_by": created_by,
    });
    Ok((changed_record(fields, settings)?, key.api_key))
}

fn changed_record(mut fields: Value, patch: Map<String, Value>) -> Result<KeyRecord, ChangeError> {
    for (field, value) in patch {
        if FIXED_FIELDS.contains(&field.as_str()) {
            return Err(bad_request(format!("the field '{field}' can't be changed")));
        }
        match (&mut fields[&field], value) {
            (Value::Object(limits), Value::Object(changes)) => limits.extend(changes),
            (current, value) => *current = value,
        }
    }
    let mut record: KeyRecord = serde_json::from_value(fields)
        .map_err(|error| bad_request(format!("invalid key: {error}")))?;
    let expires_at = record.expires_at.take();
    if let Some(text) = expires_at.filter(|text| !text.trim().is_empty()) {
        let time: jiff::Timestamp = text
            .trim()
            .parse()
            .map_err(|error| bad_request(format!("invalid expires_at '{text}': {error}")))?;
        record.expires_at = Some(format!("{time:.0}"));
    }
    Ok(record)
}

/// Whether an admin key stays usable for good: unblocked, without an expiry and with
/// access to the whole admin API.
fn is_lasting_admin(record: &KeyRecord) -> bool {
    let reaches_admin_api = |routes: &[String]| {
        let patterns: Vec<PathPattern> = routes.iter().cloned().map(PathPattern::from).collect();
        [
            "/admin/api/session",
            "/admin/api/keys",
            "/admin/api/keys/x",
            "/admin/api/keys/x/rotate",
        ]
        .iter()
        .all(|path| matches_any(&patterns, path))
    };
    let routes = record.allowed_routes.as_deref();
    record.role == Role::Admin
        && !record.blocked
        && record.expires_at.is_none()
        && routes.is_none_or(reaches_admin_api)
}

/// A change mustn't leave no lasting admin key if there was one: it would lock every
/// admin out, now or once a near expiry passes, and only editing the file would get
/// back in. Files without one (e.g. admins which all expire) can still be changed.
fn require_admin_key(before: &KeysFile, after: &KeysFile) -> Result<(), ChangeError> {
    let has_lasting_admin = |file: &KeysFile| file.keys.iter().any(is_lasting_admin);
    if has_lasting_admin(before) && !has_lasting_admin(after) {
        let message = "at least one unblocked admin key without an expiry and with access to /admin/api/* must remain";
        return Err(ChangeError::new(StatusCode::CONFLICT, message));
    }
    Ok(())
}

pub async fn handle(
    gateway: Arc<Gateway>,
    config: Arc<Config>,
    scheduler: Scheduler,
    request: LocalRequest,
) -> Response {
    let path = request.head.path().to_string();
    if PROBE_PATHS.contains(&path.as_str()) {
        if path == READINESS_PATH && config.backend.is_none() {
            let body = r#"{"status":"no backend configured"}"#;
            return response(StatusCode::SERVICE_UNAVAILABLE, JSON, body);
        }
        return response(StatusCode::OK, JSON, r#"{"status":"healthy"}"#);
    }
    if path == "/admin" || path == "/admin/" {
        return response(StatusCode::OK, "text/html; charset=utf-8", PANEL);
    }
    let principal = match authorize(&gateway, &config, &request, &path) {
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
    let segments: Vec<&str> = route.split('/').filter(|part| !part.is_empty()).collect();
    let body = &request.body;
    let result = match (request.head.method.clone(), segments.as_slice()) {
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
        Ok((status, value)) => response(status, JSON, value.to_string()),
        Err(failure) => error(failure.status, &failure.message),
    }
}

fn authorize(
    gateway: &Gateway,
    config: &Config,
    request: &LocalRequest,
    path: &str,
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
    let forbidden = (
        StatusCode::FORBIDDEN,
        "forbidden_role",
        "this API key has no access here",
    );
    // Keys of accounts.json are known, but only for inference.
    if config.accounts.account(api_key).is_some() {
        return Err(forbidden);
    }
    let Some(entry) = gateway.authenticate(api_key) else {
        return unauthorized("unknown_api_key", "unknown API key");
    };
    entry.check()?;
    match (entry.record.role, path == METRICS_PATH) {
        (Role::Admin, _) | (Role::Metrics, true) if entry.allows_route(path) => Ok(entry),
        (Role::Admin, _) | (Role::Metrics, true) => Err(ROUTE_NOT_ALLOWED),
        _ => Err(forbidden),
    }
}

fn find<'a>(file: &'a mut KeysFile, id: &str) -> Result<&'a mut KeyRecord, ChangeError> {
    let record = file.keys.iter_mut().find(|record| record.id == id);
    record.ok_or_else(|| ChangeError::new(StatusCode::NOT_FOUND, format!("no key with id '{id}'")))
}

fn parse_body(body: &[u8]) -> Result<Map<String, Value>, ChangeError> {
    serde_json::from_slice(body)
        .map_err(|error| bad_request(format!("invalid request body: {error}")))
}

async fn create(gateway: &Gateway, body: &[u8], actor: Actor<'_>) -> ApiResult {
    let settings = parse_body(body)?;
    let created_by = actor.user.map(str::to_string);
    let (record, api_key) = gateway
        .change_keys(move |file| {
            let (record, api_key) = new_record(file, settings, created_by)?;
            file.keys.push(record.clone());
            Ok((record, api_key))
        })
        .await?;
    let key = key_view(&record);
    gateway.audit.record(AuditEvent::KeyCreated { actor, key });
    let body = json!({"key": key_view(&record), "api_key": api_key});
    Ok((StatusCode::CREATED, body))
}

async fn update(gateway: &Gateway, id: &str, body: &[u8], actor: Actor<'_>) -> ApiResult {
    let patch = parse_body(body)?;
    let id = id.to_string();
    let (before, after) = gateway
        .change_keys(move |file| {
            let original = file.clone();
            let record = find(file, &id)?;
            let before = record.clone();
            let fields = serde_json::to_value(&*record).expect("unserializable key record");
            let updated = changed_record(fields, patch)?;
            *record = updated.clone();
            require_admin_key(&original, file)?;
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
    let key = key_view(&record);
    gateway.audit.record(AuditEvent::KeyRotated { actor, key });
    let body = json!({"key": key_view(&record), "api_key": api_key});
    Ok((StatusCode::OK, body))
}

async fn delete(gateway: &Gateway, id: &str, actor: Actor<'_>) -> ApiResult {
    let id = id.to_string();
    let record = gateway
        .change_keys(move |file| {
            let original = file.clone();
            let record = find(file, &id)?.clone();
            file.keys.retain(|key| key.id != id);
            require_admin_key(&original, file)?;
            Ok(record)
        })
        .await?;
    let key = key_view(&record);
    gateway.audit.record(AuditEvent::KeyDeleted { actor, key });
    Ok((StatusCode::OK, json!({"deleted": key_view(&record)})))
}

/// Every key with its state and usage.
fn key_gauges<'a>(gateway: &Gateway, store: &'a KeyStore) -> Vec<KeyGauge<'a>> {
    let (now, unix) = (jiff::Timestamp::now(), unix_now());
    let mut limiter = gateway.limiter.lock();
    store
        .entries()
        .iter()
        .map(|entry| KeyGauge {
            record: &entry.record,
            state: entry.state(now),
            usage: limiter.view(&entry.record.id, unix),
        })
        .collect()
}

fn class_names(accounts: &AccountsConfig) -> Vec<&str> {
    accounts
        .classes
        .iter()
        .map(|class| class.name.as_str())
        .collect()
}

fn key_list(gateway: &Gateway) -> Value {
    let store = gateway.keys();
    let keys: Vec<Value> = key_gauges(gateway, &store)
        .into_iter()
        .map(|key| {
            let mut value = key_view(key.record);
            value["state"] = json!(key.state);
            value["usage"] = json!(key.usage);
            value
        })
        .collect();
    json!({"keys": keys, "classes": class_names(gateway.accounts())})
}

fn metrics(gateway: &Gateway, config: &Config, scheduler: &Scheduler) -> Response {
    let scheduler = scheduler.lock();
    let status = scheduler.status(Timestamp::now());
    let scheduling_log = scheduler.scheduling_log_health();
    drop(scheduler);
    let store = gateway.keys();
    let text = gateway.metrics.render(&Gauges {
        status,
        classes: class_names(&config.accounts),
        rank_caps: &config.rank_capacities,
        keys: key_gauges(gateway, &store),
        audit_log: gateway.audit.health(),
        scheduling_log,
    });
    response(StatusCode::OK, "text/plain; version=0.0.4", text)
}

fn error(status: StatusCode, message: &str) -> Response {
    response(status, JSON, error_json(message))
}

fn response(status: StatusCode, content_type: &'static str, body: impl Into<Bytes>) -> Response {
    let body = body.into();
    let length = body.len().to_string();
    let mut head = ResponseHead::new(status);
    for (name, value) in [
        ("content-type", content_type),
        ("cache-control", "no-store"),
        ("content-length", &length),
        // The panel changes keys with a click: it must never be framed by another page.
        ("x-frame-options", "DENY"),
        ("content-security-policy", "frame-ancestors 'none'"),
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
    ] {
        head.headers
            .insert(name, HeaderValue::from_str(value).unwrap());
    }
    (head, body)
}
