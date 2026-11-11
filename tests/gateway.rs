mod support;

use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::{Method, StatusCode};
use serde_json::{json, Value};

use llm_proxy::config::ApiKeyHash;
use support::{
    chat, chat_body, deploy, event_names, open_chat_stream, post_chat, raw_json_request,
    raw_status, running_stream, send, stream_text, wait_until, ClientResponse, Deployment,
    DeploymentOptions, AUDIT_LOG, KEYS_FILE,
};

const ADMIN: &str = "sk-admin-0123456789abcdef";
const USER: &str = "sk-user-0123456789abcdef";
const METRICS: &str = "sk-metrics-0123456789abcdef";

fn key_with_api_key(id: &str, user: &str, api_key: &str, settings: Value) -> Value {
    let mut key = json!({
        "id": id,
        "user": user,
        "key_hash": ApiKeyHash::from_api_key(api_key).to_hex(),
        "default_class": "batch",
        "allowed_classes": ["idle"],
    });
    for (field, value) in settings.as_object().unwrap() {
        key[field] = value.clone();
    }
    key
}

/// The key `id` of `user`, with the API key `sk-<id>-0123456789`.
fn key(id: &str, user: &str, settings: Value) -> Value {
    key_with_api_key(id, user, &format!("sk-{id}-0123456789"), settings)
}

fn base_keys() -> Vec<Value> {
    vec![
        key_with_api_key("admin", "root", ADMIN, json!({"role": "admin"})),
        key_with_api_key("user", "carol", USER, json!({})),
        key_with_api_key("metrics", "prometheus", METRICS, json!({"role": "metrics"})),
    ]
}

/// The options of a gateway with the base keys and `extra_keys`.
fn gateway_options(extra_keys: Vec<Value>, general: Value) -> DeploymentOptions {
    let keys = [base_keys(), extra_keys].concat();
    DeploymentOptions {
        keys: Some(json!({ "keys": keys })),
        general,
        ..DeploymentOptions::default()
    }
}

async fn deploy_with_keys(extra_keys: Vec<Value>) -> Deployment {
    deploy(gateway_options(extra_keys, json!({}))).await
}

fn write_keys(deployment: &Deployment, keys: &[Value]) {
    let text = json!({ "keys": keys }).to_string();
    std::fs::write(deployment.socket_path(KEYS_FILE), text).unwrap();
}

async fn request(
    deployment: &Deployment,
    method: &str,
    path: &str,
    api_key: &str,
    body: impl Into<Bytes>,
) -> ClientResponse {
    let authorization = format!("Bearer {api_key}");
    let headers = [("authorization", authorization.as_str())];
    let method = method.parse().unwrap();
    send(deployment.address, method, path, &headers, body).await
}

async fn get(deployment: &Deployment, path: &str, api_key: &str) -> ClientResponse {
    request(deployment, "GET", path, api_key, "").await
}

/// An admin API request of `api_key` (a null body is empty): its status and JSON.
async fn admin_as(
    deployment: &Deployment,
    api_key: &str,
    method: &str,
    path: &str,
    body: Value,
) -> (StatusCode, Value) {
    let body = if body.is_null() {
        String::new()
    } else {
        body.to_string()
    };
    let path = format!("/admin/api/{path}");
    let response = request(deployment, method, &path, api_key, body).await;
    let json = serde_json::from_slice(&response.body).unwrap_or(Value::Null);
    (response.status, json)
}

async fn admin(
    deployment: &Deployment,
    method: &str,
    path: &str,
    body: Value,
) -> (StatusCode, Value) {
    admin_as(deployment, ADMIN, method, path, body).await
}

/// The status of a chat completion of `api_key`.
async fn status(deployment: &Deployment, api_key: &str) -> StatusCode {
    let response = chat(deployment.address, api_key, &[], chat_body(1, false)).await;
    response.status
}

async fn metrics(deployment: &Deployment) -> String {
    let response = get(deployment, "/metrics", METRICS).await;
    String::from_utf8(response.body.to_vec()).unwrap()
}

async fn assert_metrics(deployment: &Deployment, expected: &[&str]) {
    let metrics = metrics(deployment).await;
    for expected in expected {
        assert!(
            metrics.contains(expected),
            "missing {expected} in\n{metrics}"
        );
    }
}

/// Waits up to `seconds` for the metrics line `line`.
async fn wait_for_metric(deployment: &Deployment, line: &str, seconds: u64) {
    let started = Instant::now();
    while !metrics(deployment)
        .await
        .lines()
        .any(|metric| metric == line)
    {
        assert!(
            started.elapsed() < Duration::from_secs(seconds),
            "no {line}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The events of `log` named `name`.
fn named(log: &[Value], name: &str) -> Vec<Value> {
    log.iter()
        .filter(|event| event["event"] == name)
        .cloned()
        .collect()
}

/// The audit log's events named `name`.
fn events(deployment: &Deployment, name: &str) -> Vec<Value> {
    named(&deployment.audit_log(), name)
}

/// The first of `events` which matches `predicate`.
fn find(events: &[Value], predicate: impl Fn(&Value) -> bool) -> Value {
    events
        .iter()
        .find(|event| predicate(event))
        .unwrap()
        .clone()
}

/// Asserts that `event` has each field of `expected` with its value.
fn assert_fields(event: &Value, expected: Value) {
    for (field, value) in expected.as_object().unwrap() {
        assert_eq!(&event[field], value, "{field} of {event}");
    }
}

/// The reasons of the rejections, once there are at least `count`.
async fn reasons(deployment: &Deployment, count: usize) -> Vec<Value> {
    let reasons = || -> Vec<Value> {
        let rejected = events(deployment, "rejected");
        rejected
            .iter()
            .map(|event| event["reason"].clone())
            .collect()
    };
    wait_until(|| reasons().len() >= count).await;
    reasons()
}

/// The request events, once `count` of them have their tokens.
async fn finished_requests(deployment: &Deployment, count: usize) -> Vec<Value> {
    let finished = || {
        let requests = events(deployment, "request");
        requests
            .iter()
            .filter(|event| !event["tokens"].is_null())
            .count()
    };
    wait_until(|| finished() >= count).await;
    events(deployment, "request")
}

/// The usage of the key `id` as the admin API lists it.
async fn usage(deployment: &Deployment, id: &str) -> Value {
    let (_, list) = admin(deployment, "GET", "keys", Value::Null).await;
    let key = find(list["keys"].as_array().unwrap(), |key| key["id"] == id);
    key["usage"].clone()
}

#[tokio::test]
async fn keys_authenticate_and_are_audited() {
    let expired = json!({"expires_at": "2020-01-01T00:00:00Z"});
    let deployment = deploy_with_keys(vec![
        key("blocked", "dave", json!({"blocked": true})),
        key("expired", "erin", expired),
        key("models", "frank", json!({"allowed_routes": ["/v1/models"]})),
    ])
    .await;
    for (headers, expected) in [
        (vec![], StatusCode::OK),
        (vec![("x-priority", "idle")], StatusCode::OK),
        (vec![("x-priority", "interactive")], StatusCode::FORBIDDEN),
    ] {
        let response = chat(deployment.address, USER, &headers, chat_body(1, false)).await;
        assert_eq!(response.status, expected, "{headers:?}");
    }
    for (api_key, expected) in [
        ("sk-alice", StatusCode::OK),
        ("sk-blocked-0123456789", StatusCode::UNAUTHORIZED),
        ("sk-expired-0123456789", StatusCode::UNAUTHORIZED),
        (METRICS, StatusCode::FORBIDDEN),
        ("sk-unknown-0123456789", StatusCode::UNAUTHORIZED),
        ("sk-models-0123456789", StatusCode::FORBIDDEN),
    ] {
        assert_eq!(status(&deployment, api_key).await, expected, "{api_key}");
    }
    let models = get(&deployment, "/v1/models", "sk-models-0123456789").await;
    assert_eq!(models.status, StatusCode::OK);
    let received = deployment.engine().received();
    assert!(received
        .iter()
        .all(|request| !request.headers.contains_key("x-llm-proxy-peer")));

    let requests = finished_requests(&deployment, 3).await;
    let carol = find(&requests, |event| event["class"] == "idle");
    let expected = json!({"user": "carol", "key_id": "user", "status": 200,
        "outcome": "completed", "client": "127.0.0.1", "model": "mock-model"});
    assert_fields(&carol, expected);
    assert_eq!(carol["tokens"]["output"], 1);
    assert!(carol["time_to_first_token_seconds"].as_f64().unwrap() > 0.0);
    let alice = find(&requests, |event| event["user"] == "alice");
    assert_eq!(alice["key_id"], Value::Null);
    let expected = [
        "class_not_allowed",
        "blocked_api_key",
        "expired_api_key",
        "forbidden_role",
        "unknown_api_key",
        "route_not_allowed",
    ];
    assert_eq!(reasons(&deployment, 6).await, expected);
    let rejected = events(&deployment, "rejected");
    let unknown = find(&rejected, |event| event["reason"] == "unknown_api_key");
    assert_eq!(unknown["api_key"], "\"sk-...6789\" (21 characters)");
    let audit = std::fs::read_to_string(deployment.socket_path(AUDIT_LOG)).unwrap();
    assert!(!audit.contains("sk-unknown"));
}

#[tokio::test]
async fn per_key_limits() {
    let deployment = deploy_with_keys(vec![
        key("rpm", "rita", json!({"limits": {"rpm_limit": 2}})),
        key(
            "parallel",
            "paul",
            json!({"limits": {"max_parallel_requests": 1}}),
        ),
        // A request of chat_body(5) is estimated at 8 tokens and uses 12.
        key("tpm", "tina", json!({"limits": {"tpm_limit": 15}})),
    ])
    .await;
    let address = deployment.address;
    let rpm = "sk-rpm-0123456789";
    for _ in 0..2 {
        assert_eq!(status(&deployment, rpm).await, StatusCode::OK);
    }
    let limited = chat(address, rpm, &[], chat_body(1, false)).await;
    assert_eq!(limited.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(limited.headers["x-ratelimit-limit-requests"], "2");
    assert_eq!(limited.headers["x-ratelimit-remaining-requests"], "0");
    let retry_after = limited.headers["retry-after"].to_str().unwrap();
    assert!((1..=60).contains(&retry_after.parse::<u64>().unwrap()));

    let parallel = "sk-parallel-0123456789";
    let stream = open_chat_stream(address, parallel, &[], 30).await;
    let second = chat(address, parallel, &[], chat_body(1, false)).await;
    assert_eq!(second.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(second.headers["retry-after"], "1");
    stream_text(stream).await;
    assert_eq!(status(&deployment, parallel).await, StatusCode::OK);

    let tpm = || chat(address, "sk-tpm-0123456789", &[], chat_body(5, false));
    assert_eq!(tpm().await.status, StatusCode::OK);
    let is_tpm = |event: &Value| event["key_id"] == "tpm";
    wait_until(|| events(&deployment, "request").iter().any(is_tpm)).await;
    let limited = tpm().await;
    assert_eq!(limited.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(limited.headers["x-ratelimit-limit-tokens"], "15");
    let expected = ["rpm_limit", "max_parallel_requests", "tpm_limit"];
    assert_eq!(reasons(&deployment, 3).await, expected);
}

#[tokio::test]
async fn admin_api_manages_keys() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let panel = get(&deployment, "/admin", "").await;
    assert_eq!(panel.status, StatusCode::OK);
    assert!(String::from_utf8_lossy(&panel.body).contains("llm-proxy admin"));
    for (api_key, expected) in [
        ("", StatusCode::UNAUTHORIZED),
        (USER, StatusCode::FORBIDDEN),
        (METRICS, StatusCode::FORBIDDEN),
    ] {
        let (code, _) = admin_as(&deployment, api_key, "GET", "keys", Value::Null).await;
        assert_eq!(code, expected);
    }
    let (_, session) = admin(&deployment, "GET", "session", Value::Null).await;
    let expected = json!({"user": "root", "key_id": "admin", "role": "admin"});
    assert_eq!(session, expected);
    let (_, list) = admin(&deployment, "GET", "keys", Value::Null).await;
    assert_eq!(list["keys"].as_array().unwrap().len(), 3);
    assert_eq!(list["classes"][0], "interactive");
    assert!(!list.to_string().contains("key_hash"));

    let settings = json!({"user": "grace", "default_class": "batch", "limits": {"rpm_limit": 5}});
    let (code, created) = admin(&deployment, "POST", "keys", settings).await;
    assert_eq!(code, StatusCode::CREATED);
    let api_key = created["api_key"].as_str().unwrap().to_string();
    let path = format!("keys/{}", created["key"]["id"].as_str().unwrap());
    assert_fields(
        &created["key"],
        json!({"created_by": "root", "key_prefix": api_key[..8]}),
    );
    assert_eq!(status(&deployment, &api_key).await, StatusCode::OK);
    let keys = std::fs::read_to_string(deployment.socket_path(KEYS_FILE)).unwrap();
    let keys: Value = serde_json::from_str(&keys).unwrap();
    let hash = ApiKeyHash::from_api_key(&api_key).to_hex();
    assert_eq!(keys["keys"][3]["key_hash"], hash);
    let invalid = json!({"user": "x", "default_class": "gold"});
    let (code, _) = admin(&deployment, "POST", "keys", invalid).await;
    assert_eq!(code, StatusCode::BAD_REQUEST);

    let (code, _) = admin(&deployment, "PATCH", &path, json!({"blocked": true})).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(
        status(&deployment, &api_key).await,
        StatusCode::UNAUTHORIZED
    );
    let change = json!({"blocked": false, "limits": {"tpm_limit": 1000},
        "expires_at": "2099-01-01T00:00:00+01:00"});
    let (_, updated) = admin(&deployment, "PATCH", &path, change).await;
    assert_eq!(updated["key"]["limits"]["rpm_limit"], 5);
    assert_eq!(updated["key"]["limits"]["tpm_limit"], 1000);
    assert_eq!(updated["key"]["expires_at"], "2098-12-31T23:00:00Z");
    let (code, _) = admin(&deployment, "PATCH", &path, json!({"key_hash": "00"})).await;
    assert_eq!(code, StatusCode::BAD_REQUEST);

    let rotate = format!("{path}/rotate");
    let (_, rotated) = admin(&deployment, "POST", &rotate, Value::Null).await;
    let rotated_key = rotated["api_key"].as_str().unwrap().to_string();
    assert_eq!(
        status(&deployment, &api_key).await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(status(&deployment, &rotated_key).await, StatusCode::OK);
    let (code, _) = admin(&deployment, "DELETE", &path, Value::Null).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(
        status(&deployment, &rotated_key).await,
        StatusCode::UNAUTHORIZED
    );
    for (method, path, body, expected) in [
        ("DELETE", path.as_str(), Value::Null, StatusCode::NOT_FOUND),
        ("DELETE", "keys/admin", Value::Null, StatusCode::CONFLICT),
        (
            "PATCH",
            "keys/admin",
            json!({"role": "user"}),
            StatusCode::CONFLICT,
        ),
    ] {
        assert_eq!(admin(&deployment, method, path, body).await.0, expected);
    }

    wait_until(|| !events(&deployment, "key_deleted").is_empty()).await;
    let log = deployment.audit_log();
    let created = named(&log, "key_created");
    let actor = json!({"user": "root", "source": "admin_api"});
    assert_fields(&created[0]["actor"], actor);
    let updates = named(&log, "key_updated");
    assert_eq!(updates.len(), 2);
    assert_eq!(updates[0]["after"]["blocked"], true);
    assert_eq!(named(&log, "key_rotated").len(), 1);
    assert_eq!(named(&log, "key_deleted").len(), 1);
    let text = std::fs::read_to_string(deployment.socket_path(AUDIT_LOG)).unwrap();
    assert!(!text.contains("key_hash") && !text.contains(&api_key) && !text.contains(&rotated_key));
    let expected = ["missing_api_key", "forbidden_role", "forbidden_role"];
    assert_eq!(reasons(&deployment, 3).await[..3], expected);
}

#[tokio::test]
async fn keys_file_changes_apply_without_a_restart() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let added = "sk-added-0123456789";
    let mut keys = base_keys();
    keys.push(key("added", "heidi", json!({})));
    write_keys(&deployment, &keys);
    wait_until(|| !events(&deployment, "keys_reloaded").is_empty()).await;
    assert_eq!(status(&deployment, added).await, StatusCode::OK);
    let reloaded = events(&deployment, "keys_reloaded");
    assert_eq!(reloaded[0]["changed"], json!(["added"]));

    std::fs::write(deployment.socket_path(KEYS_FILE), "{not json").unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(status(&deployment, added).await, StatusCode::OK);
    let metrics = metrics(&deployment).await;
    assert!(!metrics.contains("llm_proxy_keys_file_errors_total 0"));
    let settings = json!({"user": "x", "default_class": "batch"});
    let (code, failure) = admin(&deployment, "POST", "keys", settings).await;
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    let message = failure["error"]["message"].as_str().unwrap();
    assert!(message.contains("is invalid"));

    keys.retain(|key| key["id"] != "user");
    write_keys(&deployment, &keys);
    wait_until(|| events(&deployment, "keys_reloaded").len() == 2).await;
    assert_eq!(status(&deployment, USER).await, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn metrics_endpoint() {
    let limits = json!({"limits": {"rpm_limit": 10, "tpd_limit": 1000}});
    let deployment = deploy_with_keys(vec![key("limited", "ivan", limits)]).await;
    let response = chat(deployment.address, USER, &[], chat_body(3, false)).await;
    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(
        status(&deployment, "sk-limited-0123456789").await,
        StatusCode::OK
    );
    let nobody = status(&deployment, "sk-nobody-0123456789").await;
    assert_eq!(nobody, StatusCode::UNAUTHORIZED);
    finished_requests(&deployment, 2).await;

    for (api_key, expected) in [
        ("", StatusCode::UNAUTHORIZED),
        (USER, StatusCode::FORBIDDEN),
        (ADMIN, StatusCode::OK),
    ] {
        assert_eq!(get(&deployment, "/metrics", api_key).await.status, expected);
    }
    let expected = [
        r#"llm_proxy_requests_total{user="carol",key="user",class="batch",path="/v1/chat/completions",status="200"} 1"#,
        r#"llm_proxy_tokens_total{user="carol",key="user",class="batch",type="output"} 3"#,
        r#"llm_proxy_rejected_requests_total{status="401",reason="unknown_api_key"} 1"#,
        r#"llm_proxy_request_duration_seconds_count{class="batch"} 2"#,
        r#"llm_proxy_keys{state="active"} 4"#,
        r#"llm_proxy_key_remaining_requests{key="limited",user="ivan"} 9"#,
        r#"llm_proxy_key_remaining_tokens{key="limited",user="ivan",window="day"}"#,
        r#"llm_proxy_rank_parallel_request_cap{rank="0"} 16"#,
    ];
    assert_metrics(&deployment, &expected).await;

    let closed = deploy(DeploymentOptions::default()).await;
    for path in ["/admin", "/metrics"] {
        assert_eq!(
            get(&closed, path, "sk-alice").await.status,
            StatusCode::NOT_FOUND
        );
    }
}

#[tokio::test]
async fn restarts_keep_running_requests_attributed_and_limits() {
    let limits = json!({"limits": {"rpm_limit": 2, "max_parallel_requests": 1}});
    let mut deployment = deploy_with_keys(vec![key("rpm", "rita", limits)]).await;
    let rpm = "sk-rpm-0123456789";
    let mut stream = open_chat_stream(deployment.address, rpm, &[], 60).await;
    stream.body_mut().frame().await.unwrap().unwrap();
    deployment.restart_middle_end().await;
    assert_eq!(
        status(&deployment, rpm).await,
        StatusCode::TOO_MANY_REQUESTS
    );
    stream_text(stream).await;
    assert_eq!(status(&deployment, rpm).await, StatusCode::OK);
    let limited = chat(deployment.address, rpm, &[], chat_body(1, false)).await;
    assert_eq!(limited.headers["x-ratelimit-limit-requests"], "2");
    let requests = finished_requests(&deployment, 2).await;
    let streamed = find(&requests, |event| event["tokens"]["output"] == 60);
    assert_fields(&streamed, json!({"key_id": "rpm", "stream": true}));
    let time_to_first_token = streamed["time_to_first_token_seconds"].as_f64().unwrap();
    assert!(time_to_first_token > 0.0);
    // The restart hands over the class and the times since the arrival.
    assert_eq!(streamed["class"], "batch");
    assert!(streamed["total_seconds"].as_f64().unwrap() >= time_to_first_token);
    assert!(streamed["queued_seconds"].as_f64().is_some());
}

#[tokio::test]
async fn aborted_requests_keep_their_status_and_first_token_time() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let headers = [("user-agent", "tester/1")];
    let mut stream = open_chat_stream(deployment.address, USER, &headers, 500).await;
    assert_eq!(stream.status(), StatusCode::OK);
    stream.body_mut().frame().await.unwrap().unwrap();
    drop(stream);
    let aborted = finished_requests(&deployment, 1).await.remove(0);
    let expected = json!({"outcome": "aborted", "status": 200, "user_agent": "tester/1"});
    assert_fields(&aborted, expected);
    assert!(aborted["time_to_first_token_seconds"].as_f64().is_some());
    assert_metrics(&deployment, &[r#"status="aborted"} 1"#]).await;
    // Keys of accounts.json have no access to /metrics or the admin API.
    let forbidden = get(&deployment, "/metrics", "sk-alice").await;
    assert_eq!(forbidden.status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn starts_without_the_keys_file() {
    let general = json!({"gateway": {"client_ip_header": "x-forwarded-for"}});
    let options = DeploymentOptions {
        keys: Some(Value::Null),
        ..gateway_options(Vec::new(), general)
    };
    let deployment = deploy(options).await;
    assert_eq!(status(&deployment, "sk-alice").await, StatusCode::OK);
    assert_eq!(status(&deployment, USER).await, StatusCode::UNAUTHORIZED);
    write_keys(&deployment, &base_keys());
    wait_until(|| !events(&deployment, "keys_reloaded").is_empty()).await;
    let forwarded = [("x-forwarded-for", "203.0.113.9, 198.51.100.7")];
    let response = chat(deployment.address, USER, &forwarded, chat_body(1, false)).await;
    assert_eq!(response.status, StatusCode::OK);
    let requests = finished_requests(&deployment, 2).await;
    let carol = find(&requests, |event| event["user"] == "carol");
    assert_eq!(carol["client"], "198.51.100.7");
}

#[tokio::test]
async fn create_key_command() {
    let directory = tempfile::tempdir().unwrap();
    let file = |name: &str| directory.path().join(name);
    let read = |name: &str| std::fs::read_to_string(file(name)).unwrap();
    let general = json!({
        "frontend": {"listen": "127.0.0.1:0"},
        "middle_end_socket": file("middle-end.sock"),
        "gateway": {"keys_file": file("keys.json"), "audit_log": file("audit.jsonl")},
    });
    std::fs::write(file("general.json"), general.to_string()).unwrap();
    let accounts = DeploymentOptions::default().accounts.to_string();
    std::fs::write(file("accounts.json"), accounts).unwrap();
    let run = |command: &str| {
        std::process::Command::new(env!("CARGO_BIN_EXE_llm-proxy"))
            .args(command.split(' '))
            .args(["--general", file("general.json").to_str().unwrap()])
            .args(["--accounts", file("accounts.json").to_str().unwrap()])
            .args(["--backend", file("backend.json").to_str().unwrap()])
            .output()
            .unwrap()
    };
    let create = "create-key --user root --default-class batch --role admin";
    // A missing keys file may be a mount which is down: only created on request.
    let refused = run(create);
    assert!(!refused.status.success() && !file("keys.json").exists());
    let output = run(&format!("{create} --new-keys-file"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    let api_key = String::from_utf8(output.stdout).unwrap().trim().to_string();
    let keys: Value = serde_json::from_str(&read("keys.json")).unwrap();
    let hash = ApiKeyHash::from_api_key(&api_key).to_hex();
    let expected = json!({"key_hash": hash, "role": "admin", "created_by": "cli"});
    assert_fields(&keys["keys"][0], expected);
    assert!(read("audit.jsonl").contains("\"key_created\""));
    assert!(run("check-config").status.success());
    assert!(!run("create-key --user x --default-class gold")
        .status
        .success());
    std::fs::remove_file(file("keys.json")).unwrap();
    assert!(!run("check-config").status.success());
}

#[tokio::test]
async fn passthrough_requests_count_against_limits_and_are_audited() {
    let limits = json!({"limits": {"rpm_limit": 2}});
    let deployment = deploy_with_keys(vec![key("rpm", "rita", limits)]).await;
    let rpm = "sk-rpm-0123456789";
    let body = json!({"prompt": "hi"}).to_string();
    let tokenize = || request(&deployment, "POST", "/tokenize", rpm, body.clone());
    // Requests with a body count like LiteLLM counts LLM calls, reads don't.
    for _ in 0..3 {
        let models = get(&deployment, "/v1/models", rpm).await;
        assert_eq!(models.status, StatusCode::OK);
    }
    assert_ne!(tokenize().await.status, StatusCode::TOO_MANY_REQUESTS);
    assert_ne!(tokenize().await.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(tokenize().await.status, StatusCode::TOO_MANY_REQUESTS);
    let finished = || -> Vec<String> {
        events(&deployment, "request")
            .iter()
            .map(|event| {
                assert!(event["request"].is_null() && event["class"].is_null());
                assert_fields(event, json!({"key_id": "rpm", "outcome": "completed"}));
                let method = event["method"].as_str().unwrap();
                format!("{method} {}", event["path"].as_str().unwrap())
            })
            .collect()
    };
    wait_until(|| finished().len() == 5).await;
    let mut paths = finished();
    paths.sort();
    let models = "GET /v1/models";
    assert_eq!(
        paths,
        [models, models, models, "POST /tokenize", "POST /tokenize"]
    );
    let expected = r#"llm_proxy_requests_total{user="rita",key="rpm",class="",path="/v1/models",status="200"} 3"#;
    assert_metrics(&deployment, &[expected]).await;
}

#[tokio::test]
async fn admin_api_guards() {
    let settings = json!({"role": "admin", "allowed_routes": ["/v1/models"]});
    let deployment = deploy_with_keys(vec![key("scoped-admin", "sam", settings)]).await;
    // The last lasting admin can't be given an expiry, which would lock everyone out later.
    let soon = jiff::Timestamp::now() + jiff::SignedDuration::from_secs(5);
    let change = json!({"expires_at": format!("{soon:.0}")});
    let (code, _) = admin(&deployment, "PATCH", "keys/admin", change).await;
    assert_eq!(code, StatusCode::CONFLICT);
    // Route patterns which can't match anything are refused.
    for route in ["*", "/v1/*/completions", "v1/models"] {
        let settings = json!({"user": "x", "default_class": "batch", "allowed_routes": [route]});
        let (code, _) = admin(&deployment, "POST", "keys", settings).await;
        assert_eq!(code, StatusCode::BAD_REQUEST, "{route}");
    }
    // allowed_routes also covers the admin API.
    let scoped = "sk-scoped-admin-0123456789";
    let (code, _) = admin_as(&deployment, scoped, "GET", "keys", Value::Null).await;
    assert_eq!(code, StatusCode::FORBIDDEN);
    // A mistake in a request isn't a keys file failure.
    let (code, _) = admin(&deployment, "DELETE", "keys/nope", Value::Null).await;
    assert_eq!(code, StatusCode::NOT_FOUND);
    let metrics = metrics(&deployment).await;
    assert!(!metrics.contains("llm_proxy_keys_file_errors_total 1"));
    // The panel can't be framed.
    let panel = send(deployment.address, Method::GET, "/admin", &[], "").await;
    let headers = &panel.headers;
    assert_eq!(headers["x-frame-options"], "DENY");
    assert_eq!(headers["content-security-policy"], "frame-ancestors 'none'");
    // Keys in api-key, and x-api-key next to an empty Authorization header.
    let azure = post_chat(
        deployment.address,
        &[("api-key", USER)],
        chat_body(1, false),
    )
    .await;
    assert_eq!(azure.status, StatusCode::OK);
    let headers = [("authorization", ""), ("x-api-key", USER)];
    let empty = post_chat(deployment.address, &headers, chat_body(1, false)).await;
    assert_eq!(empty.status, StatusCode::OK);
    let received = deployment.engine().received();
    assert!(received
        .iter()
        .all(|request| !request.headers.contains_key("api-key")));
}

#[tokio::test]
async fn closed_clients_release_their_requests_at_once() {
    let limits = json!({"limits": {"max_parallel_requests": 1}});
    let deployment = deploy_with_keys(vec![key("one", "olga", limits)]).await;
    // One token per 30 seconds: nothing is written to the client for a long time.
    deployment
        .engine()
        .set_large_batch_token_interval(0, Duration::from_secs(30));
    let stream = open_chat_stream(deployment.address, "sk-one-0123456789", &[], 5).await;
    assert_eq!(stream.status(), StatusCode::OK);
    drop(stream);
    let released = r#"llm_proxy_key_in_flight_requests{key="one",user="olga"} 0"#;
    wait_for_metric(&deployment, released, 5).await;
    // Also when the client sent more than the request (here a stray CRLF) before it left.
    let request = raw_json_request("sk-one-0123456789", &chat_body(5, true)) + "\r\n";
    let status = raw_status(&deployment, request.as_bytes(), None).await;
    assert!(status.starts_with("HTTP/1.1 200"), "{status}");
    wait_for_metric(&deployment, released, 5).await;
}

#[tokio::test]
async fn paths_like_litellm() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let post = |path: &'static str| request(&deployment, "POST", path, USER, chat_body(1, false));
    // Base URLs without /v1.
    assert_eq!(post("/chat/completions").await.status, StatusCode::OK);
    let received = deployment.engine().received();
    assert_eq!(received.last().unwrap().path, "/v1/chat/completions");
    // Non-canonical paths and wrong methods never reach the engine.
    for path in [
        "//v1/chat/completions",
        "/v1/chat/completions/",
        "/x/../v1/chat/completions",
    ] {
        assert_eq!(post(path).await.status, StatusCode::NOT_FOUND, "{path}");
    }
    let get_chat = get(&deployment, "/v1/chat/completions", USER).await;
    assert_eq!(get_chat.status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(get_chat.headers["allow"], "POST");
    assert_eq!(deployment.engine().received().len(), received.len());
    // Metrics label paths by their configured pattern.
    for model in ["a", "b", "c"] {
        get(&deployment, &format!("/v1/models/{model}"), USER).await;
    }
    assert_metrics(&deployment, &[r#"path="/v1/models/*",status="404"} 3"#]).await;
    // Probes without an API key.
    for path in ["/health/liveliness", "/health/readiness"] {
        let probe = send(deployment.address, Method::GET, path, &[], "").await;
        assert_eq!(probe.status, StatusCode::OK, "{path}");
    }
}

#[tokio::test]
async fn round_two_key_rules() {
    let settings = json!({"role": "admin", "expires_at": "2099-01-01T00:00:00Z"});
    let expiring = key("expiring", "eve", settings);
    let routes = json!({"allowed_routes": ["/chat/completions"]});
    let scoped = key("scoped", "sam", routes);
    let deployment = deploy_with_keys(vec![scoped, expiring.clone()]).await;
    // The only lasting admin can't expire, but other keys can change freely.
    let soon = json!({"expires_at": "2099-01-01T00:00:00Z"});
    let (code, _) = admin(&deployment, "PATCH", "keys/admin", soon).await;
    assert_eq!(code, StatusCode::CONFLICT);
    let (code, _) = admin(&deployment, "PATCH", "keys/user", json!({"blocked": true})).await;
    assert_eq!(code, StatusCode::OK);
    // Routes may name the path as the client sends it, before the /v1 rewrite.
    let scoped = "sk-scoped-0123456789";
    let body = chat_body(1, false);
    let short = request(&deployment, "POST", "/chat/completions", scoped, body).await;
    assert_eq!(short.status, StatusCode::OK);
    // Without any lasting admin (e.g. a policy of expiring admins), changes still work.
    write_keys(&deployment, &[expiring, base_keys()[1].clone()]);
    wait_until(|| events(&deployment, "keys_reloaded").len() == 1).await;
    let block = json!({"blocked": true});
    let expiring = "sk-expiring-0123456789";
    let (code, _) = admin_as(&deployment, expiring, "PATCH", "keys/user", block).await;
    assert_eq!(code, StatusCode::OK);
    // api-key wins over x-api-key, like in LiteLLM, and foreign credentials are stripped.
    let headers = [
        ("api-key", expiring),
        ("x-api-key", "sk-wrong"),
        ("x-goog-api-key", "secret"),
    ];
    let both = post_chat(deployment.address, &headers, chat_body(1, false)).await;
    assert_eq!(both.status, StatusCode::OK);
    let forwarded = deployment.engine().received().pop().unwrap();
    assert!(!forwarded.headers.contains_key("x-goog-api-key"));
    // LiteLLM's other liveness path.
    let probe = send(deployment.address, Method::GET, "/health/liveness", &[], "").await;
    assert_eq!(probe.status, StatusCode::OK);
}

#[tokio::test]
async fn reservations_survive_restarts() {
    // The stream reserves 93 tokens, the second request 43.
    let limits = json!({"limits": {"tpm_limit": 120}});
    let mut deployment = deploy_with_keys(vec![key("tpm", "tina", limits)]).await;
    let address = deployment.address;
    let tpm = "sk-tpm-0123456789";
    // The running stream reserves about 90 output tokens plus its prompt.
    let mut stream = open_chat_stream(address, tpm, &[], 90).await;
    stream.body_mut().frame().await.unwrap().unwrap();
    let second = || chat(address, tpm, &[], chat_body(40, false));
    assert_eq!(second().await.status, StatusCode::TOO_MANY_REQUESTS);
    deployment.restart_middle_end().await;
    assert_eq!(second().await.status, StatusCode::TOO_MANY_REQUESTS);
    stream_text(stream).await;
}

#[tokio::test]
async fn unwritten_audit_events_survive_restarts() {
    // The audit log's directory doesn't exist yet: every write fails.
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("later").join("audit.jsonl");
    let general = json!({"gateway": {"audit_log": path}});
    let mut deployment = deploy(gateway_options(Vec::new(), general)).await;
    let wrong = status(&deployment, "sk-wrong-0123456789").await;
    assert_eq!(wrong, StatusCode::UNAUTHORIZED);
    wait_for_metric(&deployment, "llm_proxy_audit_log_failing 1", 10).await;
    deployment.restart_middle_end().await;
    std::fs::create_dir(path.parent().unwrap()).unwrap();
    assert_eq!(status(&deployment, USER).await, StatusCode::OK);
    // The rejection which the old process couldn't write comes first.
    let log = || -> Vec<Value> {
        std::fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    wait_until(|| named(&log(), "request").len() == 1).await;
    let log = log();
    let names = event_names(&log);
    assert_eq!(names[0], "rejected", "{names:?}");
}

// Regression tests for the findings of the A/B hunt against LiteLLM (docs/litellm-findings.md).

/// Hand edits which an admin API change picks up are audited too.
#[tokio::test]
async fn hand_edits_picked_up_by_admin_changes_are_audited() {
    let general = json!({"gateway": {"keys_file_poll_seconds": 3600}});
    let deployment = deploy(gateway_options(Vec::new(), general)).await;
    let mut keys = base_keys();
    keys[1]["blocked"] = json!(true);
    write_keys(&deployment, &keys);
    let change = json!({"description": "changed"});
    let (code, _) = admin(&deployment, "PATCH", "keys/metrics", change).await;
    assert_eq!(code, StatusCode::OK);
    assert_eq!(status(&deployment, USER).await, StatusCode::UNAUTHORIZED);
    wait_until(|| !events(&deployment, "key_updated").is_empty()).await;
    let log = deployment.audit_log();
    assert_eq!(event_names(&log)[..2], ["keys_reloaded", "key_updated"]);
    assert_eq!(log[0]["changed"], json!(["user"]));
}

/// Engine errors are recorded at once (not after waiting for token reports), and an
/// engine which breaks off a started response is `failed`, not a client abort.
#[tokio::test]
async fn engine_errors_and_failures_in_the_audit_log() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let body = json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 5, "mock_status": 400});
    let rejected = chat(deployment.address, USER, &[], body.to_string()).await;
    assert_eq!(rejected.status, StatusCode::BAD_REQUEST);
    let started = Instant::now();
    wait_until(|| !events(&deployment, "request").is_empty()).await;
    assert!(started.elapsed() < Duration::from_secs(2), "recorded late");
    assert_eq!(events(&deployment, "request")[0]["status"], 400);

    let body = json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 10, "stream": true, "mock_drop_after": 2});
    let authorization = format!("Bearer {USER}");
    let headers = [("authorization", authorization.as_str())];
    let response = support::open(
        deployment.address,
        Method::POST,
        "/v1/chat/completions",
        &headers,
        body.to_string(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.into_body().collect().await.is_err());
    wait_until(|| events(&deployment, "request").len() == 2).await;
    let failed = events(&deployment, "request").remove(1);
    assert_fields(&failed, json!({"outcome": "failed", "status": 200}));
    assert!(failed["time_to_first_token_seconds"].as_f64().is_some());
}

/// Fields which clients control are cut in the audit log.
#[tokio::test]
async fn long_fields_are_cut_in_the_audit_log() {
    let general = json!({"gateway": {"client_ip_header": "x-forwarded-for"}});
    let deployment = deploy(gateway_options(Vec::new(), general)).await;
    let path = format!("/v1/models/{}", "p".repeat(2000));
    let agent = "a".repeat(2000);
    let headers = [
        ("authorization", "Bearer sk-unknown-0123456789"),
        ("user-agent", agent.as_str()),
    ];
    let rejected = send(deployment.address, Method::GET, &path, &headers, "").await;
    assert_eq!(rejected.status, StatusCode::UNAUTHORIZED);
    let client = "c".repeat(2000);
    let headers = [("x-forwarded-for", client.as_str())];
    let body = json!({"model": "m".repeat(2000), "messages": [{"role": "user", "content": "x"}], "max_tokens": 1});
    chat(deployment.address, USER, &headers, body.to_string()).await;
    wait_until(|| !events(&deployment, "request").is_empty()).await;
    let rejected = events(&deployment, "rejected").remove(0);
    let request = events(&deployment, "request").remove(0);
    for value in [
        &rejected["path"],
        &rejected["user_agent"],
        &request["model"],
        &request["client"],
    ] {
        assert!(value.as_str().unwrap().len() <= 259, "{value}");
    }
}

/// The proxy's own 405, 408 and 413 count in the rejection metrics.
#[tokio::test]
async fn the_proxy_s_own_rejections_are_counted() {
    let general = json!({"limits": {"max_request_body_bytes": 1000}});
    let deployment = deploy(gateway_options(Vec::new(), general)).await;
    let big = json!({"messages": [{"role": "user", "content": "x".repeat(2000)}]});
    let response = chat(deployment.address, USER, &[], big.to_string()).await;
    assert_eq!(response.status, StatusCode::PAYLOAD_TOO_LARGE);
    let wrong_method = get(&deployment, "/v1/chat/completions", USER).await;
    assert_eq!(wrong_method.status, StatusCode::METHOD_NOT_ALLOWED);
    let expected = [
        r#"llm_proxy_rejected_requests_total{status="413",reason="request_body_too_large"} 1"#,
        r#"llm_proxy_rejected_requests_total{status="405",reason="method_not_allowed"} 1"#,
    ];
    assert_metrics(&deployment, &expected).await;
}

/// A request which the proxy turns away because its queue is full doesn't count against
/// its key's rpm, and its 503 says when to retry.
#[tokio::test]
async fn full_queues_refund_the_key() {
    let rpm = key("rpm", "rita", json!({"limits": {"rpm_limit": 5}}));
    let options = DeploymentOptions {
        rank_capacity: 1,
        ..gateway_options(vec![rpm], json!({"limits": {"max_queued_requests": 1}}))
    };
    let deployment = deploy(options).await;
    let address = deployment.address;
    let running = running_stream(&deployment).await;
    let queued = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-bob", &[], 1).await).await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let full = chat(address, "sk-rpm-0123456789", &[], chat_body(1, false)).await;
    assert_eq!(full.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(full.headers["retry-after"], "1");
    assert_eq!(usage(&deployment, "rpm").await["requests_last_minute"], 0);
    running.await.unwrap();
    queued.await.unwrap();
}

/// A key's hash as an account's virtual user (LiteLLM's key hash) makes the
/// request that key's: its user and limits are shared with direct requests.
#[tokio::test]
async fn virtual_user_key_hashes_act_as_the_key() {
    let deployment = deploy_with_keys(vec![key(
        "rpm",
        "rita",
        json!({"limits": {"rpm_limit": 2}}),
    )])
    .await;
    let address = deployment.address;
    let hash = ApiKeyHash::from_api_key("sk-rpm-0123456789").to_hex();
    let via_litellm = [("x-end-user", hash.as_str())];
    let direct = chat(address, "sk-rpm-0123456789", &[], chat_body(1, false)).await;
    assert_eq!(direct.status, StatusCode::OK);
    let forwarded = chat(address, "sk-litellm", &via_litellm, chat_body(1, false)).await;
    assert_eq!(forwarded.status, StatusCode::OK);
    let limited = chat(address, "sk-litellm", &via_litellm, chat_body(1, false)).await;
    assert_eq!(limited.status, StatusCode::TOO_MANY_REQUESTS);
    let requests = finished_requests(&deployment, 2).await;
    assert_eq!(requests[0]["user"], "rita");
    assert_eq!(requests[1]["user"], "rita");
    let unknown = ApiKeyHash::from_api_key("sk-unknown").to_hex();
    let other = chat(
        address,
        "sk-litellm",
        &[("x-end-user", unknown.as_str())],
        chat_body(1, false),
    )
    .await;
    assert_eq!(other.status, StatusCode::OK);
    let requests = finished_requests(&deployment, 3).await;
    assert_eq!(
        requests[2]["user"],
        format!("litellm/{unknown}"),
        "an unknown hash stays the account's virtual user"
    );
    let headers = [("authorization", "Bearer sk-litellm"), via_litellm[0]];
    let admin = send(address, Method::GET, "/admin/keys", &headers, "").await;
    assert_eq!(admin.status, StatusCode::FORBIDDEN);
    wait_until(|| events(&deployment, "rejected").len() >= 2).await;
    let rejected = events(&deployment, "rejected");
    let forbidden = find(&rejected, |event| event["reason"] == "forbidden_role");
    assert_eq!(
        (&forbidden["user"], &forbidden["key_id"]),
        (&json!("litellm"), &Value::Null),
        "the admin API takes the account's key as the account's, also in the audit log"
    );
}

/// Virtual users (a header value) are cut: they become labels and log fields.
#[tokio::test]
async fn virtual_users_are_cut() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let user = "u".repeat(500);
    let headers = [("x-end-user", user.as_str())];
    let body = chat_body(1, false);
    let response = chat(deployment.address, "sk-litellm", &headers, body).await;
    assert_eq!(response.status, StatusCode::OK);
    let requests = finished_requests(&deployment, 1).await;
    let length = requests[0]["user"].as_str().unwrap().len();
    assert_eq!(length, "litellm/".len() + 128);
}

/// The token estimates of requests in flight show in the key's usage and metrics.
#[tokio::test]
async fn reservations_show_in_usage_and_metrics() {
    let limits = json!({"limits": {"tpm_limit": 1000}});
    let deployment = deploy_with_keys(vec![key("tpm", "tina", limits)]).await;
    let mut stream = open_chat_stream(deployment.address, "sk-tpm-0123456789", &[], 100).await;
    stream.body_mut().frame().await.unwrap().unwrap();
    let usage = usage(&deployment, "tpm").await;
    let reserved = usage["reserved_tokens"].as_u64().unwrap();
    assert!(reserved >= 100, "{usage}");
    let remaining = 1000 - reserved;
    let expected = format!(
        r#"llm_proxy_key_remaining_tokens{{key="tpm",user="tina",window="minute"}} {remaining}"#
    );
    assert_metrics(&deployment, &[&expected]).await;
    stream_text(stream).await;
}

/// Passthrough requests in flight across a restart are still recorded.
#[tokio::test]
async fn passthrough_requests_survive_restarts() {
    let mut deployment = deploy_with_keys(Vec::new()).await;
    let address = deployment.address;
    let tokenize = tokio::spawn(async move {
        let body = json!({"prompt": "x", "mock_delay_ms": 300}).to_string();
        let authorization = format!("Bearer {USER}");
        let headers = [("authorization", authorization.as_str())];
        send(address, Method::POST, "/tokenize", &headers, body).await
    });
    wait_until(|| {
        let received = deployment.engine().received();
        received.iter().any(|request| request.path == "/tokenize")
    })
    .await;
    deployment.restart_middle_end().await;
    assert_eq!(tokenize.await.unwrap().status, StatusCode::NOT_FOUND);
    let is_recorded = |event: &Value| event["path"] == "/tokenize" && event["status"] == 404;
    wait_until(|| events(&deployment, "request").iter().any(is_recorded)).await;
}

/// A queued stream whose client left after the proxy's early head got that head's 200.
#[tokio::test]
async fn queued_streams_left_after_the_early_head_are_aborted_with_200() {
    let general = json!({"limits": {"keepalive_interval_seconds": 0.05}});
    let options = DeploymentOptions {
        rank_capacity: 1,
        ..gateway_options(Vec::new(), general)
    };
    let deployment = deploy(options).await;
    let running = running_stream(&deployment).await;
    let mut queued = open_chat_stream(deployment.address, USER, &[], 5).await;
    assert_eq!(queued.status(), StatusCode::OK);
    queued.body_mut().frame().await.unwrap().unwrap();
    drop(queued);
    wait_until(|| !events(&deployment, "request").is_empty()).await;
    let aborted = events(&deployment, "request").remove(0);
    let expected = json!({"key_id": "user", "outcome": "aborted", "status": 200});
    assert_fields(&aborted, expected);
    running.await.unwrap();
}

/// A restart keeps the time a request generated before it.
#[tokio::test]
async fn generating_time_survives_restarts() {
    let mut deployment = deploy_with_keys(Vec::new()).await;
    let mut stream = open_chat_stream(deployment.address, USER, &[], 60).await;
    let mut received = String::new();
    while received.matches("token").count() < 30 {
        let frame = stream.body_mut().frame().await.unwrap().unwrap();
        received += &String::from_utf8_lossy(frame.data_ref().unwrap());
    }
    deployment.restart_middle_end().await;
    stream_text(stream).await;
    let event = finished_requests(&deployment, 1).await.remove(0);
    let generating = event["generating_seconds"].as_f64().unwrap();
    assert!(generating >= 0.5, "{event}");
}

/// A restart starts with the keys of the old process, not with the file its pre-flight
/// check read (which an admin change may have replaced since).
#[tokio::test]
async fn restarts_keep_the_current_keys() {
    let general = json!({"gateway": {"keys_file_poll_seconds": 3600}});
    let mut deployment = deploy(gateway_options(Vec::new(), general)).await;
    let settings = json!({"user": "new", "default_class": "batch"});
    let (_, created) = admin(&deployment, "POST", "keys", settings).await;
    let api_key = created["api_key"].as_str().unwrap().to_string();
    // The test deployment restarts with the configuration texts it started with.
    deployment.restart_middle_end().await;
    assert_eq!(status(&deployment, &api_key).await, StatusCode::OK);
    assert!(events(&deployment, "keys_reloaded").is_empty());
}

/// Scheduling log events which couldn't be written go to the new process.
#[tokio::test]
async fn unwritten_scheduling_log_events_survive_restarts() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("later").join("scheduling.log");
    let general = json!({"scheduling_log": path});
    let mut deployment = deploy(gateway_options(Vec::new(), general)).await;
    assert_eq!(status(&deployment, USER).await, StatusCode::OK);
    wait_for_metric(&deployment, "llm_proxy_scheduling_log_failing 1", 10).await;
    deployment.restart_middle_end().await;
    std::fs::create_dir(path.parent().unwrap()).unwrap();
    assert_eq!(status(&deployment, USER).await, StatusCode::OK);
    let finished = || {
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        text.lines()
            .filter(|line| line.contains(r#""event":"finished""#))
            .count()
    };
    wait_until(|| finished() == 2).await;
}

/// A restart re-admits queued requests in the order they arrived.
#[tokio::test]
async fn restarts_keep_the_queue_in_order() {
    let options = DeploymentOptions {
        rank_capacity: 1,
        ..gateway_options(Vec::new(), json!({}))
    };
    let mut deployment = deploy(options).await;
    let address = deployment.address;
    let running = running_stream(&deployment).await;
    let mut queued = Vec::new();
    for tokens in 11..=16 {
        queued.push(tokio::spawn(async move {
            chat(address, USER, &[], chat_body(tokens, false)).await
        }));
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    deployment.restart_middle_end().await;
    for request in queued {
        assert_eq!(request.await.unwrap().status, StatusCode::OK);
    }
    running.await.unwrap();
    let received = deployment.engine().received();
    let order: Vec<u64> = received
        .iter()
        .filter_map(|request| request.body["max_tokens"].as_u64())
        .filter(|tokens| (11..=16).contains(tokens))
        .collect();
    assert_eq!(order, [11, 12, 13, 14, 15, 16]);
}
