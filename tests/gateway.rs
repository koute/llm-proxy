mod support;

use std::time::Duration;

use hyper::{Method, StatusCode};
use serde_json::{json, Value};

use llm_proxy::config::ApiKeyHash;
use support::{
    chat, chat_body, deploy, open_chat_stream, send, stream_text, wait_until, ClientResponse,
    Deployment, DeploymentOptions, KEYS_FILE,
};

const ADMIN_KEY: &str = "sk-admin-0123456789abcdef";
const USER_KEY: &str = "sk-user-0123456789abcdef";
const METRICS_KEY: &str = "sk-metrics-0123456789abcdef";

fn hash(api_key: &str) -> String {
    ApiKeyHash::from_api_key(api_key).to_hex()
}

fn key(id: &str, user: &str, api_key: &str, settings: Value) -> Value {
    let mut key = json!({
        "id": id,
        "user": user,
        "key_hash": hash(api_key),
        "default_class": "batch",
        "allowed_classes": ["idle"],
    });
    for (field, value) in settings.as_object().unwrap() {
        key[field] = value.clone();
    }
    key
}

fn base_keys() -> Vec<Value> {
    vec![
        key("admin", "root", ADMIN_KEY, json!({"role": "admin"})),
        key("user", "carol", USER_KEY, json!({})),
        key(
            "metrics",
            "prometheus",
            METRICS_KEY,
            json!({"role": "metrics"}),
        ),
    ]
}

async fn deploy_with_keys(extra_keys: Vec<Value>) -> Deployment {
    let mut keys = base_keys();
    keys.extend(extra_keys);
    deploy(DeploymentOptions {
        keys: Some(json!({"keys": keys})),
        ..DeploymentOptions::default()
    })
    .await
}

async fn admin(
    deployment: &Deployment,
    method: Method,
    path: &str,
    api_key: &str,
    body: Value,
) -> (StatusCode, Value) {
    let authorization = format!("Bearer {api_key}");
    let body = if body.is_null() {
        String::new()
    } else {
        body.to_string()
    };
    let response = send(
        deployment.address,
        method,
        &format!("/admin/api/{path}"),
        &[("authorization", authorization.as_str())],
        body,
    )
    .await;
    let value = serde_json::from_slice(&response.body).unwrap_or(Value::Null);
    (response.status, value)
}

fn error_message(response: &ClientResponse) -> String {
    let value: Value = serde_json::from_slice(&response.body).unwrap();
    value["error"]["message"].as_str().unwrap().to_string()
}

fn events<'a>(log: &'a [Value], name: &str) -> Vec<&'a Value> {
    log.iter().filter(|event| event["event"] == name).collect()
}

async fn wait_for_audit(deployment: &Deployment, condition: impl Fn(&[Value]) -> bool) {
    wait_until(|| condition(&deployment.audit_log())).await;
}

async fn get_metrics(deployment: &Deployment, api_key: &str) -> ClientResponse {
    let authorization = format!("Bearer {api_key}");
    send(
        deployment.address,
        Method::GET,
        "/metrics",
        &[("authorization", authorization.as_str())],
        "",
    )
    .await
}

#[tokio::test]
async fn keys_from_the_keys_file_authenticate_requests() {
    let deployment = deploy_with_keys(vec![
        key(
            "blocked",
            "dave",
            "sk-blocked-0123456789",
            json!({"blocked": true}),
        ),
        key(
            "expired",
            "erin",
            "sk-expired-0123456789",
            json!({"expires_at": "2020-01-01T00:00:00Z"}),
        ),
        key(
            "models-only",
            "frank",
            "sk-models-0123456789",
            json!({"allowed_routes": ["/v1/models"]}),
        ),
    ])
    .await;
    let address = deployment.address;

    let response = chat(address, USER_KEY, &[], chat_body(5, false)).await;
    assert_eq!(response.status, StatusCode::OK);
    let response = chat(
        address,
        USER_KEY,
        &[("x-priority", "idle")],
        chat_body(5, false),
    )
    .await;
    assert_eq!(response.status, StatusCode::OK);
    let response = chat(
        address,
        USER_KEY,
        &[("x-priority", "interactive")],
        chat_body(5, false),
    )
    .await;
    assert_eq!(response.status, StatusCode::FORBIDDEN);
    assert_eq!(
        chat(address, "sk-alice", &[], chat_body(5, false))
            .await
            .status,
        StatusCode::OK
    );

    let blocked = chat(address, "sk-blocked-0123456789", &[], chat_body(5, false)).await;
    assert_eq!(
        (blocked.status, error_message(&blocked).as_str()),
        (StatusCode::FORBIDDEN, "API key is blocked")
    );
    let expired = chat(address, "sk-expired-0123456789", &[], chat_body(5, false)).await;
    assert_eq!(
        (expired.status, error_message(&expired).as_str()),
        (StatusCode::UNAUTHORIZED, "API key expired")
    );
    assert_eq!(
        chat(address, METRICS_KEY, &[], chat_body(5, false))
            .await
            .status,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        chat(address, "sk-unknown-0123456789", &[], chat_body(5, false))
            .await
            .status,
        StatusCode::UNAUTHORIZED
    );
    let models_only = chat(address, "sk-models-0123456789", &[], chat_body(5, false)).await;
    assert_eq!(models_only.status, StatusCode::FORBIDDEN);
    let models = send(
        address,
        Method::GET,
        "/v1/models",
        &[("authorization", "Bearer sk-models-0123456789")],
        "",
    )
    .await;
    assert_eq!(models.status, StatusCode::OK);

    let received = deployment.engine().received();
    assert!(received
        .iter()
        .all(|request| !request.headers.contains_key("x-llm-proxy-peer")));
    assert!(received
        .iter()
        .all(|request| !request.headers.contains_key("authorization")));

    wait_for_audit(&deployment, |log| events(log, "request").len() == 4).await;
    let log = deployment.audit_log();
    let requests = events(&log, "request");
    let carol = requests
        .iter()
        .find(|event| event["key_id"] == "user" && event["class"] == "idle")
        .unwrap();
    assert_eq!(carol["user"], "carol");
    assert_eq!(carol["path"], "/v1/chat/completions");
    assert_eq!(carol["status"], 200);
    assert_eq!(carol["outcome"], "completed");
    assert_eq!(carol["client"], "127.0.0.1");
    assert_eq!(carol["model"], "mock-model");
    assert_eq!(carol["tokens"]["output"], 5);
    assert!(carol["time_to_first_token_seconds"].as_f64().unwrap() > 0.0);
    let models = requests
        .iter()
        .find(|event| event["path"] == "/v1/models")
        .unwrap();
    assert_eq!(
        (&models["key_id"], &models["status"]),
        (&json!("models-only"), &json!(200))
    );
    let alice = requests
        .iter()
        .find(|event| event["user"] == "alice")
        .unwrap();
    assert_eq!(alice["key_id"], Value::Null);

    let rejected = events(&log, "rejected");
    let reasons: Vec<&str> = rejected
        .iter()
        .map(|event| event["reason"].as_str().unwrap())
        .collect();
    assert_eq!(
        reasons,
        [
            "class_not_allowed",
            "blocked_api_key",
            "expired_api_key",
            "forbidden_role",
            "unknown_api_key",
            "route_not_allowed"
        ]
    );
    let unknown = rejected
        .iter()
        .find(|event| event["reason"] == "unknown_api_key")
        .unwrap();
    assert_eq!(unknown["api_key"], "\"sk-...6789\" (21 characters)");
    assert_eq!(rejected[1]["key_id"], "blocked");
    assert_eq!(rejected[1]["user"], "dave");
    assert!(
        !std::fs::read_to_string(deployment.socket_path("audit.jsonl"))
            .unwrap()
            .contains("sk-unknown")
    );
}

#[tokio::test]
async fn per_key_limits() {
    let deployment = deploy_with_keys(vec![
        key(
            "rpm",
            "rita",
            "sk-rpm-0123456789",
            json!({"limits": {"rpm_limit": 2}}),
        ),
        key(
            "parallel",
            "paul",
            "sk-parallel-0123456789",
            json!({"limits": {"max_parallel_requests": 1}}),
        ),
        key(
            "tpm",
            "tina",
            "sk-tpm-0123456789",
            json!({"limits": {"tpm_limit": 10}}),
        ),
    ])
    .await;
    let address = deployment.address;

    for _ in 0..2 {
        assert_eq!(
            chat(address, "sk-rpm-0123456789", &[], chat_body(1, false))
                .await
                .status,
            StatusCode::OK
        );
    }
    let limited = chat(address, "sk-rpm-0123456789", &[], chat_body(1, false)).await;
    assert_eq!(limited.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(limited.headers["x-ratelimit-limit-requests"], "2");
    assert_eq!(limited.headers["x-ratelimit-remaining-requests"], "0");
    let retry_after: u64 = limited.headers["retry-after"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!((1..=60).contains(&retry_after));
    assert_eq!(
        error_message(&limited),
        "request rate limit of 2 per minute exceeded"
    );

    let stream = open_chat_stream(address, "sk-parallel-0123456789", &[], 30).await;
    let second = chat(address, "sk-parallel-0123456789", &[], chat_body(1, false)).await;
    assert_eq!(second.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(second.headers["retry-after"], "1");
    stream_text(stream).await;
    wait_until(|| deployment.engine().active_ids().is_empty()).await;
    assert_eq!(
        chat(address, "sk-parallel-0123456789", &[], chat_body(1, false))
            .await
            .status,
        StatusCode::OK
    );

    assert_eq!(
        chat(address, "sk-tpm-0123456789", &[], chat_body(5, false))
            .await
            .status,
        StatusCode::OK
    );
    wait_for_audit(&deployment, |log| {
        events(log, "request")
            .iter()
            .any(|event| event["key_id"] == "tpm" && !event["tokens"].is_null())
    })
    .await;
    let limited = chat(address, "sk-tpm-0123456789", &[], chat_body(5, false)).await;
    assert_eq!(limited.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(limited.headers["x-ratelimit-limit-tokens"], "10");
    assert_eq!(limited.headers["x-ratelimit-remaining-tokens"], "0");

    let log = deployment.audit_log();
    let reasons: Vec<&str> = events(&log, "rejected")
        .iter()
        .map(|event| event["reason"].as_str().unwrap())
        .collect();
    assert_eq!(reasons, ["rpm_limit", "max_parallel_requests", "tpm_limit"]);
}

#[tokio::test]
async fn admin_api_manages_keys() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let address = deployment.address;

    let panel = send(address, Method::GET, "/admin", &[], "").await;
    assert_eq!(panel.status, StatusCode::OK);
    assert!(panel.headers["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/html"));
    assert!(panel.headers.contains_key("content-security-policy"));
    assert!(String::from_utf8_lossy(&panel.body).contains("llm-proxy admin"));

    assert_eq!(
        admin(&deployment, Method::GET, "keys", "", Value::Null)
            .await
            .0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        admin(&deployment, Method::GET, "keys", USER_KEY, Value::Null)
            .await
            .0,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        admin(&deployment, Method::GET, "keys", METRICS_KEY, Value::Null)
            .await
            .0,
        StatusCode::FORBIDDEN
    );
    let (status, session) =
        admin(&deployment, Method::GET, "session", ADMIN_KEY, Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        session,
        json!({"user": "root", "key_id": "admin", "role": "admin"})
    );
    let (status, keys) = admin(&deployment, Method::GET, "keys", ADMIN_KEY, Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(keys["keys"].as_array().unwrap().len(), 3);
    assert!(!keys.to_string().contains("key_hash"));
    assert_eq!(keys["keys"][1]["state"], "active");

    let (status, created) = admin(
        &deployment,
        Method::POST,
        "keys",
        ADMIN_KEY,
        json!({"user": "grace", "description": "laptop", "default_class": "batch", "limits": {"rpm_limit": 5}}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let api_key = created["api_key"].as_str().unwrap().to_string();
    let id = created["key"]["id"].as_str().unwrap().to_string();
    assert!(api_key.starts_with("sk-"));
    assert_eq!(created["key"]["created_by"], "root");
    assert_eq!(created["key"]["key_prefix"], api_key[..8]);
    assert_eq!(
        chat(address, &api_key, &[], chat_body(1, false))
            .await
            .status,
        StatusCode::OK
    );
    let file = deployment.keys_file();
    let stored = file["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|key| key["id"] == id.as_str())
        .unwrap();
    assert_eq!(stored["key_hash"], hash(&api_key));
    assert!(!std::fs::read_to_string(deployment.socket_path(KEYS_FILE))
        .unwrap()
        .contains(&api_key));

    let (status, invalid) = admin(
        &deployment,
        Method::POST,
        "keys",
        ADMIN_KEY,
        json!({"user": "x", "default_class": "gold"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(invalid["error"]["message"]
        .as_str()
        .unwrap()
        .contains("unknown class 'gold'"));

    let path = format!("keys/{id}");
    let (status, _) = admin(
        &deployment,
        Method::PATCH,
        &path,
        ADMIN_KEY,
        json!({"blocked": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        chat(address, &api_key, &[], chat_body(1, false))
            .await
            .status,
        StatusCode::FORBIDDEN
    );
    let (status, updated) = admin(
        &deployment,
        Method::PATCH,
        &path,
        ADMIN_KEY,
        json!({"blocked": false, "limits": {"tpm_limit": 1000}, "expires_at": "2099-01-01T00:00:00+01:00"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["key"]["limits"]["rpm_limit"], 5);
    assert_eq!(updated["key"]["limits"]["tpm_limit"], 1000);
    assert_eq!(updated["key"]["expires_at"], "2098-12-31T23:00:00Z");
    assert_eq!(
        chat(address, &api_key, &[], chat_body(1, false))
            .await
            .status,
        StatusCode::OK
    );
    let (status, _) = admin(
        &deployment,
        Method::PATCH,
        &path,
        ADMIN_KEY,
        json!({"key_hash": "00"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, rotated) = admin(
        &deployment,
        Method::POST,
        &format!("{path}/rotate"),
        ADMIN_KEY,
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let rotated_key = rotated["api_key"].as_str().unwrap().to_string();
    assert_ne!(rotated_key, api_key);
    assert_eq!(
        chat(address, &api_key, &[], chat_body(1, false))
            .await
            .status,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        chat(address, &rotated_key, &[], chat_body(1, false))
            .await
            .status,
        StatusCode::OK
    );

    let (status, _) = admin(&deployment, Method::DELETE, &path, ADMIN_KEY, Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        chat(address, &rotated_key, &[], chat_body(1, false))
            .await
            .status,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        admin(&deployment, Method::DELETE, &path, ADMIN_KEY, Value::Null)
            .await
            .0,
        StatusCode::NOT_FOUND
    );

    let (status, conflict) = admin(
        &deployment,
        Method::DELETE,
        "keys/admin",
        ADMIN_KEY,
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(
        conflict["error"]["message"],
        "at least one active admin key must remain"
    );
    let (status, _) = admin(
        &deployment,
        Method::PATCH,
        "keys/admin",
        ADMIN_KEY,
        json!({"role": "user"}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(deployment.keys_file()["keys"].as_array().unwrap().len(), 3);

    let (status, overview) =
        admin(&deployment, Method::GET, "overview", ADMIN_KEY, Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(overview["classes"][0]["name"], "interactive");
    assert_eq!(overview["backend"]["model"], "mock-model");
    assert_eq!(overview["status"]["total"]["requests"], 0);
    assert!(!overview["recent_events"].as_array().unwrap().is_empty());

    wait_for_audit(&deployment, |log| events(log, "key_deleted").len() == 1).await;
    let log = deployment.audit_log();
    let created = events(&log, "key_created");
    assert_eq!(created[0]["actor"]["user"], "root");
    assert_eq!(created[0]["actor"]["key_id"], "admin");
    assert_eq!(created[0]["actor"]["source"], "admin_api");
    assert_eq!(created[0]["key"]["user"], "grace");
    let updates = events(&log, "key_updated");
    assert_eq!(updates.len(), 2);
    assert_eq!(updates[0]["before"]["blocked"], false);
    assert_eq!(updates[0]["after"]["blocked"], true);
    assert_eq!(events(&log, "key_rotated").len(), 1);
    assert_eq!(events(&log, "key_deleted")[0]["key"]["id"], id.as_str());
    let text = std::fs::read_to_string(deployment.socket_path("audit.jsonl")).unwrap();
    assert!(!text.contains("key_hash"));
    assert!(!text.contains(&api_key) && !text.contains(&rotated_key));
    let rejected: Vec<_> = events(&log, "rejected")
        .iter()
        .map(|event| {
            (
                event["reason"].as_str().unwrap(),
                event["path"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        rejected[..3],
        [
            ("missing_api_key", "/admin/api/keys"),
            ("forbidden_role", "/admin/api/keys"),
            ("forbidden_role", "/admin/api/keys")
        ]
    );
}

#[tokio::test]
async fn keys_file_changes_on_disk_apply_without_a_restart() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let address = deployment.address;
    let path = deployment.socket_path(KEYS_FILE);
    let mut keys = base_keys();
    keys.push(key("added", "heidi", "sk-added-0123456789", json!({})));
    std::fs::write(&path, json!({"keys": keys}).to_string()).unwrap();
    wait_for_audit(&deployment, |log| !events(log, "keys_reloaded").is_empty()).await;
    assert_eq!(
        chat(address, "sk-added-0123456789", &[], chat_body(1, false))
            .await
            .status,
        StatusCode::OK
    );
    let log = deployment.audit_log();
    assert_eq!(events(&log, "keys_reloaded")[0]["added"], json!(["added"]));

    std::fs::write(&path, "{not json").unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        chat(address, "sk-added-0123456789", &[], chat_body(1, false))
            .await
            .status,
        StatusCode::OK
    );
    let metrics =
        String::from_utf8(get_metrics(&deployment, METRICS_KEY).await.body.to_vec()).unwrap();
    let errors_line = metrics
        .lines()
        .find(|line| line.starts_with("llm_proxy_keys_file_errors_total "))
        .unwrap();
    assert_ne!(errors_line, "llm_proxy_keys_file_errors_total 0");
    let (status, failure) = admin(
        &deployment,
        Method::POST,
        "keys",
        ADMIN_KEY,
        json!({"user": "x", "default_class": "batch"}),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(failure["error"]["message"]
        .as_str()
        .unwrap()
        .contains("is invalid"));

    keys.retain(|key| key["id"] != "user");
    std::fs::write(&path, json!({"keys": keys}).to_string()).unwrap();
    wait_until(|| events(&deployment.audit_log(), "keys_reloaded").len() == 2).await;
    assert_eq!(
        chat(address, USER_KEY, &[], chat_body(1, false))
            .await
            .status,
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn metrics_endpoint() {
    let deployment = deploy_with_keys(vec![key(
        "limited",
        "ivan",
        "sk-limited-0123456789",
        json!({"limits": {"rpm_limit": 10, "tpd_limit": 1000}}),
    )])
    .await;
    let address = deployment.address;
    assert_eq!(
        chat(address, USER_KEY, &[], chat_body(3, false))
            .await
            .status,
        StatusCode::OK
    );
    assert_eq!(
        chat(address, "sk-limited-0123456789", &[], chat_body(3, false))
            .await
            .status,
        StatusCode::OK
    );
    assert_eq!(
        chat(address, "sk-nobody-0123456789", &[], chat_body(3, false))
            .await
            .status,
        StatusCode::UNAUTHORIZED
    );
    let authorization = format!("Bearer {USER_KEY}");
    send(
        address,
        Method::GET,
        "/v1/models/some-model",
        &[("authorization", authorization.as_str())],
        "",
    )
    .await;
    wait_for_audit(&deployment, |log| {
        let requests = events(log, "request");
        requests
            .iter()
            .filter(|event| !event["tokens"].is_null())
            .count()
            == 2
            && requests
                .iter()
                .any(|event| event["path"] == "/v1/models/some-model")
    })
    .await;

    assert_eq!(
        get_metrics(&deployment, "").await.status,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        get_metrics(&deployment, USER_KEY).await.status,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        get_metrics(&deployment, ADMIN_KEY).await.status,
        StatusCode::OK
    );
    let response = get_metrics(&deployment, METRICS_KEY).await;
    assert_eq!(response.status, StatusCode::OK);
    assert!(response.headers["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/plain"));
    let text = String::from_utf8(response.body.to_vec()).unwrap();
    for expected in [
        r#"llm_proxy_requests_total{user="carol",key="user",class="batch",path="/v1/chat/completions",status="200"} 1"#,
        r#"llm_proxy_tokens_total{user="carol",key="user",class="batch",type="output"} 3"#,
        r#"llm_proxy_rejected_requests_total{status="401",reason="unknown_api_key"} 1"#,
        r#"llm_proxy_rejected_requests_total{status="401",reason="missing_api_key"} 1"#,
        r#"llm_proxy_request_duration_seconds_count{class="batch"} 2"#,
        r#"class="",path="/v1/models/*",status="#,
        r#"llm_proxy_keys{state="active"} 4"#,
        r#"llm_proxy_key_remaining_requests{key="limited",user="ivan"} 9"#,
        r#"llm_proxy_key_remaining_tokens{key="limited",user="ivan",window="day"}"#,
        r#"llm_proxy_rank_parallel_request_cap{rank="0"} 16"#,
        "# TYPE llm_proxy_time_to_first_token_seconds histogram",
    ] {
        assert!(text.contains(expected), "missing {expected} in\n{text}");
    }
    let post = send(
        address,
        Method::POST,
        "/metrics",
        &[("authorization", "Bearer sk-metrics-0123456789abcdef")],
        "",
    )
    .await;
    assert_eq!(post.status, StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn admin_paths_stay_closed_without_a_gateway() {
    let deployment = deploy(DeploymentOptions::default()).await;
    for path in ["/admin", "/admin/api/keys", "/metrics"] {
        let response = send(
            deployment.address,
            Method::GET,
            path,
            &[("authorization", "Bearer sk-alice")],
            "",
        )
        .await;
        assert_eq!(response.status, StatusCode::NOT_FOUND, "{path}");
    }
}

#[tokio::test]
async fn restarts_keep_the_key_of_running_requests_and_the_limits() {
    let mut deployment = deploy_with_keys(vec![key(
        "rpm",
        "rita",
        "sk-rpm-0123456789",
        json!({"limits": {"rpm_limit": 2, "max_parallel_requests": 1}}),
    )])
    .await;
    let address = deployment.address;
    let stream = open_chat_stream(address, "sk-rpm-0123456789", &[], 60).await;
    wait_until(|| !deployment.engine().generating_ids().is_empty()).await;
    deployment.restart_middle_end().await;
    let parallel = chat(address, "sk-rpm-0123456789", &[], chat_body(1, false)).await;
    assert_eq!(parallel.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(parallel.headers.get("x-ratelimit-limit-requests"), None);
    stream_text(stream).await;
    wait_until(|| deployment.engine().active_ids().is_empty()).await;
    assert_eq!(
        chat(address, "sk-rpm-0123456789", &[], chat_body(1, false))
            .await
            .status,
        StatusCode::OK
    );
    let limited = chat(address, "sk-rpm-0123456789", &[], chat_body(1, false)).await;
    assert_eq!(limited.headers["x-ratelimit-limit-requests"], "2");
    wait_for_audit(&deployment, |log| {
        events(log, "request")
            .iter()
            .any(|event| event["key_id"] == "rpm" && event["tokens"]["output"] == 60)
    })
    .await;
    let log = deployment.audit_log();
    let streamed = events(&log, "request")
        .into_iter()
        .find(|event| event["tokens"]["output"] == 60)
        .unwrap();
    assert_eq!(streamed["user"], "rita");
    assert_eq!(streamed["path"], "/v1/chat/completions");
    assert_eq!(streamed["stream"], true);
}

#[tokio::test]
async fn create_key_command_bootstraps_the_keys_file() {
    let directory = tempfile::tempdir().unwrap();
    let keys_file = directory.path().join("keys.json");
    let audit_log = directory.path().join("audit.jsonl");
    let general = json!({
        "frontend": {"listen": "127.0.0.1:0"},
        "middle_end_socket": directory.path().join("middle-end.sock"),
        "gateway": {"keys_file": keys_file, "audit_log": audit_log},
    });
    std::fs::write(directory.path().join("general.json"), general.to_string()).unwrap();
    std::fs::write(
        directory.path().join("accounts.json"),
        DeploymentOptions::default().accounts.to_string(),
    )
    .unwrap();
    let run = |arguments: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_llm-proxy"))
            .args(arguments)
            .arg("--general")
            .arg(directory.path().join("general.json"))
            .arg("--accounts")
            .arg(directory.path().join("accounts.json"))
            .arg("--backend")
            .arg(directory.path().join("backend.json"))
            .output()
            .unwrap()
    };
    let output = run(&[
        "create-key",
        "--user",
        "root",
        "--default-class",
        "batch",
        "--role",
        "admin",
        "--allowed-class",
        "idle",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let api_key = String::from_utf8(output.stdout).unwrap().trim().to_string();
    assert!(api_key.starts_with("sk-") && api_key.len() == 51);
    let file: Value = serde_json::from_str(&std::fs::read_to_string(&keys_file).unwrap()).unwrap();
    let record = &file["keys"][0];
    assert_eq!(
        (&record["user"], &record["role"], &record["key_hash"]),
        (&json!("root"), &json!("admin"), &json!(hash(&api_key)))
    );
    assert_eq!(record["allowed_classes"], json!(["idle"]));
    assert!(record["created_by"].as_str().unwrap().starts_with("cli"));
    let audit: Value =
        serde_json::from_str(std::fs::read_to_string(&audit_log).unwrap().trim()).unwrap();
    assert_eq!(
        (&audit["event"], &audit["actor"]["source"]),
        (&json!("key_created"), &json!("cli"))
    );

    assert!(run(&["check-config"]).status.success());
    let failure = run(&["create-key", "--user", "x", "--default-class", "gold"]);
    assert!(!failure.status.success());
    assert!(String::from_utf8_lossy(&failure.stderr).contains("unknown class 'gold'"));
    std::fs::write(&keys_file, "{}").unwrap();
    assert!(run(&["check-config"]).status.success());
    std::fs::remove_file(&keys_file).unwrap();
    let missing = run(&["check-config"]);
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("keys.json"));
}
