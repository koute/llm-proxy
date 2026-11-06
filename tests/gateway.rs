mod support;

use std::time::Duration;

use hyper::{Method, StatusCode};
use serde_json::{json, Value};

use llm_proxy::config::ApiKeyHash;
use support::{
    chat, chat_body, deploy, open_chat_stream, send, stream_text, wait_until, ClientResponse,
    Deployment, DeploymentOptions, KEYS_FILE,
};

const ADMIN: &str = "sk-admin-0123456789abcdef";
const USER: &str = "sk-user-0123456789abcdef";
const METRICS: &str = "sk-metrics-0123456789abcdef";

fn key(id: &str, user: &str, api_key: &str, settings: Value) -> Value {
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

fn base_keys() -> Vec<Value> {
    vec![
        key("admin", "root", ADMIN, json!({"role": "admin"})),
        key("user", "carol", USER, json!({})),
        key("metrics", "prometheus", METRICS, json!({"role": "metrics"})),
    ]
}

async fn deploy_with_keys(extra_keys: Vec<Value>) -> Deployment {
    let keys = [base_keys(), extra_keys].concat();
    let keys = Some(json!({ "keys": keys }));
    deploy(DeploymentOptions {
        keys,
        ..DeploymentOptions::default()
    })
    .await
}

async fn request(
    deployment: &Deployment,
    method: Method,
    path: &str,
    api_key: &str,
    body: Value,
) -> ClientResponse {
    let authorization = format!("Bearer {api_key}");
    let body = if body.is_null() {
        String::new()
    } else {
        body.to_string()
    };
    send(
        deployment.address,
        method,
        path,
        &[("authorization", &authorization)],
        body,
    )
    .await
}

async fn admin(
    deployment: &Deployment,
    method: Method,
    path: &str,
    api_key: &str,
    body: Value,
) -> (StatusCode, Value) {
    let response = request(
        deployment,
        method,
        &format!("/admin/api/{path}"),
        api_key,
        body,
    )
    .await;
    (
        response.status,
        serde_json::from_slice(&response.body).unwrap_or(Value::Null),
    )
}

async fn status(deployment: &Deployment, api_key: &str, headers: &[(&str, &str)]) -> StatusCode {
    chat(deployment.address, api_key, headers, chat_body(1, false))
        .await
        .status
}

fn events<'a>(log: &'a [Value], name: &str) -> Vec<&'a Value> {
    log.iter().filter(|event| event["event"] == name).collect()
}

async fn reasons(deployment: &Deployment, count: usize) -> Vec<String> {
    let reasons = || -> Vec<String> {
        let log = deployment.audit_log();
        let rejected = events(&log, "rejected");
        rejected
            .iter()
            .map(|event| event["reason"].as_str().unwrap().to_string())
            .collect()
    };
    wait_until(|| reasons().len() >= count).await;
    reasons()
}

async fn wait_for_requests(deployment: &Deployment, count: usize) -> Vec<Value> {
    let finished = |log: &[Value]| {
        events(log, "request")
            .iter()
            .filter(|event| !event["tokens"].is_null())
            .count()
    };
    wait_until(|| finished(&deployment.audit_log()) >= count).await;
    deployment.audit_log()
}

#[tokio::test]
async fn keys_authenticate_and_are_audited() {
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
            "models",
            "frank",
            "sk-models-0123456789",
            json!({"allowed_routes": ["/v1/models"]}),
        ),
    ])
    .await;
    assert_eq!(status(&deployment, USER, &[]).await, StatusCode::OK);
    assert_eq!(
        status(&deployment, USER, &[("x-priority", "idle")]).await,
        StatusCode::OK
    );
    assert_eq!(
        status(&deployment, USER, &[("x-priority", "interactive")]).await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(status(&deployment, "sk-alice", &[]).await, StatusCode::OK);
    assert_eq!(
        status(&deployment, "sk-blocked-0123456789", &[]).await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        status(&deployment, "sk-expired-0123456789", &[]).await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        status(&deployment, METRICS, &[]).await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        status(&deployment, "sk-unknown-0123456789", &[]).await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        status(&deployment, "sk-models-0123456789", &[]).await,
        StatusCode::FORBIDDEN
    );
    let models = request(
        &deployment,
        Method::GET,
        "/v1/models",
        "sk-models-0123456789",
        Value::Null,
    )
    .await;
    assert_eq!(models.status, StatusCode::OK);
    let received = deployment.engine().received();
    assert!(received
        .iter()
        .all(|request| !request.headers.contains_key("x-llm-proxy-peer")));

    reasons(&deployment, 6).await;
    let log = wait_for_requests(&deployment, 3).await;
    let requests = events(&log, "request");
    let carol = requests
        .iter()
        .find(|event| event["class"] == "idle")
        .unwrap();
    assert_eq!(
        (&carol["user"], &carol["key_id"], &carol["status"]),
        (&json!("carol"), &json!("user"), &json!(200))
    );
    assert_eq!(
        (&carol["outcome"], &carol["client"], &carol["model"]),
        (
            &json!("completed"),
            &json!("127.0.0.1"),
            &json!("mock-model")
        )
    );
    assert_eq!(carol["tokens"]["output"], 1);
    assert!(carol["time_to_first_token_seconds"].as_f64().unwrap() > 0.0);
    let alice = requests
        .iter()
        .find(|event| event["user"] == "alice")
        .unwrap();
    assert_eq!(alice["key_id"], Value::Null);
    assert_eq!(
        reasons(&deployment, 6).await,
        [
            "class_not_allowed",
            "blocked_api_key",
            "expired_api_key",
            "forbidden_role",
            "unknown_api_key",
            "route_not_allowed"
        ]
    );
    let unknown = events(&log, "rejected")
        .into_iter()
        .find(|event| event["reason"] == "unknown_api_key")
        .unwrap();
    assert_eq!(unknown["api_key"], "\"sk-...6789\" (21 characters)");
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
    assert_eq!(
        status(&deployment, "sk-rpm-0123456789", &[]).await,
        StatusCode::OK
    );
    assert_eq!(
        status(&deployment, "sk-rpm-0123456789", &[]).await,
        StatusCode::OK
    );
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

    let stream = open_chat_stream(address, "sk-parallel-0123456789", &[], 30).await;
    let second = chat(address, "sk-parallel-0123456789", &[], chat_body(1, false)).await;
    assert_eq!(second.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(second.headers["retry-after"], "1");
    stream_text(stream).await;
    assert_eq!(
        status(&deployment, "sk-parallel-0123456789", &[]).await,
        StatusCode::OK
    );

    assert_eq!(
        chat(address, "sk-tpm-0123456789", &[], chat_body(5, false))
            .await
            .status,
        StatusCode::OK
    );
    wait_until(|| {
        events(&deployment.audit_log(), "request")
            .iter()
            .any(|event| event["key_id"] == "tpm")
    })
    .await;
    let limited = chat(address, "sk-tpm-0123456789", &[], chat_body(5, false)).await;
    assert_eq!(limited.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(limited.headers["x-ratelimit-limit-tokens"], "10");
    assert_eq!(
        reasons(&deployment, 3).await,
        ["rpm_limit", "max_parallel_requests", "tpm_limit"]
    );
}

#[tokio::test]
async fn admin_api_manages_keys() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let panel = request(&deployment, Method::GET, "/admin", "", Value::Null).await;
    assert_eq!(panel.status, StatusCode::OK);
    assert!(String::from_utf8_lossy(&panel.body).contains("llm-proxy admin"));
    for (api_key, expected) in [
        ("", StatusCode::UNAUTHORIZED),
        (USER, StatusCode::FORBIDDEN),
        (METRICS, StatusCode::FORBIDDEN),
    ] {
        assert_eq!(
            admin(&deployment, Method::GET, "keys", api_key, Value::Null)
                .await
                .0,
            expected
        );
    }
    let (_, session) = admin(&deployment, Method::GET, "session", ADMIN, Value::Null).await;
    assert_eq!(
        session,
        json!({"user": "root", "key_id": "admin", "role": "admin"})
    );
    let (_, list) = admin(&deployment, Method::GET, "keys", ADMIN, Value::Null).await;
    assert_eq!(
        (list["keys"].as_array().unwrap().len(), &list["classes"][0]),
        (3, &json!("interactive"))
    );
    assert!(!list.to_string().contains("key_hash"));

    let settings = json!({"user": "grace", "default_class": "batch", "limits": {"rpm_limit": 5}});
    let (status_code, created) = admin(&deployment, Method::POST, "keys", ADMIN, settings).await;
    assert_eq!(status_code, StatusCode::CREATED);
    let api_key = created["api_key"].as_str().unwrap().to_string();
    let path = format!("keys/{}", created["key"]["id"].as_str().unwrap());
    assert_eq!(
        (&created["key"]["created_by"], &created["key"]["key_prefix"]),
        (&json!("root"), &json!(api_key[..8]))
    );
    assert_eq!(status(&deployment, &api_key, &[]).await, StatusCode::OK);
    let stored = deployment.keys_file()["keys"][3].clone();
    assert_eq!(
        stored["key_hash"],
        ApiKeyHash::from_api_key(&api_key).to_hex()
    );
    let invalid = json!({"user": "x", "default_class": "gold"});
    assert_eq!(
        admin(&deployment, Method::POST, "keys", ADMIN, invalid)
            .await
            .0,
        StatusCode::BAD_REQUEST
    );

    assert_eq!(
        admin(
            &deployment,
            Method::PATCH,
            &path,
            ADMIN,
            json!({"blocked": true})
        )
        .await
        .0,
        StatusCode::OK
    );
    assert_eq!(
        status(&deployment, &api_key, &[]).await,
        StatusCode::FORBIDDEN
    );
    let change = json!({"blocked": false, "limits": {"tpm_limit": 1000}, "expires_at": "2099-01-01T00:00:00+01:00"});
    let (_, updated) = admin(&deployment, Method::PATCH, &path, ADMIN, change).await;
    assert_eq!(
        (
            &updated["key"]["limits"]["rpm_limit"],
            &updated["key"]["limits"]["tpm_limit"]
        ),
        (&json!(5), &json!(1000))
    );
    assert_eq!(updated["key"]["expires_at"], "2098-12-31T23:00:00Z");
    assert_eq!(
        admin(
            &deployment,
            Method::PATCH,
            &path,
            ADMIN,
            json!({"key_hash": "00"})
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );

    let (_, rotated) = admin(
        &deployment,
        Method::POST,
        &format!("{path}/rotate"),
        ADMIN,
        Value::Null,
    )
    .await;
    let rotated_key = rotated["api_key"].as_str().unwrap().to_string();
    assert_eq!(
        status(&deployment, &api_key, &[]).await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(status(&deployment, &rotated_key, &[]).await, StatusCode::OK);
    assert_eq!(
        admin(&deployment, Method::DELETE, &path, ADMIN, Value::Null)
            .await
            .0,
        StatusCode::OK
    );
    assert_eq!(
        status(&deployment, &rotated_key, &[]).await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        admin(&deployment, Method::DELETE, &path, ADMIN, Value::Null)
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        admin(
            &deployment,
            Method::DELETE,
            "keys/admin",
            ADMIN,
            Value::Null
        )
        .await
        .0,
        StatusCode::CONFLICT
    );
    let demotion = json!({"role": "user"});
    assert_eq!(
        admin(&deployment, Method::PATCH, "keys/admin", ADMIN, demotion)
            .await
            .0,
        StatusCode::CONFLICT
    );

    wait_until(|| !events(&deployment.audit_log(), "key_deleted").is_empty()).await;
    let log = deployment.audit_log();
    let created = events(&log, "key_created");
    assert_eq!(
        (&created[0]["actor"]["user"], &created[0]["actor"]["source"]),
        (&json!("root"), &json!("admin_api"))
    );
    let updates = events(&log, "key_updated");
    assert_eq!(
        (updates.len(), &updates[0]["after"]["blocked"]),
        (2, &json!(true))
    );
    assert_eq!(
        (
            events(&log, "key_rotated").len(),
            events(&log, "key_deleted").len()
        ),
        (1, 1)
    );
    let text = std::fs::read_to_string(deployment.socket_path("audit.jsonl")).unwrap();
    assert!(!text.contains("key_hash") && !text.contains(&api_key) && !text.contains(&rotated_key));
    assert_eq!(
        reasons(&deployment, 3).await[..3],
        ["missing_api_key", "forbidden_role", "forbidden_role"]
    );
}

#[tokio::test]
async fn keys_file_changes_apply_without_a_restart() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let path = deployment.socket_path(KEYS_FILE);
    let mut keys = base_keys();
    keys.push(key("added", "heidi", "sk-added-0123456789", json!({})));
    std::fs::write(&path, json!({ "keys": keys }).to_string()).unwrap();
    wait_until(|| !events(&deployment.audit_log(), "keys_reloaded").is_empty()).await;
    assert_eq!(
        status(&deployment, "sk-added-0123456789", &[]).await,
        StatusCode::OK
    );
    assert_eq!(
        events(&deployment.audit_log(), "keys_reloaded")[0]["changed"],
        json!(["added"])
    );

    std::fs::write(&path, "{not json").unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        status(&deployment, "sk-added-0123456789", &[]).await,
        StatusCode::OK
    );
    let metrics = request(&deployment, Method::GET, "/metrics", METRICS, Value::Null).await;
    assert!(!String::from_utf8_lossy(&metrics.body).contains("llm_proxy_keys_file_errors_total 0"));
    let (status_code, failure) = admin(
        &deployment,
        Method::POST,
        "keys",
        ADMIN,
        json!({"user": "x", "default_class": "batch"}),
    )
    .await;
    assert_eq!(status_code, StatusCode::SERVICE_UNAVAILABLE);
    assert!(failure["error"]["message"]
        .as_str()
        .unwrap()
        .contains("is invalid"));

    keys.retain(|key| key["id"] != "user");
    std::fs::write(&path, json!({ "keys": keys }).to_string()).unwrap();
    wait_until(|| events(&deployment.audit_log(), "keys_reloaded").len() == 2).await;
    assert_eq!(
        status(&deployment, USER, &[]).await,
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn metrics_endpoint() {
    let limits = json!({"limits": {"rpm_limit": 10, "tpd_limit": 1000}});
    let deployment = deploy_with_keys(vec![key(
        "limited",
        "ivan",
        "sk-limited-0123456789",
        limits,
    )])
    .await;
    assert_eq!(
        chat(deployment.address, USER, &[], chat_body(3, false))
            .await
            .status,
        StatusCode::OK
    );
    assert_eq!(
        status(&deployment, "sk-limited-0123456789", &[]).await,
        StatusCode::OK
    );
    assert_eq!(
        status(&deployment, "sk-nobody-0123456789", &[]).await,
        StatusCode::UNAUTHORIZED
    );
    wait_for_requests(&deployment, 2).await;

    for (api_key, expected) in [
        ("", StatusCode::UNAUTHORIZED),
        (USER, StatusCode::FORBIDDEN),
        (ADMIN, StatusCode::OK),
    ] {
        assert_eq!(
            request(&deployment, Method::GET, "/metrics", api_key, Value::Null)
                .await
                .status,
            expected
        );
    }
    let text = String::from_utf8(
        request(&deployment, Method::GET, "/metrics", METRICS, Value::Null)
            .await
            .body
            .to_vec(),
    )
    .unwrap();
    for expected in [
        r#"llm_proxy_requests_total{user="carol",key="user",class="batch",path="/v1/chat/completions",status="200"} 1"#,
        r#"llm_proxy_tokens_total{user="carol",key="user",class="batch",type="output"} 3"#,
        r#"llm_proxy_rejected_requests_total{status="401",reason="unknown_api_key"} 1"#,
        r#"llm_proxy_request_duration_seconds_count{class="batch"} 2"#,
        r#"llm_proxy_keys{state="active"} 4"#,
        r#"llm_proxy_key_remaining_requests{key="limited",user="ivan"} 9"#,
        r#"llm_proxy_key_remaining_tokens{key="limited",user="ivan",window="day"}"#,
        r#"llm_proxy_rank_parallel_request_cap{rank="0"} 16"#,
    ] {
        assert!(text.contains(expected), "missing {expected} in\n{text}");
    }

    let closed = deploy(DeploymentOptions::default()).await;
    for path in ["/admin", "/metrics"] {
        let response = send(
            closed.address,
            Method::GET,
            path,
            &[("authorization", "Bearer sk-alice")],
            "",
        )
        .await;
        assert_eq!(response.status, StatusCode::NOT_FOUND);
    }
}

#[tokio::test]
async fn restarts_keep_running_requests_attributed_and_limits() {
    let limits = json!({"limits": {"rpm_limit": 2, "max_parallel_requests": 1}});
    let mut deployment =
        deploy_with_keys(vec![key("rpm", "rita", "sk-rpm-0123456789", limits)]).await;
    let stream = open_chat_stream(deployment.address, "sk-rpm-0123456789", &[], 60).await;
    wait_until(|| !deployment.engine().generating_ids().is_empty()).await;
    deployment.restart_middle_end().await;
    assert_eq!(
        status(&deployment, "sk-rpm-0123456789", &[]).await,
        StatusCode::TOO_MANY_REQUESTS
    );
    stream_text(stream).await;
    assert_eq!(
        status(&deployment, "sk-rpm-0123456789", &[]).await,
        StatusCode::OK
    );
    let limited = chat(
        deployment.address,
        "sk-rpm-0123456789",
        &[],
        chat_body(1, false),
    )
    .await;
    assert_eq!(limited.headers["x-ratelimit-limit-requests"], "2");
    let log = wait_for_requests(&deployment, 2).await;
    let streamed = events(&log, "request")
        .into_iter()
        .find(|event| event["tokens"]["output"] == 60)
        .unwrap();
    assert_eq!(
        (&streamed["key_id"], &streamed["stream"]),
        (&json!("rpm"), &json!(true))
    );
}

#[tokio::test]
async fn create_key_command() {
    let directory = tempfile::tempdir().unwrap();
    let file = |name: &str| directory.path().join(name);
    let general = json!({
        "frontend": {"listen": "127.0.0.1:0"},
        "middle_end_socket": file("middle-end.sock"),
        "gateway": {"keys_file": file("keys.json"), "audit_log": file("audit.jsonl")},
    });
    std::fs::write(file("general.json"), general.to_string()).unwrap();
    std::fs::write(
        file("accounts.json"),
        DeploymentOptions::default().accounts.to_string(),
    )
    .unwrap();
    let run = |arguments: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_llm-proxy"))
            .args(arguments)
            .args(["--general", file("general.json").to_str().unwrap()])
            .args(["--accounts", file("accounts.json").to_str().unwrap()])
            .args(["--backend", file("backend.json").to_str().unwrap()])
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
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let api_key = String::from_utf8(output.stdout).unwrap().trim().to_string();
    let keys: Value =
        serde_json::from_str(&std::fs::read_to_string(file("keys.json")).unwrap()).unwrap();
    assert_eq!(
        keys["keys"][0]["key_hash"],
        ApiKeyHash::from_api_key(&api_key).to_hex()
    );
    assert_eq!(
        (&keys["keys"][0]["role"], &keys["keys"][0]["created_by"]),
        (&json!("admin"), &json!("cli"))
    );
    assert!(std::fs::read_to_string(file("audit.jsonl"))
        .unwrap()
        .contains("\"key_created\""));
    assert!(run(&["check-config"]).status.success());
    assert!(
        !run(&["create-key", "--user", "x", "--default-class", "gold"])
            .status
            .success()
    );
    std::fs::remove_file(file("keys.json")).unwrap();
    assert!(!run(&["check-config"]).status.success());
}
