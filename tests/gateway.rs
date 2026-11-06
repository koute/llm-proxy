mod support;

use std::time::Duration;

use http_body_util::BodyExt;
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
        StatusCode::UNAUTHORIZED
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
            // A request of chat_body(5) is estimated at 8 tokens and uses 12.
            json!({"limits": {"tpm_limit": 15}}),
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
    assert_eq!(limited.headers["x-ratelimit-limit-tokens"], "15");
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
        StatusCode::UNAUTHORIZED
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
    let mut stream = open_chat_stream(deployment.address, "sk-rpm-0123456789", &[], 60).await;
    stream.body_mut().frame().await.unwrap().unwrap();
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
    let mut stream =
        open_chat_stream(deployment.address, USER, &[("user-agent", "tester/1")], 500).await;
    assert_eq!(stream.status(), StatusCode::OK);
    stream.body_mut().frame().await.unwrap().unwrap();
    drop(stream);
    let log = wait_for_requests(&deployment, 1).await;
    let aborted = events(&log, "request")[0];
    assert_eq!(
        (&aborted["outcome"], &aborted["status"]),
        (&json!("aborted"), &json!(200))
    );
    assert!(aborted["time_to_first_token_seconds"].as_f64().is_some());
    assert_eq!(aborted["user_agent"], "tester/1");
    let metrics = request(&deployment, Method::GET, "/metrics", METRICS, Value::Null).await;
    let metrics = String::from_utf8(metrics.body.to_vec()).unwrap();
    assert!(metrics.contains(r#"status="aborted"} 1"#), "{metrics}");
    // Keys of accounts.json have no access to /metrics or the admin API.
    let forbidden = request(
        &deployment,
        Method::GET,
        "/metrics",
        "sk-alice",
        Value::Null,
    )
    .await;
    assert_eq!(forbidden.status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn starts_without_the_keys_file() {
    let general = json!({"gateway": {"client_ip_header": "x-forwarded-for"}});
    let keys = Some(Value::Null);
    let options = DeploymentOptions {
        keys,
        general,
        ..DeploymentOptions::default()
    };
    let deployment = deploy(options).await;
    assert_eq!(status(&deployment, "sk-alice", &[]).await, StatusCode::OK);
    assert_eq!(
        status(&deployment, USER, &[]).await,
        StatusCode::UNAUTHORIZED
    );
    let keys = json!({ "keys": base_keys() }).to_string();
    std::fs::write(deployment.socket_path(KEYS_FILE), keys).unwrap();
    wait_until(|| !events(&deployment.audit_log(), "keys_reloaded").is_empty()).await;
    let forwarded = [("x-forwarded-for", "203.0.113.9, 198.51.100.7")];
    assert_eq!(status(&deployment, USER, &forwarded).await, StatusCode::OK);
    let log = wait_for_requests(&deployment, 2).await;
    let requests = events(&log, "request");
    let carol = requests
        .iter()
        .find(|event| event["user"] == "carol")
        .unwrap();
    assert_eq!(carol["client"], "198.51.100.7");
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
    let create = [
        "create-key",
        "--user",
        "root",
        "--default-class",
        "batch",
        "--role",
        "admin",
    ];
    // A missing keys file may be a mount which is down: only created on request.
    let refused = run(&create);
    assert!(!refused.status.success() && !file("keys.json").exists());
    let output = run(&[&create[..], &["--new-keys-file"]].concat());
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

#[tokio::test]
async fn passthrough_requests_count_against_limits_and_are_audited() {
    let deployment = deploy_with_keys(vec![key(
        "rpm",
        "rita",
        "sk-rpm-0123456789",
        json!({"limits": {"rpm_limit": 2}}),
    )])
    .await;
    let tokenize = || {
        request(
            &deployment,
            Method::POST,
            "/tokenize",
            "sk-rpm-0123456789",
            json!({"prompt": "hi"}),
        )
    };
    // Requests with a body count like LiteLLM counts LLM calls, reads don't.
    for _ in 0..3 {
        let models = request(
            &deployment,
            Method::GET,
            "/v1/models",
            "sk-rpm-0123456789",
            Value::Null,
        );
        assert_eq!(models.await.status, StatusCode::OK);
    }
    assert_ne!(tokenize().await.status, StatusCode::TOO_MANY_REQUESTS);
    assert_ne!(tokenize().await.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(tokenize().await.status, StatusCode::TOO_MANY_REQUESTS);
    let finished = |log: &[Value]| -> Vec<(String, String)> {
        events(log, "request")
            .iter()
            .map(|event| {
                assert!(event["request"].is_null() && event["class"].is_null());
                assert_eq!(event["key_id"], "rpm");
                assert_eq!(event["outcome"], "completed");
                let method = event["method"].as_str().unwrap().to_string();
                (method, event["path"].as_str().unwrap().to_string())
            })
            .collect()
    };
    wait_until(|| finished(&deployment.audit_log()).len() == 5).await;
    let mut paths = finished(&deployment.audit_log());
    paths.sort();
    assert_eq!(
        paths,
        [
            ("GET", "/v1/models"),
            ("GET", "/v1/models"),
            ("GET", "/v1/models"),
            ("POST", "/tokenize"),
            ("POST", "/tokenize")
        ]
        .map(|(method, path)| (method.to_string(), path.to_string()))
    );
    let metrics = request(&deployment, Method::GET, "/metrics", METRICS, Value::Null).await;
    let metrics = String::from_utf8(metrics.body.to_vec()).unwrap();
    assert!(metrics.contains(
        "llm_proxy_requests_total{user=\"rita\",key=\"rpm\",class=\"\",path=\"/v1/models\",status=\"200\"} 3"
    ));
}

#[tokio::test]
async fn admin_api_guards() {
    let deployment = deploy_with_keys(vec![key(
        "scoped-admin",
        "sam",
        "sk-scoped-admin-0123456789",
        json!({"role": "admin", "allowed_routes": ["/v1/models"]}),
    )])
    .await;
    // The last lasting admin can't be given an expiry, which would lock everyone out later.
    let soon = format!(
        "{:.0}",
        jiff::Timestamp::now() + jiff::SignedDuration::from_secs(5)
    );
    let (status_code, _) = admin(
        &deployment,
        Method::PATCH,
        "keys/admin",
        ADMIN,
        json!({"expires_at": soon}),
    )
    .await;
    assert_eq!(status_code, StatusCode::CONFLICT);
    // Route patterns which can't match anything are refused.
    for routes in [
        json!(["*"]),
        json!(["/v1/*/completions"]),
        json!(["v1/models"]),
    ] {
        let settings = json!({"user": "x", "default_class": "batch", "allowed_routes": routes});
        assert_eq!(
            admin(&deployment, Method::POST, "keys", ADMIN, settings)
                .await
                .0,
            StatusCode::BAD_REQUEST
        );
    }
    // allowed_routes also covers the admin API.
    let (status_code, _) = admin(
        &deployment,
        Method::GET,
        "keys",
        "sk-scoped-admin-0123456789",
        Value::Null,
    )
    .await;
    assert_eq!(status_code, StatusCode::FORBIDDEN);
    // A mistake in a request isn't a keys file failure.
    assert_eq!(
        admin(&deployment, Method::DELETE, "keys/nope", ADMIN, Value::Null)
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    let metrics = request(&deployment, Method::GET, "/metrics", METRICS, Value::Null).await;
    let metrics = String::from_utf8(metrics.body.to_vec()).unwrap();
    assert!(!metrics.contains("llm_proxy_keys_file_errors_total 1"));
    // The panel can't be framed.
    let panel = send(deployment.address, Method::GET, "/admin", &[], "").await;
    assert_eq!(panel.headers["x-frame-options"], "DENY");
    assert_eq!(
        panel.headers["content-security-policy"],
        "frame-ancestors 'none'"
    );
    // Keys in api-key, and x-api-key next to an empty Authorization header.
    let body = chat_body(1, false);
    let azure = send(
        deployment.address,
        Method::POST,
        "/v1/chat/completions",
        &[("api-key", USER)],
        body.clone(),
    )
    .await;
    assert_eq!(azure.status, StatusCode::OK);
    let headers = [("authorization", ""), ("x-api-key", USER)];
    let empty = send(
        deployment.address,
        Method::POST,
        "/v1/chat/completions",
        &headers,
        body,
    )
    .await;
    assert_eq!(empty.status, StatusCode::OK);
    let received = deployment.engine().received();
    assert!(received
        .iter()
        .all(|request| !request.headers.contains_key("api-key")));
}

#[tokio::test]
async fn closed_clients_release_their_requests_at_once() {
    let limits = json!({"limits": {"max_parallel_requests": 1}});
    let deployment = deploy_with_keys(vec![key("one", "olga", "sk-one-0123456789", limits)]).await;
    // One token per 30 seconds: nothing is written to the client for a long time.
    deployment
        .engine()
        .set_large_batch_token_interval(0, Duration::from_secs(30));
    let stream = open_chat_stream(deployment.address, "sk-one-0123456789", &[], 5).await;
    assert_eq!(stream.status(), StatusCode::OK);
    drop(stream);
    let in_flight = || async {
        let metrics = request(&deployment, Method::GET, "/metrics", METRICS, Value::Null).await;
        String::from_utf8(metrics.body.to_vec())
            .unwrap()
            .contains(r#"llm_proxy_key_in_flight_requests{key="one",user="olga"} 0"#)
    };
    let started = std::time::Instant::now();
    while !in_flight().await {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the request still holds its slot"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Also when the client sent more than the request (here a stray CRLF) before it left.
    let body = chat_body(5, true);
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\nauthorization: Bearer sk-one-0123456789\r\n\
         content-type: application/json\r\ncontent-length: {}\r\n\r\n{body}\r\n",
        body.len()
    );
    let mut client = tokio::net::TcpStream::connect(deployment.address)
        .await
        .unwrap();
    tokio::io::AsyncWriteExt::write_all(&mut client, request.as_bytes())
        .await
        .unwrap();
    let mut head = [0; 12];
    tokio::io::AsyncReadExt::read_exact(&mut client, &mut head)
        .await
        .unwrap();
    assert_eq!(&head, b"HTTP/1.1 200");
    drop(client);
    let started = std::time::Instant::now();
    while !in_flight().await {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the slot is held after a stray CRLF"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn paths_like_litellm() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let body = || serde_json::from_str::<Value>(&chat_body(1, false)).unwrap();
    // Base URLs without /v1.
    let short = request(&deployment, Method::POST, "/chat/completions", USER, body()).await;
    assert_eq!(short.status, StatusCode::OK);
    assert_eq!(
        deployment.engine().received().last().unwrap().path,
        "/v1/chat/completions"
    );
    // Non-canonical paths and wrong methods never reach the engine.
    let received = deployment.engine().received().len();
    for path in [
        "//v1/chat/completions",
        "/v1/chat/completions/",
        "/x/../v1/chat/completions",
    ] {
        let response = request(&deployment, Method::POST, path, USER, body()).await;
        assert_eq!(response.status, StatusCode::NOT_FOUND, "{path}");
    }
    let get = request(
        &deployment,
        Method::GET,
        "/v1/chat/completions",
        USER,
        Value::Null,
    )
    .await;
    assert_eq!(
        (get.status, &get.headers["allow"]),
        (StatusCode::METHOD_NOT_ALLOWED, &"POST".parse().unwrap())
    );
    assert_eq!(deployment.engine().received().len(), received);
    // Metrics label paths by their configured pattern.
    for model in ["a", "b", "c"] {
        let path = format!("/v1/models/{model}");
        request(&deployment, Method::GET, &path, USER, Value::Null).await;
    }
    let metrics = request(&deployment, Method::GET, "/metrics", METRICS, Value::Null).await;
    let metrics = String::from_utf8(metrics.body.to_vec()).unwrap();
    assert!(
        metrics.contains(r#"path="/v1/models/*",status="404"} 3"#),
        "{metrics}"
    );
    // Probes without an API key.
    for path in ["/health/liveliness", "/health/readiness"] {
        let probe = send(deployment.address, Method::GET, path, &[], "").await;
        assert_eq!(probe.status, StatusCode::OK, "{path}");
    }
}

#[tokio::test]
async fn round_two_key_rules() {
    let expiring = json!({"role": "admin", "expires_at": "2099-01-01T00:00:00Z"});
    let deployment = deploy_with_keys(vec![
        key(
            "scoped",
            "sam",
            "sk-scoped-0123456789",
            json!({"allowed_routes": ["/chat/completions"]}),
        ),
        key("expiring", "eve", "sk-expiring-0123456789", expiring),
    ])
    .await;
    // The only lasting admin can't expire, but other keys can change freely.
    let soon = json!({"expires_at": "2099-01-01T00:00:00Z"});
    assert_eq!(
        admin(
            &deployment,
            Method::PATCH,
            "keys/admin",
            ADMIN,
            soon.clone()
        )
        .await
        .0,
        StatusCode::CONFLICT
    );
    let block = json!({"blocked": true});
    assert_eq!(
        admin(&deployment, Method::PATCH, "keys/user", ADMIN, block)
            .await
            .0,
        StatusCode::OK
    );
    // Routes may name the path as the client sends it, before the /v1 rewrite.
    let body = serde_json::from_str::<Value>(&chat_body(1, false)).unwrap();
    let short = request(
        &deployment,
        Method::POST,
        "/chat/completions",
        "sk-scoped-0123456789",
        body,
    )
    .await;
    assert_eq!(short.status, StatusCode::OK);
    // Without any lasting admin (e.g. a policy of expiring admins), changes still work.
    std::fs::write(
        deployment.socket_path(KEYS_FILE),
        json!({"keys": [
            key("expiring", "eve", "sk-expiring-0123456789", json!({"role": "admin", "expires_at": "2099-01-01T00:00:00Z"})),
            key("user", "carol", USER, json!({})),
        ]})
        .to_string(),
    )
    .unwrap();
    wait_until(|| events(&deployment.audit_log(), "keys_reloaded").len() == 1).await;
    let block = json!({"blocked": true});
    let (status_code, _) = admin(
        &deployment,
        Method::PATCH,
        "keys/user",
        "sk-expiring-0123456789",
        block,
    )
    .await;
    assert_eq!(status_code, StatusCode::OK);
    // api-key wins over x-api-key, like in LiteLLM, and foreign credentials are stripped.
    let headers = [
        ("api-key", "sk-expiring-0123456789"),
        ("x-api-key", "sk-wrong"),
        ("x-goog-api-key", "secret"),
    ];
    let both = send(
        deployment.address,
        Method::POST,
        "/v1/chat/completions",
        &headers,
        chat_body(1, false),
    )
    .await;
    assert_eq!(both.status, StatusCode::OK);
    assert!(!deployment
        .engine()
        .received()
        .last()
        .unwrap()
        .headers
        .contains_key("x-goog-api-key"));
    // LiteLLM's other liveness path.
    let probe = send(deployment.address, Method::GET, "/health/liveness", &[], "").await;
    assert_eq!(probe.status, StatusCode::OK);
}

#[tokio::test]
async fn reservations_survive_restarts() {
    // The stream reserves 93 tokens, the second request 43.
    let limits = json!({"limits": {"tpm_limit": 120}});
    let mut deployment =
        deploy_with_keys(vec![key("tpm", "tina", "sk-tpm-0123456789", limits)]).await;
    // The running stream reserves about 90 output tokens plus its prompt.
    let mut stream = open_chat_stream(deployment.address, "sk-tpm-0123456789", &[], 90).await;
    stream.body_mut().frame().await.unwrap().unwrap();
    let limited = chat(
        deployment.address,
        "sk-tpm-0123456789",
        &[],
        chat_body(40, false),
    )
    .await;
    assert_eq!(limited.status, StatusCode::TOO_MANY_REQUESTS);
    deployment.restart_middle_end().await;
    let limited = chat(
        deployment.address,
        "sk-tpm-0123456789",
        &[],
        chat_body(40, false),
    )
    .await;
    assert_eq!(limited.status, StatusCode::TOO_MANY_REQUESTS);
    stream_text(stream).await;
}

#[tokio::test]
async fn unwritten_audit_events_survive_restarts() {
    // The audit log's directory doesn't exist yet: every write fails.
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("later").join("audit.jsonl");
    let keys = Some(json!({ "keys": base_keys() }));
    let general = json!({"gateway": {"audit_log": path}});
    let mut deployment = deploy(DeploymentOptions {
        keys,
        general,
        ..DeploymentOptions::default()
    })
    .await;
    assert_eq!(
        status(&deployment, "sk-wrong-0123456789", &[]).await,
        StatusCode::UNAUTHORIZED
    );
    let is_failing = || async {
        let response = request(&deployment, Method::GET, "/metrics", METRICS, Value::Null).await;
        let text = String::from_utf8(response.body.to_vec()).unwrap();
        text.lines()
            .any(|line| line == "llm_proxy_audit_log_failing 1")
    };
    let started = std::time::Instant::now();
    while !is_failing().await {
        assert!(started.elapsed() < Duration::from_secs(10));
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    deployment.restart_middle_end().await;
    std::fs::create_dir(path.parent().unwrap()).unwrap();
    assert_eq!(status(&deployment, USER, &[]).await, StatusCode::OK);
    // The rejection which the old process couldn't write comes first.
    let log = || -> Vec<Value> {
        std::fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    wait_until(|| events(&log(), "request").len() == 1).await;
    let names: Vec<String> = log()
        .iter()
        .map(|event| event["event"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(names[0], "rejected", "{names:?}");
}

// Regression tests for the findings of the A/B hunt against LiteLLM (e2e/litellm-ab).

fn deploy_options(extra_keys: Vec<Value>, general: Value) -> DeploymentOptions {
    let keys = [base_keys(), extra_keys].concat();
    let keys = Some(json!({ "keys": keys }));
    DeploymentOptions {
        keys,
        general,
        ..DeploymentOptions::default()
    }
}

fn metrics_text(response: &ClientResponse) -> String {
    String::from_utf8(response.body.to_vec()).unwrap()
}

/// Hand edits which an admin API change picks up are audited too.
#[tokio::test]
async fn hand_edits_picked_up_by_admin_changes_are_audited() {
    let general = json!({"gateway": {"keys_file_poll_seconds": 3600}});
    let deployment = deploy(deploy_options(Vec::new(), general)).await;
    let path = deployment.socket_path(KEYS_FILE);
    let mut file: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for record in file["keys"].as_array_mut().unwrap() {
        if record["id"] == "user" {
            record["blocked"] = json!(true);
        }
    }
    std::fs::write(&path, file.to_string()).unwrap();
    let change = json!({"description": "changed"});
    assert_eq!(
        admin(&deployment, Method::PATCH, "keys/metrics", ADMIN, change)
            .await
            .0,
        StatusCode::OK
    );
    assert_eq!(
        status(&deployment, USER, &[]).await,
        StatusCode::UNAUTHORIZED
    );
    let log = deployment.audit_log();
    let names: Vec<&str> = log
        .iter()
        .map(|event| event["event"].as_str().unwrap())
        .collect();
    assert_eq!(names[..2], ["keys_reloaded", "key_updated"]);
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
    let started = std::time::Instant::now();
    wait_until(|| !events(&deployment.audit_log(), "request").is_empty()).await;
    assert!(started.elapsed() < Duration::from_secs(2), "recorded late");
    assert_eq!(events(&deployment.audit_log(), "request")[0]["status"], 400);

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
    wait_until(|| events(&deployment.audit_log(), "request").len() == 2).await;
    let failed = events(&deployment.audit_log(), "request")[1].clone();
    assert_eq!(
        (&failed["outcome"], &failed["status"]),
        (&json!("failed"), &json!(200))
    );
    assert!(failed["time_to_first_token_seconds"].as_f64().is_some());
}

/// Fields which clients control are cut in the audit log.
#[tokio::test]
async fn long_fields_are_cut_in_the_audit_log() {
    let general = json!({"gateway": {"client_ip_header": "x-forwarded-for"}});
    let deployment = deploy(deploy_options(Vec::new(), general)).await;
    let path = format!("/v1/models/{}", "p".repeat(2000));
    let agent = "a".repeat(2000);
    let headers = [
        ("authorization", "Bearer sk-unknown-0123456789"),
        ("user-agent", agent.as_str()),
    ];
    let rejected = send(deployment.address, Method::GET, &path, &headers, "").await;
    assert_eq!(rejected.status, StatusCode::UNAUTHORIZED);
    let client = "c".repeat(2000);
    let authorization = format!("Bearer {USER}");
    let headers = [
        ("authorization", authorization.as_str()),
        ("x-forwarded-for", client.as_str()),
    ];
    let body = json!({"model": "m".repeat(2000), "messages": [{"role": "user", "content": "x"}], "max_tokens": 1});
    send(
        deployment.address,
        Method::POST,
        "/v1/chat/completions",
        &headers,
        body.to_string(),
    )
    .await;
    wait_until(|| !events(&deployment.audit_log(), "request").is_empty()).await;
    let log = deployment.audit_log();
    let rejected = events(&log, "rejected")[0];
    let request = events(&log, "request")[0];
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
    let deployment = deploy(deploy_options(Vec::new(), general)).await;
    let big = json!({"messages": [{"role": "user", "content": "x".repeat(2000)}]});
    assert_eq!(
        request(&deployment, Method::POST, "/v1/chat/completions", USER, big)
            .await
            .status,
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let get = request(
        &deployment,
        Method::GET,
        "/v1/chat/completions",
        USER,
        Value::Null,
    )
    .await;
    assert_eq!(get.status, StatusCode::METHOD_NOT_ALLOWED);
    let metrics =
        metrics_text(&request(&deployment, Method::GET, "/metrics", METRICS, Value::Null).await);
    for expected in [
        r#"llm_proxy_rejected_requests_total{status="413",reason="request_body_too_large"} 1"#,
        r#"llm_proxy_rejected_requests_total{status="405",reason="method_not_allowed"} 1"#,
    ] {
        assert!(metrics.contains(expected), "missing {expected}");
    }
}

/// A request which the proxy turns away because its queue is full doesn't count against
/// its key's rpm, and its 503 says when to retry.
#[tokio::test]
async fn full_queues_refund_the_key() {
    let options = DeploymentOptions {
        rank_capacity: 1,
        ..deploy_options(
            vec![key(
                "rpm",
                "rita",
                "sk-rpm-0123456789",
                json!({"limits": {"rpm_limit": 5}}),
            )],
            json!({"limits": {"max_queued_requests": 1}}),
        )
    };
    let deployment = deploy(options).await;
    let address = deployment.address;
    let running = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-alice", &[], 60).await).await
    });
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let queued = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-bob", &[], 1).await).await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let full = chat(address, "sk-rpm-0123456789", &[], chat_body(1, false)).await;
    assert_eq!(
        (full.status, &full.headers["retry-after"]),
        (StatusCode::SERVICE_UNAVAILABLE, &"1".parse().unwrap())
    );
    let (_, keys) = admin(&deployment, Method::GET, "keys", ADMIN, Value::Null).await;
    let usage = keys["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|key| key["id"] == "rpm")
        .unwrap()["usage"]
        .clone();
    assert_eq!(usage["requests_last_minute"], 0);
    running.await.unwrap();
    queued.await.unwrap();
}

/// Virtual users (a header value) are cut: they become labels and log fields.
#[tokio::test]
async fn virtual_users_are_cut() {
    let deployment = deploy_with_keys(Vec::new()).await;
    let user = "u".repeat(500);
    let response = chat(
        deployment.address,
        "sk-litellm",
        &[("x-end-user", user.as_str())],
        chat_body(1, false),
    )
    .await;
    assert_eq!(response.status, StatusCode::OK);
    wait_for_requests(&deployment, 1).await;
    let log = deployment.audit_log();
    assert_eq!(
        events(&log, "request")[0]["user"].as_str().unwrap().len(),
        "litellm/".len() + 128
    );
}

/// The token estimates of requests in flight show in the key's usage and metrics.
#[tokio::test]
async fn reservations_show_in_usage_and_metrics() {
    let limits = json!({"limits": {"tpm_limit": 1000}});
    let deployment = deploy_with_keys(vec![key("tpm", "tina", "sk-tpm-0123456789", limits)]).await;
    let mut stream = open_chat_stream(deployment.address, "sk-tpm-0123456789", &[], 100).await;
    stream.body_mut().frame().await.unwrap().unwrap();
    let (_, keys) = admin(&deployment, Method::GET, "keys", ADMIN, Value::Null).await;
    let usage = keys["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|key| key["id"] == "tpm")
        .unwrap()["usage"]
        .clone();
    let reserved = usage["reserved_tokens"].as_u64().unwrap();
    assert!(reserved >= 100, "{usage}");
    let metrics =
        metrics_text(&request(&deployment, Method::GET, "/metrics", METRICS, Value::Null).await);
    let expected = format!(
        r#"llm_proxy_key_remaining_tokens{{key="tpm",user="tina",window="minute"}} {}"#,
        1000 - reserved
    );
    assert!(metrics.contains(&expected), "missing {expected}");
    stream_text(stream).await;
}

/// Passthrough requests in flight across a restart are still recorded.
#[tokio::test]
async fn passthrough_requests_survive_restarts() {
    let mut deployment = deploy_with_keys(Vec::new()).await;
    let address = deployment.address;
    let tokenize = tokio::spawn(async move {
        let authorization = format!("Bearer {USER}");
        let body = json!({"prompt": "x", "mock_delay_ms": 300});
        send(
            address,
            Method::POST,
            "/tokenize",
            &[("authorization", authorization.as_str())],
            body.to_string(),
        )
        .await
    });
    wait_until(|| {
        deployment
            .engine()
            .received()
            .iter()
            .any(|request| request.path == "/tokenize")
    })
    .await;
    deployment.restart_middle_end().await;
    assert_eq!(tokenize.await.unwrap().status, StatusCode::NOT_FOUND);
    wait_until(|| {
        events(&deployment.audit_log(), "request")
            .iter()
            .any(|event| event["path"] == "/tokenize" && event["status"] == 404)
    })
    .await;
}

/// A queued stream whose client left after the proxy's early head got that head's 200.
#[tokio::test]
async fn queued_streams_left_after_the_early_head_are_aborted_with_200() {
    let options = DeploymentOptions {
        rank_capacity: 1,
        ..deploy_options(
            Vec::new(),
            json!({"limits": {"keepalive_interval_seconds": 0.05}}),
        )
    };
    let deployment = deploy(options).await;
    let address = deployment.address;
    let running = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-alice", &[], 60).await).await
    });
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let mut queued = open_chat_stream(address, USER, &[], 5).await;
    assert_eq!(queued.status(), StatusCode::OK);
    queued.body_mut().frame().await.unwrap().unwrap();
    drop(queued);
    wait_until(|| !events(&deployment.audit_log(), "request").is_empty()).await;
    let aborted = events(&deployment.audit_log(), "request")[0].clone();
    assert_eq!(
        (&aborted["key_id"], &aborted["outcome"], &aborted["status"]),
        (&json!("user"), &json!("aborted"), &json!(200))
    );
    running.await.unwrap();
}

/// A restart keeps the time a request generated before it.
#[tokio::test]
async fn generating_time_survives_restarts() {
    let mut deployment = deploy_with_keys(Vec::new()).await;
    let mut stream = open_chat_stream(deployment.address, USER, &[], 60).await;
    let mut received = Vec::new();
    while received
        .windows(5)
        .filter(|window| *window == b"token")
        .count()
        < 30
    {
        let frame = stream.body_mut().frame().await.unwrap().unwrap();
        received.extend_from_slice(frame.data_ref().unwrap());
    }
    deployment.restart_middle_end().await;
    stream_text(stream).await;
    let log = wait_for_requests(&deployment, 1).await;
    let event = events(&log, "request")[0];
    assert!(
        event["generating_seconds"].as_f64().unwrap() >= 0.5,
        "{event}"
    );
}

/// A restart starts with the keys of the old process, not with the file its pre-flight
/// check read (which an admin change may have replaced since).
#[tokio::test]
async fn restarts_keep_the_current_keys() {
    let general = json!({"gateway": {"keys_file_poll_seconds": 3600}});
    let mut deployment = deploy(deploy_options(Vec::new(), general)).await;
    let settings = json!({"user": "new", "default_class": "batch"});
    let (_, created) = admin(&deployment, Method::POST, "keys", ADMIN, settings).await;
    let api_key = created["api_key"].as_str().unwrap().to_string();
    // The test deployment restarts with the configuration texts it started with.
    deployment.restart_middle_end().await;
    assert_eq!(status(&deployment, &api_key, &[]).await, StatusCode::OK);
    assert!(events(&deployment.audit_log(), "keys_reloaded").is_empty());
}

/// Scheduling log events which couldn't be written go to the new process.
#[tokio::test]
async fn unwritten_scheduling_log_events_survive_restarts() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("later").join("scheduling.log");
    let mut deployment = deploy(deploy_options(Vec::new(), json!({"scheduling_log": path}))).await;
    assert_eq!(status(&deployment, USER, &[]).await, StatusCode::OK);
    let is_failing = || async {
        let response = request(&deployment, Method::GET, "/metrics", METRICS, Value::Null).await;
        metrics_text(&response)
            .lines()
            .any(|line| line == "llm_proxy_scheduling_log_failing 1")
    };
    let started = std::time::Instant::now();
    while !is_failing().await {
        assert!(started.elapsed() < Duration::from_secs(10));
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    deployment.restart_middle_end().await;
    std::fs::create_dir(path.parent().unwrap()).unwrap();
    assert_eq!(status(&deployment, USER, &[]).await, StatusCode::OK);
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
        ..deploy_options(Vec::new(), json!({}))
    };
    let mut deployment = deploy(options).await;
    let address = deployment.address;
    let running = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-alice", &[], 60).await).await
    });
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
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
    let order: Vec<u64> = deployment
        .engine()
        .received()
        .iter()
        .filter_map(|request| request.body["max_tokens"].as_u64())
        .filter(|tokens| (11..=16).contains(tokens))
        .collect();
    assert_eq!(order, [11, 12, 13, 14, 15, 16]);
}
