mod support;

use std::time::Duration;

use bytes::{Bytes, BytesMut};
use http_body_util::BodyExt;
use hyper::{Method, StatusCode};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

use llm_proxy::wire::{decode_frame, encode_frame, FrameKind};
use support::mock_engine::TOKEN_INTERVAL;
use support::{
    chat, chat_body, deploy, event_names, open_chat_stream, send, stream_text, token_sequence,
    wait_until, DeploymentOptions,
};

#[tokio::test]
async fn idle_suspension_during_batch_request() {
    let deployment = deploy(DeploymentOptions::default()).await;
    let address = deployment.address;
    let idle_streams: Vec<_> = (0..3)
        .map(|_| {
            tokio::spawn(async move {
                let response =
                    open_chat_stream(address, "sk-bob", &[("x-priority", "idle")], 60).await;
                stream_text(response).await
            })
        })
        .collect();
    wait_until(|| deployment.engine().generating_ids().len() == 3).await;

    let batch = tokio::spawn(chat(address, "sk-alice", &[], chat_body(20, false)));
    wait_until(|| deployment.engine().suspended_ids().len() == 3).await;
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let response = batch.await.unwrap();
    assert_eq!(response.status, StatusCode::OK);

    wait_until(|| deployment.engine().suspended_ids().is_empty()).await;
    for stream in idle_streams {
        let text = stream.await.unwrap();
        assert_eq!(token_sequence(&text).len(), 60);
        assert!(text.ends_with("data: [DONE]\n\n"));
    }
}

#[tokio::test]
async fn keepalive_comments_during_suspension() {
    let deployment = deploy(DeploymentOptions {
        general: json!({"limits": {"keepalive_interval_seconds": 0.1}}),
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let idle = tokio::spawn(async move {
        let response = open_chat_stream(address, "sk-bob", &[("x-priority", "idle")], 40).await;
        stream_text(response).await
    });
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let batch = chat(address, "sk-alice", &[], chat_body(60, false)).await;
    assert_eq!(batch.status, StatusCode::OK);
    let text = idle.await.unwrap();
    assert!(text.contains(": keepalive\n\n"));
    assert_eq!(token_sequence(&text).len(), 40);
    assert!(!text.contains("}: keepalive"));
}

#[tokio::test]
async fn non_streaming_keepalive() {
    let deployment = deploy(DeploymentOptions {
        general: json!({"limits": {"keepalive_interval_seconds": 0.05}}),
        ..DeploymentOptions::default()
    })
    .await;
    let response = chat(deployment.address, "sk-alice", &[], chat_body(30, false)).await;
    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(response.headers["content-type"], "application/json");
    assert!(response.body.starts_with(b"\n\n"));
    let body: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(body["tokens"], 30);
}

#[tokio::test]
async fn queued_stream_keepalive() {
    let deployment = deploy(DeploymentOptions {
        rank_capacity: 1,
        general: json!({"limits": {"keepalive_interval_seconds": 0.05}}),
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let running = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-alice", &[], 60).await).await
    });
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let queued = tokio::time::timeout(
        TOKEN_INTERVAL * 30,
        open_chat_stream(address, "sk-bob", &[], 5),
    )
    .await
    .unwrap();
    assert_eq!(queued.status(), StatusCode::OK);
    assert_eq!(queued.headers()["content-type"], "text/event-stream");
    let text = stream_text(queued).await;
    assert!(text.starts_with(": keepalive\n\n"));
    assert_eq!(token_sequence(&text), (0..5).collect::<Vec<_>>());
    assert_eq!(token_sequence(&running.await.unwrap()).len(), 60);
    assert_eq!(deployment.engine().peak_generating(), 1);
}

#[tokio::test]
async fn single_response_head_after_unreachable_engine() {
    let deployment = deploy(DeploymentOptions {
        rank_capacity: 1,
        general: json!({"limits": {"keepalive_interval_seconds": 0.05}}),
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let running = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-alice", &[], 30).await).await
    });
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    deployment.engine().close_http().await;
    let mut middle_end = UnixStream::connect(deployment.socket_path("middle-end.sock"))
        .await
        .unwrap();
    let head = "POST /v1/chat/completions HTTP/1.1\r\nauthorization: Bearer sk-bob\r\n\r\n";
    let mut request = BytesMut::new();
    encode_frame(FrameKind::RequestHead, head.as_bytes(), &mut request);
    encode_frame(
        FrameKind::RequestBody,
        chat_body(5, true).as_bytes(),
        &mut request,
    );
    encode_frame(FrameKind::RequestEnd, b"", &mut request);
    middle_end.write_all(&request).await.unwrap();
    let mut response = BytesMut::new();
    while tokio::time::timeout(Duration::from_secs(5), middle_end.read_buf(&mut response))
        .await
        .unwrap()
        .unwrap()
        > 0
    {}
    let mut frame_kinds = Vec::new();
    while let Some(frame) = decode_frame(&mut response).unwrap() {
        frame_kinds.push(frame.kind);
    }
    assert_eq!(frame_kinds[0], FrameKind::ResponseHead);
    assert!(!frame_kinds[1..].contains(&FrameKind::ResponseHead));
    assert!(!frame_kinds.contains(&FrameKind::ResponseEnd));
    assert_eq!(token_sequence(&running.await.unwrap()).len(), 30);
}

#[tokio::test]
async fn stream_flag_normalization() {
    let deployment = deploy(DeploymentOptions::default()).await;
    let body = json!({"messages": [], "max_tokens": 3, "stream": "yes"});
    let response = chat(deployment.address, "sk-alice", &[], body.to_string()).await;
    assert_eq!(response.headers["content-type"], "text/event-stream");
    let text = String::from_utf8(response.body.to_vec()).unwrap();
    assert_eq!(token_sequence(&text), vec![0, 1, 2]);
    assert_eq!(
        deployment.engine().received().pop().unwrap().body["stream"],
        json!(true)
    );
}

#[tokio::test]
async fn head_request_content_length() {
    let deployment = deploy(DeploymentOptions::default()).await;
    let request = |method| {
        send(
            deployment.address,
            method,
            "/v1/models",
            &[("authorization", "Bearer sk-alice")],
            Bytes::new(),
        )
    };
    let get = request(Method::GET).await;
    let head = request(Method::HEAD).await;
    assert_eq!(head.status, StatusCode::OK);
    assert!(head.body.is_empty());
    assert_eq!(
        head.headers["content-length"],
        get.body.len().to_string().as_str()
    );
}

#[tokio::test]
async fn rank_capacity_limit() {
    let deployment = deploy(DeploymentOptions {
        rank_capacity: 2,
        ..DeploymentOptions::default()
    })
    .await;
    let requests: Vec<_> = (0..5)
        .map(|index| {
            let key = if index % 2 == 0 { "sk-alice" } else { "sk-bob" };
            tokio::spawn(chat(deployment.address, key, &[], chat_body(10, false)))
        })
        .collect();
    for request in requests {
        assert_eq!(request.await.unwrap().status, StatusCode::OK);
    }
    assert_eq!(deployment.engine().peak_generating(), 2);
}

#[tokio::test]
async fn class_soft_cap() {
    let mut accounts = DeploymentOptions::default().accounts;
    accounts["classes"][1]["soft_cap"] = json!(2);
    let deployment = deploy(DeploymentOptions {
        rank_capacity: 4,
        accounts,
        ..DeploymentOptions::default()
    })
    .await;
    let requests: Vec<_> = ["sk-alice", "sk-alice", "sk-bob", "sk-bob"]
        .into_iter()
        .map(|key| tokio::spawn(chat(deployment.address, key, &[], chat_body(10, false))))
        .collect();
    for request in requests {
        assert_eq!(request.await.unwrap().status, StatusCode::OK);
    }
    assert_eq!(deployment.engine().peak_generating(), 2);
}

#[tokio::test]
async fn class_max_parallel_per_user() {
    let mut accounts = DeploymentOptions::default().accounts;
    accounts["classes"][1]["max_parallel_per_user"] = json!(2);
    let deployment = deploy(DeploymentOptions {
        rank_capacity: 8,
        accounts,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let bob = tokio::spawn(chat(address, "sk-bob", &[], chat_body(200, false)));
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let alice: Vec<_> = (0..4)
        .map(|_| tokio::spawn(chat(address, "sk-alice", &[], chat_body(10, false))))
        .collect();
    for request in alice {
        assert_eq!(request.await.unwrap().status, StatusCode::OK);
    }
    assert_eq!(deployment.engine().peak_generating(), 3);
    assert_eq!(bob.await.unwrap().status, StatusCode::OK);
}

#[tokio::test]
async fn turns_over_max_parallel_per_user() {
    let mut accounts = DeploymentOptions::default().accounts;
    accounts["classes"][1]["max_parallel_per_user"] = json!(1);
    let deployment = deploy(DeploymentOptions {
        accounts,
        general: json!({"quality_of_service": {"swap_hysteresis_seconds": 0.2}}),
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let bob = tokio::spawn(chat(address, "sk-bob", &[], chat_body(300, false)));
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let tokens = 100;
    let alice: Vec<_> = (0..2)
        .map(|_| {
            tokio::spawn(async move {
                stream_text(open_chat_stream(address, "sk-alice", &[], tokens).await).await
            })
        })
        .collect();
    wait_until(|| deployment.engine().received().len() == 3).await;
    assert!(alice.iter().all(|stream| !stream.is_finished()));
    for stream in alice {
        assert_eq!(
            token_sequence(&stream.await.unwrap()),
            (0..tokens).collect::<Vec<_>>()
        );
    }
    assert_eq!(bob.await.unwrap().status, StatusCode::OK);
}

#[tokio::test]
async fn scheduling_log_process_events() {
    let mut deployment = deploy(DeploymentOptions::default()).await;
    deployment.restart_middle_end().await;
    deployment.shutdown_middle_end().await;
    let all_events = deployment.scheduling_log();
    for event in &all_events {
        assert!(event["time"]
            .as_str()
            .unwrap()
            .parse::<jiff::Timestamp>()
            .is_ok());
    }
    let events: Vec<&Value> = all_events
        .iter()
        .filter(|event| event["event"] != "caps")
        .collect();
    let names: Vec<&str> = events
        .iter()
        .map(|event| event["event"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["start", "stop", "start", "stop"]);
    assert_eq!(
        [
            &events[0]["restart"],
            &events[1]["reason"],
            &events[2]["restart"],
            &events[3]["reason"]
        ],
        [
            &json!(false),
            &json!("restart"),
            &json!(true),
            &json!("shutdown")
        ]
    );
    assert_eq!(events[0]["rank_caps"], json!([16]));
    assert_eq!(events[0]["classes"][1]["name"], "batch");
}

fn request_events<'a>(events: &'a [Value], request: &Value) -> Vec<&'a Value> {
    events
        .iter()
        .filter(|event| &event["request"] == request)
        .collect()
}

fn names_among<'a>(events: &[&'a Value], names: &[&str]) -> Vec<&'a str> {
    events
        .iter()
        .map(|event| event["event"].as_str().unwrap())
        .filter(|name| names.contains(name))
        .collect()
}

fn first_named<'a>(events: &[&'a Value], name: &str) -> &'a Value {
    events.iter().find(|event| event["event"] == name).unwrap()
}

const DECISIONS: &[&str] = &["admitted", "started", "suspended", "resumed"];

#[tokio::test]
async fn scheduling_log_request_lifecycle() {
    let deployment = deploy(DeploymentOptions {
        rank_capacity: 1,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let bob = tokio::spawn(chat(address, "sk-bob", &[], chat_body(100, false)));
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let interactive = [("x-priority", "interactive")];
    let alice = chat(address, "sk-alice", &interactive, chat_body(10, false)).await;
    assert_eq!(alice.status, StatusCode::OK);
    assert_eq!(bob.await.unwrap().status, StatusCode::OK);
    let finish_count = || {
        let events = deployment.scheduling_log();
        event_names(&events)
            .iter()
            .filter(|name| **name == "finished")
            .count()
    };
    wait_until(|| finish_count() == 2).await;
    let events = deployment.scheduling_log();
    let admitted: Vec<&Value> = events
        .iter()
        .filter(|event| event["event"] == "admitted")
        .collect();
    let (bob, alice) = (admitted[0], admitted[1]);
    assert_eq!(
        [
            &bob["user"],
            &bob["class"],
            &bob["priority"],
            &bob["path"],
            &bob["model"],
            &bob["stream"],
            &bob["max_tokens"],
            &bob["body_bytes"]
        ],
        [
            &json!("bob"),
            &json!("batch"),
            &json!(0),
            &json!("/v1/chat/completions"),
            &json!("mock-model"),
            &json!(false),
            &json!(100),
            &json!(chat_body(100, false).len())
        ]
    );
    assert_eq!(alice["class"], "interactive");

    let bob_events = request_events(&events, &bob["request"]);
    assert_eq!(
        names_among(&bob_events, DECISIONS),
        ["admitted", "started", "suspended", "resumed"]
    );
    assert_eq!(first_named(&bob_events, "suspended")["reason"], "hard_cap");
    let engine_stops: Vec<&Value> = bob_events
        .iter()
        .filter(|event| event["event"] == "engine_stopped")
        .map(|event| &event["reason"])
        .collect();
    assert_eq!(engine_stops, [&json!("suspended"), &json!("finished")]);
    assert_eq!(first_named(&bob_events, "completed")["status"], 200);
    let summary = first_named(&bob_events, "finished");
    assert_eq!(
        [&summary["user"], &summary["class"], &summary["suspensions"]],
        [&json!("bob"), &json!("batch"), &json!(1)]
    );
    let seconds = |field: &str| summary[field].as_f64().unwrap();
    assert!(seconds("generating_seconds") > 0.5);
    assert_eq!(seconds("usage_seconds"), seconds("generating_seconds"));
    assert!(seconds("total_seconds") > seconds("generating_seconds"));
    assert!(seconds("queued_seconds") < 0.5);

    let alice_events = request_events(&events, &alice["request"]);
    assert_eq!(
        names_among(&alice_events, DECISIONS),
        ["admitted", "started"]
    );
    assert_eq!(first_named(&alice_events, "finished")["suspensions"], 0);
}

#[tokio::test]
async fn scheduling_log_caps() {
    let mut accounts = DeploymentOptions::default().accounts;
    accounts["classes"][1]["soft_cap"] = json!(2);
    let deployment = deploy(DeploymentOptions {
        accounts,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let alice: Vec<_> = (0..3)
        .map(|_| tokio::spawn(chat(address, "sk-alice", &[], chat_body(50, false))))
        .collect();
    wait_until(|| deployment.engine().generating_ids().len() == 3).await;
    let bob = chat(address, "sk-bob", &[], chat_body(5, false)).await;
    assert_eq!(bob.status, StatusCode::OK);
    for request in alice {
        assert_eq!(request.await.unwrap().status, StatusCode::OK);
    }
    wait_until(|| {
        let events = deployment.scheduling_log();
        event_names(&events)
            .iter()
            .filter(|name| **name == "finished")
            .count()
            == 4
    })
    .await;
    let events = deployment.scheduling_log();
    let caps: Vec<Value> = events
        .iter()
        .filter(|event| event["event"] == "caps")
        .map(|event| {
            json!([
                event["has_multiple_users"],
                event["soft_cap"],
                event["rank_caps"]
            ])
        })
        .collect();
    assert_eq!(
        caps,
        [
            json!([false, null, [16]]),
            json!([true, 2, [2]]),
            json!([false, null, [16]]),
        ]
    );
    let suspension = events
        .iter()
        .find(|event| event["event"] == "suspended")
        .unwrap();
    assert_eq!(suspension["reason"], "soft_cap");
}

#[tokio::test]
async fn scheduling_log_aborted_request() {
    let deployment = deploy(DeploymentOptions::default()).await;
    let address = deployment.address;
    let client = tokio::spawn(async move {
        let response = open_chat_stream(address, "sk-alice", &[], 10_000).await;
        let mut body = response.into_body();
        body.frame().await;
        tokio::time::sleep(Duration::from_secs(60)).await;
    });
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    client.abort();
    wait_until(|| event_names(&deployment.scheduling_log()).contains(&"finished")).await;
    let events = deployment.scheduling_log();
    let request_events: Vec<&Value> = events
        .iter()
        .filter(|event| event.get("request").is_some())
        .collect();
    assert_eq!(
        names_among(
            &request_events,
            &["admitted", "started", "aborted", "completed", "finished"]
        ),
        ["admitted", "started", "aborted", "finished"]
    );
}

#[tokio::test]
async fn scheduling_log_rejections() {
    let deployment = deploy(DeploymentOptions {
        general: json!({"limits": {"max_queued_requests": 0}}),
        ..DeploymentOptions::default()
    })
    .await;
    let unknown = chat(deployment.address, "sk-unknown", &[], chat_body(1, false)).await;
    assert_eq!(unknown.status, StatusCode::UNAUTHORIZED);
    let queue_full = chat(deployment.address, "sk-alice", &[], chat_body(1, false)).await;
    assert_eq!(queue_full.status, StatusCode::SERVICE_UNAVAILABLE);
    let rejections = || {
        let events = deployment.scheduling_log();
        events
            .into_iter()
            .filter(|event| event["event"] == "rejected")
            .collect::<Vec<Value>>()
    };
    wait_until(|| rejections().len() == 2).await;
    let rejected: Vec<Value> = rejections()
        .iter()
        .map(|event| {
            json!([
                event["event"],
                event["status"],
                event["reason"],
                event["method"],
                event["path"],
                event["account"]
            ])
        })
        .collect();
    assert_eq!(
        rejected,
        [
            json!([
                "rejected",
                401,
                "unknown API key",
                "POST",
                "/v1/chat/completions",
                null
            ]),
            json!([
                "rejected",
                503,
                "too many queued requests; try again later",
                "POST",
                "/v1/chat/completions",
                "alice"
            ]),
        ]
    );
}

#[tokio::test]
async fn vllm_request_rewrite() {
    let deployment = deploy(DeploymentOptions {
        ranks: 2,
        ..DeploymentOptions::default()
    })
    .await;
    let body = json!({
        "messages": [], "max_tokens": 1, "priority": 5,
        "rid": "client-rid", "request_id": "client-request-id", "routed_dp_rank": 1
    });
    let response = chat(
        deployment.address,
        "sk-alice",
        &[
            ("x-priority", "interactive"),
            ("x-vllm-priority", "-100"),
            ("x-request-id", "client"),
            ("x-data-parallel-rank", "1"),
            ("accept-encoding", "gzip"),
        ],
        body.to_string(),
    )
    .await;
    assert_eq!(response.status, StatusCode::OK);
    let received = deployment.engines[0].received().pop().unwrap();
    let request_id = received.headers["x-request-id"].to_str().unwrap();
    assert!(request_id.starts_with("qos-") && request_id.len() == 4 + 16 + 1 + 16);
    assert_eq!(received.headers["x-data-parallel-rank"], "0");
    assert_eq!(received.body["priority"], json!(-10));
    for field in ["rid", "request_id", "routed_dp_rank"] {
        assert!(received.body.get(field).is_none());
    }
    for header in [
        "authorization",
        "x-vllm-priority",
        "x-priority",
        "x-api-key",
        "accept-encoding",
    ] {
        assert!(received.headers.get(header).is_none());
    }
}

#[tokio::test]
async fn sglang_request_rewrite() {
    let deployment = deploy(DeploymentOptions {
        engine: "sglang",
        ranks: 2,
        ..DeploymentOptions::default()
    })
    .await;
    let response = chat(
        deployment.address,
        "sk-alice",
        &[("x-priority", "interactive")],
        chat_body(1, false),
    )
    .await;
    assert_eq!(response.status, StatusCode::OK);
    let received = deployment.engines[0].received().pop().unwrap();
    assert!(received.body["rid"].as_str().unwrap().starts_with("qos-"));
    assert_eq!(received.body["routed_dp_rank"], json!(0));
    assert_eq!(received.body["priority"], json!(10));
    assert!(received.headers.get("x-request-id").is_none());
}

#[tokio::test]
async fn authentication() {
    let deployment = deploy(DeploymentOptions::default()).await;
    let missing = send(
        deployment.address,
        Method::POST,
        "/v1/chat/completions",
        &[],
        chat_body(1, false),
    )
    .await;
    assert_eq!(missing.status, StatusCode::UNAUTHORIZED);
    assert!(String::from_utf8_lossy(&missing.body).contains("missing API key"));
    let unknown = chat(deployment.address, "sk-unknown", &[], chat_body(1, false)).await;
    assert_eq!(unknown.status, StatusCode::UNAUTHORIZED);
    assert!(String::from_utf8_lossy(&unknown.body).contains("unknown API key"));
    let anthropic_style = send(
        deployment.address,
        Method::POST,
        "/v1/chat/completions",
        &[("x-api-key", "sk-alice")],
        chat_body(1, false),
    )
    .await;
    assert_eq!(anthropic_style.status, StatusCode::OK);
    let models_without_key = send(
        deployment.address,
        Method::GET,
        "/v1/models",
        &[],
        Bytes::new(),
    )
    .await;
    assert_eq!(models_without_key.status, StatusCode::UNAUTHORIZED);

    let mut options = DeploymentOptions::default();
    options.accounts["accounts"]
        .as_array_mut()
        .unwrap()
        .push(json!({"name": "anonymous", "api_key": "", "default_class": "idle"}));
    let with_default_user = deploy(options).await;
    let anonymous = send(
        with_default_user.address,
        Method::POST,
        "/v1/chat/completions",
        &[],
        chat_body(1, false),
    )
    .await;
    assert_eq!(anonymous.status, StatusCode::OK);
}

#[tokio::test]
async fn class_access() {
    let deployment = deploy(DeploymentOptions::default()).await;
    let forbidden = chat(
        deployment.address,
        "sk-bob",
        &[("x-priority", "interactive")],
        chat_body(1, false),
    )
    .await;
    assert_eq!(forbidden.status, StatusCode::FORBIDDEN);
    let unknown = chat(
        deployment.address,
        "sk-bob",
        &[("x-priority", "gold")],
        chat_body(1, false),
    )
    .await;
    assert_eq!(unknown.status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn client_priority_field() {
    let deployment = deploy(DeploymentOptions::default()).await;
    for priority in [json!(3), json!("interactive"), Value::Null] {
        let body = json!({"messages": [], "max_tokens": 1, "priority": priority});
        let response = chat(deployment.address, "sk-alice", &[], body.to_string()).await;
        assert_eq!(response.status, StatusCode::OK);
        let received = deployment.engine().received().pop().unwrap();
        assert_eq!(received.body["priority"], json!(0));
    }
}

#[tokio::test]
async fn virtual_users_behind_shared_key() {
    let deployment = deploy(DeploymentOptions {
        rank_capacity: 1,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    const CAROL: &[(&str, &str)] = &[("x-end-user", "carol")];
    let first = tokio::spawn(chat(address, "sk-litellm", CAROL, chat_body(30, false)));
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let second = tokio::spawn(chat(address, "sk-litellm", CAROL, chat_body(1, false)));
    tokio::time::sleep(TOKEN_INTERVAL * 5).await;
    let dave = tokio::spawn(chat(
        address,
        "sk-litellm",
        &[("x-end-user", "dave")],
        chat_body(1, false),
    ));
    tokio::time::sleep(TOKEN_INTERVAL * 5).await;
    for request in [first, second, dave] {
        assert_eq!(request.await.unwrap().status, StatusCode::OK);
    }
    let users: Vec<_> = deployment
        .engine()
        .received()
        .iter()
        .map(|request| request.headers["x-end-user"].to_str().unwrap().to_string())
        .collect();
    assert_eq!(users, vec!["carol", "dave", "carol"]);
}

#[tokio::test]
async fn path_whitelist() {
    let deployment = deploy(DeploymentOptions::default()).await;
    let blocked = send(
        deployment.address,
        Method::POST,
        "/flush_cache",
        &[("authorization", "Bearer sk-alice")],
        Bytes::new(),
    )
    .await;
    assert_eq!(blocked.status, StatusCode::NOT_FOUND);
    let traversal = send(
        deployment.address,
        Method::GET,
        "/v1/models/../../flush_cache",
        &[("authorization", "Bearer sk-alice")],
        Bytes::new(),
    )
    .await;
    assert_eq!(traversal.status, StatusCode::NOT_FOUND);
    assert!(deployment.engine().received().is_empty());
}

#[tokio::test]
async fn passthrough_with_repeated_headers() {
    let deployment = deploy(DeploymentOptions::default()).await;
    let response = send(
        deployment.address,
        Method::GET,
        "/v1/models",
        &[("authorization", "Bearer sk-alice")],
        Bytes::new(),
    )
    .await;
    assert_eq!(response.status, StatusCode::OK);
    let cookies: Vec<_> = response.headers.get_all("set-cookie").iter().collect();
    assert_eq!(cookies, vec!["first=1", "second=2"]);
    let body: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(body["data"][0]["id"], "mock-model");
    let received = deployment.engine().received().pop().unwrap();
    assert!(received.headers.get("authorization").is_none());
}

#[tokio::test]
async fn invalid_requests() {
    let deployment = deploy(DeploymentOptions {
        general: json!({"limits": {"max_request_body_bytes": 1000}}),
        ..DeploymentOptions::default()
    })
    .await;
    let not_json = chat(deployment.address, "sk-alice", &[], "not json".to_string()).await;
    assert_eq!(not_json.status, StatusCode::BAD_REQUEST);
    let oversized = chat(
        deployment.address,
        "sk-alice",
        &[],
        "x".repeat(3 * 1024 * 1024),
    )
    .await;
    assert_eq!(oversized.status, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn missing_backend() {
    let deployment = deploy(DeploymentOptions {
        has_backend: false,
        ..DeploymentOptions::default()
    })
    .await;
    let response = chat(deployment.address, "sk-alice", &[], chat_body(1, false)).await;
    assert_eq!(response.status, StatusCode::SERVICE_UNAVAILABLE);
    let body: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(body["error"]["message"], "no backend configured");
}

#[tokio::test]
async fn full_queue_rejection() {
    let deployment = deploy(DeploymentOptions {
        rank_capacity: 1,
        general: json!({"limits": {"max_queued_requests": 1}}),
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let running = tokio::spawn(chat(address, "sk-alice", &[], chat_body(50, false)));
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let queued = tokio::spawn(chat(address, "sk-bob", &[], chat_body(1, false)));
    tokio::time::sleep(Duration::from_millis(100)).await;
    let rejected = chat(address, "sk-bob", &[], chat_body(1, false)).await;
    assert_eq!(rejected.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(running.await.unwrap().status, StatusCode::OK);
    assert_eq!(queued.await.unwrap().status, StatusCode::OK);
}

#[tokio::test]
async fn unreachable_engine() {
    let deployment = deploy(DeploymentOptions {
        rank_capacity: 1,
        ..DeploymentOptions::default()
    })
    .await;
    deployment.engine().close_http().await;
    for _ in 0..2 {
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            chat(deployment.address, "sk-alice", &[], chat_body(1, false)),
        )
        .await
        .unwrap();
        assert_eq!(response.status, StatusCode::BAD_GATEWAY);
    }
}

#[tokio::test]
async fn data_parallel_placement() {
    let deployment = deploy(DeploymentOptions {
        ranks: 2,
        rank_capacity: 1,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let first = tokio::spawn(chat(address, "sk-alice", &[], chat_body(30, false)));
    let second = tokio::spawn(chat(address, "sk-bob", &[], chat_body(30, false)));
    wait_until(|| {
        deployment
            .engines
            .iter()
            .all(|engine| engine.generating_ids().len() == 1)
    })
    .await;
    assert_eq!(first.await.unwrap().status, StatusCode::OK);
    assert_eq!(second.await.unwrap().status, StatusCode::OK);
    for (rank, engine) in deployment.engines.iter().enumerate() {
        let received = engine.received();
        assert_eq!(received.len(), 1);
        assert_eq!(
            received[0].headers["x-data-parallel-rank"],
            rank.to_string().as_str()
        );
    }
}

#[tokio::test]
async fn graceful_shutdown_with_request_in_progress() {
    let mut deployment = deploy(DeploymentOptions::default()).await;
    let address = deployment.address;
    let request = tokio::spawn(chat(address, "sk-alice", &[], chat_body(30, false)));
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    deployment.shutdown_middle_end().await;
    assert_eq!(request.await.unwrap().status, StatusCode::OK);
}

#[tokio::test]
async fn suspension_resent_after_engine_reconnect() {
    let deployment = deploy(DeploymentOptions::default()).await;
    let address = deployment.address;
    let idle = tokio::spawn(async move {
        let response = open_chat_stream(address, "sk-bob", &[("x-priority", "idle")], 40).await;
        stream_text(response).await
    });
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let batch = tokio::spawn(chat(address, "sk-alice", &[], chat_body(60, false)));
    wait_until(|| deployment.engine().suspended_ids().len() == 1).await;
    deployment.engine().disconnect_control_clients();
    assert!(deployment.engine().suspended_ids().is_empty());
    wait_until(|| deployment.engine().has_control_client()).await;
    wait_until(|| deployment.engine().suspended_ids().len() == 1).await;
    assert_eq!(batch.await.unwrap().status, StatusCode::OK);
    assert_eq!(token_sequence(&idle.await.unwrap()).len(), 40);
}

#[tokio::test]
async fn request_body_timeout() {
    let deployment = deploy(DeploymentOptions {
        general: json!({"limits": {"request_body_timeout_seconds": 0.2}}),
        ..DeploymentOptions::default()
    })
    .await;
    let mut connection = tokio::net::TcpStream::connect(deployment.address)
        .await
        .unwrap();
    let head = "POST /v1/chat/completions HTTP/1.1\r\nhost: proxy\r\nauthorization: Bearer sk-alice\r\ncontent-length: 100\r\n\r\n{\"partial";
    tokio::io::AsyncWriteExt::write_all(&mut connection, head.as_bytes())
        .await
        .unwrap();
    let mut response = Vec::new();
    while !response.windows(4).any(|window| window == b"\r\n\r\n") {
        let mut chunk = [0u8; 1024];
        let length = tokio::time::timeout(
            Duration::from_secs(5),
            tokio::io::AsyncReadExt::read(&mut connection, &mut chunk),
        )
        .await
        .unwrap()
        .unwrap();
        assert_ne!(length, 0);
        response.extend_from_slice(&chunk[..length]);
    }
    assert!(response.starts_with(b"HTTP/1.1 408"));
}

#[tokio::test]
async fn rank_capacity_during_client_abort() {
    let deployment = deploy(DeploymentOptions {
        rank_capacity: 1,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let abandoned = tokio::spawn(async move {
        let response = open_chat_stream(address, "sk-alice", &[], 10_000).await;
        let mut body = response.into_body();
        body.frame().await;
        tokio::time::sleep(Duration::from_secs(60)).await;
    });
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let queued = tokio::spawn(chat(address, "sk-bob", &[], chat_body(5, false)));
    tokio::time::sleep(TOKEN_INTERVAL * 5).await;
    abandoned.abort();
    assert_eq!(queued.await.unwrap().status, StatusCode::OK);
    assert_eq!(deployment.engine().peak_generating(), 1);
    wait_until(|| deployment.engine().active_ids().is_empty()).await;
}

#[tokio::test]
async fn start_held_while_control_link_down() {
    let deployment = deploy(DeploymentOptions::default()).await;
    deployment.engine().close_control().await;
    wait_until(|| !deployment.engine().has_control_client()).await;
    let request = tokio::spawn(chat(
        deployment.address,
        "sk-alice",
        &[],
        chat_body(3, false),
    ));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(deployment.engine().received().is_empty());
    deployment.engine().open_control();
    assert_eq!(request.await.unwrap().status, StatusCode::OK);
}
