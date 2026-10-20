mod support;

use std::time::Duration;

use hyper::StatusCode;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use support::mock_engine::TOKEN_INTERVAL;
use support::{
    chat, chat_body, deploy, open_chat_stream, stream_text, token_sequence, wait_until,
    DeploymentOptions,
};

#[tokio::test]
async fn stream_continuity_across_restarts() {
    let mut deployment = deploy(DeploymentOptions {
        rank_capacity: 2,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let tokens = 80;
    let streams: Vec<_> = [
        ("sk-bob", "idle"),
        ("sk-bob", "idle"),
        ("sk-alice", "batch"),
        ("sk-litellm", "batch"),
    ]
    .into_iter()
    .map(|(key, class)| {
        tokio::spawn(async move {
            let response = open_chat_stream(address, key, &[("x-priority", class)], tokens).await;
            assert_eq!(response.status(), StatusCode::OK);
            stream_text(response).await
        })
    })
    .collect();
    wait_until(|| deployment.engine().generating_ids().len() == 2).await;
    for _ in 0..8 {
        tokio::time::sleep(TOKEN_INTERVAL * 7).await;
        deployment.restart_middle_end().await;
    }
    for stream in streams {
        let text = stream.await.unwrap();
        assert_eq!(token_sequence(&text), (0..tokens).collect::<Vec<_>>());
        assert!(text.ends_with("data: [DONE]\n\n"));
    }
    assert_eq!(deployment.engine().peak_generating(), 2);
    assert!(deployment.engine().suspended_ids().is_empty());
}

#[tokio::test]
async fn suspension_state_across_restart() {
    let mut deployment = deploy(DeploymentOptions {
        rank_capacity: 1,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let idle = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-bob", &[("x-priority", "idle")], 30).await).await
    });
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let batch = tokio::spawn(chat(address, "sk-alice", &[], chat_body(40, false)));
    wait_until(|| deployment.engine().suspended_ids().len() == 1).await;
    deployment.restart_middle_end().await;
    tokio::time::sleep(TOKEN_INTERVAL * 5).await;
    assert_eq!(deployment.engine().suspended_ids().len(), 1);
    assert_eq!(batch.await.unwrap().status, StatusCode::OK);
    let text = idle.await.unwrap();
    assert_eq!(token_sequence(&text), (0..30).collect::<Vec<_>>());
}

#[tokio::test]
async fn queued_request_across_restart() {
    let mut deployment = deploy(DeploymentOptions {
        rank_capacity: 1,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let running = tokio::spawn(chat(address, "sk-alice", &[], chat_body(30, false)));
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let queued = tokio::spawn(chat(address, "sk-bob", &[], chat_body(5, false)));
    tokio::time::sleep(TOKEN_INTERVAL * 3).await;
    deployment.restart_middle_end().await;
    assert_eq!(deployment.engine().received().len(), 1);
    assert_eq!(running.await.unwrap().status, StatusCode::OK);
    assert_eq!(queued.await.unwrap().status, StatusCode::OK);
    assert_eq!(deployment.engine().peak_generating(), 1);
}

#[tokio::test]
async fn partial_upload_across_restart() {
    let mut deployment = deploy(DeploymentOptions::default()).await;
    let body = chat_body(3, false);
    let (first_half, second_half) = body.split_at(body.len() / 2);
    let mut connection = TcpStream::connect(deployment.address).await.unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: proxy\r\nauthorization: Bearer sk-alice\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    connection.write_all(head.as_bytes()).await.unwrap();
    connection.write_all(first_half.as_bytes()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    deployment.restart_middle_end().await;
    connection.write_all(second_half.as_bytes()).await.unwrap();
    let mut response = String::new();
    connection.read_to_string(&mut response).await.unwrap();
    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(response.contains("\"tokens\":3"));
}

#[tokio::test]
async fn rank_remapping_across_restart() {
    let mut deployment = deploy(DeploymentOptions {
        ranks: 2,
        rank_capacity: 1,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let first = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-alice", &[], 60).await).await
    });
    wait_until(|| deployment.engines[0].generating_ids().len() == 1).await;
    let second = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-bob", &[], 60).await).await
    });
    wait_until(|| deployment.engines[1].generating_ids().len() == 1).await;
    let remaining_engine = &deployment.engines[1];
    let backend = json!({
        "engine": "vllm",
        "model": "mock-model",
        "engine_max_parallel_requests_per_rank": 1,
        "ranks": [{
            "upstream": format!("http://{}", remaining_engine.address),
            "control_socket": remaining_engine.control_socket,
        }],
    });
    deployment.restart_middle_end_with_backend(backend).await;
    let third = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-litellm", &[], 5).await).await
    });
    for stream in [first, second, third] {
        assert!(stream.await.unwrap().ends_with("data: [DONE]\n\n"));
    }
    assert_eq!(deployment.engines[1].peak_generating(), 1);
}

#[tokio::test]
async fn class_change_across_restart() {
    let mut deployment = deploy(DeploymentOptions {
        rank_capacity: 2,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let tokens = 40;
    let stream = |key: &'static str, class: &'static str| {
        tokio::spawn(async move {
            stream_text(open_chat_stream(address, key, &[("x-priority", class)], tokens).await)
                .await
        })
    };
    let idle = stream("sk-bob", "idle");
    wait_until(|| deployment.engine().generating_ids().len() == 1).await;
    let batch = [stream("sk-alice", "batch"), stream("sk-litellm", "batch")];
    wait_until(|| deployment.engine().suspended_ids().len() == 1).await;
    let queued = stream("sk-frank-0123456789abcdef", "batch");
    tokio::time::sleep(TOKEN_INTERVAL * 3).await;
    deployment
        .restart_middle_end_with_accounts(json!({
            "classes": [{"name": "batch", "priority": 5}],
            "accounts": [
                {"name": "alice", "api_key": "sk-alice", "default_class": "batch"},
                {"name": "bob", "api_key": "sk-bob", "default_class": "batch"},
                {"name": "litellm", "api_key": "sk-litellm", "default_class": "batch"},
                {"name": "frank", "api_key": "sk-frank-0123456789abcdef", "default_class": "batch"}
            ]
        }))
        .await;
    for stream in batch.into_iter().chain([idle, queued]) {
        let text = stream.await.unwrap();
        assert_eq!(token_sequence(&text), (0..tokens).collect::<Vec<_>>());
    }
    assert_eq!(deployment.engine().peak_generating(), 2);
    assert!(deployment.engine().suspended_ids().is_empty());
}

#[tokio::test]
async fn handed_over_user_identity() {
    let mut deployment = deploy(DeploymentOptions {
        rank_capacity: 2,
        ..DeploymentOptions::default()
    })
    .await;
    let address = deployment.address;
    let streams: Vec<_> = ["sk-alice", "sk-bob"]
        .into_iter()
        .map(|key| {
            tokio::spawn(
                async move { stream_text(open_chat_stream(address, key, &[], 40).await).await },
            )
        })
        .collect();
    wait_until(|| deployment.engine().generating_ids().len() == 2).await;
    deployment.restart_middle_end().await;
    let second_alice = tokio::spawn(chat(address, "sk-alice", &[], chat_body(5, false)));
    tokio::time::sleep(TOKEN_INTERVAL * 5).await;
    assert_eq!(deployment.engine().received().len(), 2);
    assert!(deployment.engine().suspended_ids().is_empty());
    assert_eq!(second_alice.await.unwrap().status, StatusCode::OK);
    for stream in streams {
        assert_eq!(
            token_sequence(&stream.await.unwrap()),
            (0..40).collect::<Vec<_>>()
        );
    }
}
