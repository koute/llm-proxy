mod support;

use std::time::Duration;

use hyper::StatusCode;

use support::process::{FrontendProcess, ProcessDeployment};
use support::{chat, chat_body, wait_until};

const USER_STATUS_WAIT: Duration = Duration::from_secs(25);

#[tokio::test]
async fn rejection_log() {
    let deployment = ProcessDeployment::start(true, 2).await;
    let unknown_key = "sk-unknown-0123456789abcdef";
    let unknown = chat(
        deployment.address,
        unknown_key,
        &[("user-agent", "litellm-test")],
        chat_body(1, false),
    )
    .await;
    assert_eq!(unknown.status, StatusCode::UNAUTHORIZED);
    let known_key = "sk-frank-0123456789abcdef";
    let known = chat(
        deployment.address,
        known_key,
        &[("x-priority", "gold")],
        chat_body(1, false),
    )
    .await;
    assert_eq!(known.status, StatusCode::BAD_REQUEST);
    wait_until(|| deployment.log_count("rejected POST /v1/chat/completions") == 2).await;
    let log = deployment.log();
    let lines: Vec<&str> = log
        .lines()
        .filter(|line| line.contains("rejected POST /v1/chat/completions"))
        .collect();
    assert!(lines[0].contains(
        "401 Unauthorized: unknown API key; unknown API key \"sk-unknown-0123456789abcdef\""
    ));
    assert!(lines[0].contains("user-agent \"litellm-test\""));
    assert!(lines[0].contains("authorization"));
    assert!(lines[1].contains(
        "400 Bad Request: unknown priority class 'gold'; account \"frank\", API key \"sk-...cdef\" (25 characters)"
    ));
    assert!(!log.contains(known_key));
}

#[tokio::test]
async fn request_dump() {
    let deployment = ProcessDeployment::start(true, 2).await;
    let directory = deployment.directory.path();
    let dump_path = directory.join("requests.dump");
    let mut frontend = FrontendProcess::start(
        &directory.join("general.json"),
        &["--dump-requests".as_ref(), dump_path.as_os_str()],
        &directory.join("frontend.log"),
    )
    .await;
    let body = chat_body(1, false);
    let response = chat(
        frontend.address,
        "sk-alice",
        &[("user-agent", "dump-test")],
        body.clone(),
    )
    .await;
    assert_eq!(response.status, StatusCode::OK);
    assert!(frontend.stop().success());
    let dump = std::fs::read_to_string(&dump_path).unwrap();
    let (separator, record) = dump.split_once('\n').unwrap();
    assert!(separator.starts_with("=== "));
    assert!(separator.contains(" request 1 from 127.0.0.1:"));
    assert!(separator.ends_with(&format!(", body {} bytes", body.len())));
    let (head, rest) = record.split_once("\n\n").unwrap();
    let head_lines: Vec<&str> = head.lines().collect();
    assert_eq!(head_lines[0], "POST /v1/chat/completions HTTP/1.1");
    assert!(head_lines.contains(&"authorization: Bearer sk-alice"));
    assert!(head_lines.contains(&"user-agent: dump-test"));
    assert_eq!(rest, format!("{body}\n"));
}

#[tokio::test]
async fn per_user_status_lines() {
    let deployment = ProcessDeployment::start(true, 16).await;
    let address = deployment.address;
    let requests: Vec<_> = ["sk-alice", "sk-alice", "sk-bob"]
        .into_iter()
        .map(|key| tokio::spawn(chat(address, key, &[], chat_body(3000, false))))
        .collect();
    wait_until(|| deployment.engine.generating_ids().len() == 3).await;
    let has_user_lines = || {
        let log = deployment.log();
        let user_lines: Vec<&str> = log
            .lines()
            .rev()
            .filter(|line| line.contains("status:   user "))
            .take(2)
            .collect();
        user_lines.len() == 2
            && user_lines[1].contains("status:   user alice: 2 running (2 generating); usage ")
            && user_lines[1]
                .contains("; engine tokens: 14 prompt, 0 cache hit, 14 prefill in 2 prefills, ")
            && user_lines[0].contains("status:   user bob: 1 running (1 generating); usage ")
            && user_lines[0]
                .contains("; engine tokens: 7 prompt, 0 cache hit, 7 prefill in 1 prefill, ")
    };
    tokio::time::timeout(USER_STATUS_WAIT, async {
        while !has_user_lines() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    for request in requests {
        request.abort();
    }
}
