mod support;

use hyper::StatusCode;

use support::process::{FrontendProcess, ProcessDeployment};
use support::{chat, chat_body, wait_until};

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
