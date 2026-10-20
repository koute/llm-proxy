mod support;

use std::os::fd::{AsRawFd, BorrowedFd};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::Duration;

use http_body_util::BodyExt;
use hyper::StatusCode;
use nix::fcntl::{fcntl, FcntlArg, FdFlag, OFlag};
use nix::sys::signal::Signal;
use nix::unistd::pipe2;
use tokio::io::AsyncReadExt;
use tokio::net::unix::pipe;

use support::mock_engine::TOKEN_INTERVAL;
use support::process::ProcessDeployment;
use support::{
    chat, chat_body, event_names, open_chat_stream, stream_text, token_sequence, wait_until,
    DeploymentOptions,
};

#[tokio::test]
async fn backend_activation_by_restart() {
    let deployment = ProcessDeployment::start(false, 2).await;
    let before = chat(deployment.address, "sk-alice", &[], chat_body(1, false)).await;
    assert_eq!(before.status, StatusCode::SERVICE_UNAVAILABLE);
    deployment.write_backend();
    deployment.restart().await;
    let after = chat(deployment.address, "sk-alice", &[], chat_body(1, false)).await;
    assert_eq!(after.status, StatusCode::OK);
}

#[tokio::test]
async fn scheduling_log_across_restart() {
    let deployment = ProcessDeployment::start(true, 2).await;
    let address = deployment.address;
    let tokens = 60;
    let stream = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-alice", &[], tokens).await).await
    });
    wait_until(|| deployment.engine.generating_ids().len() == 1).await;
    deployment.restart().await;
    assert_eq!(
        token_sequence(&stream.await.unwrap()),
        (0..tokens).collect::<Vec<_>>()
    );
    wait_until(|| event_names(&deployment.scheduling_log()).contains(&"finished")).await;
    let events = deployment.scheduling_log();
    let names = event_names(&events);
    let process_events: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| ["start", "stop", "handed_over", "adopted"].contains(name))
        .collect();
    assert_eq!(
        process_events,
        ["start", "handed_over", "stop", "start", "adopted"]
    );
    let request = |name: &str| {
        let event = events.iter().find(|event| event["event"] == name).unwrap();
        event["request"].clone()
    };
    assert_eq!(request("handed_over"), request("admitted"));
    assert_eq!(request("adopted"), request("admitted"));
    assert_eq!(request("finished"), request("admitted"));
    let adopted = events
        .iter()
        .find(|event| event["event"] == "adopted")
        .unwrap();
    assert_eq!(
        [&adopted["user"], &adopted["state"]],
        [&serde_json::json!("alice"), &serde_json::json!("running")]
    );
    let starts: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["event"] == "start")
        .collect();
    assert_eq!(
        [&starts[0]["restart"], &starts[1]["restart"]],
        [&serde_json::json!(false), &serde_json::json!(true)]
    );
}

#[tokio::test]
async fn invalid_configuration_aborts_restart() {
    let mut deployment = ProcessDeployment::start(true, 2).await;
    let accounts = std::fs::read_to_string(deployment.accounts_path()).unwrap();
    std::fs::write(deployment.accounts_path(), "{ invalid").unwrap();
    deployment.signal(Signal::SIGUSR1);
    wait_until(|| deployment.log_count("restart aborted") == 1).await;
    let during = chat(deployment.address, "sk-alice", &[], chat_body(1, false)).await;
    assert_eq!(during.status, StatusCode::OK);
    std::fs::write(deployment.accounts_path(), accounts).unwrap();
    deployment.restart().await;
    let after = chat(deployment.address, "sk-alice", &[], chat_body(1, false)).await;
    assert_eq!(after.status, StatusCode::OK);
    assert!(deployment.is_running());
}

#[tokio::test]
async fn sigterm_shutdown() {
    let mut deployment = ProcessDeployment::start(true, 2).await;
    deployment.signal(Signal::SIGTERM);
    let status = tokio::task::spawn_blocking(move || {
        let status = deployment.child.wait().unwrap();
        (status, deployment)
    })
    .await
    .unwrap();
    assert!(status.0.success());
}

#[tokio::test]
async fn restart_stress() {
    let mut deployment = ProcessDeployment::start(true, 4).await;
    let process_id = deployment.child.id();
    let address = deployment.address;
    let tokens = 60;
    let keys = ["sk-alice", "sk-bob", "sk-litellm"];
    let classes = ["batch", "idle", "batch", "interactive"];
    let streams: Vec<_> = (0..12)
        .map(|index| {
            let key = keys[index % keys.len()];
            let class = match (key, classes[index % classes.len()]) {
                ("sk-bob", "interactive") | ("sk-litellm", "interactive" | "idle") => "batch",
                (_, class) => class,
            };
            tokio::spawn(async move {
                stream_text(open_chat_stream(address, key, &[("x-priority", class)], tokens).await)
                    .await
            })
        })
        .collect();
    wait_until(|| deployment.engine.generating_ids().len() == 4).await;
    for _ in 0..10 {
        tokio::time::sleep(TOKEN_INTERVAL * 6).await;
        deployment.restart().await;
    }
    for stream in streams {
        let text = stream.await.unwrap();
        assert_eq!(token_sequence(&text), (0..tokens).collect::<Vec<_>>());
        assert!(text.ends_with("data: [DONE]\n\n"));
    }
    assert_eq!(deployment.engine.peak_generating(), 4);
    assert!(deployment.engine.suspended_ids().is_empty());
    assert!(deployment.is_running());
    assert_eq!(deployment.child.id(), process_id);
    assert_eq!(deployment.log_count("restart complete"), 10);
}

#[tokio::test]
async fn restart_with_changed_backend() {
    let mut deployment = ProcessDeployment::start(true, 1).await;
    let address = deployment.address;
    let first = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-alice", &[], 100).await).await
    });
    wait_until(|| deployment.engine.generating_ids().len() == 1).await;
    deployment.rank_capacity = 2;
    deployment.write_backend();
    deployment.restart().await;
    let second = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-bob", &[], 5).await).await
    });
    wait_until(|| deployment.engine.generating_ids().len() == 2).await;
    assert_eq!(
        token_sequence(&second.await.unwrap()),
        (0..5).collect::<Vec<_>>()
    );
    assert_eq!(
        token_sequence(&first.await.unwrap()),
        (0..100).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn restart_with_changed_classes() {
    let deployment = ProcessDeployment::start(true, 1).await;
    let address = deployment.address;
    let idle = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-bob", &[("x-priority", "idle")], 40).await).await
    });
    wait_until(|| deployment.engine.generating_ids().len() == 1).await;
    let batch = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-alice", &[], 40).await).await
    });
    wait_until(|| deployment.engine.suspended_ids().len() == 1).await;
    let accounts = serde_json::json!({
        "classes": [{"name": "batch", "priority": 5}],
        "accounts": [
            {"name": "alice", "api_key": "sk-alice", "default_class": "batch"},
            {"name": "bob", "api_key": "sk-bob", "default_class": "batch"}
        ]
    });
    std::fs::write(deployment.accounts_path(), accounts.to_string()).unwrap();
    deployment.restart().await;
    for stream in [batch, idle] {
        assert_eq!(
            token_sequence(&stream.await.unwrap()),
            (0..40).collect::<Vec<_>>()
        );
    }
    assert!(deployment.engine.suspended_ids().is_empty());
}

#[tokio::test]
async fn execve_failure_fallback() {
    let launcher_directory = tempfile::tempdir().unwrap();
    let launcher = launcher_directory.path().join("llm-proxy-once");
    let script = format!(
        "#!/bin/sh\n'{}' \"$@\"\nstatus=$?\nrm -f \"$0\"\nexit $status\n",
        env!("CARGO_BIN_EXE_llm-proxy")
    );
    std::fs::write(&launcher, script).unwrap();
    std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut deployment = ProcessDeployment::start_as(true, 2, &launcher).await;
    let address = deployment.address;
    let stream = tokio::spawn(async move {
        stream_text(open_chat_stream(address, "sk-alice", &[], 60).await).await
    });
    wait_until(|| deployment.engine.generating_ids().len() == 1).await;
    deployment.restart().await;
    assert!(!launcher.exists());
    assert_eq!(deployment.log_count("restart failed"), 1);
    assert_eq!(
        token_sequence(&stream.await.unwrap()),
        (0..60).collect::<Vec<_>>()
    );
    let after = chat(address, "sk-bob", &[], chat_body(1, false)).await;
    assert_eq!(after.status, StatusCode::OK);
    assert!(deployment.is_running());
}

#[tokio::test]
async fn restart_request_during_preflight() {
    let launcher_directory = tempfile::tempdir().unwrap();
    let launcher = launcher_directory.path().join("llm-proxy-slow-preflight");
    let marker = launcher_directory.path().join("preflight-started");
    let script = format!(
        "#!/bin/sh\nif [ \"$1\" = preflight ]; then touch '{}'; sleep 1; fi\nexec '{}' \"$@\"\n",
        marker.display(),
        env!("CARGO_BIN_EXE_llm-proxy")
    );
    std::fs::write(&launcher, script).unwrap();
    std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755)).unwrap();
    let deployment = ProcessDeployment::start_as(false, 2, &launcher).await;
    deployment.signal(Signal::SIGUSR1);
    wait_until(|| marker.exists()).await;
    deployment.write_backend();
    deployment.signal(Signal::SIGUSR1);
    wait_until(|| deployment.log_count("restart complete") == 1).await;
    let after = chat(deployment.address, "sk-alice", &[], chat_body(1, false)).await;
    assert_eq!(after.status, StatusCode::OK);
}

#[tokio::test]
async fn engine_unaffected_by_stopped_proxy() {
    let mut deployment = ProcessDeployment::start(true, 2).await;
    let address = deployment.address;
    let idle = tokio::spawn(async move {
        let response = open_chat_stream(address, "sk-bob", &[("x-priority", "idle")], 300).await;
        let _ = response.into_body().collect().await;
    });
    wait_until(|| deployment.engine.generating_ids().len() == 1).await;
    let batch = tokio::spawn(chat(address, "sk-alice", &[], chat_body(300, false)));
    wait_until(|| deployment.engine.suspended_ids().len() == 1).await;
    deployment.signal(Signal::SIGKILL);
    let _ = deployment.child.wait();
    wait_until(|| !deployment.engine.has_control_client()).await;
    assert!(deployment.engine.suspended_ids().is_empty());
    idle.abort();
    batch.abort();
}

#[tokio::test]
async fn sigterm_during_restart() {
    let mut deployment = ProcessDeployment::start(true, 2).await;
    deployment.signal(Signal::SIGUSR1);
    deployment.signal(Signal::SIGTERM);
    let exited = tokio::task::spawn_blocking(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while std::time::Instant::now() < deadline {
            if deployment.child.try_wait().unwrap().is_some() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        false
    })
    .await
    .unwrap();
    assert!(exited);
}

#[tokio::test]
async fn frontend_unaffected_by_sigusr1() {
    let directory = tempfile::tempdir().unwrap();
    let general = serde_json::json!({
        "frontend": {"listen": "127.0.0.1:0"},
        "middle_end_socket": directory.path().join("middle-end.sock"),
    });
    std::fs::write(directory.path().join("general.json"), general.to_string()).unwrap();
    let mut frontend = std::process::Command::new(env!("CARGO_BIN_EXE_llm-proxy"))
        .arg("frontend")
        .arg("--general")
        .arg(directory.path().join("general.json"))
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let process_id = nix::unistd::Pid::from_raw(frontend.id() as i32);
    nix::sys::signal::kill(process_id, Signal::SIGUSR1).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let is_running = frontend.try_wait().unwrap().is_none();
    let _ = frontend.kill();
    let _ = frontend.wait();
    assert!(is_running);
}

#[tokio::test]
async fn invalid_handover_descriptor_cleanup() {
    let directory = tempfile::tempdir().unwrap();
    let general = serde_json::json!({
        "frontend": {"listen": "127.0.0.1:0"},
        "middle_end_socket": directory.path().join("middle-end.sock"),
    });
    std::fs::write(directory.path().join("general.json"), general.to_string()).unwrap();
    std::fs::write(
        directory.path().join("accounts.json"),
        DeploymentOptions::default().accounts.to_string(),
    )
    .unwrap();
    let (inherited_reader, inherited_writer) = pipe2(OFlag::O_CLOEXEC).unwrap();
    let inherited_fd = inherited_writer.as_raw_fd();
    let mut command = Command::new(env!("CARGO_BIN_EXE_llm-proxy"));
    command
        .arg("serve")
        .arg("--general")
        .arg(directory.path().join("general.json"))
        .arg("--accounts")
        .arg(directory.path().join("accounts.json"))
        .arg("--backend")
        .arg(directory.path().join("backend.json"))
        .env("LLM_PROXY_HANDOVER_FD", "invalid")
        .stderr(Stdio::null());
    unsafe {
        command.pre_exec(move || {
            fcntl(
                BorrowedFd::borrow_raw(inherited_fd),
                FcntlArg::F_SETFD(FdFlag::empty()),
            )?;
            Ok(())
        });
    }
    let mut serve = command.spawn().unwrap();
    drop(inherited_writer);
    let mut reader = pipe::Receiver::from_owned_fd(inherited_reader).unwrap();
    let read = tokio::time::timeout(Duration::from_secs(10), reader.read(&mut [0u8; 1])).await;
    let is_running = serve.try_wait().unwrap().is_none();
    let _ = serve.kill();
    let _ = serve.wait();
    assert_eq!(read.unwrap().unwrap(), 0);
    assert!(is_running);
}
