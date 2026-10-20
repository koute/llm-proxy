use std::ffi::OsStr;
use std::net::SocketAddr;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::Duration;

use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use serde_json::json;

use super::mock_engine::MockEngine;
use super::{scheduling_log, wait_until, DeploymentOptions, SCHEDULING_LOG};
use llm_proxy::frontend;

pub struct ProcessDeployment {
    pub rank_capacity: usize,
    pub address: SocketAddr,
    pub engine: MockEngine,
    pub directory: tempfile::TempDir,
    pub child: Child,
}

impl ProcessDeployment {
    pub async fn start(has_backend: bool, rank_capacity: usize) -> ProcessDeployment {
        let executable = PathBuf::from(env!("CARGO_BIN_EXE_llm-proxy"));
        ProcessDeployment::start_as(has_backend, rank_capacity, &executable).await
    }

    pub async fn start_as(
        has_backend: bool,
        rank_capacity: usize,
        invocation_path: &Path,
    ) -> ProcessDeployment {
        let directory = tempfile::tempdir().unwrap();
        let engine = MockEngine::start(directory.path().join("engine.sock")).await;
        let middle_end_socket = directory.path().join("middle-end.sock");
        let general = json!({
            "frontend": {"listen": "127.0.0.1:0"},
            "middle_end_socket": middle_end_socket,
            "scheduling_log": directory.path().join(SCHEDULING_LOG),
            "process": {"restart_flush_timeout_seconds": 2, "shutdown_drain_timeout_seconds": 2}
        });
        std::fs::write(directory.path().join("general.json"), general.to_string()).unwrap();
        std::fs::write(
            directory.path().join("accounts.json"),
            DeploymentOptions::default().accounts.to_string(),
        )
        .unwrap();
        if has_backend {
            write_backend(
                &directory.path().join("backend.json"),
                &engine,
                rank_capacity,
            );
        }
        let child = Command::new(env!("CARGO_BIN_EXE_llm-proxy"))
            .arg0(invocation_path)
            .arg("serve")
            .arg("--general")
            .arg(directory.path().join("general.json"))
            .arg("--accounts")
            .arg(directory.path().join("accounts.json"))
            .arg("--backend")
            .arg(directory.path().join("backend.json"))
            .env("RUST_LOG", "info")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(directory.path().join("serve.log")).unwrap())
            .spawn()
            .unwrap();
        wait_until(|| middle_end_socket.exists()).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(frontend::serve(
            listener,
            frontend::Transport::Plain,
            middle_end_socket,
            None,
            Duration::from_secs(1),
            std::future::pending(),
        ));
        if has_backend {
            wait_until(|| engine.has_control_client()).await;
        }
        ProcessDeployment {
            rank_capacity,
            address,
            engine,
            directory,
            child,
        }
    }

    pub fn backend_path(&self) -> PathBuf {
        self.directory.path().join("backend.json")
    }

    pub fn write_backend(&self) {
        write_backend(&self.backend_path(), &self.engine, self.rank_capacity);
    }

    pub fn accounts_path(&self) -> PathBuf {
        self.directory.path().join("accounts.json")
    }

    pub fn log(&self) -> String {
        std::fs::read_to_string(self.directory.path().join("serve.log")).unwrap_or_default()
    }

    pub fn scheduling_log(&self) -> Vec<serde_json::Value> {
        scheduling_log(&self.directory.path().join(SCHEDULING_LOG))
    }

    pub fn log_count(&self, text: &str) -> usize {
        self.log().matches(text).count()
    }

    pub fn signal(&self, signal: Signal) {
        kill(Pid::from_raw(self.child.id() as i32), signal).unwrap();
    }

    pub async fn restart(&self) {
        let completed = self.log_count("restart complete");
        self.signal(Signal::SIGUSR1);
        wait_until(|| self.log_count("restart complete") > completed).await;
    }

    pub fn is_running(&mut self) -> bool {
        self.child.try_wait().unwrap().is_none()
    }
}

impl Drop for ProcessDeployment {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn write_backend(path: &Path, engine: &MockEngine, rank_capacity: usize) {
    let backend = json!({
        "engine": "vllm",
        "model": "mock-model",
        "engine_max_parallel_requests_per_rank": rank_capacity,
        "ranks": [{
            "upstream": format!("http://{}", engine.address),
            "control_socket": engine.control_socket,
        }],
    });
    std::fs::write(path, backend.to_string()).unwrap();
}

pub struct FrontendProcess {
    pub address: SocketAddr,
    child: Child,
}

impl FrontendProcess {
    pub async fn start(general: &Path, arguments: &[&OsStr], log: &Path) -> FrontendProcess {
        let child = Command::new(env!("CARGO_BIN_EXE_llm-proxy"))
            .arg("frontend")
            .arg("--general")
            .arg(general)
            .args(arguments)
            .env("RUST_LOG", "info")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(log).unwrap())
            .spawn()
            .unwrap();
        wait_until(|| listening_address(log).is_some()).await;
        FrontendProcess {
            address: listening_address(log).unwrap(),
            child,
        }
    }

    pub fn stop(&mut self) -> ExitStatus {
        kill(Pid::from_raw(self.child.id() as i32), Signal::SIGTERM).unwrap();
        self.child.wait().unwrap()
    }
}

impl Drop for FrontendProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn listening_address(log: &Path) -> Option<SocketAddr> {
    let log = std::fs::read_to_string(log).ok()?;
    let (_, rest) = log.split_once("frontend listening on http://")?;
    let (address, _) = rest.split_once(',')?;
    address.parse().ok()
}
