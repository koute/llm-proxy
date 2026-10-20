#![allow(dead_code)]

pub mod mock_engine;
pub mod process;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::client::conn::http1 as client_http1;
use hyper::header::HeaderMap;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use llm_proxy::config::{Config, ConfigPaths, ConfigTexts};
use llm_proxy::middle_end::{Controls, Outcome};
use llm_proxy::{frontend, middle_end};
use mock_engine::MockEngine;

pub const SCHEDULING_LOG: &str = "scheduling.jsonl";

pub struct DeploymentOptions {
    pub engine: &'static str,
    pub ranks: usize,
    pub rank_capacity: usize,
    pub priority_scheduling: bool,
    pub has_backend: bool,
    pub accounts: Value,
    pub general: Value,
}

pub struct Deployment {
    pub address: SocketAddr,
    pub engines: Vec<MockEngine>,
    pub directory: tempfile::TempDir,
    config_texts: ConfigTexts,
    next_config: Arc<Mutex<Option<Config>>>,
    middle_end_shutdown: watch::Sender<bool>,
    restart_sender: mpsc::UnboundedSender<()>,
    restart_count: watch::Receiver<usize>,
    middle_end: JoinHandle<()>,
}

pub struct ClientResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl Default for DeploymentOptions {
    fn default() -> DeploymentOptions {
        DeploymentOptions {
            engine: "vllm",
            ranks: 1,
            rank_capacity: 16,
            priority_scheduling: true,
            has_backend: true,
            accounts: json!({
                "classes": [
                    {"name": "interactive", "priority": 10},
                    {"name": "batch", "priority": 0},
                    {"name": "idle", "priority": -10, "idle_only": true}
                ],
                "accounts": [
                    {"name": "alice", "api_key": "sk-alice", "default_class": "batch", "allowed_classes": ["interactive", "idle"]},
                    {"name": "bob", "api_key": "sk-bob", "default_class": "batch", "allowed_classes": ["idle"]},
                    {"name": "litellm", "api_key": "sk-litellm", "default_class": "batch", "virtual_user_http_header": "x-end-user"},
                    {"name": "frank", "api_key": "sk-frank-0123456789abcdef", "default_class": "batch"}
                ]
            }),
            general: json!({}),
        }
    }
}

fn merged(base: Value, overrides: &Value) -> Value {
    match (base, overrides) {
        (Value::Object(mut base), Value::Object(overrides)) => {
            for (key, value) in overrides {
                let current = base.remove(key).unwrap_or(Value::Null);
                base.insert(key.clone(), merged(current, value));
            }
            Value::Object(base)
        }
        (_, overrides) => overrides.clone(),
    }
}

pub async fn deploy(options: DeploymentOptions) -> Deployment {
    let directory = tempfile::tempdir().unwrap();
    let mut engines = Vec::new();
    for rank in 0..options.ranks {
        let socket = directory.path().join(format!("engine-{rank}.sock"));
        engines.push(MockEngine::start(socket).await);
    }
    let middle_end_socket = directory.path().join("middle-end.sock");
    let general = merged(
        json!({
            "frontend": {"listen": "127.0.0.1:0"},
            "middle_end_socket": middle_end_socket,
            "scheduling_log": directory.path().join(SCHEDULING_LOG),
            "process": {"shutdown_drain_timeout_seconds": 5}
        }),
        &options.general,
    );
    let backend = json!({
        "engine": options.engine,
        "model": "mock-model",
        "engine_max_parallel_requests_per_rank": options.rank_capacity,
        "priority_scheduling": options.priority_scheduling,
        "ranks": engines.iter().enumerate().map(|(rank, engine)| json!({
            "upstream": format!("http://{}", engine.address),
            "control_socket": engine.control_socket,
            "data_parallel_rank": (options.ranks > 1).then_some(rank),
        })).collect::<Vec<_>>(),
    });
    let texts = ConfigTexts {
        general: general.to_string(),
        accounts: options.accounts.to_string(),
        backend: options.has_backend.then(|| backend.to_string()),
    };
    let paths = ConfigPaths {
        general: "general.json".into(),
        accounts: "accounts.json".into(),
        backend: "backend.json".into(),
    };
    let mut config = Config::parse(&texts, &paths).unwrap();
    let next_config: Arc<Mutex<Option<Config>>> = Arc::default();
    let restart_config = next_config.clone();
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    let (restart_sender, mut restart_requests) = mpsc::unbounded_channel();
    let (restart_count_sender, restart_count) = watch::channel(0);
    let middle_end = tokio::spawn(async move {
        let mut handover = None;
        loop {
            let mut shutdown = shutdown_receiver.clone();
            let controls = Controls {
                shutdown: async move {
                    let _ = shutdown.wait_for(|is_shutdown| *is_shutdown).await;
                },
                restart_requests: &mut restart_requests,
                restart_plan: None,
            };
            match middle_end::serve(config.clone(), handover.take(), controls).await {
                Ok(Outcome::Shutdown) => return,
                Ok(Outcome::Restart(next_handover)) => {
                    if let Some(next) = restart_config.lock().unwrap().take() {
                        config = next;
                    }
                    restart_count_sender.send_modify(|count| *count += 1);
                    handover = Some(next_handover);
                }
                Err(message) => panic!("{message}"),
            }
        }
    });
    let frontend_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = frontend_listener.local_addr().unwrap();
    tokio::spawn(frontend::serve(
        frontend_listener,
        frontend::Transport::Plain,
        middle_end_socket,
        None,
        Duration::from_secs(5),
        std::future::pending(),
    ));
    let deployment = Deployment {
        address,
        engines,
        directory,
        config_texts: texts,
        next_config,
        middle_end_shutdown: shutdown_sender,
        restart_sender,
        restart_count,
        middle_end,
    };
    if options.has_backend {
        wait_until(|| {
            deployment
                .engines
                .iter()
                .all(MockEngine::has_control_client)
        })
        .await;
    }
    deployment
}

impl Deployment {
    pub fn engine(&self) -> &MockEngine {
        &self.engines[0]
    }

    pub async fn shutdown_middle_end(&mut self) {
        let _ = self.middle_end_shutdown.send(true);
        (&mut self.middle_end).await.unwrap();
    }

    pub async fn restart_middle_end_with_backend(&mut self, backend: Value) {
        let texts = ConfigTexts {
            backend: Some(backend.to_string()),
            ..self.config_texts.clone()
        };
        self.restart_middle_end_with_texts(texts).await;
    }

    pub async fn restart_middle_end_with_accounts(&mut self, accounts: Value) {
        let texts = ConfigTexts {
            accounts: accounts.to_string(),
            ..self.config_texts.clone()
        };
        self.restart_middle_end_with_texts(texts).await;
    }

    async fn restart_middle_end_with_texts(&mut self, texts: ConfigTexts) {
        let paths = ConfigPaths {
            general: "general.json".into(),
            accounts: "accounts.json".into(),
            backend: "backend.json".into(),
        };
        *self.next_config.lock().unwrap() = Some(Config::parse(&texts, &paths).unwrap());
        self.restart_middle_end().await;
    }

    pub async fn restart_middle_end(&mut self) {
        let previous = *self.restart_count.borrow();
        self.restart_sender.send(()).unwrap();
        self.restart_count
            .wait_for(|count| *count > previous)
            .await
            .unwrap();
    }

    pub fn socket_path(&self, name: &str) -> std::path::PathBuf {
        self.directory.path().join(name)
    }

    pub fn scheduling_log(&self) -> Vec<Value> {
        scheduling_log(&self.directory.path().join(SCHEDULING_LOG))
    }
}

pub fn scheduling_log(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

pub fn event_names(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|event| event["event"].as_str().unwrap())
        .collect()
}

pub async fn wait_until(condition: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("condition not reached within 10 s");
}

pub fn chat_body(max_tokens: u64, is_streaming: bool) -> String {
    json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": max_tokens,
        "stream": is_streaming,
    })
    .to_string()
}

pub async fn open(
    address: SocketAddr,
    method: Method,
    path: &str,
    headers: &[(&str, &str)],
    body: impl Into<Bytes>,
) -> Response<Incoming> {
    let stream = TcpStream::connect(address).await.unwrap();
    let (mut sender, connection) = client_http1::handshake(TokioIo::new(stream)).await.unwrap();
    tokio::spawn(connection);
    let mut builder = Request::builder()
        .method(method)
        .uri(path)
        .header("host", address.to_string())
        .header("content-type", "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let request = builder.body(Full::new(body.into())).unwrap();
    sender.send_request(request).await.unwrap()
}

pub async fn send(
    address: SocketAddr,
    method: Method,
    path: &str,
    headers: &[(&str, &str)],
    body: impl Into<Bytes>,
) -> ClientResponse {
    let response = open(address, method, path, headers, body).await;
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    ClientResponse {
        status,
        headers,
        body,
    }
}

pub async fn chat(
    address: SocketAddr,
    api_key: &str,
    extra_headers: &[(&str, &str)],
    body: String,
) -> ClientResponse {
    let authorization = format!("Bearer {api_key}");
    let mut headers = vec![("authorization", authorization.as_str())];
    headers.extend_from_slice(extra_headers);
    send(
        address,
        Method::POST,
        "/v1/chat/completions",
        &headers,
        body,
    )
    .await
}

pub async fn open_chat_stream(
    address: SocketAddr,
    api_key: &str,
    extra_headers: &[(&str, &str)],
    max_tokens: u64,
) -> Response<Incoming> {
    let authorization = format!("Bearer {api_key}");
    let mut headers = vec![("authorization", authorization.as_str())];
    headers.extend_from_slice(extra_headers);
    open(
        address,
        Method::POST,
        "/v1/chat/completions",
        &headers,
        chat_body(max_tokens, true),
    )
    .await
}

pub async fn stream_text(response: Response<Incoming>) -> String {
    let body = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(body.to_vec()).unwrap()
}

pub fn token_sequence(text: &str) -> Vec<u64> {
    text.split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter_map(|value| value["token"].as_u64())
        .collect()
}
