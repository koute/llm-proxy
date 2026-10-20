use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::channel::Channel;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::header::HeaderMap;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::mpsc;

use llm_proxy::engine_protocol::{
    Command, CommandLine, EngineInfo, EngineKind, EngineLine, EngineStats, Event, ProxyId,
    RequestPhase, RequestStats, StopReason,
};

pub const TOKEN_INTERVAL: Duration = Duration::from_millis(10);
const TOKENS_PER_PROMPT_WORD: u64 = 3;
const PROMPT_TEMPLATE_TOKENS: u64 = 4;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone, Debug)]
pub struct ReceivedRequest {
    pub path: String,
    pub headers: HeaderMap,
    pub body: Value,
}

struct EngineState {
    max_running: usize,
    kv_cache_tokens: usize,
    prefill_interval: Duration,
    prefill_queue: VecDeque<ProxyId>,
    large_batch_slowdown: Option<LargeBatchSlowdown>,
    received: Vec<ReceivedRequest>,
    suspended_ids: HashSet<ProxyId>,
    generations: HashMap<ProxyId, Generation>,
    peak_generating: usize,
    control_clients: Vec<mpsc::UnboundedSender<Vec<u8>>>,
    control_tasks: Vec<tokio::task::AbortHandle>,
}

impl Default for EngineState {
    fn default() -> EngineState {
        EngineState {
            max_running: usize::MAX,
            kv_cache_tokens: 1 << 20,
            prefill_interval: Duration::ZERO,
            prefill_queue: VecDeque::new(),
            large_batch_slowdown: None,
            received: Vec::new(),
            suspended_ids: HashSet::new(),
            generations: HashMap::new(),
            peak_generating: 0,
            control_clients: Vec::new(),
            control_tasks: Vec::new(),
        }
    }
}

struct LargeBatchSlowdown {
    threshold: usize,
    token_interval: Duration,
}

struct Generation {
    is_generating: bool,
    prompt_tokens: u64,
    output_tokens: u64,
}

#[derive(Clone)]
pub struct MockEngine {
    pub address: SocketAddr,
    pub control_socket: PathBuf,
    state: Arc<Mutex<EngineState>>,
    http_listener: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    control_listener: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl MockEngine {
    pub async fn start(control_socket: PathBuf) -> MockEngine {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let engine = MockEngine {
            address: listener.local_addr().unwrap(),
            control_socket,
            state: Arc::default(),
            http_listener: Arc::default(),
            control_listener: Arc::default(),
        };
        engine.open_control();
        let http_engine = engine.clone();
        let accept_task = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let engine = http_engine.clone();
                tokio::spawn(async move {
                    let service = service_fn(move |request| engine.clone().handle(request));
                    let _ = http1::Builder::new()
                        .serve_connection(TokioIo::new(stream), service)
                        .await;
                });
            }
        });
        *engine.http_listener.lock().unwrap() = Some(accept_task);
        engine
    }

    pub fn open_control(&self) {
        let listener = UnixListener::bind(&self.control_socket).unwrap();
        let task = tokio::spawn(self.clone().serve_control(listener));
        *self.control_listener.lock().unwrap() = Some(task);
    }

    pub async fn close_control(&self) {
        let task = self.control_listener.lock().unwrap().take().unwrap();
        task.abort();
        let _ = task.await;
        let _ = std::fs::remove_file(&self.control_socket);
        self.disconnect_control_clients();
    }

    pub async fn close_http(&self) {
        let task = self.http_listener.lock().unwrap().take().unwrap();
        task.abort();
        let _ = task.await;
    }

    pub fn disconnect_control_clients(&self) {
        let mut state = self.state.lock().unwrap();
        for task in state.control_tasks.drain(..) {
            task.abort();
        }
        state.control_clients.clear();
        state.suspended_ids.clear();
    }

    pub fn limit_running(&self, max_running: usize, tokens_per_request: usize) {
        let mut state = self.state.lock().unwrap();
        state.max_running = max_running;
        state.kv_cache_tokens = max_running * tokens_per_request;
    }

    pub fn set_prefill_interval(&self, interval: Duration) {
        self.state.lock().unwrap().prefill_interval = interval;
    }

    pub fn set_large_batch_token_interval(&self, threshold: usize, token_interval: Duration) {
        self.state.lock().unwrap().large_batch_slowdown = Some(LargeBatchSlowdown {
            threshold,
            token_interval,
        });
    }

    pub fn has_control_client(&self) -> bool {
        !self.state.lock().unwrap().control_clients.is_empty()
    }

    pub fn received(&self) -> Vec<ReceivedRequest> {
        self.state.lock().unwrap().received.clone()
    }

    pub fn peak_generating(&self) -> usize {
        self.state.lock().unwrap().peak_generating
    }

    pub fn generating_ids(&self) -> Vec<ProxyId> {
        let state = self.state.lock().unwrap();
        state
            .generations
            .iter()
            .filter(|(_, generation)| generation.is_generating)
            .map(|(id, _)| *id)
            .collect()
    }

    pub fn suspended_ids(&self) -> Vec<ProxyId> {
        self.state
            .lock()
            .unwrap()
            .suspended_ids
            .iter()
            .copied()
            .collect()
    }

    pub fn active_ids(&self) -> Vec<ProxyId> {
        self.state
            .lock()
            .unwrap()
            .generations
            .keys()
            .copied()
            .collect()
    }

    async fn serve_control(self, listener: UnixListener) {
        while let Ok((stream, _)) = listener.accept().await {
            let (sender, mut receiver) = mpsc::unbounded_channel::<Vec<u8>>();
            self.state
                .lock()
                .unwrap()
                .control_clients
                .push(sender.clone());
            let (reader, mut writer) = stream.into_split();
            let writer_task = tokio::spawn(async move {
                while let Some(line) = receiver.recv().await {
                    if writer.write_all(&line).await.is_err() {
                        return;
                    }
                }
            });
            let engine = self.clone();
            let reader_task = tokio::spawn(async move {
                let mut lines = BufReader::new(reader).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let command: CommandLine = serde_json::from_str(&line).unwrap();
                    let reply = engine.apply(command);
                    let _ = sender.send(reply.encode());
                }
                let mut state = engine.state.lock().unwrap();
                state
                    .control_clients
                    .retain(|client| !client.same_channel(&sender));
                if state.control_clients.is_empty() {
                    state.suspended_ids.clear();
                }
            });
            self.state
                .lock()
                .unwrap()
                .control_tasks
                .extend([writer_task.abort_handle(), reader_task.abort_handle()]);
        }
    }

    fn apply(&self, line: CommandLine) -> EngineLine {
        let mut state = self.state.lock().unwrap();
        let result = match line.command {
            Command::Suspend { ids } => {
                for id in &ids {
                    MockEngine::mark_generating(&mut state, *id, false, StopReason::Suspended);
                }
                state.suspended_ids.extend(ids);
                Value::Null
            }
            Command::Resume { ids } => {
                for id in ids {
                    state.suspended_ids.remove(&id);
                }
                Value::Null
            }
            Command::Info => serde_json::to_value(EngineInfo {
                engine: EngineKind::Vllm,
                model: "mock-model".to_string(),
                data_parallel_rank: None,
                max_running_requests: 64,
                priority_scheduling: true,
                kv_cache_tokens: state.kv_cache_tokens,
            })
            .unwrap(),
            Command::Stats => serde_json::to_value(EngineStats {
                requests: state
                    .generations
                    .iter()
                    .map(|(id, generation)| RequestStats {
                        id: *id,
                        state: if state.suspended_ids.contains(id) {
                            RequestPhase::Suspended
                        } else if generation.is_generating {
                            RequestPhase::Running
                        } else {
                            RequestPhase::Waiting
                        },
                        prompt_tokens: generation.prompt_tokens,
                        output_tokens: generation.output_tokens,
                    })
                    .collect(),
            })
            .unwrap(),
        };
        EngineLine::Success {
            seq: line.seq,
            result,
        }
    }

    fn broadcast(state: &mut EngineState, event: Event) {
        let line = EngineLine::Event(event).encode();
        state
            .control_clients
            .retain(|client| client.send(line.clone()).is_ok());
    }

    fn set_generating(&self, id: ProxyId, is_generating: bool, reason: StopReason) {
        let mut state = self.state.lock().unwrap();
        MockEngine::mark_generating(&mut state, id, is_generating, reason);
    }

    fn mark_generating(
        state: &mut EngineState,
        id: ProxyId,
        is_generating: bool,
        reason: StopReason,
    ) {
        let Some(generation) = state.generations.get_mut(&id) else {
            return;
        };
        if generation.is_generating == is_generating {
            return;
        }
        generation.is_generating = is_generating;
        let event = if is_generating {
            Event::Running { id }
        } else {
            Event::Stopped { id, reason }
        };
        let generating = state
            .generations
            .values()
            .filter(|generation| generation.is_generating)
            .count();
        state.peak_generating = state.peak_generating.max(generating);
        MockEngine::broadcast(state, event);
    }

    async fn next_token(&self, id: ProxyId, index: u64) {
        let interval = loop {
            {
                let mut state = self.state.lock().unwrap();
                let generating = state
                    .generations
                    .values()
                    .filter(|generation| generation.is_generating)
                    .count();
                let is_generating = state
                    .generations
                    .get(&id)
                    .is_some_and(|generation| generation.is_generating);
                let has_slot = is_generating || generating < state.max_running;
                let prefill_head = state.prefill_queue.front().copied();
                let is_engine_free = prefill_head.is_none_or(|head| head == id);
                if !state.suspended_ids.contains(&id) && has_slot && is_engine_free {
                    MockEngine::mark_generating(&mut state, id, true, StopReason::Suspended);
                    break if prefill_head == Some(id) {
                        state.prefill_interval
                    } else {
                        MockEngine::decode_interval(&state)
                    };
                }
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        };
        tokio::time::sleep(interval).await;
        let mut state = self.state.lock().unwrap();
        if state.prefill_queue.front() == Some(&id) {
            state.prefill_queue.pop_front();
        }
        if let Some(generation) = state.generations.get_mut(&id) {
            generation.output_tokens = index + 1;
        }
    }

    fn decode_interval(state: &EngineState) -> Duration {
        let generating = state
            .generations
            .values()
            .filter(|generation| generation.is_generating)
            .count();
        match &state.large_batch_slowdown {
            Some(slowdown) if generating > slowdown.threshold => slowdown.token_interval,
            _ => TOKEN_INTERVAL,
        }
    }

    fn finish(&self, id: ProxyId) {
        self.set_generating(id, false, StopReason::Finished);
        let mut state = self.state.lock().unwrap();
        state.generations.remove(&id);
        state.prefill_queue.retain(|queued| *queued != id);
    }

    async fn handle(
        self,
        request: Request<Incoming>,
    ) -> Result<Response<BoxBody<Bytes, BoxError>>, Infallible> {
        let (parts, body) = request.into_parts();
        let body = body.collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        let path = parts.uri.path().to_string();
        self.state.lock().unwrap().received.push(ReceivedRequest {
            path: path.clone(),
            headers: parts.headers.clone(),
            body: body.clone(),
        });
        if path == "/v1/models" {
            let mut response = full_response(json!({"data": [{"id": "mock-model"}]}).to_string());
            response
                .headers_mut()
                .append("set-cookie", "first=1".parse().unwrap());
            response
                .headers_mut()
                .append("set-cookie", "second=2".parse().unwrap());
            return Ok(response);
        }
        if !matches!(
            path.as_str(),
            "/v1/chat/completions" | "/v1/completions" | "/generate"
        ) {
            let mut response = full_response("{}".to_string());
            *response.status_mut() = StatusCode::NOT_FOUND;
            return Ok(response);
        }
        let id = parts
            .headers
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .or_else(|| body["rid"].as_str())
            .map(|id| ProxyId::try_from(id.to_string()).unwrap())
            .unwrap();
        let tokens = body["max_tokens"].as_u64().unwrap_or(5);
        {
            let mut state = self.state.lock().unwrap();
            state.generations.insert(
                id,
                Generation {
                    is_generating: false,
                    prompt_tokens: prompt_tokens(&body),
                    output_tokens: 0,
                },
            );
            if !state.prefill_interval.is_zero() {
                state.prefill_queue.push_back(id);
            }
        }
        if body["stream"].as_bool() == Some(true) {
            let (mut sender, channel) = Channel::<Bytes, BoxError>::new(4);
            let engine = self.clone();
            tokio::spawn(async move {
                for index in 0..tokens {
                    engine.next_token(id, index).await;
                    let event = format!("data: {{\"token\": {index}}}\n\n");
                    if sender.send_data(Bytes::from(event)).await.is_err() {
                        engine.finish(id);
                        return;
                    }
                }
                let _ = sender
                    .send_data(Bytes::from_static(b"data: [DONE]\n\n"))
                    .await;
                engine.finish(id);
            });
            let mut response = Response::new(channel.boxed());
            response
                .headers_mut()
                .insert("content-type", "text/event-stream".parse().unwrap());
            return Ok(response);
        }
        let generation = GenerationGuard {
            engine: self.clone(),
            id,
        };
        for index in 0..tokens {
            self.next_token(id, index).await;
        }
        drop(generation);
        Ok(full_response(
            json!({"id": id, "tokens": tokens}).to_string(),
        ))
    }
}

struct GenerationGuard {
    engine: MockEngine,
    id: ProxyId,
}

impl Drop for GenerationGuard {
    fn drop(&mut self) {
        self.engine.finish(self.id);
    }
}

pub fn prompt_tokens(body: &Value) -> u64 {
    let words: usize = body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|message| message["content"].as_str())
        .map(|content| content.split_whitespace().count())
        .sum();
    PROMPT_TEMPLATE_TOKENS + TOKENS_PER_PROMPT_WORD * words as u64
}

fn full_response(body: String) -> Response<BoxBody<Bytes, BoxError>> {
    let mut response = Response::new(
        Full::new(Bytes::from(body))
            .map_err(|never| match never {})
            .boxed(),
    );
    response
        .headers_mut()
        .insert("content-type", "application/json".parse().unwrap());
    response
}
