//! A mock of a patched vLLM serving SmolLM2-135M-Instruct, for local end-to-end
//! and A/B testing (e.g. against LiteLLM). It speaks the OpenAI API with
//! deterministic output and the llm-proxy control protocol on a Unix socket.
//!
//! cargo run --release --example mock_smollm -- --listen 127.0.0.1:8001 \
//!     --control-socket /tmp/mock-a.sock --log /tmp/mock-a.jsonl
//!
//! Fault injection by prompt content (any message or prompt containing it):
//!   MOCK:error500   500 with an OpenAI error body
//!   MOCK:error400   400 with an OpenAI error body
//!   MOCK:slow       100 ms per token
//!   MOCK:hang       never answers (until the client goes away)
//!   MOCK:drop       streams 3 tokens, then drops the connection
//!   MOCK:badjson    200 with a body which isn't JSON
//!   MOCK:tokens=N   generates exactly N tokens unless max_tokens is lower

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::channel::Channel;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
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
    RequestPhase, RequestStats, StopReason, TokenCounts,
};

type BoxError = Box<dyn std::error::Error + Send + Sync>;
type Body = BoxBody<Bytes, BoxError>;

const CONTEXT_LENGTH: u64 = 8192;
const TEMPLATE_TOKENS: u64 = 4;
const WORDS: &[&str] = &[
    "the",
    "model",
    "answers",
    "with",
    "a",
    "small",
    "deterministic",
    "sentence",
    "about",
    "llamas",
    "and",
    "proxies",
    "which",
    "is",
    "quite",
    "helpful",
    "for",
    "testing",
    "every",
    "path",
    "through",
    "gateway",
    "scheduler",
    "engine",
    "token",
    "stream",
];

struct Options {
    listen: String,
    control_socket: PathBuf,
    model: String,
    /// Further names it answers to, like vLLM's `--served-model-name a b c`.
    aliases: Vec<String>,
    token_interval: Duration,
    max_running: usize,
    log: Option<PathBuf>,
}

fn parse_options() -> Options {
    let mut options = Options {
        listen: "127.0.0.1:8001".into(),
        control_socket: "/tmp/mock-smollm.sock".into(),
        model: "SmolLM2-135M-Instruct".into(),
        aliases: Vec::new(),
        token_interval: Duration::from_millis(5),
        max_running: 64,
        log: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--listen" => options.listen = value(),
            "--control-socket" => options.control_socket = value().into(),
            "--model" => options.model = value(),
            "--alias" => options.aliases.push(value()),
            "--token-ms" => {
                options.token_interval = Duration::from_millis(value().parse().unwrap())
            }
            "--max-running" => options.max_running = value().parse().unwrap(),
            "--log" => options.log = Some(value().into()),
            other => panic!("unknown argument {other}"),
        }
    }
    options
}

struct Generation {
    is_generating: bool,
    prompt_tokens: u64,
    output_tokens: u64,
}

impl Generation {
    fn tokens(&self) -> TokenCounts {
        TokenCounts {
            prompt: self.prompt_tokens,
            output: self.output_tokens,
            cache_hit: 0,
            prefill: self.prompt_tokens,
            prefill_count: 1,
        }
    }
}

#[derive(Default)]
struct State {
    generations: HashMap<ProxyId, Generation>,
    suspended: HashSet<ProxyId>,
    clients: Vec<mpsc::UnboundedSender<Vec<u8>>>,
}

#[derive(Clone)]
struct Engine {
    options: Arc<Options>,
    state: Arc<Mutex<State>>,
    log: Arc<Mutex<Option<std::fs::File>>>,
}

impl Engine {
    fn broadcast(state: &mut State, event: Event) {
        let line = EngineLine::Event(event).encode();
        state
            .clients
            .retain(|client| client.send(line.clone()).is_ok());
    }

    fn mark(state: &mut State, id: ProxyId, is_generating: bool, reason: StopReason) {
        let Some(generation) = state.generations.get_mut(&id) else {
            return;
        };
        if generation.is_generating == is_generating {
            return;
        }
        generation.is_generating = is_generating;
        let tokens = Some(generation.tokens());
        let event = if is_generating {
            Event::Running { id, tokens }
        } else {
            Event::Stopped { id, reason, tokens }
        };
        Engine::broadcast(state, event);
    }

    fn apply(&self, line: CommandLine) -> EngineLine {
        let mut state = self.state.lock().unwrap();
        let result = match line.command {
            Command::Suspend { ids } => {
                for id in &ids {
                    Engine::mark(&mut state, *id, false, StopReason::Suspended);
                }
                state.suspended.extend(ids);
                Value::Null
            }
            Command::Resume { ids } => {
                for id in ids {
                    state.suspended.remove(&id);
                }
                Value::Null
            }
            Command::Info => serde_json::to_value(EngineInfo {
                engine: EngineKind::Vllm,
                model: self.options.model.clone(),
                data_parallel_rank: None,
                max_running_requests: self.options.max_running,
                priority_scheduling: true,
                kv_cache_tokens: 1 << 20,
            })
            .unwrap(),
            Command::Stats => serde_json::to_value(EngineStats {
                requests: state
                    .generations
                    .iter()
                    .map(|(id, generation)| RequestStats {
                        id: *id,
                        state: if state.suspended.contains(id) {
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

    async fn serve_control(self, listener: UnixListener) {
        while let Ok((stream, _)) = listener.accept().await {
            let (sender, mut receiver) = mpsc::unbounded_channel::<Vec<u8>>();
            self.state.lock().unwrap().clients.push(sender.clone());
            let (reader, mut writer) = stream.into_split();
            tokio::spawn(async move {
                while let Some(line) = receiver.recv().await {
                    if writer.write_all(&line).await.is_err() {
                        return;
                    }
                }
            });
            let engine = self.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(reader).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let reply = match serde_json::from_str::<CommandLine>(&line) {
                        Ok(command) => engine.apply(command),
                        Err(error) => {
                            eprintln!("mock: bad control line {line:?}: {error}");
                            continue;
                        }
                    };
                    let _ = sender.send(reply.encode());
                }
                let mut state = engine.state.lock().unwrap();
                state.clients.retain(|client| !client.same_channel(&sender));
                if state.clients.is_empty() {
                    state.suspended.clear();
                }
            });
        }
    }

    /// Waits until the request may produce its next token, then sleeps one token interval.
    async fn next_token(&self, id: Option<ProxyId>, index: u64, interval: Duration) {
        if let Some(id) = id {
            loop {
                {
                    let mut state = self.state.lock().unwrap();
                    let running = state
                        .generations
                        .values()
                        .filter(|g| g.is_generating)
                        .count();
                    let is_generating = state.generations.get(&id).is_some_and(|g| g.is_generating);
                    let has_slot = is_generating || running < self.options.max_running;
                    if !state.suspended.contains(&id) && has_slot {
                        Engine::mark(&mut state, id, true, StopReason::Suspended);
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        }
        tokio::time::sleep(interval).await;
        if let Some(id) = id {
            if let Some(generation) = self.state.lock().unwrap().generations.get_mut(&id) {
                generation.output_tokens = index + 1;
            }
        }
    }

    fn finish(&self, id: Option<ProxyId>) {
        if let Some(id) = id {
            let mut state = self.state.lock().unwrap();
            Engine::mark(&mut state, id, false, StopReason::Finished);
            state.generations.remove(&id);
            state.suspended.remove(&id);
        }
    }

    fn record(&self, entry: Value) {
        if let Some(file) = self.log.lock().unwrap().as_mut() {
            let _ = writeln!(file, "{entry}");
        }
    }

    async fn handle(self, request: Request<Incoming>) -> Result<Response<Body>, Infallible> {
        let (parts, body) = request.into_parts();
        let bytes = match body.collect().await {
            Ok(body) => body.to_bytes(),
            Err(_) => return Ok(error(StatusCode::BAD_REQUEST, "unreadable body")),
        };
        let path = parts.uri.path().to_string();
        let headers: serde_json::Map<String, Value> = parts
            .headers
            .iter()
            .map(|(name, value)| {
                (
                    name.to_string(),
                    Value::from(String::from_utf8_lossy(value.as_bytes())),
                )
            })
            .collect();
        let parsed: Option<Value> = serde_json::from_slice(&bytes).ok();
        self.record(json!({
            "time": now_seconds_f64(),
            "method": parts.method.as_str(),
            "path": path,
            "query": parts.uri.query(),
            "headers": headers,
            "body": parsed.clone().unwrap_or_else(|| Value::from(String::from_utf8_lossy(&bytes))),
        }));
        let names: Vec<&String> = std::iter::once(&self.options.model)
            .chain(&self.options.aliases)
            .collect();
        match (parts.method.as_str(), path.as_str()) {
            ("GET", "/health") => return Ok(full(StatusCode::OK, "", "text/plain")),
            ("GET", "/version") => {
                return Ok(json_response(
                    StatusCode::OK,
                    json!({"version": "0.29.0-mock"}),
                ))
            }
            ("GET", "/v1/models") => {
                let data: Vec<Value> = names
                    .iter()
                    .map(|name| {
                        json!({
                            "id": name, "object": "model", "created": 1700000000,
                            "owned_by": "vllm", "root": "HuggingFaceTB/SmolLM2-135M-Instruct",
                            "parent": null, "max_model_len": CONTEXT_LENGTH
                        })
                    })
                    .collect();
                return Ok(json_response(
                    StatusCode::OK,
                    json!({"object": "list", "data": data}),
                ));
            }
            ("GET", p) if p.starts_with("/v1/models/") => {
                let name = &p["/v1/models/".len()..];
                if names.iter().any(|known| *known == name) {
                    return Ok(json_response(
                        StatusCode::OK,
                        json!({
                            "id": name, "object": "model", "created": 1700000000, "owned_by": "vllm"
                        }),
                    ));
                }
                return Ok(error(
                    StatusCode::NOT_FOUND,
                    &format!("The model `{name}` does not exist."),
                ));
            }
            ("POST", "/tokenize") => {
                let Some(body) = parsed else {
                    return Ok(error(StatusCode::BAD_REQUEST, "invalid JSON"));
                };
                let words = prompt_text(&body);
                let tokens: Vec<u64> = words.split_whitespace().map(word_id).collect();
                return Ok(json_response(
                    StatusCode::OK,
                    json!({
                        "count": tokens.len(), "max_model_len": CONTEXT_LENGTH, "tokens": tokens
                    }),
                ));
            }
            ("POST", "/detokenize") => {
                let Some(body) = parsed else {
                    return Ok(error(StatusCode::BAD_REQUEST, "invalid JSON"));
                };
                let text: Vec<&str> = body["tokens"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_u64)
                    .map(|token| WORDS[token as usize % WORDS.len()])
                    .collect();
                return Ok(json_response(
                    StatusCode::OK,
                    json!({"prompt": text.join(" ")}),
                ));
            }
            ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {}
            _ => return Ok(error(StatusCode::NOT_FOUND, "Not Found")),
        }
        let is_chat = path == "/v1/chat/completions";
        let Some(body) = parsed.filter(Value::is_object) else {
            return Ok(error(
                StatusCode::BAD_REQUEST,
                "Expecting value: line 1 column 1 (char 0)",
            ));
        };
        let requested_model = body["model"].as_str().unwrap_or_default();
        // vLLM answers with the name the request used.
        let model = requested_model.to_string();
        if !names.iter().any(|name| *name == requested_model) {
            return Ok(error(
                StatusCode::NOT_FOUND,
                &format!("The model `{requested_model}` does not exist."),
            ));
        }
        if is_chat && !body["messages"].as_array().is_some_and(|m| !m.is_empty()) {
            return Ok(error(
                StatusCode::BAD_REQUEST,
                "messages: field required and must be a non-empty list",
            ));
        }
        if !is_chat && body.get("prompt").is_none() {
            return Ok(error(StatusCode::BAD_REQUEST, "prompt: field required"));
        }
        let text = prompt_text(&body);
        let prompt_tokens = TEMPLATE_TOKENS + text.split_whitespace().count() as u64;
        let max_tokens = body
            .get("max_completion_tokens")
            .filter(|v| !v.is_null())
            .or_else(|| body.get("max_tokens").filter(|v| !v.is_null()));
        let max_tokens = match max_tokens {
            None => None,
            Some(value) => match value.as_u64() {
                Some(n) if n >= 1 => Some(n),
                _ => {
                    return Ok(error(
                        StatusCode::BAD_REQUEST,
                        "max_tokens must be at least 1",
                    ))
                }
            },
        };
        if prompt_tokens + max_tokens.unwrap_or(0) > CONTEXT_LENGTH {
            return Ok(error(StatusCode::BAD_REQUEST, &format!(
                "This model's maximum context length is {CONTEXT_LENGTH} tokens. However, you requested {} tokens ({prompt_tokens} in the messages, {} in the completion). Please reduce the length of the messages or completion.",
                prompt_tokens + max_tokens.unwrap_or(0), max_tokens.unwrap_or(0)
            )));
        }
        let n = body["n"].as_u64().unwrap_or(1);
        if !(1..=8).contains(&n) {
            return Ok(error(StatusCode::BAD_REQUEST, "n must be between 1 and 8"));
        }
        if text.contains("MOCK:error500") {
            return Ok(error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "mock internal error",
            ));
        }
        if text.contains("MOCK:error400") {
            return Ok(error(StatusCode::BAD_REQUEST, "mock bad request"));
        }
        if text.contains("MOCK:badjson") {
            return Ok(full(StatusCode::OK, "{not json", "application/json"));
        }
        let natural = text
            .split("MOCK:tokens=")
            .nth(1)
            .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|digits| digits.parse::<u64>().ok())
            .unwrap_or_else(|| 8 + hash(&text) % 33);
        let output_tokens = max_tokens.map_or(natural, |max| max.min(natural));
        let finish_reason = if max_tokens.is_some_and(|max| max < natural) {
            "length"
        } else {
            "stop"
        };
        let interval = if text.contains("MOCK:slow") {
            Duration::from_millis(100)
        } else {
            self.options.token_interval
        };
        let hang = text.contains("MOCK:hang");
        let drop_after = text.contains("MOCK:drop").then_some(3);
        let id = parts
            .headers
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| ProxyId::try_from(value.to_string()).ok());
        if let Some(id) = id {
            self.state.lock().unwrap().generations.insert(
                id,
                Generation {
                    is_generating: false,
                    prompt_tokens,
                    output_tokens: 0,
                },
            );
        }
        let guard = Guard {
            engine: self.clone(),
            id,
        };
        let words: Vec<&'static str> = (0..output_tokens)
            .map(|index| WORDS[(hash(&(text.as_str(), index)) % WORDS.len() as u64) as usize])
            .collect();
        let response_id = format!(
            "{}-{:016x}",
            if is_chat { "chatcmpl" } else { "cmpl" },
            hash(&(text.as_str(), now_seconds_f64().to_bits()))
        );
        let created = now_seconds_f64() as u64;
        let usage = json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens * n,
            "total_tokens": prompt_tokens + output_tokens * n,
            "prompt_tokens_details": null
        });
        let stream = body["stream"].as_bool() == Some(true);
        if !stream {
            if hang {
                std::future::pending::<()>().await;
            }
            for index in 0..output_tokens {
                self.next_token(id, index, interval).await;
            }
            drop(guard);
            let content = words.join(" ");
            let choices: Vec<Value> = (0..n)
                .map(|index| {
                    if is_chat {
                        json!({
                            "index": index, "logprobs": null, "finish_reason": finish_reason,
                            "stop_reason": null,
                            "message": {"role": "assistant", "content": content, "tool_calls": [], "reasoning_content": null}
                        })
                    } else {
                        json!({"index": index, "text": content, "logprobs": null, "finish_reason": finish_reason, "stop_reason": null})
                    }
                })
                .collect();
            return Ok(json_response(
                StatusCode::OK,
                json!({
                    "id": response_id,
                    "object": if is_chat { "chat.completion" } else { "text_completion" },
                    "created": created, "model": model, "choices": choices, "usage": usage,
                    "system_fingerprint": null
                }),
            ));
        }
        let include_usage = body["stream_options"]["include_usage"].as_bool() == Some(true);
        let object = if is_chat {
            "chat.completion.chunk"
        } else {
            "text_completion"
        };
        let (mut sender, channel) = Channel::<Bytes, BoxError>::new(4);
        let engine = self.clone();
        tokio::spawn(async move {
            let _guard = guard;
            let chunk = |choices: Value, usage: Value| {
                let mut value = json!({"id": response_id, "object": object, "created": created, "model": model, "choices": choices});
                if include_usage {
                    value["usage"] = usage;
                }
                Bytes::from(format!("data: {value}\n\n"))
            };
            if is_chat {
                let first: Vec<Value> = (0..n)
                    .map(|i| json!({"index": i, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null}))
                    .collect();
                if sender
                    .send_data(chunk(Value::from(first), Value::Null))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            if hang {
                // Empty frames send nothing, but fail once the client is gone.
                loop {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    if sender.send_data(Bytes::new()).await.is_err() {
                        return;
                    }
                }
            }
            for (index, word) in words.iter().enumerate() {
                engine.next_token(id, index as u64, interval).await;
                if drop_after == Some(index) {
                    sender.abort("mock dropped the connection".into());
                    return;
                }
                let piece = if index == 0 {
                    word.to_string()
                } else {
                    format!(" {word}")
                };
                let choices: Vec<Value> = (0..n)
                    .map(|i| {
                        if is_chat {
                            json!({"index": i, "delta": {"content": piece}, "logprobs": null, "finish_reason": null})
                        } else {
                            json!({"index": i, "text": piece, "logprobs": null, "finish_reason": null, "stop_reason": null})
                        }
                    })
                    .collect();
                if sender
                    .send_data(chunk(Value::from(choices), Value::Null))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            let last: Vec<Value> = (0..n)
                .map(|i| {
                    if is_chat {
                        json!({"index": i, "delta": {"content": ""}, "logprobs": null, "finish_reason": finish_reason, "stop_reason": null})
                    } else {
                        json!({"index": i, "text": "", "logprobs": null, "finish_reason": finish_reason, "stop_reason": null})
                    }
                })
                .collect();
            if sender
                .send_data(chunk(Value::from(last), Value::Null))
                .await
                .is_err()
            {
                return;
            }
            if include_usage {
                let value = json!({"id": response_id, "object": object, "created": created, "model": model, "choices": [], "usage": usage});
                if sender
                    .send_data(Bytes::from(format!("data: {value}\n\n")))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            let _ = sender
                .send_data(Bytes::from_static(b"data: [DONE]\n\n"))
                .await;
        });
        let mut response = Response::new(channel.boxed());
        response.headers_mut().insert(
            "content-type",
            "text/event-stream; charset=utf-8".parse().unwrap(),
        );
        Ok(response)
    }
}

struct Guard {
    engine: Engine,
    id: Option<ProxyId>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.engine.finish(self.id);
    }
}

fn prompt_text(body: &Value) -> String {
    let mut parts = Vec::new();
    for message in body["messages"].as_array().into_iter().flatten() {
        match &message["content"] {
            Value::String(text) => parts.push(text.clone()),
            Value::Array(items) => parts.extend(
                items
                    .iter()
                    .filter_map(|item| item["text"].as_str().map(str::to_string)),
            ),
            _ => {}
        }
    }
    match &body["prompt"] {
        Value::String(text) => parts.push(text.clone()),
        Value::Array(items) => {
            parts.extend(items.iter().filter_map(|i| i.as_str().map(str::to_string)))
        }
        _ => {}
    }
    if let Some(text) = body["text"].as_str() {
        parts.push(text.to_string());
    }
    parts.join("\n")
}

fn hash(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn word_id(word: &str) -> u64 {
    hash(&word) % 49152
}

fn now_seconds_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

fn full(status: StatusCode, body: &str, content_type: &str) -> Response<Body> {
    let mut response = Response::new(
        Full::new(Bytes::from(body.to_string()))
            .map_err(|never| match never {})
            .boxed(),
    );
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert("content-type", content_type.parse().unwrap());
    response
}

fn json_response(status: StatusCode, body: Value) -> Response<Body> {
    full(status, &body.to_string(), "application/json")
}

fn error(status: StatusCode, message: &str) -> Response<Body> {
    let kind = match status.as_u16() {
        400 => "BadRequestError",
        404 => "NotFoundError",
        _ => "InternalServerError",
    };
    json_response(
        status,
        json!({
            "error": {"message": message, "type": kind, "param": null, "code": status.as_u16()}
        }),
    )
}

#[tokio::main]
async fn main() {
    let options = parse_options();
    // The HTTP port first: a second mock with the same arguments must fail before it
    // removes the control socket of the first.
    let listener = TcpListener::bind(&options.listen).await.expect("bind http");
    let _ = std::fs::remove_file(&options.control_socket);
    let control = UnixListener::bind(&options.control_socket).expect("bind control socket");
    let log = options.log.as_ref().map(|path| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("open log")
    });
    eprintln!(
        "mock-smollm: serving {} on http://{} with the control socket {}",
        options.model,
        options.listen,
        options.control_socket.display()
    );
    let engine = Engine {
        options: Arc::new(options),
        state: Arc::default(),
        log: Arc::new(Mutex::new(log)),
    };
    tokio::spawn(engine.clone().serve_control(control));
    while let Ok((stream, _)) = listener.accept().await {
        let engine = engine.clone();
        tokio::spawn(async move {
            let service = service_fn(move |request| engine.clone().handle(request));
            let _ = http1::Builder::new()
                // Like uvicorn: no low limit on the number of headers.
                .max_headers(1000)
                .serve_connection(TokioIo::new(stream), service)
                .await;
        });
    }
}
