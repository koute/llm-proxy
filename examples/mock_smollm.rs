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
use clap::Parser;
use http_body_util::channel::{Channel, SendError};
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::http::request::Parts;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
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
/// The vocabulary of token ids and generated text.
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

#[derive(Parser)]
struct Options {
    #[arg(long, default_value = "127.0.0.1:8001")]
    listen: String,
    #[arg(long, default_value = "/tmp/mock-smollm.sock")]
    control_socket: PathBuf,
    #[arg(long, default_value = "SmolLM2-135M-Instruct")]
    model: String,
    /// Further names it answers to, like vLLM's `--served-model-name a b c`.
    #[arg(long = "alias")]
    aliases: Vec<String>,
    /// Milliseconds per generated token.
    #[arg(long, default_value_t = 5)]
    token_ms: u64,
    #[arg(long, default_value_t = 64)]
    max_running: usize,
    #[arg(long)]
    log: Option<PathBuf>,
}

struct Generation {
    is_generating: bool,
    prompt_tokens: u64,
    output_tokens: u64,
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
    /// Tells the control clients when a request starts or stops generating.
    fn mark(state: &mut State, id: ProxyId, is_generating: bool, reason: StopReason) {
        let Some(generation) = state.generations.get_mut(&id) else {
            return;
        };
        if generation.is_generating == is_generating {
            return;
        }
        generation.is_generating = is_generating;
        let tokens = Some(TokenCounts {
            prompt: generation.prompt_tokens,
            output: generation.output_tokens,
            cache_hit: 0,
            prefill: generation.prompt_tokens,
            prefill_count: 1,
        });
        let event = if is_generating {
            Event::Running { id, tokens }
        } else {
            Event::Stopped { id, reason, tokens }
        };
        let line = EngineLine::Event(event).encode();
        state
            .clients
            .retain(|client| client.send(line.clone()).is_ok());
    }

    fn apply(&self, CommandLine { seq, command }: CommandLine) -> EngineLine {
        let mut state = self.state.lock().unwrap();
        let result = match command {
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
            Command::Stats => {
                let stats = |(id, generation): (&ProxyId, &Generation)| RequestStats {
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
                };
                let requests = state.generations.iter().map(stats).collect();
                serde_json::to_value(EngineStats { requests }).unwrap()
            }
        };
        EngineLine::Success { seq, result }
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
                    match serde_json::from_str::<CommandLine>(&line) {
                        Ok(command) => {
                            let _ = sender.send(engine.apply(command).encode());
                        }
                        Err(error) => eprintln!("mock: bad control line {line:?}: {error}"),
                    }
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

    async fn handle(self, request: Request<Incoming>) -> Result<Response<Body>, Infallible> {
        let (parts, body) = request.into_parts();
        Ok(match body.collect().await {
            Ok(body) => self.respond(parts, body.to_bytes()).await,
            Err(_) => error(400, "unreadable body"),
        })
    }

    async fn respond(&self, parts: Parts, bytes: Bytes) -> Response<Body> {
        let path = parts.uri.path();
        let headers: serde_json::Map<String, Value> = parts
            .headers
            .iter()
            .map(|(name, value)| (name.to_string(), lossy(value.as_bytes())))
            .collect();
        let parsed: Option<Value> = serde_json::from_slice(&bytes).ok();
        let entry = json!({
            "time": now_seconds_f64(),
            "method": parts.method.as_str(),
            "path": path,
            "query": parts.uri.query(),
            "headers": headers,
            "body": parsed.clone().unwrap_or_else(|| lossy(&bytes)),
        });
        if let Some(file) = self.log.lock().unwrap().as_mut() {
            let _ = writeln!(file, "{entry}");
        }
        let names: Vec<&String> = std::iter::once(&self.options.model)
            .chain(&self.options.aliases)
            .collect();
        let unknown_model = |name: &str| error(404, &format!("The model `{name}` does not exist."));
        match (parts.method.as_str(), path) {
            ("GET", "/health") => return full(200, "", "text/plain"),
            ("GET", "/version") => return json_response(200, json!({"version": "0.29.0-mock"})),
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
                return json_response(200, json!({"object": "list", "data": data}));
            }
            ("GET", p) if p.starts_with("/v1/models/") => {
                let name = &p["/v1/models/".len()..];
                if !names.iter().any(|known| *known == name) {
                    return unknown_model(name);
                }
                let model = json!({
                    "id": name, "object": "model", "created": 1700000000, "owned_by": "vllm"
                });
                return json_response(200, model);
            }
            ("POST", "/tokenize") => {
                let Some(body) = parsed else {
                    return error(400, "invalid JSON");
                };
                let words = prompt_text(&body);
                let tokens: Vec<u64> = words.split_whitespace().map(word_id).collect();
                let tokenized = json!({
                    "count": tokens.len(), "max_model_len": CONTEXT_LENGTH, "tokens": tokens
                });
                return json_response(200, tokenized);
            }
            ("POST", "/detokenize") => {
                let Some(body) = parsed else {
                    return error(400, "invalid JSON");
                };
                let tokens = body["tokens"].as_array().into_iter().flatten();
                let text: Vec<&str> = tokens.filter_map(Value::as_u64).map(word).collect();
                return json_response(200, json!({"prompt": text.join(" ")}));
            }
            ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {}
            _ => return error(404, "Not Found"),
        }
        let is_chat = path == "/v1/chat/completions";
        let Some(body) = parsed.filter(Value::is_object) else {
            return error(400, "Expecting value: line 1 column 1 (char 0)");
        };
        let requested_model = body["model"].as_str().unwrap_or_default();
        // vLLM answers with the name the request used.
        let model = requested_model.to_string();
        if !names.iter().any(|name| *name == requested_model) {
            return unknown_model(requested_model);
        }
        if is_chat && !body["messages"].as_array().is_some_and(|m| !m.is_empty()) {
            return error(400, "messages: field required and must be a non-empty list");
        }
        if !is_chat && body.get("prompt").is_none() {
            return error(400, "prompt: field required");
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
                _ => return error(400, "max_tokens must be at least 1"),
            },
        };
        let completion_tokens = max_tokens.unwrap_or(0);
        if prompt_tokens + completion_tokens > CONTEXT_LENGTH {
            return error(400, &format!(
                "This model's maximum context length is {CONTEXT_LENGTH} tokens. However, you requested {} tokens ({prompt_tokens} in the messages, {completion_tokens} in the completion). Please reduce the length of the messages or completion.",
                prompt_tokens + completion_tokens
            ));
        }
        let n = body["n"].as_u64().unwrap_or(1);
        if !(1..=8).contains(&n) {
            return error(400, "n must be between 1 and 8");
        }
        if text.contains("MOCK:error500") {
            return error(500, "mock internal error");
        }
        if text.contains("MOCK:error400") {
            return error(400, "mock bad request");
        }
        if text.contains("MOCK:badjson") {
            return full(200, "{not json", "application/json");
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
            Duration::from_millis(self.options.token_ms)
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
        let words: Vec<&str> = (0..output_tokens)
            .map(|index| word(hash(&(text.as_str(), index))))
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
        if body["stream"].as_bool() != Some(true) {
            if hang {
                std::future::pending::<()>().await;
            }
            for index in 0..output_tokens {
                self.next_token(id, index, interval).await;
            }
            drop(guard);
            let content = words.join(" ");
            let choice = |index| {
                if is_chat {
                    json!({
                        "index": index, "logprobs": null, "finish_reason": finish_reason,
                        "stop_reason": null,
                        "message": {"role": "assistant", "content": content, "tool_calls": [], "reasoning_content": null}
                    })
                } else {
                    json!({"index": index, "text": content, "logprobs": null, "finish_reason": finish_reason, "stop_reason": null})
                }
            };
            let choices: Vec<Value> = (0..n).map(choice).collect();
            let completion = json!({
                "id": response_id,
                "object": if is_chat { "chat.completion" } else { "text_completion" },
                "created": created, "model": model, "choices": choices, "usage": usage,
                "system_fingerprint": null
            });
            return json_response(200, completion);
        }

        // The stream's frames, prepared up front; the task below paces them.
        let include_usage = body["stream_options"]["include_usage"].as_bool() == Some(true);
        let chunk = |choices: Vec<Value>, usage: Value| {
            let mut value = json!({
                "id": response_id,
                "object": if is_chat { "chat.completion.chunk" } else { "text_completion" },
                "created": created, "model": model, "choices": choices
            });
            if include_usage {
                value["usage"] = usage;
            }
            Bytes::from(format!("data: {value}\n\n"))
        };
        // Chat deltas have no `stop_reason` until the last one.
        let choices = |piece: &str, finish_reason: Option<&str>| {
            let choice = |index| match (is_chat, finish_reason) {
                (true, None) => {
                    json!({"index": index, "delta": {"content": piece}, "logprobs": null, "finish_reason": null})
                }
                (true, Some(reason)) => {
                    json!({"index": index, "delta": {"content": piece}, "logprobs": null, "finish_reason": reason, "stop_reason": null})
                }
                (false, reason) => {
                    json!({"index": index, "text": piece, "logprobs": null, "finish_reason": reason, "stop_reason": null})
                }
            };
            (0..n).map(choice).collect()
        };
        let role = |index| json!({"index": index, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null});
        let first = is_chat.then(|| chunk((0..n).map(role).collect(), Value::Null));
        let mut tokens = Vec::new();
        for (index, word) in words.iter().enumerate() {
            let piece = if index == 0 {
                word.to_string()
            } else {
                format!(" {word}")
            };
            tokens.push(chunk(choices(&piece, None), Value::Null));
        }
        let mut last = vec![chunk(choices("", Some(finish_reason)), Value::Null)];
        if include_usage {
            last.push(chunk(Vec::new(), usage));
        }
        last.push(Bytes::from_static(b"data: [DONE]\n\n"));

        let (mut sender, channel) = Channel::<Bytes, BoxError>::new(4);
        let engine = self.clone();
        tokio::spawn(async move {
            let _guard = guard;
            if let Some(first) = first {
                sender.send_data(first).await?;
            }
            if hang {
                // Empty frames send nothing, but fail once the client is gone.
                loop {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    sender.send_data(Bytes::new()).await?;
                }
            }
            for (index, token) in tokens.into_iter().enumerate() {
                engine.next_token(id, index as u64, interval).await;
                if drop_after == Some(index) {
                    sender.abort("mock dropped the connection".into());
                    return Ok(());
                }
                sender.send_data(token).await?;
            }
            for frame in last {
                sender.send_data(frame).await?;
            }
            Ok::<(), SendError>(())
        });
        response(200, "text/event-stream; charset=utf-8", channel.boxed())
    }
}

/// Ends the request's generation when its response is done or abandoned.
struct Guard {
    engine: Engine,
    id: Option<ProxyId>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            let mut state = self.engine.state.lock().unwrap();
            Engine::mark(&mut state, id, false, StopReason::Finished);
            state.generations.remove(&id);
            state.suspended.remove(&id);
        }
    }
}

fn prompt_text(body: &Value) -> String {
    let mut parts = Vec::new();
    for message in body["messages"].as_array().into_iter().flatten() {
        match &message["content"] {
            Value::String(text) => parts.push(text.as_str()),
            Value::Array(items) => parts.extend(items.iter().filter_map(|i| i["text"].as_str())),
            _ => {}
        }
    }
    match &body["prompt"] {
        Value::String(text) => parts.push(text),
        Value::Array(items) => parts.extend(items.iter().filter_map(Value::as_str)),
        _ => {}
    }
    if let Some(text) = body["text"].as_str() {
        parts.push(text);
    }
    parts.join("\n")
}

fn hash(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// The word of a token id or index, cycling through the vocabulary.
fn word(index: u64) -> &'static str {
    WORDS[(index % WORDS.len() as u64) as usize]
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

fn lossy(bytes: &[u8]) -> Value {
    Value::from(String::from_utf8_lossy(bytes))
}

fn response(status: u16, content_type: &str, body: Body) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", content_type)
        .body(body)
        .unwrap()
}

fn full(status: u16, body: impl Into<Bytes>, content_type: &str) -> Response<Body> {
    let body = Full::new(body.into()).map_err(|never| match never {});
    response(status, content_type, body.boxed())
}

fn json_response(status: u16, body: Value) -> Response<Body> {
    full(status, body.to_string(), "application/json")
}

fn error(status: u16, message: &str) -> Response<Body> {
    let kind = match status {
        400 => "BadRequestError",
        404 => "NotFoundError",
        _ => "InternalServerError",
    };
    json_response(
        status,
        json!({
            "error": {"message": message, "type": kind, "param": null, "code": status}
        }),
    )
}

#[tokio::main]
async fn main() {
    let options = Options::parse();
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
