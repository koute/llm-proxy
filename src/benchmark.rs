use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1 as client_http1;
use hyper::{Method, Request};
use hyper_util::rt::TokioIo;
use log::{info, warn};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpStream, UnixStream};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio::time::Instant;

use crate::config::{
    throughput_limit, BackendConfig, BenchmarkMeasurement, BenchmarkResult, GeneralConfig,
    QualityOfServiceConfig, RankConfig, Seconds, UpstreamAddress,
};
use crate::engine_protocol::{
    Command, CommandLine, EngineInfo, EngineKind, EngineLine, EngineStats, ProxyId, RequestPhase,
};

const STATS_POLL_INTERVAL: Duration = Duration::from_millis(20);
const WINDOW_SAMPLES: u32 = 50;
const RANK_PLACEHOLDER: &str = "{rank}";
const DECLINE_LIMIT: f64 = 0.5;
const WARM_UP_OUTPUT_TOKENS: usize = 16;
const CALIBRATION_OUTPUT_TOKENS: usize = 256;
const LEVELS: &[usize] = &[
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
];

#[derive(Clone, Debug, clap::Args)]
pub struct BenchmarkOptions {
    #[arg(
        long,
        value_parser = |url: &str| UpstreamAddress::try_from(url.to_string()),
        help = "Engine base URL, for example http://127.0.0.1:8000"
    )]
    pub upstream: UpstreamAddress,
    #[arg(
        long,
        help = "Engine control socket; {rank} is replaced with the data parallel rank"
    )]
    pub control_socket: String,
    #[arg(long, default_value = "config/backend.json")]
    pub output: PathBuf,
    #[arg(
        long,
        default_value = "config/general.json",
        help = "General configuration with minimum_peak_throughput and peak_throughput_tolerance for the search of the throughput limit; default values if the file does not exist"
    )]
    pub general: PathBuf,
    #[arg(
        long,
        help = "Model name for the requests; default: the engine's model"
    )]
    pub model: Option<String>,
    #[arg(long, default_value_t = 1000)]
    pub prompt_tokens: usize,
    #[arg(long, default_value_t = 512)]
    pub output_tokens: usize,
    #[arg(long, help = "Highest number of parallel requests per rank to measure")]
    pub max_parallel_requests_per_rank: Option<usize>,
    #[arg(
        long = "level-seconds",
        default_value = "10",
        help = "Measurement window per level; it starts when every request slot has completed its first request"
    )]
    pub level_duration: Seconds,
}

struct ControlClient {
    reader: BufReader<OwnedReadHalf>,
    writer: OwnedWriteHalf,
    next_seq: u64,
}

struct Rank {
    socket: PathBuf,
    info: EngineInfo,
    control: ControlClient,
}

struct Workload {
    upstream: UpstreamAddress,
    engine: EngineKind,
    model: String,
    prompt: String,
    instance: u64,
    sequence: AtomicU64,
}

struct Sample {
    time: Instant,
    output_tokens: HashMap<ProxyId, u64>,
    running: usize,
}

impl ControlClient {
    async fn connect(socket: &Path) -> Result<ControlClient, String> {
        let stream = UnixStream::connect(socket).await.map_err(|error| {
            format!(
                "failed to connect to the engine control socket '{}': {error}",
                socket.display()
            )
        })?;
        let (reader, writer) = stream.into_split();
        Ok(ControlClient {
            reader: BufReader::new(reader),
            writer,
            next_seq: 0,
        })
    }

    async fn call(&mut self, command: Command) -> Result<Value, String> {
        let seq = self.next_seq;
        self.next_seq += 1;
        let line = CommandLine { seq, command }.encode();
        self.writer
            .write_all(&line)
            .await
            .map_err(|error| format!("failed to write to the engine control socket: {error}"))?;
        loop {
            let mut text = String::new();
            let length = self.reader.read_line(&mut text).await.map_err(|error| {
                format!("failed to read from the engine control socket: {error}")
            })?;
            if length == 0 {
                return Err("the engine closed the control socket".to_string());
            }
            match EngineLine::parse(text.trim_end().as_bytes())? {
                EngineLine::Success { seq: reply, result } if reply == seq => return Ok(result),
                EngineLine::Failure { seq: reply, error } if reply == seq => {
                    return Err(format!("the engine rejected the command: {error}"))
                }
                _ => {}
            }
        }
    }

    async fn info(&mut self) -> Result<EngineInfo, String> {
        serde_json::from_value(self.call(Command::Info).await?)
            .map_err(|error| format!("invalid info reply: {error}"))
    }

    async fn stats(&mut self) -> Result<EngineStats, String> {
        serde_json::from_value(self.call(Command::Stats).await?)
            .map_err(|error| format!("invalid stats reply: {error}"))
    }
}

impl Workload {
    fn request(
        &self,
        output_tokens: usize,
        data_parallel_rank: Option<usize>,
    ) -> Request<Full<Bytes>> {
        let id = ProxyId {
            instance: self.instance,
            sequence: self.sequence.fetch_add(1, Ordering::Relaxed),
        };
        let mut body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": format!("{id} {}", self.prompt)}],
            "max_tokens": output_tokens,
            "min_tokens": output_tokens,
            "ignore_eos": true,
            "temperature": 0.0,
            "stream": true,
        });
        let mut request = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("host", self.upstream.authority())
            .header("content-type", "application/json")
            .body(())
            .expect("invalid benchmark request");
        self.engine.label_request(
            request.headers_mut(),
            body.as_object_mut().expect("non-object benchmark body"),
            id,
            data_parallel_rank,
        );
        request.map(|()| Full::new(Bytes::from(body.to_string())))
    }

    async fn send(&self, request: Request<Full<Bytes>>) -> Result<(), String> {
        let upstream = &self.upstream;
        let stream = TcpStream::connect(upstream.authority())
            .await
            .map_err(|error| format!("failed to connect to {upstream}: {error}"))?;
        let (mut sender, connection) = client_http1::handshake(TokioIo::new(stream))
            .await
            .map_err(|error| format!("HTTP handshake with {upstream} failed: {error}"))?;
        let exchange = async {
            let response = sender
                .send_request(request)
                .await
                .map_err(|error| format!("request to {upstream} failed: {error}"))?;
            let status = response.status();
            let body = response
                .into_body()
                .collect()
                .await
                .map_err(|error| format!("response from {upstream} failed: {error}"))?
                .to_bytes();
            if !status.is_success() {
                warn!(
                    "benchmark request failed with {status}: {}",
                    String::from_utf8_lossy(&body)
                );
                return Err(format!("the engine returned {status}"));
            }
            Ok(())
        };
        tokio::select! {
            result = exchange => result,
            Err(error) = connection => Err(format!("connection to {upstream} failed: {error}")),
        }
    }

    fn is_own(&self, id: &ProxyId) -> bool {
        id.instance == self.instance
    }
}

pub async fn run(options: BenchmarkOptions) -> Result<BackendConfig, String> {
    let quality_of_service = quality_of_service_settings(&options.general)?;
    let mut ranks = discover_ranks(&options.control_socket).await?;
    let engine = ranks[0].info.engine;
    let model = options
        .model
        .clone()
        .unwrap_or_else(|| ranks[0].info.model.clone());
    let engine_max = ranks
        .iter()
        .map(|rank| rank.info.max_running_requests)
        .min()
        .unwrap_or(1);
    let level_limit = options
        .max_parallel_requests_per_rank
        .map_or(engine_max, |limit| limit.min(engine_max));
    info!(
        "benchmark: {engine} '{model}', {} rank(s), engine limit {engine_max} parallel requests per rank",
        ranks.len()
    );
    let mut workload = Workload {
        upstream: options.upstream.clone(),
        engine,
        model: model.clone(),
        prompt: String::new(),
        instance: u64::from(std::process::id()),
        sequence: AtomicU64::new(0),
    };
    let prompt_tokens =
        calibrate_prompt(&mut workload, &mut ranks[0], options.prompt_tokens).await?;
    let workload = Arc::new(workload);

    let mut warm_up = JoinSet::new();
    for rank in &ranks {
        let workload = workload.clone();
        let request = workload.request(WARM_UP_OUTPUT_TOKENS, rank.info.data_parallel_rank);
        warm_up.spawn(async move { workload.send(request).await });
    }
    while let Some(result) = warm_up.join_next().await {
        result.map_err(|error| format!("warm-up request failed: {error}"))??;
    }

    let tokens_per_request = prompt_tokens as usize + options.output_tokens;
    let kv_cache_tokens = ranks
        .iter()
        .map(|rank| rank.info.kv_cache_tokens)
        .min()
        .unwrap_or_default();
    let kv_cache_limit = (kv_cache_tokens / tokens_per_request).max(1);
    let sustainable_max = kv_cache_limit.min(engine_max);
    info!(
        "benchmark: {tokens_per_request} prompt and output tokens per request; the KV cache contains {kv_cache_limit} requests per rank"
    );
    println!(
        "prompt: {prompt_tokens} tokens (requested {})",
        options.prompt_tokens
    );
    println!(
        "{:>22} {:>22} {:>24} {:>22}",
        "requests per rank", "running per rank", "tokens/s per request", "tokens/s per rank"
    );

    let mut measurements: Vec<BenchmarkMeasurement> = Vec::new();
    let mut declining_levels = 0;
    let level_limit = level_limit.min(sustainable_max);
    let levels = LEVELS
        .iter()
        .copied()
        .filter(|level| *level < level_limit)
        .chain([level_limit]);
    for level in levels {
        let measurement = measure_level(&options, &workload, &mut ranks, level).await?;
        let peak = measurements
            .iter()
            .map(|measurement| measurement.tokens_per_second_per_rank)
            .fold(measurement.tokens_per_second_per_rank, f64::max);
        declining_levels = if measurement.tokens_per_second_per_rank < peak * DECLINE_LIMIT {
            declining_levels + 1
        } else {
            0
        };
        measurements.push(measurement);
        if declining_levels == 2 {
            info!("benchmark: throughput fell below {DECLINE_LIMIT} of the peak twice; stopping");
            break;
        }
    }
    let throughput_cap = loop {
        let cap = throughput_limit(
            &measurements,
            quality_of_service.minimum_peak_throughput,
            quality_of_service.peak_throughput_tolerance,
        );
        let levels = measurements
            .iter()
            .map(|measurement| measurement.parallel_requests_per_rank);
        let lower = levels.clone().filter(|level| *level <= cap).max();
        let upper = levels.filter(|level| *level > cap).min();
        let (Some(lower), Some(upper)) = (lower, upper) else {
            break cap;
        };
        if upper - lower <= 1 {
            break cap;
        }
        let level = (lower + upper) / 2;
        let measurement = measure_level(&options, &workload, &mut ranks, level).await?;
        let index =
            measurements.partition_point(|existing| existing.parallel_requests_per_rank < level);
        measurements.insert(index, measurement);
    };
    info!(
        "benchmark: throughput limit {throughput_cap} parallel requests per rank (minimum_peak_throughput {}, peak_throughput_tolerance {})",
        quality_of_service.minimum_peak_throughput, quality_of_service.peak_throughput_tolerance
    );
    let single_stream_tokens_per_second = measurements
        .first()
        .map(|measurement| measurement.tokens_per_second_per_request)
        .ok_or("no level produced a measurement")?;

    let backend = BackendConfig {
        engine,
        model,
        engine_max_parallel_requests_per_rank: sustainable_max,
        priority_scheduling: ranks.iter().all(|rank| rank.info.priority_scheduling),
        ranks: ranks
            .iter()
            .map(|rank| RankConfig {
                upstream: options.upstream.clone(),
                control_socket: rank.socket.clone(),
                data_parallel_rank: (ranks.len() > 1)
                    .then_some(rank.info.data_parallel_rank)
                    .flatten(),
            })
            .collect(),
        benchmark: Some(BenchmarkResult {
            prompt_tokens: prompt_tokens as usize,
            output_tokens: options.output_tokens,
            single_stream_tokens_per_second,
            measurements,
        }),
    };
    let text = backend.to_json();
    BackendConfig::parse(&text)
        .map_err(|error| format!("generated an invalid backend file: {error}"))?;
    std::fs::write(&options.output, text)
        .map_err(|error| format!("failed to write '{}': {error}", options.output.display()))?;
    info!("benchmark: wrote '{}'", options.output.display());
    Ok(backend)
}

fn quality_of_service_settings(path: &Path) -> Result<QualityOfServiceConfig, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            info!(
                "benchmark: '{}' does not exist; the search of the throughput limit uses the default thresholds",
                path.display()
            );
            return Ok(QualityOfServiceConfig::default());
        }
        Err(error) => return Err(format!("failed to read '{}': {error}", path.display())),
    };
    GeneralConfig::parse(&text)
        .map(|general| general.quality_of_service)
        .map_err(|error| format!("invalid '{}': {error}", path.display()))
}

async fn discover_ranks(template: &str) -> Result<Vec<Rank>, String> {
    let sockets: Vec<PathBuf> = if template.contains(RANK_PLACEHOLDER) {
        (0..)
            .map(|rank: usize| PathBuf::from(template.replace(RANK_PLACEHOLDER, &rank.to_string())))
            .take_while(|socket| socket.exists())
            .collect()
    } else {
        vec![PathBuf::from(template)]
    };
    if sockets.is_empty() {
        return Err(format!("no engine control socket matches '{template}'"));
    }
    let mut ranks = Vec::with_capacity(sockets.len());
    for socket in sockets {
        let mut control = ControlClient::connect(&socket).await?;
        let info = control.info().await?;
        ranks.push(Rank {
            socket,
            info,
            control,
        });
    }
    Ok(ranks)
}

fn prompt_text(words: usize) -> String {
    (0..words)
        .map(|index| format!("w{}", index % 89))
        .collect::<Vec<_>>()
        .join(" ")
}

async fn calibrate_prompt(
    workload: &mut Workload,
    rank: &mut Rank,
    target_tokens: usize,
) -> Result<u64, String> {
    let full_words = target_tokens.max(1);
    let half_words = full_words / 2;
    workload.prompt = prompt_text(full_words);
    let full_tokens = measure_prompt_tokens(workload, rank).await?;
    workload.prompt = prompt_text(half_words);
    let half_tokens = measure_prompt_tokens(workload, rank).await?;
    if full_tokens <= half_tokens {
        return Err(format!(
            "the engine counts {full_tokens} prompt tokens for {full_words} words and {half_tokens} for {half_words} words"
        ));
    }
    let tokens_per_word = (full_tokens - half_tokens) as f64 / (full_words - half_words) as f64;
    let template_tokens = full_tokens as f64 - tokens_per_word * full_words as f64;
    let words = ((target_tokens as f64 - template_tokens) / tokens_per_word).round();
    workload.prompt = prompt_text(words.max(0.0) as usize);
    measure_prompt_tokens(workload, rank).await
}

async fn measure_prompt_tokens(workload: &Workload, rank: &mut Rank) -> Result<u64, String> {
    let request = workload.request(CALIBRATION_OUTPUT_TOKENS, rank.info.data_parallel_rank);
    let mut exchange = Box::pin(workload.send(request));
    let prompt_tokens = loop {
        tokio::select! {
            result = &mut exchange => {
                result?;
                return Err("the engine completed a calibration request before reporting it in the statistics".to_string());
            }
            () = tokio::time::sleep(STATS_POLL_INTERVAL) => {}
        }
        let requests = rank.control.stats().await?.requests;
        if let Some(request) = requests.iter().find(|request| workload.is_own(&request.id)) {
            break request.prompt_tokens;
        }
    };
    drop(exchange);
    wait_until_idle(workload, std::slice::from_mut(rank)).await?;
    Ok(prompt_tokens)
}

async fn wait_until_idle(workload: &Workload, ranks: &mut [Rank]) -> Result<(), String> {
    loop {
        let mut is_idle = true;
        for rank in ranks.iter_mut() {
            let requests = rank.control.stats().await?.requests;
            is_idle &= !requests.iter().any(|request| workload.is_own(&request.id));
        }
        if is_idle {
            return Ok(());
        }
        tokio::time::sleep(STATS_POLL_INTERVAL).await;
    }
}

async fn measure_level(
    options: &BenchmarkOptions,
    workload: &Arc<Workload>,
    ranks: &mut [Rank],
    level: usize,
) -> Result<BenchmarkMeasurement, String> {
    let output_tokens = options.output_tokens;
    let (first_completion_sender, mut first_completions) = mpsc::unbounded_channel();
    let mut slots = JoinSet::new();
    for rank in ranks.iter() {
        for slot in 0..level {
            let workload = workload.clone();
            let first_completion_sender = first_completion_sender.clone();
            let first_output_tokens = (output_tokens * (slot + 1)).div_ceil(level);
            let data_parallel_rank = rank.info.data_parallel_rank;
            slots.spawn(async move {
                let send = |tokens| workload.send(workload.request(tokens, data_parallel_rank));
                if let Err(error) = send(first_output_tokens).await {
                    return error;
                }
                let _ = first_completion_sender.send(());
                loop {
                    if let Err(error) = send(output_tokens).await {
                        return error;
                    }
                }
            });
        }
    }
    let slot_failure = |result: Result<String, tokio::task::JoinError>| {
        result.unwrap_or_else(|error| format!("benchmark request task failed: {error}"))
    };
    let mut pending_first_completions = level * ranks.len();
    while pending_first_completions > 0 {
        tokio::select! {
            Some(()) = first_completions.recv() => pending_first_completions -= 1,
            Some(result) = slots.join_next() => return Err(slot_failure(result)),
        }
    }

    let window = options.level_duration.duration();
    let window_end = Instant::now() + window;
    let mut samples: Vec<Sample> = Vec::new();
    loop {
        let mut sample = Sample {
            time: Instant::now(),
            output_tokens: HashMap::new(),
            running: 0,
        };
        for rank in ranks.iter_mut() {
            let requests = rank.control.stats().await?.requests;
            for request in requests
                .iter()
                .filter(|request| workload.is_own(&request.id))
            {
                if request.state == RequestPhase::Running {
                    sample.running += 1;
                }
                sample
                    .output_tokens
                    .insert(request.id, request.output_tokens);
            }
        }
        samples.push(sample);
        if Instant::now() >= window_end {
            break;
        }
        tokio::select! {
            () = tokio::time::sleep(window / WINDOW_SAMPLES) => {}
            Some(result) = slots.join_next() => return Err(slot_failure(result)),
        }
    }
    slots.shutdown().await;
    wait_until_idle(workload, ranks).await?;

    let no_measurement = || format!("no measurement at {level} parallel requests per rank");
    let (Some(first), Some(last)) = (samples.first(), samples.last()) else {
        return Err(no_measurement());
    };
    let seconds = last.time.duration_since(first.time).as_secs_f64();
    if seconds <= 0.0 {
        return Err(no_measurement());
    }
    let mean_running =
        samples.iter().map(|sample| sample.running).sum::<usize>() as f64 / samples.len() as f64;
    let tokens_at = |sample: &Sample, id: &ProxyId| -> u64 {
        let is_finished = samples
            .iter()
            .take_while(|earlier| earlier.time < sample.time)
            .any(|earlier| earlier.output_tokens.contains_key(id));
        match sample.output_tokens.get(id) {
            Some(tokens) => *tokens,
            None if is_finished => output_tokens as u64,
            None => 0,
        }
    };
    let request_ids: HashSet<&ProxyId> = samples
        .iter()
        .flat_map(|sample| sample.output_tokens.keys())
        .collect();
    let window_tokens: u64 = request_ids
        .into_iter()
        .map(|id| tokens_at(last, id).saturating_sub(tokens_at(first, id)))
        .sum();
    let rank_rate = window_tokens as f64 / seconds / ranks.len() as f64;
    let measurement = BenchmarkMeasurement {
        parallel_requests_per_rank: level,
        tokens_per_second_per_request: rank_rate / level as f64,
        tokens_per_second_per_rank: rank_rate,
    };
    println!(
        "{:>22} {:>22.1} {:>24.1} {:>22.1}",
        level,
        mean_running / ranks.len() as f64,
        measurement.tokens_per_second_per_request,
        measurement.tokens_per_second_per_rank
    );
    Ok(measurement)
}
