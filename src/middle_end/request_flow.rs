use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use bytes::{Buf, Bytes, BytesMut};
use http::header::{HeaderValue, CONTENT_LENGTH, CONTENT_TYPE, TRANSFER_ENCODING};
use http::StatusCode;
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::net::{TcpStream, UnixStream};
use tokio::sync::oneshot;
use tokio::time::Instant;

use super::routing::{
    engine_request, error_response, route, GeneratingRequest, Rejection, Route, UpstreamRequest,
};
use super::scheduler::{handed_over_class, Registration, Scheduler, StartedRequest};
use super::scheduling_log::{Event, RequestDescription, SchedulingLog};
use super::{quiesce_deadline, Quiesce};
use crate::config::{BackendConfig, Config};
use crate::engine_protocol::ProxyId;
use crate::http1::{without_transport_headers, BodyDecoder, Decoded, RequestHead, ResponseHead};
use crate::quality_of_service::{Rank, RequestId, UserId};
use crate::utils::{
    base64_bytes, request_head_serde, BufferedConnection, ConnectionSnapshot, ReadResult, Timestamp,
};
use crate::wire::{decode_frame, encode_body_frames, encode_frame, request_end_offset, FrameKind};

const HEAD_ALLOWANCE: usize = 1024 * 1024;
const FRONTEND_WRITE_TIMEOUT: Duration = Duration::from_secs(60);
const EVENT_STREAM_COMMENT: &[u8] = b": keepalive\n\n";
const JSON_WHITESPACE: &[u8] = b"\n";
const MAX_RESPONSE_HEAD: usize = 1024 * 1024;
const FRONTEND_HIGH_WATER: usize = 1024 * 1024;

pub struct FlowContext {
    pub config: Arc<Config>,
    pub scheduler: Scheduler,
    pub scheduling_log: SchedulingLog,
}

impl FlowContext {
    fn backend(&self) -> Option<&BackendConfig> {
        self.config.backend.as_ref()
    }

    fn keepalive_interval(&self) -> Duration {
        self.config
            .general
            .limits
            .keepalive_interval_seconds
            .duration()
    }
}

pub struct RequestFlow {
    frontend: BufferedConnection<UnixStream>,
    stage: Stage,
}

enum Stage {
    Receiving {
        deadline: Timestamp,
    },
    Queued {
        request: GeneratingRequest,
        proxy_id: ProxyId,
        permit: oneshot::Receiver<Rank>,
        registration: Registration,
        early_response: EarlyResponse,
    },
    Exchanging {
        exchange: Exchange,
        registration: Option<Registration>,
    },
    Closing,
}

enum Progress {
    Continue,
    Finished,
}

impl RequestFlow {
    pub fn new(stream: UnixStream, context: &FlowContext) -> RequestFlow {
        let timeout = context
            .config
            .general
            .limits
            .request_body_timeout_seconds
            .duration();
        RequestFlow {
            frontend: BufferedConnection::new(stream),
            stage: Stage::Receiving {
                deadline: Timestamp::now() + timeout,
            },
        }
    }

    pub async fn run(mut self, context: &FlowContext, mut quiesce: Quiesce) -> Option<RequestFlow> {
        loop {
            let progress = tokio::select! {
                biased;
                deadline = quiesce_deadline(&mut quiesce) => {
                    self.flush_until(deadline).await;
                    return Some(self);
                }
                progress = self.step(context) => progress,
            };
            if let Progress::Finished = progress {
                return None;
            }
        }
    }

    async fn flush_until(&mut self, deadline: Instant) {
        let flush = async {
            if let Stage::Exchanging { exchange, .. } = &mut self.stage {
                if let Some(upstream) = exchange.upstream() {
                    let _ = upstream.flush().await;
                }
            }
            let _ = self.frontend.flush().await;
        };
        let _ = tokio::time::timeout_at(deadline, flush).await;
    }

    async fn step(&mut self, context: &FlowContext) -> Progress {
        match &mut self.stage {
            Stage::Receiving { deadline } => {
                let deadline = *deadline;
                self.receive(context, deadline).await
            }
            Stage::Queued {
                permit,
                early_response,
                ..
            } => {
                let frontend = &mut self.frontend;
                let keepalive_at =
                    Some(early_response.keepalive_at).filter(|_| !frontend.has_pending_writes());
                tokio::select! {
                    rank = permit => {
                        match rank {
                            Ok(rank) => self.start(context, rank),
                            Err(_) if early_response.is_head_sent() => return Progress::Finished,
                            Err(_) => self.respond(StatusCode::SERVICE_UNAVAILABLE, "request removed from the queue"),
                        }
                        Progress::Continue
                    }
                    result = frontend.writable(), if frontend.has_pending_writes() => {
                        match result.and_then(|()| frontend.write_available()) {
                            Ok(()) => Progress::Continue,
                            Err(_) => Progress::Finished,
                        }
                    }
                    result = frontend.readable() => match result.and_then(|()| frontend.read_available()) {
                        Ok(ReadResult::NotReady) => Progress::Continue,
                        _ => Progress::Finished,
                    },
                    _ = tokio::time::sleep_until(keepalive_at.map_or_else(Instant::now, Timestamp::instant)), if keepalive_at.is_some() => {
                        early_response.send(&mut frontend.write_buffer, context.keepalive_interval());
                        Progress::Continue
                    }
                }
            }
            Stage::Exchanging {
                exchange,
                registration,
            } => {
                match exchange
                    .step(&mut self.frontend, context.keepalive_interval())
                    .await
                {
                    ExchangeProgress::Continue => Progress::Continue,
                    ExchangeProgress::Complete => {
                        if let Some(registration) = registration.take() {
                            registration.complete(exchange.engine_status());
                        }
                        self.stage = Stage::Closing;
                        Progress::Continue
                    }
                    ExchangeProgress::Aborted => Progress::Finished,
                }
            }
            Stage::Closing => {
                if !self.frontend.has_pending_writes() {
                    return Progress::Finished;
                }
                let frontend = &mut self.frontend;
                let write = async {
                    tokio::select! {
                        result = frontend.writable() => match result.and_then(|()| frontend.write_available()) {
                            Ok(()) => Progress::Continue,
                            Err(_) => Progress::Finished,
                        },
                        result = frontend.readable() => match result.and_then(|()| frontend.read_available()) {
                            Ok(ReadResult::NotReady) => Progress::Continue,
                            _ => Progress::Finished,
                        },
                    }
                };
                tokio::time::timeout(FRONTEND_WRITE_TIMEOUT, write)
                    .await
                    .unwrap_or(Progress::Finished)
            }
        }
    }

    async fn receive(&mut self, context: &FlowContext, deadline: Timestamp) -> Progress {
        let limits = &context.config.general.limits;
        match complete_request(&mut self.frontend.read_buffer) {
            Ok(Some((head, body))) => {
                self.dispatch(context, head, body);
                return Progress::Continue;
            }
            Ok(None) => {}
            Err(error) => {
                debug!("invalid request from the frontend: {error}");
                return Progress::Finished;
            }
        }
        if self.frontend.read_buffer.len() > limits.max_request_body_bytes + HEAD_ALLOWANCE {
            self.respond(StatusCode::PAYLOAD_TOO_LARGE, "request body too large");
            return Progress::Continue;
        }
        tokio::select! {
            result = self.frontend.fill() => match result {
                Ok(ReadResult::Data) => Progress::Continue,
                _ => Progress::Finished,
            },
            _ = tokio::time::sleep_until(deadline.instant()) => {
                self.respond(StatusCode::REQUEST_TIMEOUT, "request body read timed out");
                Progress::Continue
            }
        }
    }

    fn dispatch(&mut self, context: &FlowContext, head: RequestHead, body: Bytes) {
        let body_bytes = body.len();
        match route(&context.config, head, body) {
            Route::Rejection(rejection) => self.reject(context, rejection),
            Route::Passthrough(upstream_request) => {
                self.stage = Stage::Exchanging {
                    exchange: Exchange::new(upstream_request, None),
                    registration: None,
                };
            }
            Route::Generating(request) => {
                let description = RequestDescription::new(&request, body_bytes);
                match context.scheduler.admit(
                    request.user.clone(),
                    request.class.clone(),
                    &description,
                ) {
                    Ok(admission) => {
                        let early_response = EarlyResponse::new(
                            request.is_streaming(),
                            context.keepalive_interval(),
                        );
                        self.stage = Stage::Queued {
                            request,
                            proxy_id: admission.proxy_id,
                            permit: admission.permit,
                            registration: admission.registration,
                            early_response,
                        };
                    }
                    Err(_) => {
                        let rejection = Rejection {
                            status: StatusCode::SERVICE_UNAVAILABLE,
                            message: "too many queued requests; try again later".to_string(),
                            method: request.head.method.to_string(),
                            path: description.path,
                            account: Some(request.user.account().to_string()),
                            client: format!("user {}", request.user),
                        };
                        self.reject(context, rejection);
                    }
                }
            }
        }
    }

    fn reject(&mut self, context: &FlowContext, rejection: Rejection) {
        warn!(
            "rejected {} {} with {}: {}; {}",
            rejection.method, rejection.path, rejection.status, rejection.message, rejection.client
        );
        context.scheduling_log.record(Event::Rejected {
            status: rejection.status.as_u16(),
            reason: &rejection.message,
            method: &rejection.method,
            path: &rejection.path,
            account: rejection.account.as_deref(),
        });
        self.respond(rejection.status, &rejection.message);
    }

    fn start(&mut self, context: &FlowContext, rank: Rank) {
        let Stage::Queued {
            request,
            proxy_id,
            registration,
            early_response,
            ..
        } = std::mem::replace(&mut self.stage, Stage::Closing)
        else {
            unreachable!("start outside of the queue stage");
        };
        let backend = context
            .backend()
            .expect("generating request without backend");
        let upstream_request = engine_request(backend, rank, &request, proxy_id);
        self.stage = Stage::Exchanging {
            exchange: Exchange::new(upstream_request, Some(early_response)),
            registration: Some(registration),
        };
    }

    fn respond(&mut self, status: StatusCode, message: &str) {
        let (head, body) = error_response(status, message);
        write_response(&mut self.frontend, head, body);
        self.stage = Stage::Closing;
    }
}

fn write_response(frontend: &mut BufferedConnection<UnixStream>, head: ResponseHead, body: Bytes) {
    let output = &mut frontend.write_buffer;
    encode_frame(FrameKind::ResponseHead, &head.encode(), output);
    encode_body_frames(FrameKind::ResponseBody, &body, output);
    encode_frame(FrameKind::ResponseEnd, b"", output);
}

fn write_error(frontend: &mut BufferedConnection<UnixStream>, status: StatusCode, message: &str) {
    let (head, body) = error_response(status, message);
    write_response(frontend, head, body);
}

fn complete_request(buffer: &mut BytesMut) -> Result<Option<(RequestHead, Bytes)>, String> {
    let Some(request_end) = request_end_offset(buffer)? else {
        return Ok(None);
    };
    let mut frames = buffer.split_to(request_end);
    let mut head = None;
    let mut body = BytesMut::new();
    while let Some(frame) = decode_frame(&mut frames)? {
        match frame.kind {
            FrameKind::RequestHead => {
                head = RequestHead::parse(&frame.payload)?.map(|(head, _)| head);
            }
            FrameKind::RequestBody => body.extend_from_slice(&frame.payload),
            FrameKind::RequestEnd => {}
            other => return Err(format!("unexpected frame {other:?} from the frontend")),
        }
    }
    let head = head.ok_or("request without a complete head")?;
    Ok(Some((head, body.freeze())))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EarlyResponse {
    is_event_stream: bool,
    is_head_sent: bool,
    pub keepalive_at: Timestamp,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EventStreamKeepalive {
    pub keepalive_at: Timestamp,
    trailing_newlines: usize,
}

impl EarlyResponse {
    pub fn new(is_event_stream: bool, interval: Duration) -> EarlyResponse {
        EarlyResponse {
            is_event_stream,
            is_head_sent: false,
            keepalive_at: Timestamp::now() + interval,
        }
    }

    pub fn is_head_sent(&self) -> bool {
        self.is_head_sent
    }

    pub fn send(&mut self, output: &mut BytesMut, interval: Duration) {
        let (content_type, padding) = if self.is_event_stream {
            ("text/event-stream", EVENT_STREAM_COMMENT)
        } else {
            ("application/json", JSON_WHITESPACE)
        };
        if !self.is_head_sent {
            let mut head = ResponseHead::new(StatusCode::OK);
            head.headers
                .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
            encode_frame(FrameKind::ResponseHead, &head.encode(), output);
            self.is_head_sent = true;
        }
        encode_frame(FrameKind::ResponseBody, padding, output);
        self.keepalive_at = Timestamp::now() + interval;
    }
}

impl EventStreamKeepalive {
    pub fn new(interval: Duration) -> EventStreamKeepalive {
        EventStreamKeepalive {
            keepalive_at: Timestamp::now() + interval,
            trailing_newlines: 2,
        }
    }

    pub fn record(&mut self, data: &[u8], interval: Duration) {
        let significant = data.iter().rev().filter(|byte| **byte != b'\r');
        let newlines = significant
            .clone()
            .take_while(|byte| **byte == b'\n')
            .count();
        let is_only_newlines = newlines == significant.count();
        self.trailing_newlines = if is_only_newlines {
            self.trailing_newlines + newlines
        } else {
            newlines
        };
        self.keepalive_at = Timestamp::now() + interval;
    }

    pub fn send(&mut self, output: &mut BytesMut, interval: Duration) {
        if self.trailing_newlines >= 2 {
            encode_frame(FrameKind::ResponseBody, EVENT_STREAM_COMMENT, output);
        }
        self.keepalive_at = Timestamp::now() + interval;
    }
}

pub struct Exchange {
    authority: String,
    is_head_request: bool,
    upstream: Upstream,
    response: ResponsePhase,
    write_stall_deadline: Option<Timestamp>,
}

enum Upstream {
    Unconnected(Bytes),
    Connected(BufferedConnection<TcpStream>),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum ResponsePhase {
    Head {
        early_response: Option<EarlyResponse>,
    },
    Body {
        decoder: BodyDecoder,
        keepalive: Option<EventStreamKeepalive>,
        #[serde(default)]
        status: Option<u16>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExchangeSnapshot {
    authority: String,
    is_head_request: bool,
    upstream: UpstreamSnapshot,
    response: ResponsePhase,
    write_stall_deadline: Option<Timestamp>,
}

#[derive(Debug, Serialize, Deserialize)]
enum UpstreamSnapshot {
    Unconnected(#[serde(with = "base64_bytes")] Bytes),
    Connected(ConnectionSnapshot),
}

pub enum ExchangeProgress {
    Continue,
    Complete,
    Aborted,
}

impl Exchange {
    pub fn new(request: UpstreamRequest, early_response: Option<EarlyResponse>) -> Exchange {
        Exchange {
            authority: request.authority,
            is_head_request: request.is_head_request,
            upstream: Upstream::Unconnected(request.bytes),
            response: ResponsePhase::Head { early_response },
            write_stall_deadline: None,
        }
    }

    pub fn engine_status(&self) -> Option<u16> {
        match self.response {
            ResponsePhase::Body { status, .. } => status,
            ResponsePhase::Head { .. } => None,
        }
    }

    pub fn upstream(&mut self) -> Option<&mut BufferedConnection<TcpStream>> {
        match &mut self.upstream {
            Upstream::Connected(connection) => Some(connection),
            Upstream::Unconnected(_) => None,
        }
    }

    pub async fn step(
        &mut self,
        frontend: &mut BufferedConnection<UnixStream>,
        keepalive_interval: Duration,
    ) -> ExchangeProgress {
        let Exchange {
            authority,
            is_head_request,
            upstream,
            response,
            write_stall_deadline,
        } = self;
        let connection = match upstream {
            Upstream::Unconnected(request) => {
                return match TcpStream::connect(authority.as_str()).await {
                    Ok(stream) => {
                        let mut connection = BufferedConnection::new(stream);
                        connection.write_buffer.extend_from_slice(request);
                        *upstream = Upstream::Connected(connection);
                        ExchangeProgress::Continue
                    }
                    Err(error) => {
                        warn!("failed to connect to the engine at {authority}: {error}");
                        if !response.is_head_sent() {
                            write_error(frontend, StatusCode::BAD_GATEWAY, "engine unreachable");
                        }
                        ExchangeProgress::Complete
                    }
                };
            }
            Upstream::Connected(connection) => connection,
        };
        match response.process(
            *is_head_request,
            &mut connection.read_buffer,
            frontend,
            keepalive_interval,
        ) {
            Ok(true) => return ExchangeProgress::Complete,
            Ok(false) => {}
            Err(error) => return response.fail(authority, frontend, &error),
        }
        let keepalive_at = response
            .keepalive_at()
            .filter(|_| !frontend.has_pending_writes());
        if !frontend.has_pending_writes() {
            *write_stall_deadline = None;
        }
        let stall_deadline = frontend.has_pending_writes().then(|| {
            *write_stall_deadline.get_or_insert_with(|| Timestamp::now() + FRONTEND_WRITE_TIMEOUT)
        });
        tokio::select! {
            result = connection.readable(), if frontend.write_buffer.len() < FRONTEND_HIGH_WATER => {
                match result.and_then(|()| connection.read_available()) {
                    Ok(ReadResult::EndOfStream) => response.end_of_upstream(authority, frontend),
                    Ok(_) => ExchangeProgress::Continue,
                    Err(error) => response.fail(authority, frontend, &format!("engine connection failed: {error}")),
                }
            }
            result = connection.writable(), if connection.has_pending_writes() => {
                match result.and_then(|()| connection.write_available()) {
                    Ok(()) => ExchangeProgress::Continue,
                    Err(error) => response.fail(authority, frontend, &format!("engine connection failed: {error}")),
                }
            }
            result = frontend.writable(), if frontend.has_pending_writes() => {
                let pending = frontend.write_buffer.len();
                match result.and_then(|()| frontend.write_available()) {
                    Ok(()) => {
                        if frontend.write_buffer.len() < pending {
                            *write_stall_deadline = None;
                        }
                        ExchangeProgress::Continue
                    }
                    Err(_) => ExchangeProgress::Aborted,
                }
            }
            result = frontend.readable() => match result.and_then(|()| frontend.read_available()) {
                Ok(ReadResult::NotReady) => ExchangeProgress::Continue,
                _ => ExchangeProgress::Aborted,
            },
            _ = tokio::time::sleep_until(stall_deadline.map_or_else(Instant::now, Timestamp::instant)), if stall_deadline.is_some() => {
                warn!("the client did not read the response for {FRONTEND_WRITE_TIMEOUT:?}; aborting the request");
                ExchangeProgress::Aborted
            }
            _ = tokio::time::sleep_until(keepalive_at.map_or_else(Instant::now, Timestamp::instant)), if keepalive_at.is_some() => {
                response.send_keepalive(&mut frontend.write_buffer, keepalive_interval);
                ExchangeProgress::Continue
            }
        }
    }

    pub fn snapshot(self) -> io::Result<ExchangeSnapshot> {
        Ok(ExchangeSnapshot {
            authority: self.authority,
            is_head_request: self.is_head_request,
            upstream: match self.upstream {
                Upstream::Unconnected(request) => UpstreamSnapshot::Unconnected(request),
                Upstream::Connected(connection) => {
                    UpstreamSnapshot::Connected(connection.snapshot()?)
                }
            },
            response: self.response,
            write_stall_deadline: self.write_stall_deadline,
        })
    }

    pub fn restore(snapshot: ExchangeSnapshot) -> io::Result<Exchange> {
        Ok(Exchange {
            authority: snapshot.authority,
            is_head_request: snapshot.is_head_request,
            upstream: match snapshot.upstream {
                UpstreamSnapshot::Unconnected(request) => Upstream::Unconnected(request),
                UpstreamSnapshot::Connected(connection) => {
                    Upstream::Connected(BufferedConnection::restore(connection)?)
                }
            },
            response: snapshot.response,
            write_stall_deadline: snapshot.write_stall_deadline,
        })
    }
}

impl ResponsePhase {
    fn is_head_sent(&self) -> bool {
        match self {
            ResponsePhase::Head { early_response } => {
                early_response.is_some_and(|early_response| early_response.is_head_sent())
            }
            ResponsePhase::Body { .. } => true,
        }
    }

    fn end_of_upstream(
        &self,
        authority: &str,
        frontend: &mut BufferedConnection<UnixStream>,
    ) -> ExchangeProgress {
        match self {
            ResponsePhase::Body { decoder, .. } if decoder.end_of_input().is_ok() => {
                encode_frame(FrameKind::ResponseEnd, b"", &mut frontend.write_buffer);
                ExchangeProgress::Complete
            }
            _ => self.fail(
                authority,
                frontend,
                "the engine closed the connection before the end of the response",
            ),
        }
    }

    fn fail(
        &self,
        authority: &str,
        frontend: &mut BufferedConnection<UnixStream>,
        error: &str,
    ) -> ExchangeProgress {
        warn!("request to the engine at {authority} failed: {error}");
        if self.is_head_sent() {
            return ExchangeProgress::Aborted;
        }
        write_error(
            frontend,
            StatusCode::BAD_GATEWAY,
            "invalid response from the engine",
        );
        ExchangeProgress::Complete
    }

    fn process(
        &mut self,
        is_head_request: bool,
        input: &mut BytesMut,
        frontend: &mut BufferedConnection<UnixStream>,
        keepalive_interval: Duration,
    ) -> Result<bool, String> {
        loop {
            match self {
                ResponsePhase::Head { .. } => {
                    let Some((head, length)) = ResponseHead::parse(input)? else {
                        if input.len() > MAX_RESPONSE_HEAD {
                            return Err("response head too large".to_string());
                        }
                        return Ok(false);
                    };
                    input.advance(length);
                    if head.status.is_informational() {
                        continue;
                    }
                    let decoder = BodyDecoder::for_response(&head, is_head_request)?;
                    let is_event_stream = head
                        .headers
                        .get(CONTENT_TYPE)
                        .and_then(|value| value.to_str().ok())
                        .is_some_and(|value| value.starts_with("text/event-stream"));
                    if !self.is_head_sent() {
                        encode_frame(
                            FrameKind::ResponseHead,
                            &forwarded_head(&head).encode(),
                            &mut frontend.write_buffer,
                        );
                    }
                    *self = ResponsePhase::Body {
                        decoder,
                        keepalive: (is_event_stream && decoder.content_length().is_none())
                            .then(|| EventStreamKeepalive::new(keepalive_interval)),
                        status: Some(head.status.as_u16()),
                    };
                }
                ResponsePhase::Body {
                    decoder, keepalive, ..
                } => match decoder.decode(input)? {
                    Decoded::Data(data) => {
                        if let Some(keepalive) = keepalive {
                            keepalive.record(&data, keepalive_interval);
                        }
                        encode_body_frames(
                            FrameKind::ResponseBody,
                            &data,
                            &mut frontend.write_buffer,
                        );
                    }
                    Decoded::End => {
                        encode_frame(FrameKind::ResponseEnd, b"", &mut frontend.write_buffer);
                        return Ok(true);
                    }
                    Decoded::Incomplete => return Ok(false),
                },
            }
        }
    }

    fn keepalive_at(&self) -> Option<Timestamp> {
        match self {
            ResponsePhase::Head { early_response } => {
                early_response.map(|early_response| early_response.keepalive_at)
            }
            ResponsePhase::Body { keepalive, .. } => {
                keepalive.map(|keepalive| keepalive.keepalive_at)
            }
        }
    }

    fn send_keepalive(&mut self, output: &mut BytesMut, interval: Duration) {
        match self {
            ResponsePhase::Head {
                early_response: Some(early_response),
            } => early_response.send(output, interval),
            ResponsePhase::Body {
                keepalive: Some(keepalive),
                ..
            } => keepalive.send(output, interval),
            _ => {}
        }
    }
}

fn forwarded_head(head: &ResponseHead) -> ResponseHead {
    let mut headers = without_transport_headers(&head.headers);
    let content_length = head
        .headers
        .get(CONTENT_LENGTH)
        .filter(|_| !head.headers.contains_key(TRANSFER_ENCODING));
    if let Some(content_length) = content_length {
        headers.insert(CONTENT_LENGTH, content_length.clone());
    }
    ResponseHead {
        status: head.status,
        headers,
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FlowSnapshot {
    frontend: ConnectionSnapshot,
    stage: StageSnapshot,
}

#[derive(Debug, Serialize, Deserialize)]
enum StageSnapshot {
    Receiving {
        deadline: Timestamp,
    },
    Queued {
        request: QueuedRequestSnapshot,
        early_response: EarlyResponse,
    },
    Exchanging {
        exchange: ExchangeSnapshot,
        engine_request: Option<EngineRequestSnapshot>,
    },
    Closing,
}

#[derive(Debug, Serialize, Deserialize)]
struct QueuedRequestSnapshot {
    #[serde(with = "request_head_serde")]
    head: RequestHead,
    body: Map<String, Value>,
    user: String,
    priority: i32,
}

#[derive(Debug, Serialize, Deserialize)]
struct EngineRequestSnapshot {
    sequence: u64,
    control_socket: PathBuf,
    is_suspended: bool,
    user: String,
    priority: i32,
}

impl RequestFlow {
    pub fn snapshot(mut self, context: &FlowContext) -> io::Result<FlowSnapshot> {
        if let Stage::Queued { permit, .. } = &mut self.stage {
            if let Ok(rank) = permit.try_recv() {
                self.start(context, rank);
            }
        }
        let frontend = self.frontend.snapshot()?;
        let stage = match self.stage {
            Stage::Receiving { deadline } => StageSnapshot::Receiving { deadline },
            Stage::Queued {
                request,
                registration,
                early_response,
                ..
            } => {
                registration.detach();
                StageSnapshot::Queued {
                    request: QueuedRequestSnapshot {
                        head: request.head,
                        body: request.body,
                        user: request.user.0,
                        priority: request.class.priority,
                    },
                    early_response,
                }
            }
            Stage::Exchanging {
                exchange,
                registration,
            } => {
                let exchange = exchange.snapshot()?;
                let engine_request = registration.and_then(|registration| {
                    let id = registration.detach();
                    let started = context.scheduler.lock().started_request(id)?;
                    Some(EngineRequestSnapshot {
                        sequence: id.0,
                        control_socket: context
                            .backend()?
                            .ranks
                            .get(started.rank.0)?
                            .control_socket
                            .clone(),
                        is_suspended: started.is_suspended,
                        user: started.user.0,
                        priority: started.priority,
                    })
                });
                StageSnapshot::Exchanging {
                    exchange,
                    engine_request,
                }
            }
            Stage::Closing => StageSnapshot::Closing,
        };
        Ok(FlowSnapshot { frontend, stage })
    }

    pub fn restore(snapshot: FlowSnapshot, context: &FlowContext) -> io::Result<RequestFlow> {
        let mut flow = RequestFlow {
            frontend: BufferedConnection::restore(snapshot.frontend)?,
            stage: Stage::Closing,
        };
        match snapshot.stage {
            StageSnapshot::Receiving { deadline } => flow.stage = Stage::Receiving { deadline },
            StageSnapshot::Queued { early_response, .. } if context.backend().is_none() => {
                if !early_response.is_head_sent() {
                    flow.respond(StatusCode::SERVICE_UNAVAILABLE, "no backend configured");
                }
            }
            StageSnapshot::Queued {
                request,
                early_response,
            } => {
                let request = GeneratingRequest {
                    head: request.head,
                    body: request.body,
                    user: UserId(request.user),
                    class: handed_over_class(request.priority),
                };
                let body_bytes = serde_json::to_vec(&request.body).map_or(0, |body| body.len());
                let admission = context.scheduler.readmit(
                    request.user.clone(),
                    request.class.clone(),
                    &RequestDescription::new(&request, body_bytes),
                );
                flow.stage = Stage::Queued {
                    request,
                    proxy_id: admission.proxy_id,
                    permit: admission.permit,
                    registration: admission.registration,
                    early_response,
                };
            }
            StageSnapshot::Exchanging {
                exchange,
                engine_request,
            } => {
                let exchange = Exchange::restore(exchange)?;
                let registration = engine_request.and_then(|engine_request| {
                    let rank =
                        context.backend()?.ranks.iter().position(|rank| {
                            rank.control_socket == engine_request.control_socket
                        })?;
                    let started = StartedRequest {
                        user: UserId(engine_request.user),
                        priority: engine_request.priority,
                        rank: Rank(rank),
                        is_suspended: engine_request.is_suspended,
                    };
                    Some(
                        context
                            .scheduler
                            .adopt(RequestId(engine_request.sequence), started),
                    )
                });
                flow.stage = Stage::Exchanging {
                    exchange,
                    registration,
                };
            }
            StageSnapshot::Closing => {}
        }
        Ok(flow)
    }
}
