use std::future::Future;
use std::io;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use bytes::{Buf, Bytes, BytesMut};
use http::header::{HeaderValue, CONTENT_LENGTH, CONTENT_TYPE, RETRY_AFTER, TRANSFER_ENCODING};
use http::StatusCode;
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::net::{TcpStream, UnixStream};
use tokio::sync::oneshot;
use tokio::time::Instant;

use super::routing::{
    engine_request, error_response, route, GeneratingRequest, PassthroughRecord, Rejection, Route,
    UpstreamRequest,
};
use super::scheduler::{
    restored_class, Ending, Registration, RequestRecord, Scheduler, StartedRequest,
};
use super::scheduling_log::{Event, RequestDescription, SchedulingLog};
use super::{quiesce_deadline, Quiesce};
use crate::config::{BackendConfig, Config};
use crate::engine_protocol::ProxyId;
use crate::gateway::limits::InFlightGuard;
use crate::gateway::{admin, Attribution, Gateway, Outcome as GatewayOutcome};
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
type LocalResponse = Pin<Box<dyn Future<Output = (ResponseHead, Bytes)> + Send>>;

pub struct FlowContext {
    pub config: Arc<Config>,
    pub scheduler: Scheduler,
    pub scheduling_log: SchedulingLog,
    pub gateway: Option<Arc<Gateway>>,
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

#[allow(clippy::large_enum_variant)]
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
        passthrough: Option<PassthroughRecord>,
        /// The record of a request the engine finished before a restart, which the
        /// flow reports when its response ends.
        finished: Option<Box<RequestRecord>>,
        _in_flight: Option<InFlightGuard>,
    },
    Local {
        response: LocalResponse,
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
        // An admin API request may be changing the keys file: let it finish, so that
        // its client learns the outcome and the change is audited by this process.
        if let Stage::Local { response } = &mut self.stage {
            if let Ok((head, body)) = tokio::time::timeout_at(deadline, response).await {
                write_response(&mut self.frontend, head, body);
                self.stage = Stage::Closing;
            }
        }
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
                let keepalive_at = (early_response.is_event_stream()
                    && !frontend.has_pending_writes())
                .then_some(early_response.keepalive_at);
                let progress = tokio::select! {
                    rank = permit => {
                        match rank {
                            Ok(rank) => self.start(context, rank),
                            Err(_) if early_response.is_head_sent() => return Progress::Finished,
                            Err(_) => self.respond_rejected(context, StatusCode::SERVICE_UNAVAILABLE, "removed_from_queue", "request removed from the queue"),
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
                };
                if let Progress::Finished = progress {
                    self.abort_queued();
                }
                progress
            }
            Stage::Exchanging {
                exchange,
                registration,
                passthrough,
                finished,
                ..
            } => {
                match exchange
                    .step(&mut self.frontend, context.keepalive_interval())
                    .await
                {
                    ExchangeProgress::Continue => Progress::Continue,
                    ExchangeProgress::Complete => {
                        // Without the engine's status the proxy answered with a 502.
                        let status = exchange.engine_status().or(Some(502));
                        if let Some(registration) = registration.take() {
                            registration.complete(status, exchange.first_data_at);
                        }
                        if let (Some(gateway), Some(record)) =
                            (&context.gateway, passthrough.take())
                        {
                            gateway.passthrough_finished(record, status, GatewayOutcome::Completed);
                        }
                        if let (Some(gateway), Some(mut record)) =
                            (&context.gateway, finished.take())
                        {
                            let first_data_at = exchange.first_data_at;
                            record.end(
                                GatewayOutcome::Completed,
                                status,
                                first_data_at,
                                Timestamp::now(),
                            );
                            record.report(gateway);
                        }
                        self.stage = Stage::Closing;
                        Progress::Continue
                    }
                    progress @ (ExchangeProgress::Aborted | ExchangeProgress::Failed) => {
                        let outcome = match progress {
                            ExchangeProgress::Failed => GatewayOutcome::Failed,
                            _ => GatewayOutcome::Aborted,
                        };
                        if let Some(registration) = registration.take() {
                            registration.abort(exchange.ending(outcome));
                        }
                        if let (Some(gateway), Some(record)) =
                            (&context.gateway, passthrough.take())
                        {
                            gateway.passthrough_finished(record, exchange.client_status(), outcome);
                        }
                        if let (Some(gateway), Some(mut record)) =
                            (&context.gateway, finished.take())
                        {
                            let ending = exchange.ending(outcome);
                            record.end(
                                ending.outcome,
                                ending.engine_status,
                                ending.first_data_at,
                                Timestamp::now(),
                            );
                            record.report(gateway);
                        }
                        Progress::Finished
                    }
                }
            }
            Stage::Local { response } => {
                let (head, body) = response.await;
                write_response(&mut self.frontend, head, body);
                self.stage = Stage::Closing;
                Progress::Continue
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
                            // The rest of a request body we refused (e.g. a 413), which a
                            // proxy like nginx streams: discard it until the response is out.
                            Ok(ReadResult::Data) => {
                                frontend.read_buffer.clear();
                                Progress::Continue
                            }
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
            Ok(Some((_, body))) if body.len() > limits.max_request_body_bytes => {
                self.respond_rejected(
                    context,
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "request_body_too_large",
                    "request body too large",
                );
                return Progress::Continue;
            }
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
            self.respond_rejected(
                context,
                StatusCode::PAYLOAD_TOO_LARGE,
                "request_body_too_large",
                "request body too large",
            );
            return Progress::Continue;
        }
        tokio::select! {
            result = self.frontend.fill() => match result {
                Ok(ReadResult::Data) => {
                    // An idle timeout: nginx streams bodies as they come.
                    let timeout = limits.request_body_timeout_seconds.duration();
                    self.stage = Stage::Receiving { deadline: Timestamp::now() + timeout };
                    Progress::Continue
                }
                _ => Progress::Finished,
            },
            _ = tokio::time::sleep_until(deadline.instant()) => {
                self.respond_rejected(context, StatusCode::REQUEST_TIMEOUT, "request_body_timeout", "request body read timed out");
                Progress::Continue
            }
        }
    }

    fn dispatch(&mut self, context: &FlowContext, head: RequestHead, body: Bytes) {
        let body_bytes = body.len();
        match route(&context.config, context.gateway.as_deref(), head, body) {
            Route::Rejection(rejection) => self.reject(context, rejection),
            Route::Local(request) => {
                let gateway = context
                    .gateway
                    .clone()
                    .expect("local route without gateway");
                self.stage = Stage::Local {
                    response: Box::pin(admin::handle(
                        gateway,
                        context.config.clone(),
                        context.scheduler.clone(),
                        request,
                    )),
                };
            }
            Route::Passthrough(request) => {
                self.stage = Stage::Exchanging {
                    exchange: Exchange::new(request.upstream, None),
                    registration: None,
                    passthrough: request.record,
                    finished: None,
                    _in_flight: request.in_flight,
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
                            request.head.path() == ANTHROPIC_MESSAGES_PATH,
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
                        // The request never ran: it doesn't count against its key's limits.
                        if let (Some(gateway), Some(key_id)) =
                            (&context.gateway, &request.attribution.key_id)
                        {
                            gateway.limiter.lock().refund(key_id);
                        }
                        let rejection = Rejection {
                            status: StatusCode::SERVICE_UNAVAILABLE,
                            reason: "queue_full",
                            message: "too many queued requests; try again later".to_string(),
                            method: request.head.method.to_string(),
                            path: description.path,
                            account: Some(request.user.account().to_string()),
                            key_id: request.attribution.key_id.clone(),
                            masked_key: None,
                            client: format!("user {}", request.user),
                            client_address: request.attribution.client.clone(),
                            user_agent: None,
                            headers: vec![(RETRY_AFTER, HeaderValue::from_static("1"))],
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
        if let Some(gateway) = &context.gateway {
            gateway.record_rejection(&rejection);
        }
        let (mut head, body) = error_response(rejection.status, &rejection.message);
        head.headers.extend(rejection.headers);
        write_response(&mut self.frontend, head, body);
        self.stage = Stage::Closing;
    }

    fn start(&mut self, context: &FlowContext, rank: Rank) {
        let Stage::Queued {
            mut request,
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
            exchange: Exchange::new(
                upstream_request,
                Some(early_response).filter(EarlyResponse::is_event_stream),
            ),
            registration: Some(registration),
            passthrough: None,
            finished: None,
            _in_flight: request.in_flight.take(),
        };
    }

    /// The client of a queued request went away: if it had got the early head, the
    /// request ends with the status it got.
    fn abort_queued(&mut self) {
        let Stage::Queued { early_response, .. } = &self.stage else {
            return;
        };
        if !early_response.is_head_sent() {
            return;
        }
        if let Stage::Queued { registration, .. } =
            std::mem::replace(&mut self.stage, Stage::Closing)
        {
            registration.abort(Ending {
                outcome: GatewayOutcome::Aborted,
                engine_status: Some(200),
                first_data_at: None,
            });
        }
    }

    fn respond(&mut self, status: StatusCode, message: &str) {
        let (head, body) = error_response(status, message);
        write_response(&mut self.frontend, head, body);
        self.stage = Stage::Closing;
    }

    /// Responds with an error of the proxy's own, which the rejection metrics count.
    fn respond_rejected(
        &mut self,
        context: &FlowContext,
        status: StatusCode,
        reason: &'static str,
        message: &str,
    ) {
        warn!("rejected a request with {status}: {message}");
        if let Some(gateway) = &context.gateway {
            gateway.metrics.observe_rejection(status.as_u16(), reason);
        }
        self.respond(status, message);
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
/// The keepalives of a streaming request before the engine's response head: the proxy
/// commits to a `200` with an event stream and sends SSE comments. Non-streaming
/// requests get no early head, so that their status stays the engine's.
pub struct EarlyResponse {
    is_event_stream: bool,
    is_head_sent: bool,
    pub keepalive_at: Timestamp,
    /// An Anthropic `/v1/messages` stream, whose clients only see errors as
    /// `event: error`.
    #[serde(default)]
    is_anthropic: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EventStreamKeepalive {
    pub keepalive_at: Timestamp,
    trailing_newlines: usize,
}

impl EarlyResponse {
    pub fn new(is_event_stream: bool, is_anthropic: bool, interval: Duration) -> EarlyResponse {
        EarlyResponse {
            is_event_stream,
            is_head_sent: false,
            keepalive_at: Timestamp::now() + interval,
            is_anthropic,
        }
    }

    pub fn is_head_sent(&self) -> bool {
        self.is_head_sent
    }

    pub fn is_event_stream(&self) -> bool {
        self.is_event_stream
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
    first_data_at: Option<Timestamp>,
}

enum Upstream {
    Unconnected(Bytes),
    Connected(BufferedConnection<TcpStream>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum ResponsePhase {
    Head {
        early_response: Option<EarlyResponse>,
    },
    Body {
        decoder: BodyDecoder,
        keepalive: Option<EventStreamKeepalive>,
        #[serde(default)]
        status: Option<u16>,
        /// The body of an engine error after the proxy had sent its own `200` event
        /// stream head: it's sent as an SSE error event at the end.
        #[serde(default)]
        error_event: Option<Vec<u8>>,
        #[serde(default)]
        is_anthropic: bool,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExchangeSnapshot {
    authority: String,
    is_head_request: bool,
    upstream: UpstreamSnapshot,
    response: ResponsePhase,
    write_stall_deadline: Option<Timestamp>,
    #[serde(default)]
    first_data_at: Option<Timestamp>,
}

#[derive(Debug, Serialize, Deserialize)]
enum UpstreamSnapshot {
    Unconnected(#[serde(with = "base64_bytes")] Bytes),
    Connected(ConnectionSnapshot),
}

pub enum ExchangeProgress {
    Continue,
    Complete,
    /// The client went away.
    Aborted,
    /// The engine failed after the response had started; the client's response is cut off.
    Failed,
}

impl Exchange {
    pub fn new(request: UpstreamRequest, early_response: Option<EarlyResponse>) -> Exchange {
        Exchange {
            authority: request.authority,
            is_head_request: request.is_head_request,
            upstream: Upstream::Unconnected(request.bytes),
            response: ResponsePhase::Head { early_response },
            write_stall_deadline: None,
            first_data_at: None,
        }
    }

    pub fn engine_status(&self) -> Option<u16> {
        match self.response {
            ResponsePhase::Body { status, .. } => status,
            ResponsePhase::Head { .. } => None,
        }
    }

    /// The status the client got: the engine's, or 200 from an early keepalive head.
    pub fn client_status(&self) -> Option<u16> {
        match self.response {
            ResponsePhase::Body { status, .. } => status,
            ResponsePhase::Head { .. } => self.response.is_head_sent().then_some(200),
        }
    }

    fn ending(&self, outcome: GatewayOutcome) -> Ending {
        Ending {
            outcome,
            engine_status: self.client_status(),
            first_data_at: self.first_data_at,
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
            first_data_at,
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
                        } else {
                            // After the proxy's own event stream head: an error event.
                            let is_anthropic = matches!(response, ResponsePhase::Head { early_response: Some(early) } if early.is_anthropic);
                            let body = br#"{"error": {"message": "engine unreachable"}}"#;
                            let event = sse_error_event(502, body, is_anthropic);
                            let output = &mut frontend.write_buffer;
                            encode_body_frames(FrameKind::ResponseBody, &event, output);
                            encode_frame(FrameKind::ResponseEnd, b"", output);
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
            first_data_at,
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
            first_data_at: self.first_data_at,
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
            first_data_at: snapshot.first_data_at,
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
            return ExchangeProgress::Failed;
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
        first_data_at: &mut Option<Timestamp>,
    ) -> Result<bool, String> {
        loop {
            match self {
                ResponsePhase::Head { early_response } => {
                    let is_anthropic = early_response.is_some_and(|early| early.is_anthropic);
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
                    let is_head_sent = self.is_head_sent();
                    if !is_head_sent {
                        encode_frame(
                            FrameKind::ResponseHead,
                            &forwarded_head(&head).encode(),
                            &mut frontend.write_buffer,
                        );
                    }
                    // After the proxy's own event stream head, anything but an event
                    // stream from the engine can only reach the client as an SSE event.
                    let is_late_error =
                        is_head_sent && (!head.status.is_success() || !is_event_stream);
                    *self = ResponsePhase::Body {
                        decoder,
                        keepalive: (is_event_stream && decoder.content_length().is_none())
                            .then(|| EventStreamKeepalive::new(keepalive_interval)),
                        status: Some(head.status.as_u16()),
                        error_event: is_late_error.then(Vec::new),
                        is_anthropic,
                    };
                }
                ResponsePhase::Body {
                    decoder,
                    keepalive,
                    status,
                    error_event,
                    is_anthropic,
                } => match decoder.decode(input)? {
                    Decoded::Data(data) => {
                        if !data.is_empty() {
                            first_data_at.get_or_insert_with(Timestamp::now);
                        }
                        if let Some(error_event) = error_event {
                            let room = MAX_ERROR_EVENT_BYTES.saturating_sub(error_event.len());
                            error_event.extend_from_slice(&data[..data.len().min(room)]);
                            continue;
                        }
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
                        if let Some(error_event) = error_event {
                            let event =
                                sse_error_event(status.unwrap_or(502), error_event, *is_anthropic);
                            encode_body_frames(
                                FrameKind::ResponseBody,
                                &event,
                                &mut frontend.write_buffer,
                            );
                        }
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

const MAX_ERROR_EVENT_BYTES: usize = 64 * 1024;

/// An engine error as an SSE event, which OpenAI clients raise as an error:
/// `data: {"error": ...}`. The engine's body is kept if it already has that shape.
const ANTHROPIC_MESSAGES_PATH: &str = "/v1/messages";

/// An engine error as an SSE event which clients raise as an error: for OpenAI clients
/// `data: {"error": ...}` (the engine's body if it already has that shape), for
/// Anthropic clients `event: error` with Anthropic's error shape.
fn sse_error_event(status: u16, body: &[u8], is_anthropic: bool) -> Vec<u8> {
    let (status, error) = match (200..300).contains(&status) {
        true => {
            let text = String::from_utf8_lossy(body);
            let message = format!(
                "the engine's response isn't an event stream: {}",
                text.trim()
            );
            (
                502,
                serde_json::json!({"error": {"message": message, "code": 502}}),
            )
        }
        false => (status, openai_error(status, body)),
    };
    if !is_anthropic {
        return format!("\ndata: {error}\n\n").into_bytes();
    }
    let kind = match status {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    };
    let message = match &error["error"] {
        Value::String(text) => text.clone(),
        inner => inner["message"]
            .as_str()
            .map_or_else(|| inner.to_string(), str::to_string),
    };
    let event = serde_json::json!({"type": "error", "error": {"type": kind, "message": message}});
    format!("\nevent: error\ndata: {event}\n\n").into_bytes()
}

fn openai_error(status: u16, body: &[u8]) -> Value {
    match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(object)) if object.contains_key("error") => Value::Object(object),
        Ok(Value::Object(object)) => {
            let message = object
                .get("message")
                .and_then(Value::as_str)
                .map_or_else(|| Value::Object(object.clone()).to_string(), str::to_string);
            serde_json::json!({"error": {"message": message, "code": status}})
        }
        _ => {
            let message = String::from_utf8_lossy(body).trim().to_string();
            serde_json::json!({"error": {"message": message, "code": status}})
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

impl FlowSnapshot {
    /// When the request arrived, if it's queued or in the engine: the order in which
    /// a restart re-admits queued requests.
    pub fn arrival(&self) -> Option<Timestamp> {
        match &self.stage {
            StageSnapshot::Queued { request, .. } => request.attribution.arrival,
            StageSnapshot::Exchanging {
                engine_request: Some(engine_request),
                ..
            } => engine_request.attribution.arrival,
            _ => None,
        }
    }
}

// Only built once per request at a restart: its size doesn't matter.
#[allow(clippy::large_enum_variant)]
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
        #[serde(default)]
        passthrough: Option<PassthroughRecord>,
        #[serde(default)]
        finished: Option<Box<RequestRecord>>,
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
    #[serde(default)]
    attribution: Attribution,
}

#[derive(Debug, Serialize, Deserialize)]
struct EngineRequestSnapshot {
    sequence: u64,
    control_socket: PathBuf,
    is_suspended: bool,
    user: String,
    priority: i32,
    #[serde(default)]
    attribution: Attribution,
}

impl RequestFlow {
    pub fn snapshot(mut self, context: &FlowContext) -> io::Result<FlowSnapshot> {
        if let Stage::Queued { permit, .. } = &mut self.stage {
            if let Ok(rank) = permit.try_recv() {
                self.start(context, rank);
            }
        }
        if let Stage::Local { .. } = self.stage {
            self.respond(
                StatusCode::SERVICE_UNAVAILABLE,
                "the proxy restarted during this request; check whether it took effect",
            );
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
                        attribution: request.attribution,
                    },
                    early_response,
                }
            }
            Stage::Exchanging {
                exchange,
                registration,
                passthrough,
                mut finished,
                ..
            } => {
                let exchange = exchange.snapshot()?;
                let engine_request = registration.and_then(|registration| {
                    let id = registration.detach();
                    let mut scheduler = context.scheduler.lock();
                    let Some(started) = scheduler.started_request(id) else {
                        // The engine finished it, the response is still on its way.
                        finished = scheduler.take_finished_record(id).map(Box::new);
                        return None;
                    };
                    drop(scheduler);
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
                        attribution: started.attribution,
                    })
                });
                StageSnapshot::Exchanging {
                    exchange,
                    engine_request,
                    passthrough,
                    finished,
                }
            }
            Stage::Local { .. } | Stage::Closing => StageSnapshot::Closing,
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
                let in_flight = in_flight_guard(context, &request.attribution);
                let request = GeneratingRequest {
                    head: request.head,
                    body: request.body,
                    user: UserId(request.user),
                    class: restored_class(
                        &context.config.accounts.classes,
                        request.attribution.class.as_deref(),
                        request.priority,
                    ),
                    attribution: request.attribution,
                    in_flight,
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
                passthrough,
                finished,
            } => {
                let exchange = Exchange::restore(exchange)?;
                let in_flight = match (&engine_request, &passthrough) {
                    (Some(engine_request), _) => {
                        in_flight_guard(context, &engine_request.attribution)
                    }
                    (None, Some(record)) => {
                        let (key_id, reserved_tokens) =
                            (record.key_id.as_deref(), record.reserved_tokens);
                        reacquired_guard(context, key_id, reserved_tokens)
                    }
                    (None, None) => None,
                };
                let registration = engine_request.and_then(|engine_request| {
                    let rank =
                        context.backend()?.ranks.iter().position(|rank| {
                            rank.control_socket == engine_request.control_socket
                        })?;
                    let class = restored_class(
                        &context.config.accounts.classes,
                        engine_request.attribution.class.as_deref(),
                        engine_request.priority,
                    );
                    let started = StartedRequest {
                        user: UserId(engine_request.user),
                        priority: engine_request.priority,
                        class: Some(class),
                        rank: Rank(rank),
                        is_suspended: engine_request.is_suspended,
                        attribution: engine_request.attribution,
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
                    passthrough,
                    finished,
                    _in_flight: in_flight,
                };
            }
            StageSnapshot::Closing => {}
        }
        Ok(flow)
    }
}

fn in_flight_guard(context: &FlowContext, attribution: &Attribution) -> Option<InFlightGuard> {
    let key_id = attribution.key_id.as_deref();
    reacquired_guard(context, key_id, Some(attribution.reserved_tokens))
}

/// Counts a handed-over request against its key's limits again, if it counted before.
fn reacquired_guard(
    context: &FlowContext,
    key_id: Option<&str>,
    reserved_tokens: Option<u64>,
) -> Option<InFlightGuard> {
    let gateway = context.gateway.as_ref()?;
    let (key_id, reserved_tokens) = (key_id?, reserved_tokens?);
    gateway.limiter.lock().reacquire(key_id, reserved_tokens);
    Some(gateway.limiter.guard(key_id, reserved_tokens))
}

#[cfg(test)]
mod sse_error_tests {
    use super::*;

    fn event(status: u16, body: &str, is_anthropic: bool) -> String {
        String::from_utf8(sse_error_event(status, body.as_bytes(), is_anthropic)).unwrap()
    }

    #[test]
    fn late_engine_errors_as_sse_events() {
        // An OpenAI error body is kept as it is.
        let openai = event(400, r#"{"error": {"message": "bad", "code": 400}}"#, false);
        assert_eq!(
            openai,
            "\ndata: {\"error\":{\"message\":\"bad\",\"code\":400}}\n\n"
        );
        // Other bodies become one.
        assert!(event(500, r#"{"message": "boom"}"#, false).contains(r#""message":"boom""#));
        assert!(event(502, "plain text", false).contains(r#""message":"plain text""#));
        // A 2xx which isn't an event stream is an error too.
        assert!(event(200, r#"{"id": 1}"#, false).contains("isn't an event stream"));
        // Anthropic clients only see `event: error` with their error shape.
        let anthropic = event(404, r#"{"error": "Not Found"}"#, true);
        assert!(
            anthropic.starts_with("\nevent: error\ndata: "),
            "{anthropic}"
        );
        assert!(
            anthropic.contains(r#""type":"not_found_error""#)
                && anthropic.contains(r#""message":"Not Found""#)
        );
        assert!(event(429, "{}", true).contains("rate_limit_error"));
        assert!(event(503, "{}", true).contains("overloaded_error"));
    }
}
