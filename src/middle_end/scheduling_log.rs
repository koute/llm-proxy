use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

use log::{error, warn};
use serde::Serialize;
use tokio::task::JoinHandle;

use super::routing::GeneratingRequest;
use crate::config::ClassConfig;
use crate::engine_protocol::{ProxyId, StopReason};
use crate::quality_of_service::{RequestState, WaitReason};
use crate::utils::AppendFile;

const FORMAT: u32 = 1;

#[derive(Clone, Default)]
pub struct SchedulingLog {
    file: Option<AppendFile>,
}

pub struct SchedulingLogWriter(Option<JoinHandle<()>>);

#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event<'a> {
    Start {
        pid: u32,
        format: u32,
        restart: bool,
        rank_caps: &'a [usize],
        swap_hysteresis_seconds: f64,
        classes: Vec<ClassSummary<'a>>,
    },
    Stop {
        reason: ProcessStop,
    },
    Rejected {
        status: u16,
        reason: &'a str,
        method: &'a str,
        path: &'a str,
        account: Option<&'a str>,
    },
    Admitted {
        request: ProxyId,
        user: &'a str,
        class: &'a str,
        priority: i32,
        path: &'a str,
        model: Option<&'a str>,
        stream: bool,
        max_tokens: Option<u64>,
        body_bytes: usize,
    },
    Started {
        request: ProxyId,
        rank: usize,
        queued_seconds: f64,
    },
    Suspended {
        request: ProxyId,
        rank: usize,
        reason: Option<WaitReason>,
    },
    Resumed {
        request: ProxyId,
        rank: usize,
    },
    EngineRunning {
        request: ProxyId,
        rank: usize,
    },
    EngineStopped {
        request: ProxyId,
        rank: usize,
        reason: StopReason,
    },
    Completed {
        request: ProxyId,
        status: Option<u16>,
    },
    Aborted {
        request: ProxyId,
    },
    Finished {
        request: ProxyId,
        user: &'a str,
        class: &'a str,
        priority: i32,
        total_seconds: f64,
        queued_seconds: Option<f64>,
        generating_seconds: f64,
        usage_seconds: f64,
        suspensions: u32,
    },
    HandedOver {
        request: ProxyId,
        state: StateName,
    },
    Adopted {
        request: ProxyId,
        user: &'a str,
        priority: i32,
        rank: usize,
        state: StateName,
    },
    Caps {
        has_multiple_users: bool,
        soft_cap: Option<NonZeroUsize>,
        rank_caps: &'a [usize],
    },
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessStop {
    Shutdown,
    Restart,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StateName {
    Queued,
    Running,
    Suspended,
    Abandoned,
}

impl From<RequestState> for StateName {
    fn from(state: RequestState) -> StateName {
        match state {
            RequestState::New => StateName::Queued,
            RequestState::Running => StateName::Running,
            RequestState::Suspended => StateName::Suspended,
            RequestState::Abandoned { .. } => StateName::Abandoned,
        }
    }
}

pub struct RequestDescription {
    pub path: String,
    pub model: Option<String>,
    pub is_streaming: bool,
    pub max_tokens: Option<u64>,
    pub body_bytes: usize,
}

impl RequestDescription {
    pub fn new(request: &GeneratingRequest, body_bytes: usize) -> RequestDescription {
        let body = &request.body;
        RequestDescription {
            path: request.head.path().to_string(),
            model: body
                .get("model")
                .and_then(|model| model.as_str())
                .map(str::to_string),
            is_streaming: request.is_streaming(),
            max_tokens: ["max_tokens", "max_completion_tokens"]
                .iter()
                .find_map(|field| body.get(*field).and_then(|value| value.as_u64())),
            body_bytes,
        }
    }
}

#[derive(Serialize)]
pub struct ClassSummary<'a> {
    name: &'a str,
    priority: i32,
    idle_only: bool,
    cost: f32,
    soft_cap: Option<NonZeroUsize>,
    max_parallel_per_user: Option<NonZeroUsize>,
}

#[derive(Serialize)]
struct Line<'a> {
    time: String,
    #[serde(flatten)]
    event: &'a Event<'a>,
}

impl SchedulingLog {
    pub fn open(path: Option<&Path>) -> (SchedulingLog, SchedulingLogWriter) {
        let Some(path) = path else {
            return (SchedulingLog::default(), SchedulingLogWriter(None));
        };
        match AppendFile::open(path, "scheduling log") {
            Ok((file, writer)) => (
                SchedulingLog { file: Some(file) },
                SchedulingLogWriter(Some(writer)),
            ),
            Err(message) => {
                error!("{message}; the scheduling log is off");
                (SchedulingLog::default(), SchedulingLogWriter(None))
            }
        }
    }

    pub fn record(&self, event: Event) {
        let Some(file) = &self.file else {
            return;
        };
        let line = Line {
            time: format!("{:.6}", jiff::Timestamp::now()),
            event: &event,
        };
        let mut bytes = serde_json::to_vec(&line).expect("unserializable scheduling log event");
        bytes.push(b'\n');
        file.append(bytes);
    }

    pub fn start_event<'a>(
        restart: bool,
        rank_caps: &'a [usize],
        swap_hysteresis: Duration,
        classes: &'a [ClassConfig],
    ) -> Event<'a> {
        Event::Start {
            pid: std::process::id(),
            format: FORMAT,
            restart,
            rank_caps,
            swap_hysteresis_seconds: swap_hysteresis.as_secs_f64(),
            classes: classes
                .iter()
                .map(|class| ClassSummary {
                    name: &class.name,
                    priority: class.priority,
                    idle_only: class.idle_only,
                    cost: class.cost,
                    soft_cap: class.soft_cap,
                    max_parallel_per_user: class.max_parallel_per_user,
                })
                .collect(),
        }
    }

    pub async fn stop(
        &self,
        reason: ProcessStop,
        writer: &mut SchedulingLogWriter,
        timeout: Duration,
    ) {
        self.record(Event::Stop { reason });
        let (Some(file), Some(writer)) = (&self.file, writer.0.take()) else {
            return;
        };
        file.close();
        if tokio::time::timeout(timeout, writer).await.is_err() {
            warn!("the scheduling log was not complete after {timeout:?}");
        }
    }
}
