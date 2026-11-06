use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

use super::routing::truncated;
use log::warn;
use serde::{Deserialize, Serialize};

use super::routing::GeneratingRequest;
use crate::config::ClassConfig;
use crate::engine_protocol::{ProxyId, StopReason, TokenCounts};
use crate::gateway::Attribution;
use crate::quality_of_service::{RequestState, WaitReason};
use crate::utils::{AppendFile, AppendHealth, AppendWriter};

const FORMAT: u32 = 1;

#[derive(Clone, Default)]
pub struct SchedulingLog {
    file: Option<AppendFile>,
}

pub struct SchedulingLogWriter(Option<AppendWriter>);

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
        tokens: Option<TokenCounts>,
    },
    EngineStopped {
        request: ProxyId,
        rank: usize,
        reason: StopReason,
        tokens: Option<TokenCounts>,
    },
    Completed {
        request: ProxyId,
        status: Option<u16>,
    },
    Aborted {
        request: ProxyId,
    },
    Finished(&'a RequestSummary),
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestSummary {
    pub request: ProxyId,
    pub user: String,
    pub class: String,
    pub priority: i32,
    pub total_seconds: f64,
    pub queued_seconds: Option<f64>,
    pub generating_seconds: f64,
    pub usage_seconds: f64,
    pub suspensions: u32,
    pub tokens: Option<TokenCounts>,
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
    pub attribution: Attribution,
}

impl RequestDescription {
    pub fn new(request: &GeneratingRequest, body_bytes: usize) -> RequestDescription {
        let body = &request.body;
        RequestDescription {
            path: truncated(request.head.path()),
            model: body
                .get("model")
                .and_then(|model| model.as_str())
                .map(truncated),
            is_streaming: request.is_streaming(),
            max_tokens: ["max_tokens", "max_completion_tokens"]
                .iter()
                .find_map(|field| body.get(*field).and_then(|value| value.as_u64())),
            body_bytes,
            attribution: request.attribution.clone(),
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
        let (file, writer) = AppendFile::open_lazily(path, "scheduling log");
        (
            SchedulingLog { file: Some(file) },
            SchedulingLogWriter(Some(writer)),
        )
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

    /// Writes what's pending, for up to `timeout`, and returns what couldn't be written,
    /// which a restart hands over to the new process.
    pub async fn stop(
        &self,
        reason: ProcessStop,
        writer: &mut SchedulingLogWriter,
        timeout: Duration,
    ) -> Vec<u8> {
        self.record(Event::Stop { reason });
        let (Some(file), Some(writer)) = (&self.file, writer.0.take()) else {
            return Vec::new();
        };
        let unwritten = file.stop(writer, timeout).await;
        if !unwritten.is_empty() {
            warn!(
                "the scheduling log can't be written; {} bytes of events are left",
                unwritten.len()
            );
        }
        unwritten
    }

    /// Appends events which a previous process couldn't write.
    pub fn append_unwritten(&self, events: Vec<u8>) {
        if let (Some(file), false) = (&self.file, events.is_empty()) {
            file.append_handed_over(events);
        }
    }

    pub fn health(&self) -> Option<AppendHealth> {
        self.file.as_ref().map(AppendFile::health)
    }
}
