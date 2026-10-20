use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use log::debug;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use super::control_link::LinkMessage;
use super::scheduling_log::{Event as LogEvent, RequestDescription, SchedulingLog};
use crate::config::ClassConfig;
use crate::engine_protocol::{Command, Event, ProxyId, StopReason};
use crate::quality_of_service::{
    abandon_request, calculate_quality_of_service, execute_quality_of_service,
    handle_engine_message, remove_expired_requests, remove_request, status, Action, Caps,
    EngineMessage, Rank, Request, RequestId, RequestState, Settings, Status, UsageHistory, User,
    UserId, WaitReason,
};
use crate::utils::Timestamp;

#[derive(Clone)]
pub struct Scheduler {
    state: Arc<Mutex<SchedulerState>>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ProxyIdCounter {
    instance: u64,
    next_sequence: u64,
}

pub struct StartedRequest {
    pub user: UserId,
    pub priority: i32,
    pub rank: Rank,
    pub is_suspended: bool,
}

pub struct SchedulerState {
    proxy_ids: ProxyIdCounter,
    users: Vec<User>,
    requests: Vec<Request>,
    permits: HashMap<RequestId, oneshot::Sender<Rank>>,
    settings: Settings,
    max_queued_requests: usize,
    control_links: Vec<mpsc::UnboundedSender<LinkMessage>>,
    log: SchedulingLog,
    caps_in_log: Option<Caps>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct QueueFull;

pub struct Admission {
    pub proxy_id: ProxyId,
    pub permit: oneshot::Receiver<Rank>,
    pub registration: Registration,
}

pub struct Registration {
    scheduler: Option<Scheduler>,
    id: RequestId,
}

impl ProxyIdCounter {
    pub fn new() -> ProxyIdCounter {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        ProxyIdCounter {
            instance: start_time.as_nanos() as u64 ^ u64::from(std::process::id()).rotate_left(48),
            next_sequence: 0,
        }
    }

    fn next(&mut self) -> RequestId {
        let id = RequestId(self.next_sequence);
        self.next_sequence += 1;
        id
    }

    fn proxy_id(&self, id: RequestId) -> ProxyId {
        ProxyId {
            instance: self.instance,
            sequence: id.0,
        }
    }
}

pub fn handed_over_class(priority: i32) -> ClassConfig {
    ClassConfig {
        name: "handed over".to_string(),
        priority,
        idle_only: false,
        cost: 0.0,
        soft_cap: None,
        max_parallel_per_user: None,
    }
}

impl Scheduler {
    pub fn new(
        proxy_ids: ProxyIdCounter,
        settings: Settings,
        max_queued_requests: usize,
        control_links: Vec<mpsc::UnboundedSender<LinkMessage>>,
        log: SchedulingLog,
    ) -> Scheduler {
        Scheduler {
            state: Arc::new(Mutex::new(SchedulerState {
                proxy_ids,
                users: Vec::new(),
                requests: Vec::new(),
                permits: HashMap::new(),
                settings,
                max_queued_requests,
                control_links,
                log,
                caps_in_log: None,
            })),
        }
    }

    pub fn lock(&self) -> MutexGuard<'_, SchedulerState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn admit(
        &self,
        user: UserId,
        class: ClassConfig,
        description: &RequestDescription,
    ) -> Result<Admission, QueueFull> {
        let mut state = self.lock();
        if state.queued_count() >= state.max_queued_requests {
            return Err(QueueFull);
        }
        let now = Timestamp::now();
        let (id, proxy_id, permit) = state.admit(user, class, description, now);
        state.schedule(now);
        drop(state);
        Ok(self.admission(id, proxy_id, permit))
    }

    pub fn readmit(
        &self,
        user: UserId,
        class: ClassConfig,
        description: &RequestDescription,
    ) -> Admission {
        let (id, proxy_id, permit) = self
            .lock()
            .admit(user, class, description, Timestamp::now());
        self.admission(id, proxy_id, permit)
    }

    fn admission(
        &self,
        id: RequestId,
        proxy_id: ProxyId,
        permit: oneshot::Receiver<Rank>,
    ) -> Admission {
        Admission {
            proxy_id,
            permit,
            registration: self.registration(id),
        }
    }

    pub fn adopt(&self, id: RequestId, started: StartedRequest) -> Registration {
        self.lock().adopt(id, started, Timestamp::now());
        self.registration(id)
    }

    pub fn registration(&self, id: RequestId) -> Registration {
        Registration {
            scheduler: Some(self.clone()),
            id,
        }
    }
}

impl Registration {
    pub fn complete(mut self, engine_status: Option<u16>) {
        if let Some(scheduler) = self.scheduler.take() {
            scheduler
                .lock()
                .complete(self.id, engine_status, Timestamp::now());
        }
    }

    pub fn detach(mut self) -> RequestId {
        self.scheduler = None;
        self.id
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Some(scheduler) = &self.scheduler {
            scheduler.lock().abandon(self.id, Timestamp::now());
        }
    }
}

impl SchedulerState {
    fn queued_count(&self) -> usize {
        self.requests
            .iter()
            .filter(|request| request.request_state == RequestState::New)
            .count()
    }

    fn user_index(&mut self, user: UserId, now: Timestamp) -> usize {
        if let Some(index) = self.users.iter().position(|existing| existing.id == user) {
            return index;
        }
        self.users.push(User {
            id: user,
            usage_history: UsageHistory::new(now),
        });
        self.users.len() - 1
    }

    fn admit(
        &mut self,
        user: UserId,
        class: ClassConfig,
        description: &RequestDescription,
        now: Timestamp,
    ) -> (RequestId, ProxyId, oneshot::Receiver<Rank>) {
        let user_index = self.user_index(user, now);
        let id = self.proxy_ids.next();
        let proxy_id = self.proxy_ids.proxy_id(id);
        debug!(
            "admission {proxy_id}: user {}, class {}",
            self.users[user_index].id, class.name
        );
        self.log.record(LogEvent::Admitted {
            request: proxy_id,
            user: &self.users[user_index].id.0,
            class: &class.name,
            priority: class.priority,
            path: &description.path,
            model: description.model.as_deref(),
            stream: description.is_streaming,
            max_tokens: description.max_tokens,
            body_bytes: description.body_bytes,
        });
        self.requests.push(Request::new(id, user_index, class, now));
        let (sender, permit) = oneshot::channel();
        self.permits.insert(id, sender);
        (id, proxy_id, permit)
    }

    fn adopt(&mut self, id: RequestId, started: StartedRequest, now: Timestamp) {
        let user_index = self.user_index(started.user, now);
        let mut request = Request::new(id, user_index, handed_over_class(started.priority), now);
        request.rank = Some(started.rank);
        request.request_state = if started.is_suspended {
            RequestState::Suspended
        } else {
            request
                .usage
                .start(&mut self.users[user_index].usage_history, 0.0, now);
            RequestState::Running
        };
        self.log.record(LogEvent::Adopted {
            request: self.proxy_ids.proxy_id(id),
            user: &self.users[user_index].id.0,
            priority: started.priority,
            rank: started.rank.0,
            state: request.request_state.into(),
        });
        self.requests.push(request);
    }

    fn complete(&mut self, id: RequestId, engine_status: Option<u16>, now: Timestamp) {
        self.log.record(LogEvent::Completed {
            request: self.proxy_ids.proxy_id(id),
            status: engine_status,
        });
        self.permits.remove(&id);
        if let Some(request) = remove_request(&mut self.users, &mut self.requests, id, now) {
            self.finish(request, now);
        }
        self.schedule(now);
    }

    fn abandon(&mut self, id: RequestId, now: Timestamp) {
        self.log.record(LogEvent::Aborted {
            request: self.proxy_ids.proxy_id(id),
        });
        self.permits.remove(&id);
        if let Some(request) = abandon_request(&mut self.users, &mut self.requests, id, now) {
            self.finish(request, now);
        }
        self.schedule(now);
    }

    pub fn schedule(&mut self, now: Timestamp) {
        for request in remove_expired_requests(&mut self.users, &mut self.requests, now) {
            self.finish(request, now);
        }
        let allocation =
            calculate_quality_of_service(&mut self.users, &mut self.requests, &self.settings, now);
        let actions = execute_quality_of_service(&mut self.requests, now);
        if self.caps_in_log.as_ref() != Some(&allocation.caps) {
            let caps = &allocation.caps;
            self.log.record(LogEvent::Caps {
                has_multiple_users: caps.has_multiple_users,
                soft_cap: caps.soft_cap,
                rank_caps: &caps.rank_caps,
            });
            self.caps_in_log = Some(allocation.caps.clone());
        }
        self.log_actions(&actions, &allocation.wait_reasons);
        self.apply(actions);
    }

    fn log_actions(&self, actions: &[Action], wait_reasons: &HashMap<RequestId, WaitReason>) {
        let proxy_id = |id: RequestId| self.proxy_ids.proxy_id(id);
        for action in actions {
            match action {
                Action::Start { request, rank } => {
                    let Some(request) = self.requests.iter().find(|item| item.id == *request)
                    else {
                        continue;
                    };
                    let queue_time = request
                        .start_time
                        .map(|start_time| start_time.saturating_duration_since(request.arrival))
                        .unwrap_or_default();
                    self.log.record(LogEvent::Started {
                        request: proxy_id(request.id),
                        rank: rank.0,
                        queued_seconds: queue_time.as_secs_f64(),
                    });
                }
                Action::Suspend { rank, requests } => {
                    for id in requests {
                        self.log.record(LogEvent::Suspended {
                            request: proxy_id(*id),
                            rank: rank.0,
                            reason: wait_reasons.get(id).copied(),
                        });
                    }
                }
                Action::Resume { rank, requests } => {
                    for id in requests {
                        self.log.record(LogEvent::Resumed {
                            request: proxy_id(*id),
                            rank: rank.0,
                        });
                    }
                }
            }
        }
    }

    fn finish(&mut self, request: Request, now: Timestamp) {
        let generating_seconds = request.usage.generating_time().as_secs_f64();
        self.log.record(LogEvent::Finished {
            request: self.proxy_ids.proxy_id(request.id),
            user: &self.users[request.user_index].id.0,
            class: &request.class.name,
            priority: request.class.priority,
            total_seconds: now.saturating_duration_since(request.arrival).as_secs_f64(),
            queued_seconds: request.start_time.map(|start_time| {
                start_time
                    .saturating_duration_since(request.arrival)
                    .as_secs_f64()
            }),
            generating_seconds,
            usage_seconds: generating_seconds * f64::from(request.class.cost),
            suspensions: request.suspension_count,
        });
        self.apply(request.release_action());
    }

    pub fn log_handover(&self) {
        for request in &self.requests {
            self.log.record(LogEvent::HandedOver {
                request: self.proxy_ids.proxy_id(request.id),
                state: request.request_state.into(),
            });
        }
    }

    fn apply(&mut self, actions: impl IntoIterator<Item = Action>) {
        for action in actions {
            let (rank, message) = match action {
                Action::Start { request, rank } => {
                    let Some(permit) = self.permits.remove(&request) else {
                        continue;
                    };
                    (rank, LinkMessage::GrantStart(permit))
                }
                Action::Suspend { rank, requests } => {
                    let ids = requests
                        .into_iter()
                        .map(|id| self.proxy_ids.proxy_id(id))
                        .collect();
                    (rank, LinkMessage::Command(Command::Suspend { ids }))
                }
                Action::Resume { rank, requests } => {
                    let ids = requests
                        .into_iter()
                        .map(|id| self.proxy_ids.proxy_id(id))
                        .collect();
                    (rank, LinkMessage::Command(Command::Resume { ids }))
                }
            };
            let _ = self.control_links[rank.0].send(message);
        }
    }

    pub fn tick(&mut self, now: Timestamp) {
        self.schedule(now);
        self.remove_idle_users();
    }

    fn remove_idle_users(&mut self) {
        let mut is_active = vec![false; self.users.len()];
        for request in &self.requests {
            is_active[request.user_index] = true;
        }
        let mut new_indices = vec![usize::MAX; self.users.len()];
        let mut retained = Vec::with_capacity(self.users.len());
        for (index, user) in std::mem::take(&mut self.users).into_iter().enumerate() {
            if is_active[index] || !user.usage_history.is_empty() {
                new_indices[index] = retained.len();
                retained.push(user);
            }
        }
        self.users = retained;
        for request in &mut self.requests {
            request.user_index = new_indices[request.user_index];
        }
    }

    pub fn engine_event(&mut self, rank: Rank, event: Event, now: Timestamp) {
        let (proxy_id, message) = match event {
            Event::Running { id } => (id, EngineMessage::Running(RequestId(id.sequence))),
            Event::Stopped { id, reason } => {
                (id, EngineMessage::Stopped(RequestId(id.sequence), reason))
            }
        };
        if proxy_id.instance != self.proxy_ids.instance {
            return;
        }
        self.log.record(match message {
            EngineMessage::Running(_) => LogEvent::EngineRunning {
                request: proxy_id,
                rank: rank.0,
            },
            EngineMessage::Stopped(_, reason) => LogEvent::EngineStopped {
                request: proxy_id,
                rank: rank.0,
                reason,
            },
        });
        if let Some(request) =
            handle_engine_message(&mut self.users, &mut self.requests, rank, message, now)
        {
            self.finish(request, now);
        }
        if let EngineMessage::Stopped(_, StopReason::Finished) = message {
            self.schedule(now);
        }
    }

    pub fn resynchronization_commands(&self, rank: Rank) -> Vec<Command> {
        let mut suspended = Vec::new();
        let mut running = Vec::new();
        for request in self
            .requests
            .iter()
            .filter(|request| request.rank == Some(rank))
        {
            let ids = match request.request_state {
                RequestState::Suspended | RequestState::Abandoned { is_suspended: true } => {
                    &mut suspended
                }
                RequestState::Running => &mut running,
                _ => continue,
            };
            ids.push(self.proxy_ids.proxy_id(request.id));
        }
        [
            (!suspended.is_empty()).then_some(Command::Suspend { ids: suspended }),
            (!running.is_empty()).then_some(Command::Resume { ids: running }),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    pub fn status(&self) -> Status {
        status(&self.requests)
    }

    pub fn proxy_ids(&self) -> ProxyIdCounter {
        self.proxy_ids
    }

    pub fn started_request(&self, id: RequestId) -> Option<StartedRequest> {
        let request = self.requests.iter().find(|request| request.id == id)?;
        let is_suspended = match request.request_state {
            RequestState::Running => false,
            RequestState::Suspended => true,
            RequestState::New | RequestState::Abandoned { .. } => return None,
        };
        Some(StartedRequest {
            user: self.users[request.user_index].id.clone(),
            priority: request.class.priority,
            rank: request.rank?,
            is_suspended,
        })
    }

    pub fn release_abandoned(&mut self, now: Timestamp) {
        let abandoned: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|request| matches!(request.request_state, RequestState::Abandoned { .. }))
            .map(|request| request.id)
            .collect();
        for id in abandoned {
            if let Some(request) = remove_request(&mut self.users, &mut self.requests, id, now) {
                self.finish(request, now);
            }
        }
    }
}
