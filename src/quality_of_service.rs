use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::num::NonZeroUsize;
use std::time::Duration;

use serde::Serialize;

use crate::config::{rank_share, ClassConfig};
use crate::engine_protocol::StopReason;
use crate::utils::Timestamp;

pub const ABANDONED_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
pub const USER_ID_SEPARATOR: char = '/';
pub const USAGE_BUCKET: Duration = Duration::from_secs(60);
const RETAINED_HALF_LIVES: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rank(pub usize);

impl fmt::Display for Rank {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UserId(pub String);

impl UserId {
    pub fn new(account: &str, virtual_user: Option<&str>) -> UserId {
        match virtual_user {
            Some(virtual_user) => UserId(format!("{account}{USER_ID_SEPARATOR}{virtual_user}")),
            None => UserId(account.to_string()),
        }
    }

    pub fn account(&self) -> &str {
        self.0.split(USER_ID_SEPARATOR).next().unwrap_or_default()
    }
}

impl fmt::Display for UserId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct User {
    pub id: UserId,
    pub usage_history: UsageHistory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestState {
    New,
    Running,
    Suspended,
    Abandoned { is_suspended: bool },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub id: RequestId,
    pub user_index: usize,
    pub class: ClassConfig,
    pub arrival: Timestamp,
    pub rank: Option<Rank>,
    pub request_state: RequestState,
    pub should_be_running: bool,
    pub state_since: Timestamp,
    pub start_time: Option<Timestamp>,
    pub suspension_count: u32,
    pub usage: RequestUsage,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Allocation {
    pub caps: Caps,
    pub wait_reasons: HashMap<RequestId, WaitReason>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Caps {
    pub has_multiple_users: bool,
    pub soft_cap: Option<NonZeroUsize>,
    pub rank_caps: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitReason {
    IdleOnly,
    MaxParallelPerUser,
    SoftCap,
    HardCap,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub rank_capacities: Vec<usize>,
    pub swap_hysteresis: Duration,
    pub usage_half_life: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineMessage {
    Running(RequestId),
    Stopped(RequestId, StopReason),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    Start {
        request: RequestId,
        rank: Rank,
    },
    Suspend {
        rank: Rank,
        requests: Vec<RequestId>,
    },
    Resume {
        rank: Rank,
        requests: Vec<RequestId>,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub total: RequestCounts,
    pub classes: Vec<ClassStatus>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestCounts {
    pub requests: usize,
    pub users: usize,
    pub running: usize,
    pub generating: usize,
    pub suspended: usize,
    pub queued: usize,
    pub abandoned: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassStatus {
    pub name: String,
    pub priority: i32,
    pub counts: RequestCounts,
}

impl Request {
    pub fn new(
        id: RequestId,
        user_index: usize,
        class: ClassConfig,
        arrival: Timestamp,
    ) -> Request {
        Request {
            id,
            user_index,
            class,
            arrival,
            rank: None,
            request_state: RequestState::New,
            should_be_running: false,
            state_since: arrival,
            start_time: None,
            suspension_count: 0,
            usage: RequestUsage::default(),
        }
    }

    fn is_protected(&self, now: Timestamp, swap_hysteresis: Duration) -> bool {
        self.request_state == RequestState::Running
            && now.saturating_duration_since(self.state_since) < swap_hysteresis
    }

    fn is_started(&self) -> bool {
        self.request_state != RequestState::New
    }

    fn is_abandoned(&self) -> bool {
        matches!(self.request_state, RequestState::Abandoned { .. })
    }

    fn is_expired(&self, now: Timestamp) -> bool {
        self.is_abandoned()
            && now.saturating_duration_since(self.state_since) >= ABANDONED_REQUEST_TIMEOUT
    }

    fn is_suspended_in_engine(&self) -> bool {
        matches!(
            self.request_state,
            RequestState::Suspended | RequestState::Abandoned { is_suspended: true }
        )
    }

    pub fn release_action(&self) -> Option<Action> {
        let rank = self.rank.filter(|_| self.is_suspended_in_engine())?;
        Some(Action::Resume {
            rank,
            requests: vec![self.id],
        })
    }
}

pub fn calculate_quality_of_service(
    users: &mut [User],
    requests: &mut [Request],
    settings: &Settings,
    now: Timestamp,
) -> Allocation {
    for request in requests.iter_mut() {
        let history = &mut users[request.user_index].usage_history;
        request.usage.tick(history, now);
        request.should_be_running = false;
    }
    for user in users.iter_mut() {
        user.usage_history
            .discard_expired(now, settings.usage_half_life);
    }

    let rank_count = settings.rank_capacities.len();
    let mut assigned = vec![0usize; rank_count];
    let mut current_load = vec![0usize; rank_count];
    for request in requests.iter() {
        match (request.request_state, request.rank) {
            (
                RequestState::Abandoned {
                    is_suspended: false,
                },
                Some(rank),
            ) => assigned[rank.0] += 1,
            (RequestState::Running, Some(rank)) => current_load[rank.0] += 1,
            _ => {}
        }
    }

    let candidates: Vec<usize> = (0..requests.len())
        .filter(|&index| !requests[index].is_abandoned())
        .collect();
    let heaviness: Vec<f32> = users
        .iter()
        .map(|user| user.usage_history.heaviness(now, settings.usage_half_life))
        .collect();
    let highest_class_priority = candidates
        .iter()
        .map(|&index| requests[index].class.priority)
        .max()
        .unwrap_or(i32::MIN);
    let has_multiple_users = candidates
        .iter()
        .map(|&index| requests[index].user_index)
        .collect::<HashSet<_>>()
        .len()
        > 1;
    let soft_cap = candidates
        .iter()
        .filter_map(|&index| requests[index].class.soft_cap)
        .min()
        .filter(|_| has_multiple_users);
    let tier_positions = positions_within_user_tiers(
        requests,
        &candidates,
        has_multiple_users,
        settings.swap_hysteresis,
        now,
    );
    let is_guaranteed = |index: usize| tier_positions[index] == 0;
    let effective_rank_caps: Vec<usize> = settings
        .rank_capacities
        .iter()
        .enumerate()
        .map(|(rank, &hard_cap)| {
            soft_cap.map_or(hard_cap, |soft_cap| {
                hard_cap.min(rank_share(soft_cap.get(), rank_count, rank))
            })
        })
        .collect();

    let mut order = candidates;
    order.sort_by(|&left, &right| {
        let precedence = |index: usize| {
            (
                Reverse(requests[index].class.priority),
                Reverse(is_guaranteed(index)),
                Reverse(requests[index].is_protected(now, settings.swap_hysteresis)),
                tier_positions[index],
            )
        };
        let (first, second) = (&requests[left], &requests[right]);
        precedence(left)
            .cmp(&precedence(right))
            .then(heaviness[first.user_index].total_cmp(&heaviness[second.user_index]))
            .then(second.is_started().cmp(&first.is_started()))
            .then(first.arrival.cmp(&second.arrival))
            .then(first.id.cmp(&second.id))
    });

    let mut running_per_user_class: HashMap<(usize, String), usize> = HashMap::new();
    let mut wait_reasons = HashMap::new();
    for index in order {
        let request = &requests[index];
        let user_class = (request.user_index, request.class.name.clone());
        let placement = 'placement: {
            if request.class.idle_only && request.class.priority < highest_class_priority {
                break 'placement Err(WaitReason::IdleOnly);
            }
            let user_class_running = running_per_user_class
                .get(&user_class)
                .copied()
                .unwrap_or(0);
            if has_multiple_users
                && request
                    .class
                    .max_parallel_per_user
                    .is_some_and(|limit| user_class_running >= limit.get())
            {
                break 'placement Err(WaitReason::MaxParallelPerUser);
            }
            let is_soft_cap_exempt =
                is_guaranteed(index) && request.class.priority == highest_class_priority;
            let rank_caps = if is_soft_cap_exempt {
                &settings.rank_capacities
            } else {
                &effective_rank_caps
            };
            let free_capacity = |rank: usize| rank_caps[rank].saturating_sub(assigned[rank]);
            let mut candidate_ranks = match (request.is_started(), request.rank) {
                (true, Some(rank)) => rank.0..rank.0 + 1,
                _ => 0..rank_count,
            };
            candidate_ranks
                .clone()
                .filter(|&rank| free_capacity(rank) > 0)
                .min_by_key(|&rank| (Reverse(free_capacity(rank)), current_load[rank], rank))
                .ok_or_else(|| {
                    let is_below_hard_cap =
                        candidate_ranks.any(|rank| assigned[rank] < settings.rank_capacities[rank]);
                    if is_below_hard_cap && !is_soft_cap_exempt {
                        WaitReason::SoftCap
                    } else {
                        WaitReason::HardCap
                    }
                })
        };
        match placement {
            Ok(rank) => {
                assigned[rank] += 1;
                *running_per_user_class.entry(user_class).or_default() += 1;
                let request = &mut requests[index];
                request.rank = Some(Rank(rank));
                request.should_be_running = true;
            }
            Err(reason) => {
                wait_reasons.insert(request.id, reason);
            }
        }
    }
    Allocation {
        caps: Caps {
            has_multiple_users,
            soft_cap,
            rank_caps: effective_rank_caps,
        },
        wait_reasons,
    }
}

fn positions_within_user_tiers(
    requests: &[Request],
    candidates: &[usize],
    has_multiple_users: bool,
    turn: Duration,
    now: Timestamp,
) -> Vec<usize> {
    let mut groups: HashMap<(usize, i32), Vec<usize>> = HashMap::new();
    for &index in candidates {
        let request = &requests[index];
        groups
            .entry((request.user_index, request.class.priority))
            .or_default()
            .push(index);
    }
    let mut positions = vec![usize::MAX; requests.len()];
    for mut members in groups.into_values() {
        let has_turns = has_multiple_users
            && members
                .iter()
                .any(|&index| requests[index].class.max_parallel_per_user.is_some());
        if has_turns {
            members.sort_by_key(|&index| turn_order(&requests[index], turn, now));
        } else {
            members.sort_by_key(|&index| {
                let request = &requests[index];
                (!request.is_started(), request.arrival, request.id)
            });
        }
        for (position, index) in members.into_iter().enumerate() {
            positions[index] = position;
        }
    }
    positions
}

fn turn_order(request: &Request, turn: Duration, now: Timestamp) -> impl Ord {
    let is_running = request.request_state == RequestState::Running;
    let queue_time = if is_running {
        let completed_turns = now
            .saturating_duration_since(request.state_since)
            .as_nanos()
            .checked_div(turn.as_nanos())
            .unwrap_or(0);
        request.state_since + turn * u32::try_from(completed_turns).unwrap_or(u32::MAX)
    } else {
        request.state_since
    };
    (
        !request.is_protected(now, turn),
        queue_time,
        !is_running,
        Reverse(request.state_since),
        request.id,
    )
}

pub fn remove_expired_requests(
    users: &mut [User],
    requests: &mut Vec<Request>,
    now: Timestamp,
) -> Vec<Request> {
    let expired: Vec<RequestId> = requests
        .iter()
        .filter(|request| request.is_expired(now))
        .map(|request| request.id)
        .collect();
    expired
        .into_iter()
        .filter_map(|id| remove_request(users, requests, id, now))
        .collect()
}

pub fn execute_quality_of_service(requests: &mut [Request], now: Timestamp) -> Vec<Action> {
    let mut suspensions: BTreeMap<Rank, Vec<RequestId>> = BTreeMap::new();
    let mut resumptions: BTreeMap<Rank, Vec<RequestId>> = BTreeMap::new();
    let mut starts = Vec::new();
    for request in requests.iter_mut() {
        let next_state = match (
            request.request_state,
            request.should_be_running,
            request.rank,
        ) {
            (RequestState::New, true, Some(rank)) => {
                starts.push(Action::Start {
                    request: request.id,
                    rank,
                });
                request.start_time = Some(now);
                RequestState::Running
            }
            (RequestState::Running, false, Some(rank)) => {
                suspensions.entry(rank).or_default().push(request.id);
                request.suspension_count += 1;
                RequestState::Suspended
            }
            (RequestState::Suspended, true, Some(rank)) => {
                resumptions.entry(rank).or_default().push(request.id);
                RequestState::Running
            }
            _ => continue,
        };
        request.request_state = next_state;
        request.state_since = now;
    }
    suspensions
        .into_iter()
        .map(|(rank, requests)| Action::Suspend { rank, requests })
        .chain(
            resumptions
                .into_iter()
                .map(|(rank, requests)| Action::Resume { rank, requests }),
        )
        .chain(starts)
        .collect()
}

pub fn handle_engine_message(
    users: &mut [User],
    requests: &mut Vec<Request>,
    rank: Rank,
    message: EngineMessage,
    now: Timestamp,
) -> Option<Request> {
    let id = match message {
        EngineMessage::Stopped(id, StopReason::Finished) => {
            return remove_request(users, requests, id, now)
        }
        EngineMessage::Running(id) | EngineMessage::Stopped(id, _) => id,
    };
    let request = requests.iter_mut().find(|request| request.id == id)?;
    let history = &mut users[request.user_index].usage_history;
    match message {
        EngineMessage::Running(_) => {
            request.rank = Some(rank);
            request.usage.start(history, request.class.cost, now);
        }
        EngineMessage::Stopped(..) => request.usage.end(history, now),
    }
    None
}

pub fn remove_request(
    users: &mut [User],
    requests: &mut Vec<Request>,
    id: RequestId,
    now: Timestamp,
) -> Option<Request> {
    let index = requests.iter().position(|request| request.id == id)?;
    let mut request = requests.swap_remove(index);
    request
        .usage
        .end(&mut users[request.user_index].usage_history, now);
    Some(request)
}

pub fn abandon_request(
    users: &mut [User],
    requests: &mut Vec<Request>,
    id: RequestId,
    now: Timestamp,
) -> Option<Request> {
    let request = requests.iter_mut().find(|request| request.id == id)?;
    request.request_state = match request.request_state {
        RequestState::New => return remove_request(users, requests, id, now),
        RequestState::Running => RequestState::Abandoned {
            is_suspended: false,
        },
        RequestState::Suspended => RequestState::Abandoned { is_suspended: true },
        abandoned @ RequestState::Abandoned { .. } => abandoned,
    };
    request.state_since = now;
    None
}

pub fn status(requests: &[Request]) -> Status {
    let mut classes: BTreeMap<(Reverse<i32>, &str), Vec<&Request>> = BTreeMap::new();
    for request in requests {
        let class = &request.class;
        classes
            .entry((Reverse(class.priority), class.name.as_str()))
            .or_default()
            .push(request);
    }
    Status {
        total: RequestCounts::of(requests),
        classes: classes
            .into_iter()
            .map(|((Reverse(priority), name), requests)| ClassStatus {
                name: name.to_string(),
                priority,
                counts: RequestCounts::of(requests),
            })
            .collect(),
    }
}

impl RequestCounts {
    fn of<'a>(requests: impl IntoIterator<Item = &'a Request>) -> RequestCounts {
        let mut counts = RequestCounts::default();
        let mut users = HashSet::new();
        for request in requests {
            counts.requests += 1;
            users.insert(request.user_index);
            match request.request_state {
                RequestState::New => counts.queued += 1,
                RequestState::Running => counts.running += 1,
                RequestState::Suspended => counts.suspended += 1,
                RequestState::Abandoned { .. } => counts.abandoned += 1,
            }
            if request.usage.is_generating() {
                counts.generating += 1;
            }
        }
        counts.users = users.len();
        counts
    }

    fn requests_and_users(&self) -> String {
        format!(
            "{} from {}",
            counted(self.requests, "request"),
            counted(self.users, "user")
        )
    }
}

fn counted(count: usize, noun: &str) -> String {
    match count {
        1 => format!("1 {noun}"),
        _ => format!("{count} {noun}s"),
    }
}

impl fmt::Display for RequestCounts {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}: {} running ({} generating in the engine), {} suspended, {} queued, {} abandoned",
            self.requests_and_users(),
            self.running,
            self.generating,
            self.suspended,
            self.queued,
            self.abandoned
        )
    }
}

impl fmt::Display for ClassStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let counts = &self.counts;
        let states = [
            (
                counts.running + counts.generating,
                format!(
                    "{} running ({} generating)",
                    counts.running, counts.generating
                ),
            ),
            (counts.suspended, format!("{} suspended", counts.suspended)),
            (counts.queued, format!("{} queued", counts.queued)),
            (counts.abandoned, format!("{} abandoned", counts.abandoned)),
        ];
        let states: Vec<String> = states
            .into_iter()
            .filter(|(count, _)| *count > 0)
            .map(|(_, text)| text)
            .collect();
        write!(
            formatter,
            "{} (priority {}): {}: {}",
            self.name,
            self.priority,
            counts.requests_and_users(),
            states.join(", ")
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UsageHistory {
    usage_buckets: Vec<f32>,
    usage_start_at: Timestamp,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestUsage {
    generating_since: Option<Timestamp>,
    generating_time: Duration,
    cost: f32,
}

impl UsageHistory {
    pub fn new(start: Timestamp) -> UsageHistory {
        UsageHistory {
            usage_buckets: Vec::new(),
            usage_start_at: start,
        }
    }

    pub fn heaviness(&self, now: Timestamp, half_life: Duration) -> f32 {
        self.usage_buckets
            .iter()
            .enumerate()
            .map(|(index, seconds)| {
                let bucket_end = self.bucket_start(index + 1);
                let age = now.saturating_duration_since(bucket_end);
                seconds * 0.5f32.powf(age.as_secs_f32() / half_life.as_secs_f32())
            })
            .sum()
    }

    pub fn discard_expired(&mut self, now: Timestamp, half_life: Duration) {
        let retention = half_life * RETAINED_HALF_LIVES;
        let expired = (0..self.usage_buckets.len())
            .take_while(|index| {
                now.saturating_duration_since(self.bucket_start(index + 1)) > retention
            })
            .count();
        self.usage_buckets.drain(..expired);
        self.usage_start_at = self.bucket_start(expired);
    }

    pub fn is_empty(&self) -> bool {
        self.usage_buckets.iter().all(|seconds| *seconds == 0.0)
    }

    fn record(&mut self, from: Timestamp, to: Timestamp, cost: f32) {
        let mut position = from.max(self.usage_start_at);
        while position < to {
            let index = (position
                .saturating_duration_since(self.usage_start_at)
                .as_nanos()
                / USAGE_BUCKET.as_nanos()) as usize;
            let segment_end = to.min(self.bucket_start(index + 1));
            if self.usage_buckets.len() <= index {
                self.usage_buckets.resize(index + 1, 0.0);
            }
            self.usage_buckets[index] += segment_end
                .saturating_duration_since(position)
                .as_secs_f32()
                * cost;
            position = segment_end;
        }
    }

    fn bucket_start(&self, index: usize) -> Timestamp {
        self.usage_start_at + USAGE_BUCKET * index as u32
    }
}

impl RequestUsage {
    pub fn start(&mut self, history: &mut UsageHistory, cost: f32, timestamp: Timestamp) {
        self.tick(history, timestamp);
        self.cost = cost;
        self.generating_since.get_or_insert(timestamp);
    }

    pub fn end(&mut self, history: &mut UsageHistory, timestamp: Timestamp) {
        self.tick(history, timestamp);
        self.generating_since = None;
    }

    pub fn tick(&mut self, history: &mut UsageHistory, timestamp: Timestamp) {
        if let Some(since) = self.generating_since {
            history.record(since, timestamp, self.cost);
            self.generating_time += timestamp.saturating_duration_since(since);
            self.generating_since = Some(since.max(timestamp));
        }
    }

    pub fn is_generating(&self) -> bool {
        self.generating_since.is_some()
    }

    pub fn generating_time(&self) -> Duration {
        self.generating_time
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::time::Duration;

    use super::*;

    const HYSTERESIS: Duration = Duration::from_secs(30);
    const TICK: Duration = Duration::from_millis(250);
    const HALF_LIFE: Duration = Duration::from_secs(3600);

    fn class(name: &str, priority: i32) -> ClassConfig {
        ClassConfig {
            name: name.to_string(),
            priority,
            idle_only: false,
            cost: 1.0,
            soft_cap: None,
            max_parallel_per_user: None,
        }
    }

    fn soft_capped(name: &str, priority: i32, soft_cap: usize) -> ClassConfig {
        ClassConfig {
            soft_cap: NonZeroUsize::new(soft_cap),
            ..class(name, priority)
        }
    }

    fn limited_per_user(name: &str, priority: i32, max_parallel_per_user: usize) -> ClassConfig {
        ClassConfig {
            max_parallel_per_user: NonZeroUsize::new(max_parallel_per_user),
            ..class(name, priority)
        }
    }

    fn idle_class() -> ClassConfig {
        ClassConfig {
            idle_only: true,
            ..class("idle", -10)
        }
    }

    struct Simulation {
        users: Vec<User>,
        requests: Vec<Request>,
        settings: Settings,
        now: Timestamp,
        next_id: u64,
        actions: Vec<Action>,
        allocation: Option<Allocation>,
    }

    impl Simulation {
        fn new(rank_capacities: Vec<usize>) -> Simulation {
            Simulation {
                users: Vec::new(),
                requests: Vec::new(),
                settings: Settings {
                    rank_capacities,
                    swap_hysteresis: HYSTERESIS,
                    usage_half_life: HALF_LIFE,
                },
                now: Timestamp::default() + Duration::from_secs(1_000_000),
                next_id: 0,
                actions: Vec::new(),
                allocation: None,
            }
        }

        fn user_index(&mut self, account: &str) -> usize {
            let id = UserId::new(account, None);
            if let Some(index) = self.users.iter().position(|user| user.id == id) {
                return index;
            }
            self.users.push(User {
                id,
                usage_history: UsageHistory::new(self.now),
            });
            self.users.len() - 1
        }

        fn arrival(&mut self, account: &str, class: &ClassConfig) -> RequestId {
            let id = self.arrival_without_step(account, class);
            self.step();
            id
        }

        fn arrivals(&mut self, account: &str, class: &ClassConfig, count: usize) -> Vec<RequestId> {
            (0..count).map(|_| self.arrival(account, class)).collect()
        }

        fn arrivals_from_users(&mut self, class: &ClassConfig, count: usize) -> Vec<RequestId> {
            (0..count)
                .map(|index| self.arrival(&format!("user-{index}"), class))
                .collect()
        }

        fn arrival_without_step(&mut self, account: &str, class: &ClassConfig) -> RequestId {
            let user_index = self.user_index(account);
            let id = RequestId(self.next_id);
            self.next_id += 1;
            self.requests
                .push(Request::new(id, user_index, class.clone(), self.now));
            id
        }

        fn step(&mut self) {
            let expired = remove_expired_requests(&mut self.users, &mut self.requests, self.now);
            self.allocation = Some(calculate_quality_of_service(
                &mut self.users,
                &mut self.requests,
                &self.settings,
                self.now,
            ));
            self.actions = expired.iter().filter_map(Request::release_action).collect();
            self.actions
                .extend(execute_quality_of_service(&mut self.requests, self.now));
            for action in self.actions.clone() {
                let (rank, ids, message): (Rank, Vec<RequestId>, fn(RequestId) -> EngineMessage) =
                    match action {
                        Action::Start { request, rank } => {
                            (rank, vec![request], EngineMessage::Running)
                        }
                        Action::Resume { rank, requests } => {
                            (rank, requests, EngineMessage::Running)
                        }
                        Action::Suspend { rank, requests } => (rank, requests, |id| {
                            EngineMessage::Stopped(id, StopReason::Suspended)
                        }),
                    };
                for id in ids {
                    self.engine(rank, message(id));
                }
            }
        }

        fn engine(&mut self, rank: Rank, message: EngineMessage) -> Option<Request> {
            handle_engine_message(&mut self.users, &mut self.requests, rank, message, self.now)
        }

        fn advance(&mut self, duration: Duration) {
            let end = self.now + duration;
            while self.now < end {
                self.now = self.now + TICK;
                self.step();
            }
        }

        fn finish(&mut self, id: RequestId) {
            remove_request(&mut self.users, &mut self.requests, id, self.now);
            self.step();
        }

        fn abandon(&mut self, id: RequestId) {
            abandon_request(&mut self.users, &mut self.requests, id, self.now);
            self.step();
        }

        fn request(&self, id: RequestId) -> &Request {
            self.requests
                .iter()
                .find(|request| request.id == id)
                .unwrap()
        }

        fn state(&self, id: RequestId) -> RequestState {
            self.request(id).request_state
        }

        fn states(&self, ids: &[RequestId]) -> Vec<RequestState> {
            ids.iter().map(|id| self.state(*id)).collect()
        }

        fn running_count(&self, ids: &[RequestId]) -> usize {
            ids.iter().filter(|id| self.state(**id) == Running).count()
        }

        fn wait_reason(&self, id: RequestId) -> Option<WaitReason> {
            let allocation = self.allocation.as_ref().unwrap();
            allocation.wait_reasons.get(&id).copied()
        }

        fn rank(&self, id: RequestId) -> Option<Rank> {
            self.request(id).rank
        }

        fn heaviness(&self, account: &str) -> f32 {
            let user = self
                .users
                .iter()
                .find(|user| user.id == UserId::new(account, None))
                .unwrap();
            user.usage_history.heaviness(self.now, HALF_LIFE)
        }
    }

    use RequestState::{New, Running, Suspended};

    #[test]
    fn higher_priority_preemption_at_capacity() {
        let mut simulation = Simulation::new(vec![2]);
        let batch = class("batch", 0);
        let first_batch = simulation.arrival("bob", &batch);
        let second_batch = simulation.arrival("bob", &batch);
        let interactive = simulation.arrival("alice", &class("interactive", 10));
        assert_eq!(simulation.state(interactive), Running);
        assert_eq!(
            simulation.states(&[first_batch, second_batch]),
            vec![Running, Suspended]
        );
        simulation.finish(interactive);
        assert_eq!(
            simulation.states(&[first_batch, second_batch]),
            vec![Running, Running]
        );
    }

    #[test]
    fn suspension_before_start_on_engine() {
        let mut simulation = Simulation::new(vec![1]);
        let batch = simulation.arrival("bob", &class("batch", 0));
        let interactive = simulation.arrival("alice", &class("interactive", 10));
        assert_eq!(
            simulation.actions,
            vec![
                Action::Suspend {
                    rank: Rank(0),
                    requests: vec![batch]
                },
                Action::Start {
                    request: interactive,
                    rank: Rank(0)
                }
            ]
        );
    }

    #[test]
    fn idle_only_suspension_below_capacity() {
        let mut simulation = Simulation::new(vec![8]);
        let idle: Vec<RequestId> = (0..3)
            .map(|_| simulation.arrival("night-batch", &idle_class()))
            .collect();
        assert_eq!(simulation.states(&idle), vec![Running; 3]);
        let batch = simulation.arrival("bob", &class("batch", 0));
        assert_eq!(simulation.state(batch), Running);
        assert_eq!(simulation.states(&idle), vec![Suspended; 3]);
        simulation.advance(Duration::from_secs(5));
        simulation.finish(batch);
        assert_eq!(simulation.states(&idle), vec![Running; 3]);
    }

    #[test]
    fn idle_only_suspension_by_queued_request() {
        let mut simulation = Simulation::new(vec![1]);
        let batch = class("batch", 0);
        let running_batch = simulation.arrival("bob", &batch);
        let idle = simulation.arrival("night-batch", &idle_class());
        assert_eq!(simulation.state(idle), New);
        simulation.finish(running_batch);
        assert_eq!(simulation.state(idle), Running);
        let queued_batch = simulation.arrival("carol", &batch);
        assert_eq!(simulation.state(queued_batch), Running);
        assert_eq!(simulation.state(idle), Suspended);
    }

    #[test]
    fn guaranteed_request_at_hard_cap() {
        let mut simulation = Simulation::new(vec![2]);
        let normal = class("normal", 0);
        let heavy_first = simulation.arrival("heavy", &normal);
        let heavy_second = simulation.arrival("heavy", &normal);
        let light = simulation.arrival("light", &normal);
        assert_eq!(simulation.state(light), Running);
        assert_eq!(
            simulation.states(&[heavy_first, heavy_second]),
            vec![Running, Suspended]
        );
    }

    #[test]
    fn higher_class_before_guaranteed_request() {
        let mut simulation = Simulation::new(vec![1]);
        let higher = simulation.arrival("bob", &class("higher", 1));
        let guaranteed = simulation.arrival("alice", &class("normal", 0));
        assert_eq!(simulation.states(&[higher, guaranteed]), vec![Running, New]);
    }

    #[test]
    fn soft_cap_of_present_classes() {
        let mut simulation = Simulation::new(vec![64]);
        let low_class = soft_capped("low", 0, 32);
        let mut low = simulation.arrivals("bob", &low_class, 32);
        low.push(simulation.arrival("carol", &low_class));
        assert_eq!(simulation.running_count(&low), 32);
        let medium = simulation.arrival("alice", &soft_capped("medium", 10, 16));
        assert_eq!(simulation.state(medium), Running);
        assert_eq!(simulation.running_count(&low), 15);
        simulation.finish(medium);
        assert_eq!(simulation.running_count(&low), 32);
    }

    #[test]
    fn single_user_above_soft_cap() {
        let mut simulation = Simulation::new(vec![64]);
        let low = soft_capped("low", 0, 32);
        let bob = simulation.arrivals("bob", &low, 40);
        assert_eq!(simulation.running_count(&bob), 40);
        let carol = simulation.arrival("carol", &low);
        assert_eq!(simulation.state(carol), Running);
        assert_eq!(simulation.running_count(&bob), 31);
        simulation.finish(carol);
        assert_eq!(simulation.running_count(&bob), 40);
    }

    #[test]
    fn guaranteed_requests_above_soft_cap() {
        let mut simulation = Simulation::new(vec![64]);
        let low = simulation.arrivals_from_users(&soft_capped("low", 0, 32), 33);
        assert_eq!(simulation.running_count(&low), 33);
        let medium = simulation.arrival("alice", &soft_capped("medium", 10, 16));
        assert_eq!(simulation.state(medium), Running);
        assert_eq!(simulation.running_count(&low), 15);
        simulation.finish(medium);
        assert_eq!(simulation.running_count(&low), 33);
    }

    #[test]
    fn heavy_user_suspension_at_soft_cap() {
        let mut simulation = Simulation::new(vec![64]);
        let batch = soft_capped("batch", 0, 32);
        let heavy = simulation.arrivals("heavy", &batch, 32);
        let light = simulation.arrival("light", &batch);
        assert_eq!(simulation.state(light), Running);
        assert_eq!(simulation.running_count(&heavy), 31);
    }

    #[test]
    fn lower_class_guaranteed_request_at_soft_cap() {
        let mut simulation = Simulation::new(vec![64]);
        let medium = simulation.arrivals_from_users(&soft_capped("medium", 10, 32), 32);
        let low = simulation.arrival("newcomer", &soft_capped("low", 0, 32));
        assert_eq!(simulation.running_count(&medium), 32);
        assert_eq!(simulation.state(low), New);
    }

    #[test]
    fn hard_cap_below_soft_cap() {
        let mut simulation = Simulation::new(vec![4]);
        let batch = soft_capped("batch", 0, 16);
        let heavy = simulation.arrivals("heavy", &batch, 6);
        assert_eq!(simulation.running_count(&heavy), 4);
        let others = simulation.arrivals_from_users(&batch, 4);
        assert_eq!(
            simulation.running_count(&heavy) + simulation.running_count(&others),
            4
        );
    }

    #[test]
    fn soft_cap_share_per_rank() {
        let mut simulation = Simulation::new(vec![8, 8]);
        let bob = simulation.arrivals("bob", &class("batch", 0), 6);
        let (bob_on_first_rank, bob_on_second_rank): (Vec<RequestId>, Vec<RequestId>) = bob
            .into_iter()
            .partition(|id| simulation.rank(*id) == Some(Rank(0)));
        for id in bob_on_second_rank {
            simulation.finish(id);
        }
        assert_eq!(simulation.running_count(&bob_on_first_rank), 3);
        let carol = simulation.arrival("carol", &soft_capped("capped-batch", 0, 4));
        assert_eq!(simulation.state(carol), Running);
        assert_eq!(simulation.running_count(&bob_on_first_rank), 2);
    }

    #[test]
    fn max_parallel_per_user_and_single_user_exemption() {
        let mut simulation = Simulation::new(vec![32, 32]);
        let medium = limited_per_user("medium", 0, 4);
        let bob = simulation.arrivals("bob", &medium, 10);
        assert_eq!(simulation.running_count(&bob), 10);
        let carol = simulation.arrival("carol", &medium);
        assert_eq!(simulation.state(carol), Running);
        assert_eq!(simulation.running_count(&bob), 4);
        simulation.finish(carol);
        assert_eq!(simulation.running_count(&bob), 10);
    }

    #[test]
    fn max_parallel_per_user_per_class() {
        let mut simulation = Simulation::new(vec![64]);
        let medium = simulation.arrivals("bob", &limited_per_user("medium", 0, 2), 4);
        let low = simulation.arrivals("bob", &limited_per_user("low", -1, 3), 4);
        let unlimited = simulation.arrivals("bob", &class("batch", -1), 4);
        simulation.arrival("carol", &class("batch", -1));
        assert_eq!(simulation.running_count(&medium), 2);
        assert_eq!(simulation.running_count(&low), 3);
        assert_eq!(simulation.running_count(&unlimited), 4);
    }

    #[test]
    fn turns_over_max_parallel_per_user() {
        let mut simulation = Simulation::new(vec![64]);
        let medium = limited_per_user("medium", 0, 2);
        let bob = simulation.arrivals("bob", &medium, 3);
        simulation.arrival("carol", &medium);
        let turn_ticks = (HYSTERESIS.as_millis() / TICK.as_millis()) as u32;
        let turn_count = 6;
        let mut running_ticks = vec![0; bob.len()];
        let mut suspensions = 0;
        let mut previous = simulation.states(&bob);
        for _ in 0..turn_count * turn_ticks {
            simulation.advance(TICK);
            let current = simulation.states(&bob);
            assert!(simulation.running_count(&bob) <= 2);
            for (index, state) in current.iter().enumerate() {
                running_ticks[index] += u32::from(*state == Running);
                suspensions += u32::from(previous[index] == Running && *state == Suspended);
            }
            previous = current;
        }
        let least = *running_ticks.iter().min().unwrap();
        let most = *running_ticks.iter().max().unwrap();
        assert!(least >= 3 * turn_ticks);
        assert!(most - least <= turn_ticks + 1);
        assert!(suspensions <= turn_count);
    }

    #[test]
    fn first_come_order_without_active_limit() {
        let mut at_hard_cap = Simulation::new(vec![3]);
        let bob = at_hard_cap.arrivals("bob", &class("batch", 0), 3);
        at_hard_cap.arrival("carol", &class("batch", 0));
        at_hard_cap.advance(HYSTERESIS * 3);
        assert_eq!(at_hard_cap.states(&bob), vec![Running, Running, Suspended]);

        let mut single_user = Simulation::new(vec![2]);
        let bob = single_user.arrivals("bob", &limited_per_user("medium", 0, 1), 3);
        single_user.advance(HYSTERESIS * 3);
        assert_eq!(single_user.states(&bob), vec![Running, Running, New]);
    }

    #[test]
    fn wait_reasons() {
        let mut at_hard_cap = Simulation::new(vec![1]);
        let batch = at_hard_cap.arrival("bob", &class("batch", 0));
        let interactive = at_hard_cap.arrival("alice", &class("interactive", 10));
        assert_eq!(at_hard_cap.wait_reason(batch), Some(WaitReason::HardCap));
        assert_eq!(at_hard_cap.wait_reason(interactive), None);

        let mut at_soft_cap = Simulation::new(vec![64]);
        let bob = at_soft_cap.arrivals("bob", &soft_capped("batch", 0, 2), 2);
        at_soft_cap.arrival("carol", &soft_capped("batch", 0, 2));
        assert_eq!(at_soft_cap.wait_reason(bob[1]), Some(WaitReason::SoftCap));

        let mut at_user_limit = Simulation::new(vec![64]);
        let bob = at_user_limit.arrivals("bob", &limited_per_user("medium", 0, 1), 2);
        at_user_limit.arrival("carol", &class("batch", 0));
        assert_eq!(
            at_user_limit.wait_reason(bob[1]),
            Some(WaitReason::MaxParallelPerUser)
        );

        let mut idle_only = Simulation::new(vec![64]);
        let idle = idle_only.arrival("night-batch", &idle_class());
        idle_only.arrival("bob", &class("batch", 0));
        assert_eq!(idle_only.wait_reason(idle), Some(WaitReason::IdleOnly));
    }

    #[test]
    fn round_robin_between_users_with_equal_usage() {
        let mut simulation = Simulation::new(vec![2]);
        let batch = class("batch", 0);
        let alice_first = simulation.arrival_without_step("alice", &batch);
        let alice_second = simulation.arrival_without_step("alice", &batch);
        let bob = simulation.arrival_without_step("bob", &batch);
        simulation.step();
        assert_eq!(
            simulation.states(&[alice_first, alice_second, bob]),
            vec![Running, New, Running]
        );
    }

    #[test]
    fn lighter_user_first_within_tier() {
        let mut simulation = Simulation::new(vec![1]);
        let batch = class("batch", 0);
        let heavy_history = simulation.arrival("heavy", &batch);
        simulation.advance(Duration::from_secs(600));
        simulation.finish(heavy_history);
        let heavy = simulation.arrival("heavy", &batch);
        let light = simulation.arrival("light", &batch);
        simulation.advance(HYSTERESIS + TICK);
        assert_eq!(simulation.states(&[heavy, light]), vec![Suspended, Running]);
        assert!(simulation.heaviness("heavy") > simulation.heaviness("light"));
    }

    #[test]
    fn swap_hysteresis_within_tier() {
        let mut simulation = Simulation::new(vec![1]);
        let batch = class("batch", 0);
        let heavy = simulation.arrival("heavy", &batch);
        simulation.advance(Duration::from_secs(5));
        let light = simulation.arrival("light", &batch);
        assert_eq!(simulation.states(&[heavy, light]), vec![Running, New]);
        simulation.advance(HYSTERESIS - Duration::from_secs(5) - TICK * 2);
        assert_eq!(simulation.states(&[heavy, light]), vec![Running, New]);
        simulation.advance(TICK * 2);
        assert_eq!(simulation.states(&[heavy, light]), vec![Suspended, Running]);
    }

    #[test]
    fn swap_frequency_between_equal_users() {
        let mut simulation = Simulation::new(vec![1]);
        let alice = simulation.arrival("alice", &idle_class());
        let bob = simulation.arrival("bob", &idle_class());
        let mut swaps = 0;
        let mut previous = simulation.states(&[alice, bob]);
        for _ in 0..(10 * 60 * 4) {
            simulation.advance(TICK);
            let current = simulation.states(&[alice, bob]);
            swaps += usize::from(current != previous);
            previous = current;
        }
        assert!(swaps > 0);
        assert!(swaps as f64 <= 600.0 / HYSTERESIS.as_secs_f64());
        assert!((simulation.heaviness("alice") - simulation.heaviness("bob")).abs() <= 31.0);
    }

    #[test]
    fn data_parallel_placement() {
        let mut simulation = Simulation::new(vec![2, 2]);
        let batch = class("batch", 0);
        let requests: Vec<RequestId> = (0..4)
            .map(|index| simulation.arrival(&format!("user-{index}"), &batch))
            .collect();
        let ranks: Vec<Option<Rank>> = requests.iter().map(|id| simulation.rank(*id)).collect();
        assert_eq!(
            ranks,
            vec![Some(Rank(0)), Some(Rank(1)), Some(Rank(0)), Some(Rank(1))]
        );
    }

    #[test]
    fn suspended_request_bound_to_rank() {
        let mut simulation = Simulation::new(vec![1, 1]);
        let batch = class("batch", 0);
        let interactive = class("interactive", 10);
        let by_rank = [
            simulation.arrival("bob", &batch),
            simulation.arrival("carol", &batch),
        ];
        let first_urgent = simulation.arrival("alice", &interactive);
        let urgent_rank = simulation.rank(first_urgent).unwrap();
        let displaced = by_rank[urgent_rank.0];
        assert_eq!(simulation.state(displaced), Suspended);
        let other_rank = Rank(1 - urgent_rank.0);
        let second_urgent = simulation.arrival("dave", &interactive);
        assert_eq!(simulation.rank(second_urgent), Some(other_rank));
        simulation.finish(first_urgent);
        assert_eq!(simulation.state(by_rank[other_rank.0]), Suspended);
        assert_eq!(simulation.state(displaced), Running);
        assert_eq!(simulation.rank(displaced), Some(urgent_rank));
    }

    #[test]
    fn abandoned_request_slot_until_engine_finish() {
        let mut simulation = Simulation::new(vec![1]);
        let batch = class("batch", 0);
        let abandoned = simulation.arrival("alice", &batch);
        let queued = simulation.arrival("bob", &batch);
        simulation.abandon(abandoned);
        simulation.engine(
            Rank(0),
            EngineMessage::Stopped(abandoned, StopReason::Preempted),
        );
        simulation.advance(TICK);
        assert_eq!(simulation.state(queued), New);
        simulation.engine(
            Rank(0),
            EngineMessage::Stopped(abandoned, StopReason::Finished),
        );
        simulation.step();
        assert_eq!(simulation.state(queued), Running);
    }

    #[test]
    fn abandoned_request_timeout() {
        let mut simulation = Simulation::new(vec![1]);
        let batch = class("batch", 0);
        let abandoned = simulation.arrival("alice", &batch);
        let queued = simulation.arrival("bob", &batch);
        simulation.abandon(abandoned);
        simulation.advance(ABANDONED_REQUEST_TIMEOUT - TICK);
        assert_eq!(simulation.state(queued), New);
        simulation.advance(TICK);
        assert_eq!(simulation.state(queued), Running);
    }

    #[test]
    fn abandoned_suspended_request_cleanup() {
        let mut simulation = Simulation::new(vec![1]);
        let idle = simulation.arrival("night-batch", &idle_class());
        simulation.arrival("bob", &class("batch", 0));
        assert_eq!(simulation.state(idle), Suspended);
        simulation.abandon(idle);
        assert_eq!(simulation.actions, Vec::new());
        simulation.advance(ABANDONED_REQUEST_TIMEOUT);
        assert!(simulation.actions.contains(&Action::Resume {
            rank: Rank(0),
            requests: vec![idle]
        }));
    }

    #[test]
    fn engine_finish_of_suspended_request() {
        let mut simulation = Simulation::new(vec![1]);
        let idle = simulation.arrival("night-batch", &idle_class());
        simulation.arrival("bob", &class("batch", 0));
        let removed =
            simulation.engine(Rank(0), EngineMessage::Stopped(idle, StopReason::Finished));
        assert_eq!(
            removed.and_then(|request| request.release_action()),
            Some(Action::Resume {
                rank: Rank(0),
                requests: vec![idle]
            })
        );
        assert!(simulation.requests.iter().all(|request| request.id != idle));
    }

    #[test]
    fn usage_history_decay() {
        let start = Timestamp::default() + Duration::from_secs(1_000_000);
        let mut history = UsageHistory::new(start);
        let mut usage = RequestUsage::default();
        usage.start(&mut history, 0.5, start + Duration::from_secs(30));
        usage.tick(&mut history, start + Duration::from_secs(90));
        usage.end(&mut history, start + Duration::from_secs(150));
        let now = start + Duration::from_secs(180);
        let fresh = history.heaviness(now, HALF_LIFE);
        assert!((fresh - 60.0).abs() < 1.0);
        let one_half_life_later = history.heaviness(now + HALF_LIFE, HALF_LIFE);
        assert!((one_half_life_later - fresh / 2.0).abs() < 0.01);
        history.discard_expired(now + HALF_LIFE * 9, HALF_LIFE);
        assert!(history.is_empty());
    }

    #[test]
    fn usage_only_while_generating_in_engine() {
        let mut simulation = Simulation::new(vec![1]);
        let batch = ClassConfig {
            cost: 2.0,
            ..class("batch", 0)
        };
        let request = simulation.arrival("bob", &batch);
        simulation.advance(Duration::from_secs(10));
        simulation.engine(
            Rank(0),
            EngineMessage::Stopped(request, StopReason::Preempted),
        );
        simulation.advance(Duration::from_secs(10));
        assert!((simulation.heaviness("bob") - 20.0).abs() < 0.1);
        assert_eq!(simulation.state(request), Running);
        assert_eq!(status(&simulation.requests).total.generating, 0);
    }

    #[test]
    fn status_counts() {
        let mut simulation = Simulation::new(vec![2]);
        let batch = class("batch", 0);
        let abandoned = simulation.arrival("erin", &class("urgent", 20));
        simulation.abandon(abandoned);
        simulation.arrival("bob", &batch);
        simulation.arrival("carol", &batch);
        simulation.arrival("dave", &class("interactive", 10));
        simulation.arrival("bob", &batch);
        let status = status(&simulation.requests);
        let counts =
            |requests, users, running, generating, suspended, queued, abandoned| RequestCounts {
                requests,
                users,
                running,
                generating,
                suspended,
                queued,
                abandoned,
            };
        let class_status = |name: &str, priority, counts| ClassStatus {
            name: name.to_string(),
            priority,
            counts,
        };
        assert_eq!(
            status,
            Status {
                total: counts(5, 4, 1, 2, 1, 2, 1),
                classes: vec![
                    class_status("urgent", 20, counts(1, 1, 0, 1, 0, 0, 1)),
                    class_status("interactive", 10, counts(1, 1, 1, 1, 0, 0, 0)),
                    class_status("batch", 0, counts(3, 2, 0, 0, 1, 2, 0)),
                ],
            }
        );
        assert_eq!(
            status.total.to_string(),
            "5 requests from 4 users: 1 running (2 generating in the engine), 1 suspended, 2 queued, 1 abandoned"
        );
        let class_lines: Vec<String> = status.classes.iter().map(ToString::to_string).collect();
        assert_eq!(
            class_lines,
            [
                "urgent (priority 20): 1 request from 1 user: 0 running (1 generating), 1 abandoned",
                "interactive (priority 10): 1 request from 1 user: 1 running (1 generating)",
                "batch (priority 0): 3 requests from 2 users: 1 suspended, 2 queued",
            ]
        );
    }
}
