use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use serde::{Deserialize, Serialize};

use super::keys::KeyLimits;

const MINUTE: u64 = 60;
const DAY: u64 = 24 * 60 * MINUTE;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Limiter {
    keys: HashMap<String, KeyUsage>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct KeyUsage {
    request_times: VecDeque<u64>,
    minute_tokens: VecDeque<(u64, u64)>,
    day_tokens: VecDeque<(u64, u64)>,
    #[serde(skip)]
    in_flight: u64,
    requests: u64,
    prompt_tokens: u64,
    output_tokens: u64,
    last_used: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitKind {
    ParallelRequests,
    RequestsPerMinute,
    TokensPerMinute,
    TokensPerDay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LimitExceeded {
    pub kind: LimitKind,
    pub limit: u64,
    pub retry_after_seconds: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct UsageView {
    pub in_flight: u64,
    pub requests_last_minute: u64,
    pub tokens_last_minute: u64,
    pub tokens_last_day: u64,
    pub requests: u64,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub last_used: Option<u64>,
}

#[derive(Clone, Default)]
pub struct SharedLimiter(Arc<Mutex<Limiter>>);

pub struct InFlightGuard {
    limiter: SharedLimiter,
    key_id: String,
}

impl LimitKind {
    pub fn reason(self) -> &'static str {
        match self {
            LimitKind::ParallelRequests => "max_parallel_requests",
            LimitKind::RequestsPerMinute => "rpm_limit",
            LimitKind::TokensPerMinute => "tpm_limit",
            LimitKind::TokensPerDay => "tpd_limit",
        }
    }
}

impl fmt::Display for LimitExceeded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limit = self.limit;
        match self.kind {
            LimitKind::ParallelRequests => {
                write!(
                    formatter,
                    "too many parallel requests for this API key (limit {limit})"
                )
            }
            LimitKind::RequestsPerMinute => {
                write!(
                    formatter,
                    "request rate limit of {limit} per minute exceeded"
                )
            }
            LimitKind::TokensPerMinute => {
                write!(formatter, "token rate limit of {limit} per minute exceeded")
            }
            LimitKind::TokensPerDay => {
                write!(formatter, "token limit of {limit} per day exceeded")
            }
        }
    }
}

impl KeyUsage {
    fn prune(&mut self, now: u64) {
        while self
            .request_times
            .front()
            .is_some_and(|time| time + MINUTE <= now)
        {
            self.request_times.pop_front();
        }
        while self
            .minute_tokens
            .front()
            .is_some_and(|(time, _)| time + MINUTE <= now)
        {
            self.minute_tokens.pop_front();
        }
        while self
            .day_tokens
            .front()
            .is_some_and(|(time, _)| time + DAY <= now)
        {
            self.day_tokens.pop_front();
        }
    }

    fn view(&self) -> UsageView {
        UsageView {
            in_flight: self.in_flight,
            requests_last_minute: self.request_times.len() as u64,
            tokens_last_minute: self.minute_tokens.iter().map(|(_, tokens)| tokens).sum(),
            tokens_last_day: self.day_tokens.iter().map(|(_, tokens)| tokens).sum(),
            requests: self.requests,
            prompt_tokens: self.prompt_tokens,
            output_tokens: self.output_tokens,
            last_used: self.last_used,
        }
    }
}

fn retry_after(entries: &VecDeque<(u64, u64)>, limit: u64, window: u64, now: u64) -> u64 {
    let mut total: u64 = entries.iter().map(|(_, tokens)| tokens).sum();
    for (time, tokens) in entries {
        total -= tokens;
        if total < limit {
            return (time + window).saturating_sub(now).max(1);
        }
    }
    1
}

impl Limiter {
    pub fn admit(
        &mut self,
        key_id: &str,
        limits: &KeyLimits,
        now: u64,
    ) -> Result<(), LimitExceeded> {
        let usage = self.keys.entry(key_id.to_string()).or_default();
        usage.prune(now);
        let view = usage.view();
        let exceeded = |kind, limit: u64, retry_after_seconds| {
            Err(LimitExceeded {
                kind,
                limit,
                retry_after_seconds,
            })
        };
        if let Some(limit) = limits.max_parallel_requests {
            if view.in_flight >= limit.get() {
                return exceeded(LimitKind::ParallelRequests, limit.get(), 1);
            }
        }
        if let Some(limit) = limits.rpm_limit {
            if view.requests_last_minute >= limit.get() {
                let oldest = usage.request_times.front().copied().unwrap_or(now);
                let retry = (oldest + MINUTE).saturating_sub(now).max(1);
                return exceeded(LimitKind::RequestsPerMinute, limit.get(), retry);
            }
        }
        if let Some(limit) = limits.tpm_limit {
            if view.tokens_last_minute >= limit.get() {
                let retry = retry_after(&usage.minute_tokens, limit.get(), MINUTE, now);
                return exceeded(LimitKind::TokensPerMinute, limit.get(), retry);
            }
        }
        if let Some(limit) = limits.tpd_limit {
            if view.tokens_last_day >= limit.get() {
                let retry = retry_after(&usage.day_tokens, limit.get(), DAY, now);
                return exceeded(LimitKind::TokensPerDay, limit.get(), retry);
            }
        }
        usage.request_times.push_back(now);
        usage.in_flight += 1;
        usage.requests += 1;
        usage.last_used = Some(now);
        Ok(())
    }

    pub fn reacquire(&mut self, key_id: &str) {
        self.keys.entry(key_id.to_string()).or_default().in_flight += 1;
    }

    pub fn release(&mut self, key_id: &str) {
        if let Some(usage) = self.keys.get_mut(key_id) {
            usage.in_flight = usage.in_flight.saturating_sub(1);
        }
    }

    pub fn record_use(&mut self, key_id: &str, now: u64) {
        let usage = self.keys.entry(key_id.to_string()).or_default();
        usage.requests += 1;
        usage.last_used = Some(now);
    }

    pub fn record_tokens(&mut self, key_id: &str, prompt: u64, output: u64, now: u64) {
        let usage = self.keys.entry(key_id.to_string()).or_default();
        usage.prune(now);
        usage.prompt_tokens += prompt;
        usage.output_tokens += output;
        let tokens = prompt + output;
        if tokens == 0 {
            return;
        }
        match usage.minute_tokens.back_mut() {
            Some((time, total)) if *time == now => *total += tokens,
            _ => usage.minute_tokens.push_back((now, tokens)),
        }
        let minute_start = now - now % MINUTE;
        match usage.day_tokens.back_mut() {
            Some((time, total)) if *time == minute_start => *total += tokens,
            _ => usage.day_tokens.push_back((minute_start, tokens)),
        }
    }

    pub fn view(&mut self, key_id: &str, now: u64) -> UsageView {
        match self.keys.get_mut(key_id) {
            Some(usage) => {
                usage.prune(now);
                usage.view()
            }
            None => UsageView::default(),
        }
    }

    pub fn retain(&mut self, key_ids: &HashSet<&str>) {
        self.keys
            .retain(|key_id, usage| usage.in_flight > 0 || key_ids.contains(key_id.as_str()));
    }
}

impl SharedLimiter {
    pub fn new(limiter: Limiter) -> SharedLimiter {
        SharedLimiter(Arc::new(Mutex::new(limiter)))
    }

    pub fn lock(&self) -> MutexGuard<'_, Limiter> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn guard(&self, key_id: &str) -> InFlightGuard {
        InFlightGuard {
            limiter: self.clone(),
            key_id: key_id.to_string(),
        }
    }
}

impl InFlightGuard {
    pub fn key_id(&self) -> &str {
        &self.key_id
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.limiter.lock().release(&self.key_id);
    }
}

impl fmt::Debug for InFlightGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "InFlightGuard({})", self.key_id)
    }
}

pub fn unix_now() -> u64 {
    jiff::Timestamp::now().as_second().max(0) as u64
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;

    fn limits(
        parallel: Option<u64>,
        rpm: Option<u64>,
        tpm: Option<u64>,
        tpd: Option<u64>,
    ) -> KeyLimits {
        KeyLimits {
            max_parallel_requests: parallel.and_then(NonZeroU64::new),
            rpm_limit: rpm.and_then(NonZeroU64::new),
            tpm_limit: tpm.and_then(NonZeroU64::new),
            tpd_limit: tpd.and_then(NonZeroU64::new),
        }
    }

    #[test]
    fn requests_per_minute() {
        let mut limiter = Limiter::default();
        let rpm = limits(None, Some(2), None, None);
        assert!(limiter.admit("k", &rpm, 1000).is_ok());
        assert!(limiter.admit("k", &rpm, 1010).is_ok());
        let exceeded = limiter.admit("k", &rpm, 1020).unwrap_err();
        assert_eq!(exceeded.kind, LimitKind::RequestsPerMinute);
        assert_eq!(exceeded.retry_after_seconds, 40);
        assert!(limiter.admit("k", &rpm, 1060).is_ok());
        assert!(limiter.admit("other", &rpm, 1060).is_ok());
    }

    #[test]
    fn parallel_requests() {
        let shared = SharedLimiter::default();
        let parallel = limits(Some(1), None, None, None);
        assert!(shared.lock().admit("k", &parallel, 0).is_ok());
        let guard = shared.guard("k");
        assert_eq!(
            shared.lock().admit("k", &parallel, 0).unwrap_err().kind,
            LimitKind::ParallelRequests
        );
        drop(guard);
        assert!(shared.lock().admit("k", &parallel, 0).is_ok());
        assert_eq!(shared.lock().view("k", 0).in_flight, 1);
    }

    #[test]
    fn tokens_per_minute_and_day() {
        let mut limiter = Limiter::default();
        let tpm = limits(None, None, Some(100), None);
        assert!(limiter.admit("k", &tpm, 0).is_ok());
        limiter.record_tokens("k", 60, 50, 10);
        let exceeded = limiter.admit("k", &tpm, 20).unwrap_err();
        assert_eq!(exceeded.kind, LimitKind::TokensPerMinute);
        assert_eq!(exceeded.retry_after_seconds, 50);
        assert!(limiter.admit("k", &tpm, 70).is_ok());

        let tpd = limits(None, None, None, Some(1000));
        limiter.record_tokens("d", 600, 400, 120);
        let exceeded = limiter.admit("d", &tpd, 3600).unwrap_err();
        assert_eq!(exceeded.kind, LimitKind::TokensPerDay);
        assert_eq!(exceeded.retry_after_seconds, 120 + DAY - 3600);
        assert!(limiter.admit("d", &tpd, 120 + DAY).is_ok());
        let view = limiter.view("d", 120 + DAY);
        assert_eq!(
            (view.prompt_tokens, view.output_tokens, view.tokens_last_day),
            (600, 400, 0)
        );
    }

    #[test]
    fn retain_drops_deleted_keys() {
        let mut limiter = Limiter::default();
        limiter.record_use("kept", 0);
        limiter.record_use("deleted", 0);
        limiter.retain(&HashSet::from(["kept"]));
        assert_eq!(limiter.view("kept", 0).requests, 1);
        assert_eq!(limiter.view("deleted", 0).requests, 0);
    }
}
