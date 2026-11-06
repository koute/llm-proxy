use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::num::NonZeroU64;
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
    requests: VecDeque<u64>,
    tokens: VecDeque<(u64, u64)>,
    #[serde(skip)]
    in_flight: u64,
    total_requests: u64,
    total_tokens: u64,
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
    pub tokens: u64,
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
                write!(formatter, "too many parallel requests (limit {limit})")
            }
            LimitKind::RequestsPerMinute => {
                write!(formatter, "limit of {limit} requests per minute exceeded")
            }
            LimitKind::TokensPerMinute => {
                write!(formatter, "limit of {limit} tokens per minute exceeded")
            }
            LimitKind::TokensPerDay => {
                write!(formatter, "limit of {limit} tokens per day exceeded")
            }
        }
    }
}

impl KeyUsage {
    fn prune(&mut self, now: u64) {
        while self
            .requests
            .front()
            .is_some_and(|time| time + MINUTE <= now)
        {
            self.requests.pop_front();
        }
        while self
            .tokens
            .front()
            .is_some_and(|(time, _)| time + DAY <= now)
        {
            self.tokens.pop_front();
        }
    }

    fn tokens_since(&self, window: u64, now: u64) -> u64 {
        let recent = self.tokens.iter().filter(|(time, _)| time + window > now);
        recent.map(|(_, tokens)| tokens).sum()
    }

    fn token_retry_after(&self, window: u64, now: u64) -> u64 {
        let oldest = self.tokens.iter().find(|(time, _)| time + window > now);
        oldest.map_or(1, |(time, _)| time + window - now)
    }

    fn view(&self, now: u64) -> UsageView {
        UsageView {
            in_flight: self.in_flight,
            requests_last_minute: self.requests.len() as u64,
            tokens_last_minute: self.tokens_since(MINUTE, now),
            tokens_last_day: self.tokens_since(DAY, now),
            requests: self.total_requests,
            tokens: self.total_tokens,
            last_used: self.last_used,
        }
    }
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
        let view = usage.view(now);
        for (kind, limit, used) in [
            (
                LimitKind::ParallelRequests,
                limits.max_parallel_requests,
                view.in_flight,
            ),
            (
                LimitKind::RequestsPerMinute,
                limits.rpm_limit,
                view.requests_last_minute,
            ),
            (
                LimitKind::TokensPerMinute,
                limits.tpm_limit,
                view.tokens_last_minute,
            ),
            (
                LimitKind::TokensPerDay,
                limits.tpd_limit,
                view.tokens_last_day,
            ),
        ] {
            let Some(limit) = limit.map(NonZeroU64::get).filter(|limit| used >= *limit) else {
                continue;
            };
            let retry_after = match kind {
                LimitKind::ParallelRequests => 1,
                LimitKind::RequestsPerMinute => {
                    usage.requests.front().map_or(1, |time| time + MINUTE - now)
                }
                LimitKind::TokensPerMinute => usage.token_retry_after(MINUTE, now),
                LimitKind::TokensPerDay => usage.token_retry_after(DAY, now),
            };
            return Err(LimitExceeded {
                kind,
                limit,
                retry_after_seconds: retry_after.max(1),
            });
        }
        usage.requests.push_back(now);
        usage.in_flight += 1;
        usage.total_requests += 1;
        usage.last_used = Some(now);
        Ok(())
    }

    pub fn reacquire(&mut self, key_id: &str) {
        self.keys.entry(key_id.to_string()).or_default().in_flight += 1;
    }

    fn release(&mut self, key_id: &str) {
        if let Some(usage) = self.keys.get_mut(key_id) {
            usage.in_flight = usage.in_flight.saturating_sub(1);
        }
    }

    pub fn record_tokens(&mut self, key_id: &str, tokens: u64, now: u64) {
        let usage = self.keys.entry(key_id.to_string()).or_default();
        usage.prune(now);
        usage.total_tokens += tokens;
        if tokens > 0 {
            usage.tokens.push_back((now, tokens));
        }
    }

    pub fn view(&mut self, key_id: &str, now: u64) -> UsageView {
        let Some(usage) = self.keys.get_mut(key_id) else {
            return UsageView::default();
        };
        usage.prune(now);
        usage.view(now)
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
    use super::*;

    fn limits(values: [u64; 4]) -> KeyLimits {
        KeyLimits {
            max_parallel_requests: NonZeroU64::new(values[0]),
            rpm_limit: NonZeroU64::new(values[1]),
            tpm_limit: NonZeroU64::new(values[2]),
            tpd_limit: NonZeroU64::new(values[3]),
        }
    }

    fn exceeded(limiter: &mut Limiter, limits: &KeyLimits, now: u64) -> (LimitKind, u64) {
        let exceeded = limiter.admit("k", limits, now).unwrap_err();
        (exceeded.kind, exceeded.retry_after_seconds)
    }

    #[test]
    fn rate_limits() {
        let mut limiter = Limiter::default();
        let rpm = limits([0, 2, 0, 0]);
        assert!(limiter.admit("k", &rpm, 1000).is_ok());
        assert!(limiter.admit("k", &rpm, 1010).is_ok());
        assert_eq!(
            exceeded(&mut limiter, &rpm, 1020),
            (LimitKind::RequestsPerMinute, 40)
        );
        assert!(limiter.admit("k", &rpm, 1060).is_ok());

        let mut limiter = Limiter::default();
        limiter.record_tokens("k", 110, 10);
        assert_eq!(
            exceeded(&mut limiter, &limits([0, 0, 100, 0]), 20),
            (LimitKind::TokensPerMinute, 50)
        );
        assert!(limiter.admit("k", &limits([0, 0, 100, 0]), 70).is_ok());
        assert_eq!(
            exceeded(&mut limiter, &limits([0, 0, 0, 100]), 3600),
            (LimitKind::TokensPerDay, 10 + DAY - 3600)
        );
        assert!(limiter
            .admit("k", &limits([0, 0, 0, 100]), 10 + DAY)
            .is_ok());
        assert_eq!(limiter.view("k", 10 + DAY).tokens, 110);
    }

    #[test]
    fn parallel_requests_and_deleted_keys() {
        let shared = SharedLimiter::default();
        let parallel = limits([1, 0, 0, 0]);
        assert!(shared.lock().admit("k", &parallel, 0).is_ok());
        let guard = shared.guard("k");
        assert_eq!(
            exceeded(&mut shared.lock(), &parallel, 0).0,
            LimitKind::ParallelRequests
        );
        drop(guard);
        assert!(shared.lock().admit("k", &parallel, 0).is_ok());
        drop(shared.guard("k"));
        shared.lock().retain(&HashSet::new());
        assert_eq!(shared.lock().view("k", 0).requests, 0);
    }
}
