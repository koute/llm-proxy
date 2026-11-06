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
    requests: VecDeque<u64>,
    /// Tokens of finished requests, one entry per second.
    tokens: VecDeque<(u64, u64)>,
    #[serde(skip)]
    in_flight: u64,
    /// The token estimates of the requests in flight, which count against the token
    /// limits until the requests finish and their real counts replace them.
    #[serde(skip)]
    reserved_tokens: u64,
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
    /// What's left of the limit now.
    pub remaining: u64,
    /// When all of the key's exceeded limits have room again, at the earliest.
    pub retry_after_seconds: u64,
    /// The estimate of a request which is over the token limit on its own.
    pub too_large: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct UsageView {
    pub in_flight: u64,
    /// The token estimates of the requests in flight.
    pub reserved_tokens: u64,
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
    reserved_tokens: u64,
}

impl LimitKind {
    /// The rejection's reason, what the limit counts, and the unit of LiteLLM's
    /// `x-ratelimit-*` headers.
    pub fn names(self) -> (&'static str, &'static str, Option<&'static str>) {
        match self {
            LimitKind::ParallelRequests => ("max_parallel_requests", "parallel requests", None),
            LimitKind::RequestsPerMinute => ("rpm_limit", "requests per minute", Some("requests")),
            LimitKind::TokensPerMinute => ("tpm_limit", "tokens per minute", Some("tokens")),
            LimitKind::TokensPerDay => ("tpd_limit", "tokens per day", Some("tokens-day")),
        }
    }
}

impl fmt::Display for LimitExceeded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (_, what, _) = self.kind.names();
        let limit = self.limit;
        match self.too_large {
            Some(tokens) => write!(
                formatter,
                "the request's estimate of {tokens} tokens (its prompt and maximum output) is over the limit of {limit} {what} on its own"
            ),
            None => write!(formatter, "limit of {limit} {what} exceeded"),
        }
    }
}

fn prune<T>(entries: &mut VecDeque<T>, is_expired: impl Fn(&T) -> bool) {
    while entries.front().is_some_and(&is_expired) {
        entries.pop_front();
    }
}

impl KeyUsage {
    fn prune(&mut self, now: u64) {
        prune(&mut self.requests, |time| time + MINUTE <= now);
        prune(&mut self.tokens, |(time, _)| time + DAY <= now);
    }

    fn tokens_since(&self, window: u64, now: u64) -> u64 {
        let recent = self.tokens.iter().filter(|(time, _)| time + window > now);
        recent.map(|(_, tokens)| tokens).sum()
    }

    /// Seconds until enough of the window's oldest tokens expire for `used` to be at most
    /// `budget`, or None if only finishing requests in flight can get it there.
    fn token_retry_after(&self, mut used: u64, budget: u64, window: u64, now: u64) -> Option<u64> {
        for (time, tokens) in self.tokens.iter().filter(|(time, _)| time + window > now) {
            used = used.saturating_sub(*tokens);
            if used <= budget {
                return Some(time + window - now);
            }
        }
        None
    }

    /// When the limit of `kind` has room for a request with `estimate` tokens again, if
    /// it hasn't now. Like in LiteLLM, a request doesn't fit a token limit if the tokens
    /// of the window, the reserved ones and its own exceed it, so one with more than the
    /// limit never fits. A limit of 0 never has room. Retry after the window, then.
    fn exceeded(
        &self,
        kind: LimitKind,
        limit: u64,
        estimate: u64,
        now: u64,
    ) -> Option<LimitExceeded> {
        let reserved = self.reserved_tokens;
        let tokens = |window| self.tokens_since(window, now).saturating_add(reserved);
        // What's used, what the request adds (one request, or its estimate of tokens), and
        // the window.
        let (used, new, window) = match kind {
            LimitKind::ParallelRequests => (self.in_flight, 1, MINUTE),
            LimitKind::RequestsPerMinute => (self.requests.len() as u64, 1, MINUTE),
            LimitKind::TokensPerMinute => (tokens(MINUTE), estimate, MINUTE),
            LimitKind::TokensPerDay => (tokens(DAY), estimate, DAY),
        };
        if limit > 0 && used.saturating_add(new) <= limit {
            return None;
        }
        let retry_after = match kind {
            _ if limit == 0 || new > limit => Some(window),
            LimitKind::ParallelRequests => None,
            LimitKind::RequestsPerMinute => {
                let index = self.requests.len() - limit as usize;
                self.requests.get(index).map(|time| time + MINUTE - now)
            }
            LimitKind::TokensPerMinute | LimitKind::TokensPerDay => {
                self.token_retry_after(used, limit - new, window, now)
            }
        };
        Some(LimitExceeded {
            kind,
            limit,
            remaining: limit.saturating_sub(used),
            // Requests in flight free their slot or reservation when they finish.
            retry_after_seconds: retry_after.unwrap_or(1).max(1),
            // Only an estimate of tokens can be over a limit of at least 1.
            too_large: (limit > 0 && new > limit).then_some(new),
        })
    }
}

impl Limiter {
    /// Admits a request with `reserved_tokens` (its estimated tokens) if the key's
    /// limits allow it.
    pub fn admit(
        &mut self,
        key_id: &str,
        limits: &KeyLimits,
        reserved_tokens: u64,
        now: u64,
    ) -> Result<(), LimitExceeded> {
        let usage = self.keys.entry(key_id.to_string()).or_default();
        usage.prune(now);
        let exceeded = [
            (LimitKind::ParallelRequests, limits.max_parallel_requests),
            (LimitKind::RequestsPerMinute, limits.rpm_limit),
            (LimitKind::TokensPerMinute, limits.tpm_limit),
            (LimitKind::TokensPerDay, limits.tpd_limit),
        ]
        .into_iter()
        .filter_map(|(kind, limit)| usage.exceeded(kind, limit?, reserved_tokens, now))
        .max_by_key(|exceeded| exceeded.retry_after_seconds);
        if let Some(exceeded) = exceeded {
            return Err(exceeded);
        }
        usage.requests.push_back(now);
        usage.in_flight += 1;
        usage.reserved_tokens = usage.reserved_tokens.saturating_add(reserved_tokens);
        usage.total_requests += 1;
        usage.last_used = Some(now);
        Ok(())
    }

    /// Takes back an admission whose request the proxy then rejected (e.g. a full queue).
    pub fn refund(&mut self, key_id: &str) {
        if let Some(usage) = self.keys.get_mut(key_id) {
            usage.requests.pop_back();
            usage.total_requests = usage.total_requests.saturating_sub(1);
        }
    }

    /// Counts a request handed over by a restart, with its reservation.
    pub fn reacquire(&mut self, key_id: &str, reserved_tokens: u64) {
        let usage = self.keys.entry(key_id.to_string()).or_default();
        usage.in_flight += 1;
        usage.reserved_tokens = usage.reserved_tokens.saturating_add(reserved_tokens);
    }

    fn release(&mut self, key_id: &str, reserved_tokens: u64) {
        if let Some(usage) = self.keys.get_mut(key_id) {
            usage.in_flight = usage.in_flight.saturating_sub(1);
            usage.reserved_tokens = usage.reserved_tokens.saturating_sub(reserved_tokens);
        }
    }

    pub fn record_tokens(&mut self, key_id: &str, tokens: u64, now: u64) {
        let usage = self.keys.entry(key_id.to_string()).or_default();
        usage.prune(now);
        usage.total_tokens += tokens;
        match usage.tokens.back_mut() {
            _ if tokens == 0 => {}
            Some((time, total)) if *time == now => *total += tokens,
            _ => usage.tokens.push_back((now, tokens)),
        }
    }

    pub fn view(&mut self, key_id: &str, now: u64) -> UsageView {
        let Some(usage) = self.keys.get_mut(key_id) else {
            return UsageView::default();
        };
        usage.prune(now);
        UsageView {
            in_flight: usage.in_flight,
            reserved_tokens: usage.reserved_tokens,
            requests_last_minute: usage.requests.len() as u64,
            tokens_last_minute: usage.tokens_since(MINUTE, now),
            tokens_last_day: usage.tokens_since(DAY, now),
            requests: usage.total_requests,
            tokens: usage.total_tokens,
            last_used: usage.last_used,
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

    /// Holds a request's parallel slot and token reservation until it's dropped.
    pub fn guard(&self, key_id: &str, reserved_tokens: u64) -> InFlightGuard {
        InFlightGuard {
            limiter: self.clone(),
            key_id: key_id.to_string(),
            reserved_tokens,
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.limiter
            .lock()
            .release(&self.key_id, self.reserved_tokens);
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
    use LimitKind::*;

    fn limits(values: [Option<u64>; 4]) -> KeyLimits {
        KeyLimits {
            max_parallel_requests: values[0],
            rpm_limit: values[1],
            tpm_limit: values[2],
            tpd_limit: values[3],
        }
    }

    fn exceeded(limiter: &mut Limiter, limits: &KeyLimits, now: u64) -> (LimitKind, u64) {
        let exceeded = limiter.admit("k", limits, 0, now).unwrap_err();
        (exceeded.kind, exceeded.retry_after_seconds)
    }

    #[test]
    fn rate_limits() {
        let mut limiter = Limiter::default();
        let rpm = limits([None, Some(2), None, None]);
        assert!(limiter.admit("k", &rpm, 0, 1000).is_ok());
        assert!(limiter.admit("k", &rpm, 0, 1010).is_ok());
        assert_eq!(exceeded(&mut limiter, &rpm, 1020), (RequestsPerMinute, 40));
        assert!(limiter.admit("k", &rpm, 0, 1060).is_ok());

        let mut limiter = Limiter::default();
        limiter.record_tokens("k", 110, 10);
        let tpm = limits([None, None, Some(100), None]);
        assert_eq!(exceeded(&mut limiter, &tpm, 20), (TokensPerMinute, 50));
        assert!(limiter.admit("k", &tpm, 0, 70).is_ok());
        let tpd = limits([None, None, None, Some(100)]);
        assert_eq!(
            exceeded(&mut limiter, &tpd, 3600),
            (TokensPerDay, 10 + DAY - 3600)
        );
        assert!(limiter.admit("k", &tpd, 0, 10 + DAY).is_ok());
        assert_eq!(limiter.view("k", 10 + DAY).tokens, 110);
    }

    #[test]
    fn retry_after_is_enough() {
        // Waiting until the oldest tokens expire isn't enough if newer ones still exceed.
        let mut limiter = Limiter::default();
        let tpm = limits([None, None, Some(100), None]);
        limiter.record_tokens("k", 10, 0);
        limiter.record_tokens("k", 200, 5);
        assert_eq!(exceeded(&mut limiter, &tpm, 10), (TokensPerMinute, 55));
        assert!(limiter.admit("k", &tpm, 0, 65).is_ok());
        // With several exceeded limits, the latest one counts.
        let mut limiter = Limiter::default();
        let both = limits([None, Some(1), None, Some(50)]);
        assert!(limiter.admit("k", &both, 0, 0).is_ok());
        limiter.record_tokens("k", 60, 0);
        assert_eq!(exceeded(&mut limiter, &both, 1), (TokensPerDay, DAY - 1));
        // A limit of 0 denies everything.
        let zero = limits([None, Some(0), None, None]);
        assert_eq!(
            exceeded(&mut Limiter::default(), &zero, 0),
            (RequestsPerMinute, MINUTE)
        );
        let zero = limits([Some(0), None, None, None]);
        assert_eq!(
            exceeded(&mut Limiter::default(), &zero, 0),
            (ParallelRequests, MINUTE)
        );
        // A request whose estimate alone is over a limit never fits (like in LiteLLM),
        // and huge estimates don't wrap around.
        let mut limiter = Limiter::default();
        let huge = limiter.admit("k", &tpm, u64::MAX, 0).unwrap_err();
        assert_eq!((huge.too_large, huge.remaining), (Some(u64::MAX), 100));
        assert_eq!(
            (huge.kind, huge.retry_after_seconds),
            (TokensPerMinute, MINUTE)
        );
        assert!(limiter.admit("k", &tpm, 100, 0).is_ok());
    }

    #[test]
    fn reservations_count_until_requests_finish() {
        let shared = SharedLimiter::default();
        let tpm = limits([None, None, Some(100), None]);
        assert!(shared.lock().admit("k", &tpm, 60, 0).is_ok());
        let first = shared.guard("k", 60);
        // 60 reserved: another 60 doesn't fit until the first request finishes.
        let exceeded = shared.lock().admit("k", &tpm, 60, 0).unwrap_err();
        assert_eq!(
            (exceeded.kind, exceeded.retry_after_seconds),
            (TokensPerMinute, 1)
        );
        assert!(shared.lock().admit("k", &tpm, 40, 0).is_ok());
        let second = shared.guard("k", 40);
        shared.lock().record_tokens("k", 30, 1);
        drop(first);
        assert!(shared.lock().admit("k", &tpm, 30, 1).is_ok());
        drop(second);
        shared.lock().record_tokens("k", 30, 1);
        assert_eq!(shared.lock().keys["k"].tokens, VecDeque::from([(1, 60)]));
    }

    #[test]
    fn parallel_requests_and_deleted_keys() {
        let shared = SharedLimiter::default();
        let parallel = limits([Some(1), None, None, None]);
        assert!(shared.lock().admit("k", &parallel, 0, 0).is_ok());
        let guard = shared.guard("k", 0);
        assert_eq!(
            exceeded(&mut shared.lock(), &parallel, 0).0,
            ParallelRequests
        );
        drop(guard);
        assert!(shared.lock().admit("k", &parallel, 0, 0).is_ok());
        drop(shared.guard("k", 0));
        shared.lock().refund("k");
        assert_eq!(shared.lock().view("k", 0).requests, 1);
        shared.lock().retain(&HashSet::new());
        assert_eq!(shared.lock().view("k", 0).requests, 0);
    }
}
