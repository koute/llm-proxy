use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::keys::{KeyRecord, KeyState};
use super::limits::UsageView;
use super::FinishedRequest;
use crate::quality_of_service::{RequestCounts, Status};
use crate::utils::AppendHealth;

const BUCKETS: [f64; 15] = [
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0,
];
const KEYS_FILE_RELOADS: &str = "llm_proxy_keys_file_reloads_total";
const KEYS_FILE_ERRORS: &str = "llm_proxy_keys_file_errors_total";
/// The families in the order of the exposition.
const FAMILIES: &[(&str, &str, &str)] = &[
    (
        "llm_proxy_requests_total",
        "counter",
        "Finished requests by status: an HTTP status, aborted (the client left), failed (the engine broke off the response), error or unknown.",
    ),
    (
        "llm_proxy_rejected_requests_total",
        "counter",
        "Requests which the proxy rejected.",
    ),
    (
        "llm_proxy_tokens_total",
        "counter",
        "Engine token counts of finished requests.",
    ),
    (
        "llm_proxy_suspensions_total",
        "counter",
        "Suspensions of finished requests.",
    ),
    (
        KEYS_FILE_RELOADS,
        "counter",
        "Changes of the keys file which the proxy loaded.",
    ),
    (
        KEYS_FILE_ERRORS,
        "counter",
        "Failed reads, writes and parses of the keys file.",
    ),
    (
        "llm_proxy_request_duration_seconds",
        "histogram",
        "Time from the arrival of a request to the end of its response.",
    ),
    (
        "llm_proxy_queue_duration_seconds",
        "histogram",
        "Time requests waited before the proxy sent them to the engine.",
    ),
    (
        "llm_proxy_time_to_first_token_seconds",
        "histogram",
        "Time from the arrival of a request to the first data from the engine.",
    ),
    (
        "llm_proxy_requests",
        "gauge",
        "Requests in the proxy by class and state: queued, waiting_in_engine, generating, suspended or abandoned.",
    ),
    (
        "llm_proxy_audit_log_failing",
        "gauge",
        "1 while the audit log can't be written; the proxy keeps the records and retries.",
    ),
    (
        "llm_proxy_scheduling_log_failing",
        "gauge",
        "1 while the scheduling log can't be written; the proxy keeps the events and retries.",
    ),
    (
        "llm_proxy_audit_log_dropped_records_total",
        "counter",
        "Audit records dropped because writing the audit log lagged too far behind.",
    ),
    (
        "llm_proxy_active_users",
        "gauge",
        "Users with requests in the proxy.",
    ),
    (
        "llm_proxy_rank_parallel_request_cap",
        "gauge",
        "Maximum number of parallel requests per rank.",
    ),
    ("llm_proxy_keys", "gauge", "API keys by state."),
    (
        "llm_proxy_key_in_flight_requests",
        "gauge",
        "Requests of an API key in the proxy.",
    ),
    (
        "llm_proxy_key_remaining_requests",
        "gauge",
        "Requests an API key can still send this minute.",
    ),
    (
        "llm_proxy_key_remaining_tokens",
        "gauge",
        "Tokens an API key can still use in the window, after the estimates of its requests in flight.",
    ),
];

type Series = BTreeMap<(&'static str, String), f64>;
type Pairs<'a> = [(&'a str, &'a str)];

#[derive(Default)]
pub struct Metrics {
    state: Mutex<MetricsState>,
}

#[derive(Default)]
struct MetricsState {
    counters: Series,
    histograms: BTreeMap<(&'static str, String), Histogram>,
}

#[derive(Default)]
struct Histogram {
    buckets: [u64; BUCKETS.len()],
    count: u64,
    sum: f64,
}

pub struct KeyGauge<'a> {
    pub record: &'a KeyRecord,
    pub state: KeyState,
    pub usage: UsageView,
}

pub struct Gauges<'a> {
    pub status: Status,
    /// The configured classes, which have series even without requests.
    pub classes: Vec<&'a str>,
    pub rank_caps: &'a [usize],
    pub keys: Vec<KeyGauge<'a>>,
    pub audit_log: Option<AppendHealth>,
    pub scheduling_log: Option<AppendHealth>,
}

fn labels(pairs: &Pairs) -> String {
    if pairs.is_empty() {
        return String::new();
    }
    let escape = |value: &str| {
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    };
    let pairs: Vec<String> = pairs
        .iter()
        .map(|(name, value)| format!("{name}=\"{}\"", escape(value)))
        .collect();
    format!("{{{}}}", pairs.join(","))
}

fn with_bound(labels: &str, bound: &str) -> String {
    match labels.strip_suffix('}') {
        Some(labels) => format!("{labels},le=\"{bound}\"}}"),
        None => format!("{{le=\"{bound}\"}}"),
    }
}

impl Metrics {
    fn lock(&self) -> MutexGuard<'_, MetricsState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn add(&self, name: &'static str, pairs: &Pairs, value: f64) {
        let mut state = self.lock();
        *state.counters.entry((name, labels(pairs))).or_default() += value;
    }

    fn count_request(&self, identity: &Pairs, path: &str, status: &str) {
        let series = [identity, &[("path", path), ("status", status)]].concat();
        self.add("llm_proxy_requests_total", &series, 1.0);
    }

    /// Passthrough requests have no class; they're counted with an empty one.
    pub fn observe_passthrough(&self, user: &str, key: Option<&str>, path: &str, status: &str) {
        let key = key.unwrap_or_default();
        self.count_request(&[("user", user), ("key", key), ("class", "")], path, status);
    }

    pub fn observe_request(&self, finished: &FinishedRequest, status: &str) {
        let (summary, attribution) = (finished.summary, finished.attribution);
        let user = summary.user.as_str();
        let key = attribution.key_id.as_deref().unwrap_or_default();
        let class = finished.class();
        let identity = [("user", user), ("key", key), ("class", class)];
        let route = attribution.route.as_deref().unwrap_or(&attribution.path);
        self.count_request(&identity, route, status);
        if let Some(tokens) = summary.tokens {
            for (kind, count) in [
                ("prompt", tokens.prompt),
                ("output", tokens.output),
                ("cache_hit", tokens.cache_hit),
                ("prefill", tokens.prefill),
            ] {
                let series = [&identity[..], &[("type", kind)]].concat();
                self.add("llm_proxy_tokens_total", &series, count as f64);
            }
        }
        if summary.suspensions > 0 {
            let suspensions = f64::from(summary.suspensions);
            self.add("llm_proxy_suspensions_total", &identity, suspensions);
        }
        let total = finished.total_seconds;
        self.observe("llm_proxy_request_duration_seconds", class, total);
        if let Some(seconds) = finished.queued_seconds {
            self.observe("llm_proxy_queue_duration_seconds", class, seconds);
        }
        if let Some(seconds) = finished.time_to_first_token {
            self.observe("llm_proxy_time_to_first_token_seconds", class, seconds);
        }
    }

    fn observe(&self, name: &'static str, class: &str, value: f64) {
        let mut state = self.lock();
        let series = (name, labels(&[("class", class)]));
        let histogram = state.histograms.entry(series).or_default();
        for (bucket, bound) in histogram.buckets.iter_mut().zip(BUCKETS) {
            *bucket += u64::from(value <= bound);
        }
        histogram.count += 1;
        histogram.sum += value;
    }

    pub fn observe_rejection(&self, status: u16, reason: &str) {
        let status = status.to_string();
        let series = [("status", status.as_str()), ("reason", reason)];
        self.add("llm_proxy_rejected_requests_total", &series, 1.0);
    }

    pub fn keys_file_reloaded(&self) {
        self.add(KEYS_FILE_RELOADS, &[], 1.0);
    }

    pub fn keys_file_failed(&self) {
        self.add(KEYS_FILE_ERRORS, &[], 1.0);
    }

    pub fn render(&self, gauges: &Gauges) -> String {
        let mut series = gauge_series(gauges);
        for name in [KEYS_FILE_RELOADS, KEYS_FILE_ERRORS] {
            series.insert((name, String::new()), 0.0);
        }
        let state = self.lock();
        series.extend(state.counters.clone());
        let mut out = String::new();
        for &(name, kind, help) in FAMILIES {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
            for ((_, labels), value) in series.iter().filter(|((of, _), _)| *of == name) {
                let _ = writeln!(out, "{name}{labels} {value}");
            }
            let histograms = state.histograms.iter().filter(|((of, _), _)| *of == name);
            for ((_, labels), histogram) in histograms {
                for (bound, count) in BUCKETS.iter().zip(histogram.buckets) {
                    let labels = with_bound(labels, &bound.to_string());
                    let _ = writeln!(out, "{name}_bucket{labels} {count}");
                }
                let count = histogram.count;
                let _ = writeln!(out, "{name}_bucket{} {count}", with_bound(labels, "+Inf"));
                let _ = writeln!(out, "{name}_sum{labels} {}", histogram.sum);
                let _ = writeln!(out, "{name}_count{labels} {count}");
            }
        }
        out
    }
}

fn gauge_series(gauges: &Gauges) -> Series {
    let mut series = Series::new();
    let mut set = |name, pairs: &Pairs, value: u64| {
        series.insert((name, labels(pairs)), value as f64);
    };
    // The configured classes have series even when idle; the scheduler's counts replace
    // them. Of two classes with the same name, the first one shows (hence the reversal).
    let idle = RequestCounts::default();
    let configured = gauges.classes.iter().map(|name| (*name, &idle));
    let classes = gauges.status.classes.iter().rev();
    let classes = classes.map(|class| (class.name.as_str(), &class.counts));
    for (class, counts) in configured.chain(classes) {
        // The states partition the requests: a request sent to the engine is either
        // generating or waiting in the engine (e.g. for a prefill).
        let waiting_in_engine = counts.running - counts.running_generating;
        for (state, count) in [
            ("queued", counts.queued),
            ("waiting_in_engine", waiting_in_engine),
            ("generating", counts.running_generating),
            ("suspended", counts.suspended),
            ("abandoned", counts.abandoned),
        ] {
            let pairs = [("class", class), ("state", state)];
            set("llm_proxy_requests", &pairs, count as u64);
        }
    }
    if let Some(health) = &gauges.scheduling_log {
        let failing = u64::from(health.is_failing);
        set("llm_proxy_scheduling_log_failing", &[], failing);
    }
    if let Some(health) = &gauges.audit_log {
        let failing = u64::from(health.is_failing);
        set("llm_proxy_audit_log_failing", &[], failing);
        let dropped = health.dropped as u64;
        set("llm_proxy_audit_log_dropped_records_total", &[], dropped);
    }
    let users = gauges.status.total.users as u64;
    set("llm_proxy_active_users", &[], users);
    for (rank, cap) in gauges.rank_caps.iter().enumerate() {
        let rank = rank.to_string();
        let pairs = [("rank", rank.as_str())];
        set("llm_proxy_rank_parallel_request_cap", &pairs, *cap as u64);
    }
    for (state, name) in [
        (KeyState::Active, "active"),
        (KeyState::Blocked, "blocked"),
        (KeyState::Expired, "expired"),
    ] {
        let count = gauges.keys.iter().filter(|key| key.state == state).count();
        set("llm_proxy_keys", &[("state", name)], count as u64);
    }
    for KeyGauge { record, usage, .. } in &gauges.keys {
        let identity = [("key", record.id.as_str()), ("user", &record.user)];
        let limits = record.limits;
        set(
            "llm_proxy_key_in_flight_requests",
            &identity,
            usage.in_flight,
        );
        if let Some(limit) = limits.rpm_limit {
            let remaining = limit.saturating_sub(usage.requests_last_minute);
            set("llm_proxy_key_remaining_requests", &identity, remaining);
        }
        // Like the limits themselves, the estimates of the requests in flight count.
        for (window, limit, used) in [
            ("minute", limits.tpm_limit, usage.tokens_last_minute),
            ("day", limits.tpd_limit, usage.tokens_last_day),
        ] {
            let Some(limit) = limit else { continue };
            let remaining = limit.saturating_sub(used.saturating_add(usage.reserved_tokens));
            let series = [&identity[..], &[("window", window)]].concat();
            set("llm_proxy_key_remaining_tokens", &series, remaining);
        }
    }
    series
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_protocol::ProxyId;
    use crate::gateway::{Attribution, Outcome};
    use crate::middle_end::scheduling_log::RequestSummary;
    use crate::quality_of_service::ClassStatus;

    #[test]
    fn exposition_format() {
        let metrics = Metrics::default();
        let summary = RequestSummary {
            request: ProxyId {
                instance: 1,
                sequence: 2,
            },
            user: "litellm/bob \"b\"".to_string(),
            class: "high".to_string(),
            priority: 1,
            total_seconds: 3.0,
            queued_seconds: Some(0.2),
            generating_seconds: 2.0,
            usage_seconds: 2.0,
            suspensions: 2,
            tokens: None,
        };
        let attribution = Attribution {
            key_id: Some("key-1".to_string()),
            path: "/v1/chat/completions".to_string(),
            ..Attribution::default()
        };
        let finished = FinishedRequest {
            summary: &summary,
            attribution: &attribution,
            outcome: Outcome::Completed,
            status: Some(200),
            time_to_first_token: Some(0.7),
            total_seconds: 3.0,
            queued_seconds: Some(0.2),
        };
        metrics.observe_request(&finished, "200");
        metrics.observe_rejection(429, "rpm_limit");
        let counts = RequestCounts {
            requests: 1,
            queued: 1,
            ..RequestCounts::default()
        };
        let status = Status {
            total: RequestCounts { users: 1, ..counts },
            classes: vec![ClassStatus {
                name: "high".to_string(),
                priority: 1,
                counts,
            }],
            users: Vec::new(),
        };
        let text = metrics.render(&Gauges {
            status,
            classes: vec!["high", "low"],
            rank_caps: &[16],
            keys: Vec::new(),
            audit_log: Some(AppendHealth {
                is_failing: false,
                dropped: 0,
            }),
            scheduling_log: None,
        });
        let identity = r#"user="litellm/bob \"b\"",key="key-1",class="high""#;
        let expected = r#"llm_proxy_requests_total{IDENTITY,path="/v1/chat/completions",status="200"} 1
llm_proxy_suspensions_total{IDENTITY} 2
llm_proxy_rejected_requests_total{status="429",reason="rpm_limit"} 1
llm_proxy_request_duration_seconds_bucket{class="high",le="2.5"} 0
llm_proxy_request_duration_seconds_bucket{class="high",le="+Inf"} 1
llm_proxy_time_to_first_token_seconds_sum{class="high"} 0.7
llm_proxy_requests{class="high",state="queued"} 1
llm_proxy_requests{class="low",state="generating"} 0
llm_proxy_audit_log_failing 0
llm_proxy_active_users 1
llm_proxy_keys_file_errors_total 0"#;
        for expected in expected.replace("IDENTITY", identity).lines() {
            assert!(text.contains(expected), "missing {expected} in\n{text}");
        }
        for line in text.lines().filter(|line| !line.starts_with('#')) {
            let value = line.rsplit_once(' ').unwrap().1;
            assert!(value.parse::<f64>().is_ok(), "{line}");
        }
    }
}
