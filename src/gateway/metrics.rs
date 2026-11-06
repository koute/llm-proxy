use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::{Mutex, MutexGuard};

use super::keys::{KeyLimits, KeyState};
use super::limits::UsageView;
use super::FinishedRequest;
use crate::quality_of_service::{RequestCounts, Status};
use crate::utils::AppendHealth;

const BUCKETS: [f64; 15] = [
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0,
];
const KEYS_FILE_RELOADS: &str = "llm_proxy_keys_file_reloads_total";
const KEYS_FILE_ERRORS: &str = "llm_proxy_keys_file_errors_total";
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
    pub id: &'a str,
    pub user: &'a str,
    pub state: KeyState,
    pub limits: KeyLimits,
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

fn labels(pairs: &[(&str, &str)]) -> String {
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
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn add(&self, name: &'static str, labels: String, value: f64) {
        *self.lock().counters.entry((name, labels)).or_default() += value;
    }

    fn observe(&self, name: &'static str, labels: String, value: f64) {
        let mut state = self.lock();
        let histogram = state.histograms.entry((name, labels)).or_default();
        for (bucket, bound) in histogram.buckets.iter_mut().zip(BUCKETS) {
            *bucket += u64::from(value <= bound);
        }
        histogram.count += 1;
        histogram.sum += value;
    }

    /// Passthrough requests have no class; they're counted with an empty one.
    pub fn observe_passthrough(&self, user: &str, key: Option<&str>, path: &str, status: &str) {
        let series = [
            ("user", user),
            ("key", key.unwrap_or_default()),
            ("class", ""),
            ("path", path),
            ("status", status),
        ];
        self.add("llm_proxy_requests_total", labels(&series), 1.0);
    }

    pub fn observe_request(&self, finished: &FinishedRequest, status: &str) {
        let (summary, attribution) = (finished.summary, finished.attribution);
        let key = attribution.key_id.as_deref().unwrap_or_default();
        let identity = [
            ("user", summary.user.as_str()),
            ("key", key),
            ("class", finished.class()),
        ];
        let route = attribution.route.as_deref().unwrap_or(&attribution.path);
        let request = [("path", route), ("status", status)];
        self.add(
            "llm_proxy_requests_total",
            labels(&[&identity[..], &request].concat()),
            1.0,
        );
        if let Some(tokens) = summary.tokens {
            for (kind, count) in [
                ("prompt", tokens.prompt),
                ("output", tokens.output),
                ("cache_hit", tokens.cache_hit),
                ("prefill", tokens.prefill),
            ] {
                let series = labels(&[&identity[..], &[("type", kind)]].concat());
                self.add("llm_proxy_tokens_total", series, count as f64);
            }
        }
        if summary.suspensions > 0 {
            let suspensions = f64::from(summary.suspensions);
            self.add(
                "llm_proxy_suspensions_total",
                labels(&identity),
                suspensions,
            );
        }
        let class = || labels(&[("class", finished.class())]);
        let total = finished.total_seconds;
        self.observe("llm_proxy_request_duration_seconds", class(), total);
        if let Some(seconds) = finished.queued_seconds {
            self.observe("llm_proxy_queue_duration_seconds", class(), seconds);
        }
        if let Some(seconds) = finished.time_to_first_token {
            self.observe("llm_proxy_time_to_first_token_seconds", class(), seconds);
        }
    }

    pub fn observe_rejection(&self, status: u16, reason: &str) {
        let series = labels(&[("status", &status.to_string()), ("reason", reason)]);
        self.add("llm_proxy_rejected_requests_total", series, 1.0);
    }

    pub fn keys_file_reloaded(&self) {
        self.add(KEYS_FILE_RELOADS, String::new(), 1.0);
    }

    pub fn keys_file_failed(&self) {
        self.add(KEYS_FILE_ERRORS, String::new(), 1.0);
    }

    pub fn render(&self, gauges: &Gauges) -> String {
        let mut series = gauge_series(gauges);
        let mut state = self.lock();
        for name in [KEYS_FILE_RELOADS, KEYS_FILE_ERRORS] {
            state.counters.entry((name, String::new())).or_default();
        }
        series.extend(
            state
                .counters
                .iter()
                .map(|(key, value)| (key.clone(), *value)),
        );
        let mut out = String::new();
        for (name, kind, help) in FAMILIES {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
            let of_family = |(family, _): &&(&str, String)| family == name;
            for ((_, labels), value) in series.iter().filter(|(key, _)| of_family(key)) {
                let _ = writeln!(out, "{name}{labels} {value}");
            }
            for ((_, labels), histogram) in
                state.histograms.iter().filter(|(key, _)| of_family(key))
            {
                for (bound, count) in BUCKETS.iter().zip(histogram.buckets) {
                    let _ = writeln!(
                        out,
                        "{name}_bucket{} {count}",
                        with_bound(labels, &bound.to_string())
                    );
                }
                let count = histogram.count;
                let _ = writeln!(out, "{name}_bucket{} {count}", with_bound(labels, "+Inf"));
                let _ = writeln!(
                    out,
                    "{name}_sum{labels} {}\n{name}_count{labels} {count}",
                    histogram.sum
                );
            }
        }
        out
    }
}

fn gauge_series(gauges: &Gauges) -> Series {
    let mut series = Series::new();
    let mut set = |name, labels, value: u64| series.insert((name, labels), value as f64);
    let idle = RequestCounts::default();
    let classes = gauges.classes.iter().map(|name| {
        let class = gauges
            .status
            .classes
            .iter()
            .find(|class| class.name == *name);
        (*name, class.map_or(&idle, |class| &class.counts))
    });
    let unconfigured = gauges
        .status
        .classes
        .iter()
        .filter(|class| !gauges.classes.contains(&class.name.as_str()));
    for (name, counts) in
        classes.chain(unconfigured.map(|class| (class.name.as_str(), &class.counts)))
    {
        // The states partition the requests: a request sent to the engine is either
        // generating or waiting in the engine (e.g. for a prefill).
        for (state, count) in [
            ("queued", counts.queued),
            (
                "waiting_in_engine",
                counts.running - counts.running_generating,
            ),
            ("generating", counts.running_generating),
            ("suspended", counts.suspended),
            ("abandoned", counts.abandoned),
        ] {
            set(
                "llm_proxy_requests",
                labels(&[("class", name), ("state", state)]),
                count as u64,
            );
        }
    }
    if let Some(health) = &gauges.scheduling_log {
        let failing = u64::from(health.is_failing);
        set("llm_proxy_scheduling_log_failing", String::new(), failing);
    }
    if let Some(health) = &gauges.audit_log {
        set(
            "llm_proxy_audit_log_failing",
            String::new(),
            u64::from(health.is_failing),
        );
        set(
            "llm_proxy_audit_log_dropped_records_total",
            String::new(),
            health.dropped as u64,
        );
    }
    set(
        "llm_proxy_active_users",
        String::new(),
        gauges.status.total.users as u64,
    );
    for (rank, cap) in gauges.rank_caps.iter().enumerate() {
        set(
            "llm_proxy_rank_parallel_request_cap",
            labels(&[("rank", &rank.to_string())]),
            *cap as u64,
        );
    }
    for (state, name) in [
        (KeyState::Active, "active"),
        (KeyState::Blocked, "blocked"),
        (KeyState::Expired, "expired"),
    ] {
        let count = gauges.keys.iter().filter(|key| key.state == state).count();
        set("llm_proxy_keys", labels(&[("state", name)]), count as u64);
    }
    for key in &gauges.keys {
        let identity = [("key", key.id), ("user", key.user)];
        let usage = &key.usage;
        set(
            "llm_proxy_key_in_flight_requests",
            labels(&identity),
            usage.in_flight,
        );
        if let Some(limit) = key.limits.rpm_limit {
            let remaining = limit.saturating_sub(usage.requests_last_minute);
            set(
                "llm_proxy_key_remaining_requests",
                labels(&identity),
                remaining,
            );
        }
        // Like the limits themselves, the estimates of the requests in flight count.
        let reserved = usage.reserved_tokens;
        for (window, limit, used) in [
            (
                "minute",
                key.limits.tpm_limit,
                usage.tokens_last_minute.saturating_add(reserved),
            ),
            (
                "day",
                key.limits.tpd_limit,
                usage.tokens_last_day.saturating_add(reserved),
            ),
        ] {
            if let Some(limit) = limit {
                let series = labels(&[&identity[..], &[("window", window)]].concat());
                set(
                    "llm_proxy_key_remaining_tokens",
                    series,
                    limit.saturating_sub(used),
                );
            }
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
    use crate::quality_of_service::{ClassStatus, RequestCounts};

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
        let classes = vec![ClassStatus {
            name: "high".to_string(),
            priority: 1,
            counts,
        }];
        let status = Status {
            total: RequestCounts { users: 1, ..counts },
            classes,
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
        for expected in [
            format!(
                r#"llm_proxy_requests_total{{{identity},path="/v1/chat/completions",status="200"}} 1"#
            ),
            format!(r#"llm_proxy_suspensions_total{{{identity}}} 2"#),
            r#"llm_proxy_rejected_requests_total{status="429",reason="rpm_limit"} 1"#.to_string(),
            r#"llm_proxy_request_duration_seconds_bucket{class="high",le="2.5"} 0"#.to_string(),
            r#"llm_proxy_request_duration_seconds_bucket{class="high",le="+Inf"} 1"#.to_string(),
            r#"llm_proxy_time_to_first_token_seconds_sum{class="high"} 0.7"#.to_string(),
            r#"llm_proxy_requests{class="high",state="queued"} 1"#.to_string(),
            r#"llm_proxy_requests{class="low",state="generating"} 0"#.to_string(),
            "llm_proxy_audit_log_failing 0".to_string(),
            "llm_proxy_active_users 1".to_string(),
            "llm_proxy_keys_file_errors_total 0".to_string(),
        ] {
            assert!(text.contains(&expected), "missing {expected} in\n{text}");
        }
        for line in text.lines().filter(|line| !line.starts_with('#')) {
            assert!(
                line.rsplit_once(' ').unwrap().1.parse::<f64>().is_ok(),
                "{line}"
            );
        }
    }
}
