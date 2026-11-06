use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::{Mutex, MutexGuard};

use super::keys::{KeyLimits, KeyState};
use super::limits::UsageView;
use crate::engine_protocol::TokenCounts;
use crate::quality_of_service::Status;

const BUCKETS: [f64; 15] = [
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0,
];
const KEYS_FILE_RELOADS: &str = "llm_proxy_keys_file_reloads_total";
const KEYS_FILE_ERRORS: &str = "llm_proxy_keys_file_errors_total";
const FAMILIES: &[(&str, &str, &str)] = &[
    (
        "llm_proxy_requests_total",
        "counter",
        "Finished requests by status: an HTTP status, aborted or error.",
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
        "Requests in the proxy by class and state.",
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
        "Tokens an API key can still use in the window.",
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

pub struct RequestMetric<'a> {
    pub user: &'a str,
    pub key: &'a str,
    pub class: &'a str,
    pub path: &'a str,
    pub status: String,
    pub tokens: Option<TokenCounts>,
    pub suspensions: u32,
    pub total_seconds: f64,
    pub queued_seconds: Option<f64>,
    pub time_to_first_token: Option<f64>,
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
    pub rank_caps: &'a [usize],
    pub keys: Vec<KeyGauge<'a>>,
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

    pub fn observe_request(&self, metric: RequestMetric) {
        let identity = [
            ("user", metric.user),
            ("key", metric.key),
            ("class", metric.class),
        ];
        let request = [("path", metric.path), ("status", metric.status.as_str())];
        self.add(
            "llm_proxy_requests_total",
            labels(&[&identity[..], &request].concat()),
            1.0,
        );
        if let Some(tokens) = metric.tokens {
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
        if metric.suspensions > 0 {
            self.add(
                "llm_proxy_suspensions_total",
                labels(&identity),
                f64::from(metric.suspensions),
            );
        }
        let class = || labels(&[("class", metric.class)]);
        self.observe(
            "llm_proxy_request_duration_seconds",
            class(),
            metric.total_seconds,
        );
        if let Some(seconds) = metric.queued_seconds {
            self.observe("llm_proxy_queue_duration_seconds", class(), seconds);
        }
        if let Some(seconds) = metric.time_to_first_token {
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
    for class in &gauges.status.classes {
        let counts = &class.counts;
        for (state, count) in [
            ("running", counts.running),
            ("generating", counts.generating),
            ("suspended", counts.suspended),
            ("queued", counts.queued),
            ("abandoned", counts.abandoned),
        ] {
            set(
                "llm_proxy_requests",
                labels(&[("class", &class.name), ("state", state)]),
                count as u64,
            );
        }
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
            let remaining = limit.get().saturating_sub(usage.requests_last_minute);
            set(
                "llm_proxy_key_remaining_requests",
                labels(&identity),
                remaining,
            );
        }
        for (window, limit, used) in [
            ("minute", key.limits.tpm_limit, usage.tokens_last_minute),
            ("day", key.limits.tpd_limit, usage.tokens_last_day),
        ] {
            if let Some(limit) = limit {
                let series = labels(&[&identity[..], &[("window", window)]].concat());
                set(
                    "llm_proxy_key_remaining_tokens",
                    series,
                    limit.get().saturating_sub(used),
                );
            }
        }
    }
    series
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality_of_service::{ClassStatus, RequestCounts};

    #[test]
    fn exposition_format() {
        let metrics = Metrics::default();
        metrics.observe_request(RequestMetric {
            user: "litellm/bob \"b\"",
            key: "key-1",
            class: "high",
            path: "/v1/chat/completions",
            status: "200".to_string(),
            tokens: Some(TokenCounts {
                prompt: 10,
                output: 5,
                cache_hit: 2,
                prefill: 8,
                prefill_count: 1,
            }),
            suspensions: 2,
            total_seconds: 3.0,
            queued_seconds: Some(0.2),
            time_to_first_token: Some(0.7),
        });
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
            rank_caps: &[16],
            keys: Vec::new(),
        });
        let identity = r#"user="litellm/bob \"b\"",key="key-1",class="high""#;
        for expected in [
            format!(
                r#"llm_proxy_requests_total{{{identity},path="/v1/chat/completions",status="200"}} 1"#
            ),
            format!(r#"llm_proxy_tokens_total{{{identity},type="output"}} 5"#),
            format!(r#"llm_proxy_suspensions_total{{{identity}}} 2"#),
            r#"llm_proxy_rejected_requests_total{status="429",reason="rpm_limit"} 1"#.to_string(),
            r#"llm_proxy_request_duration_seconds_bucket{class="high",le="2.5"} 0"#.to_string(),
            r#"llm_proxy_request_duration_seconds_bucket{class="high",le="+Inf"} 1"#.to_string(),
            r#"llm_proxy_time_to_first_token_seconds_sum{class="high"} 0.7"#.to_string(),
            r#"llm_proxy_requests{class="high",state="queued"} 1"#.to_string(),
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
