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

#[derive(Default)]
pub struct Metrics {
    state: Mutex<MetricsState>,
}

#[derive(Default)]
struct MetricsState {
    requests: BTreeMap<[String; 5], u64>,
    rejections: BTreeMap<[String; 2], u64>,
    tokens: BTreeMap<[String; 4], u64>,
    suspensions: BTreeMap<[String; 3], u64>,
    generating_seconds: BTreeMap<[String; 2], f64>,
    durations: BTreeMap<[String; 1], Histogram>,
    queue_durations: BTreeMap<[String; 1], Histogram>,
    first_token_durations: BTreeMap<[String; 1], Histogram>,
    keys_file_reloads: u64,
    keys_file_errors: u64,
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
    pub generating_seconds: f64,
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

impl Histogram {
    fn observe(&mut self, value: f64) {
        for (bucket, bound) in self.buckets.iter_mut().zip(BUCKETS) {
            if value <= bound {
                *bucket += 1;
            }
        }
        self.count += 1;
        self.sum += value;
    }
}

impl Metrics {
    fn lock(&self) -> MutexGuard<'_, MetricsState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn observe_request(&self, metric: RequestMetric) {
        let mut state = self.lock();
        let (user, key, class) = (metric.user, metric.key, metric.class);
        *state
            .requests
            .entry(labels([user, key, class, metric.path, &metric.status]))
            .or_default() += 1;
        if let Some(tokens) = metric.tokens {
            for (kind, count) in [
                ("prompt", tokens.prompt),
                ("output", tokens.output),
                ("cache_hit", tokens.cache_hit),
                ("prefill", tokens.prefill),
            ] {
                *state
                    .tokens
                    .entry(labels([user, key, class, kind]))
                    .or_default() += count;
            }
        }
        if metric.suspensions > 0 {
            *state
                .suspensions
                .entry(labels([user, key, class]))
                .or_default() += u64::from(metric.suspensions);
        }
        if metric.generating_seconds > 0.0 {
            *state
                .generating_seconds
                .entry(labels([user, class]))
                .or_default() += metric.generating_seconds;
        }
        state
            .durations
            .entry(labels([class]))
            .or_default()
            .observe(metric.total_seconds);
        if let Some(queued_seconds) = metric.queued_seconds {
            state
                .queue_durations
                .entry(labels([class]))
                .or_default()
                .observe(queued_seconds);
        }
        if let Some(seconds) = metric.time_to_first_token {
            state
                .first_token_durations
                .entry(labels([class]))
                .or_default()
                .observe(seconds);
        }
    }

    pub fn observe_rejection(&self, status: u16, reason: &str) {
        *self
            .lock()
            .rejections
            .entry(labels([&status.to_string(), reason]))
            .or_default() += 1;
    }

    pub fn keys_file_reloaded(&self) {
        self.lock().keys_file_reloads += 1;
    }

    pub fn keys_file_failed(&self) {
        self.lock().keys_file_errors += 1;
    }

    pub fn render(&self, gauges: &Gauges) -> String {
        let state = self.lock();
        let mut output = String::new();
        let out = &mut output;
        family(
            out,
            "llm_proxy_requests_total",
            "counter",
            "Requests which the proxy finished, by status (an HTTP status, 'aborted' or 'error').",
            ["user", "key", "class", "path", "status"],
            state
                .requests
                .iter()
                .map(|(labels, value)| (labels, *value as f64)),
        );
        family(
            out,
            "llm_proxy_rejected_requests_total",
            "counter",
            "Requests which the proxy rejected before they reached the engine.",
            ["status", "reason"],
            state
                .rejections
                .iter()
                .map(|(labels, value)| (labels, *value as f64)),
        );
        family(
            out,
            "llm_proxy_tokens_total",
            "counter",
            "Engine token counts of finished requests.",
            ["user", "key", "class", "type"],
            state
                .tokens
                .iter()
                .map(|(labels, value)| (labels, *value as f64)),
        );
        family(
            out,
            "llm_proxy_suspensions_total",
            "counter",
            "Suspensions of finished requests.",
            ["user", "key", "class"],
            state
                .suspensions
                .iter()
                .map(|(labels, value)| (labels, *value as f64)),
        );
        family(
            out,
            "llm_proxy_generating_seconds_total",
            "counter",
            "Time which finished requests spent generating in the engine.",
            ["user", "class"],
            state
                .generating_seconds
                .iter()
                .map(|(labels, value)| (labels, *value)),
        );
        histogram(
            out,
            "llm_proxy_request_duration_seconds",
            "Time from the arrival of a request to the end of its response.",
            &state.durations,
        );
        histogram(
            out,
            "llm_proxy_queue_duration_seconds",
            "Time which requests waited before the proxy sent them to the engine.",
            &state.queue_durations,
        );
        histogram(
            out,
            "llm_proxy_time_to_first_token_seconds",
            "Time from the arrival of a request to the first response data from the engine.",
            &state.first_token_durations,
        );
        let unlabeled = [[]; 1];
        family(
            out,
            "llm_proxy_keys_file_reloads_total",
            "counter",
            "Changes of the keys file which the proxy loaded.",
            [],
            unlabeled
                .iter()
                .map(|labels| (labels, state.keys_file_reloads as f64)),
        );
        family(
            out,
            "llm_proxy_keys_file_errors_total",
            "counter",
            "Failed reads, writes and parses of the keys file.",
            [],
            unlabeled
                .iter()
                .map(|labels| (labels, state.keys_file_errors as f64)),
        );
        drop(state);
        render_gauges(out, gauges);
        output
    }
}

fn render_gauges(out: &mut String, gauges: &Gauges) {
    let mut request_states = Vec::new();
    for class in &gauges.status.classes {
        let counts = &class.counts;
        for (state, count) in [
            ("running", counts.running),
            ("generating", counts.generating),
            ("suspended", counts.suspended),
            ("queued", counts.queued),
            ("abandoned", counts.abandoned),
        ] {
            request_states.push((labels([&class.name, state]), count as f64));
        }
    }
    family(
        out,
        "llm_proxy_requests",
        "gauge",
        "Requests in the proxy by class and state; generating requests are also running.",
        ["class", "state"],
        request_states
            .iter()
            .map(|(labels, value)| (labels, *value)),
    );
    let users = [([], gauges.status.total.users as f64)];
    family(
        out,
        "llm_proxy_active_users",
        "gauge",
        "Users with requests in the proxy.",
        [],
        users.iter().map(|(labels, value)| (labels, *value)),
    );
    let caps: Vec<_> = gauges
        .rank_caps
        .iter()
        .enumerate()
        .map(|(rank, cap)| (labels([&rank.to_string()]), *cap as f64))
        .collect();
    family(
        out,
        "llm_proxy_rank_parallel_request_cap",
        "gauge",
        "Maximum number of parallel requests per data parallel rank.",
        ["rank"],
        caps.iter().map(|(labels, value)| (labels, *value)),
    );
    let mut key_states: BTreeMap<&str, f64> =
        [("active", 0.0), ("blocked", 0.0), ("expired", 0.0)].into();
    for key in &gauges.keys {
        let name = match key.state {
            KeyState::Active => "active",
            KeyState::Blocked => "blocked",
            KeyState::Expired => "expired",
        };
        *key_states.entry(name).or_default() += 1.0;
    }
    let key_states: Vec<_> = key_states
        .into_iter()
        .map(|(state, count)| (labels([state]), count))
        .collect();
    family(
        out,
        "llm_proxy_keys",
        "gauge",
        "API keys in the keys file by state.",
        ["state"],
        key_states.iter().map(|(labels, value)| (labels, *value)),
    );
    let mut per_key: BTreeMap<&str, Vec<([String; 2], f64)>> = BTreeMap::new();
    let mut remaining_tokens = Vec::new();
    for key in &gauges.keys {
        let key_labels = labels([key.id, key.user]);
        let usage = &key.usage;
        let mut push = |name, value: f64| {
            per_key
                .entry(name)
                .or_default()
                .push((key_labels.clone(), value))
        };
        push("in_flight", usage.in_flight as f64);
        push("requests_last_minute", usage.requests_last_minute as f64);
        push("tokens_last_minute", usage.tokens_last_minute as f64);
        push("tokens_last_day", usage.tokens_last_day as f64);
        if let Some(last_used) = usage.last_used {
            push("last_used", last_used as f64);
        }
        if let Some(limit) = key.limits.rpm_limit {
            push(
                "remaining_requests",
                limit.get().saturating_sub(usage.requests_last_minute) as f64,
            );
        }
        for (window, limit, used) in [
            ("minute", key.limits.tpm_limit, usage.tokens_last_minute),
            ("day", key.limits.tpd_limit, usage.tokens_last_day),
        ] {
            if let Some(limit) = limit {
                remaining_tokens.push((
                    labels([key.id, key.user, window]),
                    limit.get().saturating_sub(used) as f64,
                ));
            }
        }
    }
    for (name, kind, help) in [
        (
            "in_flight",
            "llm_proxy_key_in_flight_requests",
            "Requests of an API key in the proxy.",
        ),
        (
            "requests_last_minute",
            "llm_proxy_key_requests_last_minute",
            "Requests of an API key in the last minute.",
        ),
        (
            "tokens_last_minute",
            "llm_proxy_key_tokens_last_minute",
            "Prompt and output tokens of an API key in the last minute.",
        ),
        (
            "tokens_last_day",
            "llm_proxy_key_tokens_last_day",
            "Prompt and output tokens of an API key in the last 24 hours.",
        ),
        (
            "last_used",
            "llm_proxy_key_last_used_timestamp_seconds",
            "Time of the last request of an API key since this process started.",
        ),
        (
            "remaining_requests",
            "llm_proxy_key_remaining_requests",
            "Requests an API key can still send in the current minute.",
        ),
    ] {
        let samples = per_key.remove(name).unwrap_or_default();
        family(
            out,
            kind,
            "gauge",
            help,
            ["key", "user"],
            samples.iter().map(|(labels, value)| (labels, *value)),
        );
    }
    family(
        out,
        "llm_proxy_key_remaining_tokens",
        "gauge",
        "Tokens an API key can still use in the current window.",
        ["key", "user", "window"],
        remaining_tokens
            .iter()
            .map(|(labels, value)| (labels, *value)),
    );
}

fn labels<const N: usize>(values: [&str; N]) -> [String; N] {
    values.map(str::to_string)
}

fn family<'a, const N: usize>(
    out: &mut String,
    name: &str,
    kind: &str,
    help: &str,
    names: [&str; N],
    samples: impl Iterator<Item = (&'a [String; N], f64)>,
) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
    for (values, value) in samples {
        let _ = writeln!(
            out,
            "{name}{} {}",
            label_set(&names, values, None),
            number(value)
        );
    }
}

fn histogram(out: &mut String, name: &str, help: &str, series: &BTreeMap<[String; 1], Histogram>) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} histogram");
    for (values, histogram) in series {
        for (bound, count) in BUCKETS.iter().zip(histogram.buckets) {
            let _ = writeln!(
                out,
                "{name}_bucket{} {count}",
                label_set(&["class"], values, Some(&number(*bound)))
            );
        }
        let _ = writeln!(
            out,
            "{name}_bucket{} {}",
            label_set(&["class"], values, Some("+Inf")),
            histogram.count
        );
        let labels = label_set(&["class"], values, None);
        let _ = writeln!(out, "{name}_sum{labels} {}", number(histogram.sum));
        let _ = writeln!(out, "{name}_count{labels} {}", histogram.count);
    }
}

fn label_set(names: &[&str], values: &[String], bound: Option<&str>) -> String {
    let mut pairs: Vec<String> = names
        .iter()
        .zip(values)
        .map(|(name, value)| format!("{name}=\"{}\"", escape(value)))
        .collect();
    if let Some(bound) = bound {
        pairs.push(format!("le=\"{bound}\""));
    }
    if pairs.is_empty() {
        String::new()
    } else {
        format!("{{{}}}", pairs.join(","))
    }
}

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
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
            generating_seconds: 1.5,
            total_seconds: 3.0,
            queued_seconds: Some(0.2),
            time_to_first_token: Some(0.7),
        });
        metrics.observe_rejection(429, "rpm_limit");
        let status = Status {
            total: RequestCounts {
                users: 1,
                ..RequestCounts::default()
            },
            classes: vec![ClassStatus {
                name: "high".to_string(),
                priority: 1,
                counts: RequestCounts {
                    requests: 2,
                    running: 1,
                    queued: 1,
                    ..RequestCounts::default()
                },
            }],
            users: Vec::new(),
        };
        let text = metrics.render(&Gauges {
            status,
            rank_caps: &[16],
            keys: Vec::new(),
        });
        let user = r#"user="litellm/bob \"b\"""#;
        assert!(text.contains(&format!(
            r#"llm_proxy_requests_total{{{user},key="key-1",class="high",path="/v1/chat/completions",status="200"}} 1"#
        )));
        assert!(text.contains(&format!(
            r#"llm_proxy_tokens_total{{{user},key="key-1",class="high",type="output"}} 5"#
        )));
        assert!(text
            .contains(r#"llm_proxy_rejected_requests_total{status="429",reason="rpm_limit"} 1"#));
        assert!(
            text.contains(r#"llm_proxy_request_duration_seconds_bucket{class="high",le="2.5"} 0"#)
        );
        assert!(
            text.contains(r#"llm_proxy_request_duration_seconds_bucket{class="high",le="5"} 1"#)
        );
        assert!(
            text.contains(r#"llm_proxy_request_duration_seconds_bucket{class="high",le="+Inf"} 1"#)
        );
        assert!(text.contains(r#"llm_proxy_time_to_first_token_seconds_sum{class="high"} 0.7"#));
        assert!(text.contains(r#"llm_proxy_requests{class="high",state="queued"} 1"#));
        assert!(text.contains("llm_proxy_active_users 1"));
        assert!(text.contains(r#"llm_proxy_rank_parallel_request_cap{rank="0"} 16"#));
        assert!(text.contains("llm_proxy_keys_file_errors_total 0"));
        for line in text.lines().filter(|line| !line.starts_with('#')) {
            let (_, value) = line.rsplit_once(' ').unwrap();
            assert!(value.parse::<f64>().is_ok(), "{line}");
        }
    }
}
