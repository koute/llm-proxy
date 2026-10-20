use std::fmt;

use http::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const VLLM_REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");
const VLLM_RANK_HEADER: HeaderName = HeaderName::from_static("x-data-parallel-rank");
const VLLM_UNPINNED_PATHS: &[&str] = &["/v1/responses"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ProxyId {
    pub instance: u64,
    pub sequence: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    Vllm,
    Sglang,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Command {
    Suspend { ids: Vec<ProxyId> },
    Resume { ids: Vec<ProxyId> },
    Info,
    Stats,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandLine {
    pub seq: u64,
    #[serde(flatten)]
    pub command: Command,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EngineLine {
    Event(Event),
    Success { seq: u64, result: Value },
    Failure { seq: u64, error: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Running { id: ProxyId },
    Stopped { id: ProxyId, reason: StopReason },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Suspended,
    Preempted,
    Finished,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineInfo {
    pub engine: EngineKind,
    pub model: String,
    pub data_parallel_rank: Option<usize>,
    pub max_running_requests: usize,
    pub priority_scheduling: bool,
    pub kv_cache_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineStats {
    pub requests: Vec<RequestStats>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestStats {
    pub id: ProxyId,
    pub state: RequestPhase,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestPhase {
    Running,
    Suspended,
    Waiting,
}

impl EngineKind {
    pub fn label_request(
        self,
        headers: &mut HeaderMap,
        body: &mut Map<String, Value>,
        id: ProxyId,
        data_parallel_rank: Option<usize>,
    ) {
        match self {
            EngineKind::Vllm => {
                let id = HeaderValue::from_str(&id.to_string()).expect("invalid proxy id header");
                headers.insert(VLLM_REQUEST_ID_HEADER, id);
                if let Some(rank) = data_parallel_rank {
                    headers.insert(VLLM_RANK_HEADER, HeaderValue::from(rank));
                }
            }
            EngineKind::Sglang => {
                body.insert("rid".to_string(), Value::from(id.to_string()));
                if let Some(rank) = data_parallel_rank {
                    body.insert("routed_dp_rank".to_string(), Value::from(rank));
                }
            }
        }
    }

    pub fn priority(self, class_priority: i32) -> i64 {
        match self {
            EngineKind::Vllm => -i64::from(class_priority),
            EngineKind::Sglang => i64::from(class_priority),
        }
    }

    pub fn accepts_rank_pin(self, path: &str) -> bool {
        self != EngineKind::Vllm || !VLLM_UNPINNED_PATHS.contains(&path)
    }
}

impl fmt::Display for EngineKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            EngineKind::Vllm => "vllm",
            EngineKind::Sglang => "sglang",
        })
    }
}

impl fmt::Display for ProxyId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "qos-{:016x}-{:016x}",
            self.instance, self.sequence
        )
    }
}

impl TryFrom<String> for ProxyId {
    type Error = String;

    fn try_from(text: String) -> Result<ProxyId, String> {
        let parts = text
            .strip_prefix("qos-")
            .and_then(|rest| rest.split_once('-'))
            .filter(|(instance, sequence)| instance.len() == 16 && sequence.len() == 16);
        let parse = |hex: &str| u64::from_str_radix(hex, 16).ok();
        parts
            .and_then(|(instance, sequence)| {
                Some(ProxyId {
                    instance: parse(instance)?,
                    sequence: parse(sequence)?,
                })
            })
            .ok_or_else(|| format!("invalid proxy id '{text}'"))
    }
}

impl From<ProxyId> for String {
    fn from(id: ProxyId) -> String {
        id.to_string()
    }
}

impl CommandLine {
    pub fn encode(&self) -> Vec<u8> {
        let mut line = serde_json::to_vec(self).expect("unserializable control command");
        line.push(b'\n');
        line
    }
}

impl EngineLine {
    pub fn parse(line: &[u8]) -> Result<EngineLine, String> {
        serde_json::from_slice(line)
            .map_err(|error| format!("invalid line from the engine control socket: {error}"))
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut line = serde_json::to_vec(self).expect("unserializable engine line");
        line.push(b'\n');
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_format() {
        let id = ProxyId {
            instance: 0xab,
            sequence: 1,
        };
        let command = CommandLine {
            seq: 7,
            command: Command::Suspend { ids: vec![id] },
        };
        assert_eq!(
            command.encode(),
            b"{\"seq\":7,\"op\":\"suspend\",\"ids\":[\"qos-00000000000000ab-0000000000000001\"]}\n"
        );
        assert_eq!(
            EngineLine::parse(
                br#"{"event":"stopped","id":"qos-00000000000000ab-0000000000000001","reason":"preempted"}"#
            ),
            Ok(EngineLine::Event(Event::Stopped {
                id,
                reason: StopReason::Preempted
            }))
        );
        assert!(EngineLine::parse(br#"{"event":"running","id":"qos-1"}"#).is_err());
        assert_eq!(
            EngineLine::parse(br#"{"seq":7,"result":null}"#),
            Ok(EngineLine::Success {
                seq: 7,
                result: Value::Null
            })
        );
        assert_eq!(
            EngineLine::parse(br#"{"seq":8,"error":"unknown op"}"#),
            Ok(EngineLine::Failure {
                seq: 8,
                error: "unknown op".to_string()
            })
        );
    }
}
