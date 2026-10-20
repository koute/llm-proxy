DISCLAIMER: This project is heavily AI-assisted.

# A quality-of-service proxy for LLM inference engines

This repository contains a quality-of-service proxy for OpenAI and Anthropic compatible LLM engines (vLLM and sglang).

  * Every request belongs to a priority class. When the engine is full higher priority requests run first, and the proxy suspends lower priority requests *inside* the engine, so no generated tokens are lost.
  * Fair scheduling between users: every user's first request runs before anyone's second request, and lighter users run before heavier ones. When there's only one user that user can use the whole capacity.
  * Idle-only classes, whose requests only run when there are no requests of a higher priority.
  * Per-class soft caps (fewer parallel requests, so that each one generates faster) and per-user limits.
  * Benchmarks the engine to find out how many parallel requests it can handle before the total throughput drops.
  * Restarts in-place on `SIGUSR1`, even into a new binary, without interrupting any requests.
  * Supports tensor parallel and data parallel engines.
  * Optional scheduling log in JSON lines format.

It requires a patched engine; this repository includes patches for vLLM 0.30.0 and sglang 0.5.20.

## How it works

```
client ─HTTP(S)─> llm-proxy frontend ─Unix socket─> llm-proxy serve ─HTTP─> vLLM / sglang
                  (TLS, HTTP parsing)               (middle end: QoS)  └─Unix socket per rank─┘
```

  * `llm-proxy frontend` accepts HTTP(S) connections and forwards every request to the middle end. It doesn't contain any logic, so you should rarely need to restart it. (Restarting it closes all of its connections.)
  * `llm-proxy serve` (the middle end) does everything else: API keys, accounts, QoS, and the connections to the engine.

The patched engine works as usual when the proxy isn't running. When the middle end connects to the engine's control socket the engine suspends and resumes requests on command and reports which requests are generating. When the middle end disconnects the engine resumes all suspended requests.

## Building

```sh
cargo build --release                                     # HTTP only
cargo build --release --features tls                      # HTTP and HTTPS
cargo build --release --target x86_64-unknown-linux-musl  # static binary, HTTP only
```

The `tls` feature needs a C compiler (for `ring`); a static build with TLS needs a musl C compiler, e.g. `x86_64-linux-musl-gcc`.

## Patching the engine

```sh
SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cp patches/llm_proxy_qos.py "$SITE"/
patch -p1 -d "$SITE" < patches/vllm-0.30.0.patch      # or sglang-0.5.20.patch
```

Then start the engine with the path to the control socket in an environment variable. The engine replaces `{rank}` with the data parallel rank; it's required if there's more than one rank.

```sh
VLLM_QOS_CONTROL_SOCKET='/run/llm/qos-{rank}.sock' vllm serve MODEL --host 127.0.0.1 --port 8000 \
    --scheduling-policy priority
SGLANG_QOS_CONTROL_SOCKET='/run/llm/qos-{rank}.sock' python -m sglang.launch_server --model-path MODEL \
    --host 127.0.0.1 --port 8000 --enable-priority-scheduling
```

vLLM warns about an unknown `VLLM_QOS_CONTROL_SOCKET` environment variable at startup; you can ignore it, just don't use `--fail-on-environ-validation`.

The engine doesn't free a suspended request's KV cache until it needs the memory or the slot for other requests.

Both patches support tensor parallelism and data parallelism (including MoE waves and DP attention), although there are no tests of data parallelism on real hardware yet. There are also no tests for pipeline parallelism, prefill/decode disaggregation and speculative decoding; the patched sglang rejects pipeline parallelism and prefill/decode disaggregation at startup.

## Usage

  1. Patch and start the engine.
  2. Write `config/general.json` and `config/accounts.json` (see `config/*.example.json`).
  3. Start the middle end and the frontend:
     ```sh
     llm-proxy serve
     llm-proxy frontend
     ```
     Until `config/backend.json` exists every request gets a 503.
  4. Benchmark the engine. This writes `config/backend.json`.
     ```sh
     llm-proxy benchmark --upstream http://127.0.0.1:8000 --control-socket '/run/llm/qos-{rank}.sock' \
         --prompt-tokens 1000 --output-tokens 512
     ```
     Use prompt and output lengths similar to your workload, as they affect the results.
  5. Check `config/backend.json` and send `SIGUSR1` to `llm-proxy serve` to load it.

The configuration paths default to `config/general.json`, `config/accounts.json` and `config/backend.json`; use `--general`, `--accounts` and `--backend` to change them. `llm-proxy check-config` validates the configuration files.

To switch to another model or engine start the new engine, run the benchmark again and send `SIGUSR1`; `general.json` and `accounts.json` don't need to change.

While there are requests the middle end prints a status line every second:

```
status: 6 requests from 4 users: 4 running (4 generating in the engine), 1 suspended, 1 queued, 0 abandoned
status:   high (priority 1): 1 request from 1 user: 1 running (1 generating)
status:   medium (priority 0): 4 requests from 3 users: 3 running (3 generating), 1 suspended
status:   idle (priority -10): 1 request from 1 user: 1 queued
```

`--verbose` logs every QoS decision. `llm-proxy frontend --dump-requests <file>` appends every request to a file, unredacted and with API keys included, so only use it for debugging.

## Configuration

### general.json

| Key | Default | Description |
|---|---|---|
| `frontend.listen` | required | The frontend's address, e.g. `"0.0.0.0:8080"`. |
| `frontend.tls` | `null` | `null` (HTTP), `"self_signed"`, or `{"certificate": path, "private_key": path}` (PEM). Requires the `tls` feature. |
| `middle_end_socket` | required | Path to the Unix socket between the frontend and the middle end. |
| `paths.generating` | chat, completions, responses, messages, `/generate` | QoS-managed paths. |
| `paths.whitelist` | models, embeddings, tokenize, health, ... | Other paths which the proxy forwards as-is. Every other path gets a 404. |
| `limits.max_request_body_bytes` | 64 MiB | Bigger requests get a 413. |
| `limits.request_body_timeout_seconds` | 30 | Requests which take longer to receive get a 408. |
| `limits.max_queued_requests` | 1024 | Requests over this limit get a 503. |
| `limits.keepalive_interval_seconds` | 15 | How long a response can be silent before the proxy sends keepalive bytes. |
| `quality_of_service.minimum_peak_throughput` | 1.0 | See below. |
| `quality_of_service.peak_throughput_tolerance` | 0.05 | See below. |
| `quality_of_service.maximum_parallel_requests` | `null` | Limit on parallel requests across all ranks. |
| `quality_of_service.swap_hysteresis_seconds` | 30 | Minimum time a request runs after it starts or resumes before the proxy can swap it out for another request of the same priority. |
| `quality_of_service.usage_half_life_minutes` | 60 | Half-life of the per-user usage history. |
| `process.restart_flush_timeout_seconds` | 5 | Time to write out pending data on `SIGUSR1`. |
| `process.shutdown_drain_timeout_seconds` | 30 | Time for requests to finish after `SIGTERM` or `SIGINT`. |
| `scheduling_log` | `null` | Path to the scheduling log. |

Both path lists accept `/prefix/*` wildcards.

The maximum number of parallel requests per data parallel rank (the hard cap) is the lowest of: the engine's limit, the rank's share of `maximum_parallel_requests`, and the highest number of parallel requests whose total throughput in the benchmark is at least `peak * minimum_peak_throughput * (1 - peak_throughput_tolerance)`. For example, if the benchmark measured 400, 600, 750, 800 and 750 tokens/s for 1 to 5 parallel requests then the hard cap is 4 with a `minimum_peak_throughput` of 1.0, and 5 with 0.9.

### accounts.json

```json
{
  "classes": [
    { "name": "interactive", "priority": 10, "soft_cap": 16, "max_parallel_per_user": 4 },
    { "name": "batch", "priority": 0, "cost": 0.5, "soft_cap": 32 },
    { "name": "idle", "priority": -10, "idle_only": true, "cost": 0.1 }
  ],
  "accounts": [
    { "name": "alice", "api_key": "sk-...", "default_class": "interactive", "allowed_classes": ["batch", "idle"] },
    { "name": "litellm", "api_key": "sk-...", "default_class": "interactive", "virtual_user_http_header": "x-litellm-end-user" },
    { "name": "anonymous", "api_key": "", "default_class": "idle" }
  ]
}
```

| Class key | Default | Description |
|---|---|---|
| `name` | required | |
| `priority` | required | Signed integer; higher runs first. |
| `idle_only` | false | Only run when there are no requests of a class with a higher priority. |
| `cost` | 1.0 | Usage multiplier. |
| `soft_cap` | `null` | Maximum number of parallel requests while there's a request of this class. |
| `max_parallel_per_user` | `null` | Maximum number of parallel requests of a single user in this class. |

Account keys are `name` (can't contain `/`), `api_key`, `default_class`, `allowed_classes` (classes a client can pick in addition to the default one) and the optional `virtual_user_http_header`.

  * The proxy reads the API key from `Authorization: Bearer <key>` or `x-api-key`. Requests without an API key use the account with an empty `api_key`, if there is one.
  * Clients can pick a class with the `x-priority: <class>` header.
  * With `virtual_user_http_header` every value of that header is a separate user (`<account>/<value>`). This is useful e.g. with LiteLLM in front of the proxy, which uses a single API key but can send the name of the end user in a header.
  * The proxy strips the API key, `x-priority` and any engine-specific overrides (e.g. `x-request-id`, `x-data-parallel-rank`, or `priority` in the body) before it forwards the request.
  * The middle end logs every rejected request along with the API key (masked if it's a valid key), the user agent and the names of the headers.

### backend.json

`llm-proxy benchmark` writes this file; see `config/backend.example.json` for an example.

| Key | Description |
|---|---|
| `engine` | `"vllm"` or `"sglang"`. |
| `model` | Model name; informational only. |
| `engine_max_parallel_requests_per_rank` | How many requests the engine can run at once on a single rank. |
| `priority_scheduling` | Whether to send the class priority to the engine, which uses it only to pick which request to preempt when it runs out of KV cache. Requires `--scheduling-policy priority` (vLLM) or `--enable-priority-scheduling` (sglang). |
| `ranks[]` | `upstream`, `control_socket` and `data_parallel_rank` of every data parallel rank. |
| `benchmark` | The benchmark results. |

The benchmark runs N parallel requests per rank for N = 1, 2, 3, 4, 6, 8, 12, 16, 24, ... and measures the total throughput. It stops at the engine's limit (the lower of `max_running_requests` and how many requests fit into the KV cache) or after two levels with less than half of the peak throughput. Around the hard cap it measures extra levels so that the cap doesn't come from interpolating between two distant levels.

## Scheduling

Every 250 ms, and every time a request arrives or finishes, the middle end orders the requests and runs as many of them as the caps allow. The order is:

  1. Higher class `priority` first. Classes with the same priority form a tier.
  2. Each user's first request in a tier. These can suspend other requests immediately, without waiting for `swap_hysteresis_seconds`.
  3. Requests which started or resumed less than `swap_hysteresis_seconds` ago.
  4. Each user's second request before anyone's third, and so on.
  5. Lighter users before heavier users.
  6. Running requests before waiting ones, and then by arrival.

A user's usage is the time their requests spent generating in the engine, multiplied by the class `cost`. Older usage counts less, with a half-life of `usage_half_life_minutes`.

Requests of an `idle_only` class only run when there are no requests of a class with a higher priority.

A `soft_cap` limits the total number of parallel requests while there's a request of that class, so that each request generates faster. If multiple classes with a soft cap have requests then the smallest soft cap applies. Each user's first request in the highest tier can still go over it, up to the hard cap. For example, with a hard cap of 64, a `low` class with a `soft_cap` of 32 and a `medium` class with a `soft_cap` of 16: user A sends 32 `low` requests and user B sends one, so 32 of them run. Then user C sends a `medium` request; it runs along with 15 `low` requests, and the other `low` requests wait until it finishes.

`max_parallel_per_user` limits how many requests of a single user in that class can run at once, even if there are free slots. The user's requests over the limit take turns: every `swap_hysteresis_seconds` the proxy suspends the request which ran the longest and resumes the one which waited the longest. Each turn change can force the engine to recompute the resumed request's KV cache.

Neither `soft_cap` nor `max_parallel_per_user` applies while there's only one user.

A suspended or queued request's connection stays open, and the proxy sends keepalive bytes every `keepalive_interval_seconds` so that the client doesn't time out. If the engine hasn't sent a response head by the time the first keepalive is due then the proxy sends its own `200` head, so an engine error for that request arrives with a `200` status.

## Restarting

`kill -USR1 <pid of llm-proxy serve>` restarts the middle end in-place without interrupting any requests:

  1. It validates the configuration files with the binary on disk. If that fails it doesn't restart.
  2. It stops accepting new connections and writes out pending data.
  3. It saves the state of every connection into a memory file and `execve`s the binary on disk, which can be a new version. All of the file descriptors stay open, so streams continue byte for byte.

The saved state doesn't include the QoS configuration, so you can change the classes, the backend or the QoS code while requests are in progress. After the restart the old requests run in a special `handed over` class with their old priority, and the usage history starts from zero. If `execve` fails the old process continues with its old configuration.

## Scheduling log

With `scheduling_log` set the middle end appends a JSON object to that file for every scheduling event. Since `SIGUSR1` reloads the configuration you can turn it on or off without stopping the proxy. It doesn't contain any API keys or prompts.

| Event | Description |
|---|---|
| `start`, `stop` | The middle end started or stopped. `start` includes the caps and the classes. |
| `caps` | The caps in effect changed. |
| `rejected` | The middle end rejected a request. |
| `admitted` | A request entered the queue. |
| `started` | The proxy sent a request to the engine. |
| `suspended`, `resumed` | The proxy suspended or resumed a request. The `reason` for a suspension is `idle_only`, `max_parallel_per_user`, `soft_cap` or `hard_cap`. |
| `engine_running`, `engine_stopped` | The engine started or stopped generating a request. |
| `completed`, `aborted` | The middle end sent the whole response, or the connection closed before that. |
| `finished` | Summary of a request: queue time, generating time, usage and the number of suspensions. |
| `handed_over`, `adopted` | A request continued across a `SIGUSR1` restart. Queued requests get a new ID. |

For example, here's a batch request which the proxy suspended for an interactive request (without `time`):

```
{"event":"admitted","request":"qos-f50e9f773648ca6f-0000000000000000","user":"bob","class":"batch","priority":0,"path":"/v1/chat/completions","model":"mock-model","stream":false,"max_tokens":100,"body_bytes":101}
{"event":"started","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0,"queued_seconds":0.0}
{"event":"engine_running","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0}
{"event":"suspended","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0,"reason":"hard_cap"}
{"event":"engine_stopped","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0,"reason":"suspended"}
{"event":"resumed","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0}
{"event":"engine_running","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0}
{"event":"completed","request":"qos-f50e9f773648ca6f-0000000000000000","status":200}
{"event":"finished","request":"qos-f50e9f773648ca6f-0000000000000000","user":"bob","class":"batch","priority":0,"total_seconds":1.227,"queued_seconds":0.0,"generating_seconds":1.114,"usage_seconds":1.114,"suspensions":1}
{"event":"engine_stopped","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0,"reason":"finished"}
```

To rotate the log either use `logrotate` with `copytruncate`, or move the file and send `SIGUSR1`.

## Control protocol

The middle end and the engine exchange JSON messages, one per line, over a Unix socket:

| Proxy → engine | Engine → proxy |
|---|---|
| `{"seq":1,"op":"suspend","ids":["qos-..."]}` | `{"seq":1,"result":null}` |
| `{"seq":2,"op":"resume","ids":["qos-..."]}` | `{"seq":2,"result":null}` |
| `{"seq":3,"op":"info"}` | `{"seq":3,"result":{"engine":"vllm","model":"...","data_parallel_rank":0,"max_running_requests":256,"priority_scheduling":true,"kv_cache_tokens":1048576}}` |
| `{"seq":4,"op":"stats"}` | `{"seq":4,"result":{"requests":[{"id":"qos-...","state":"running","prompt_tokens":12,"output_tokens":40}]}}` |
| | `{"event":"running","id":"qos-..."}`, `{"event":"stopped","id":"qos-...","reason":"suspended"}` |

The proxy sends its request ID to the engine in `X-Request-Id` (vLLM) or `rid` (sglang).

## Running tests

```sh
cargo test
cargo test --features tls
cargo clippy --all-targets
python -m pytest patches   # the shared engine module; doesn't need a GPU
e2e/run-vllm.sh            # tests with a real engine; need a GPU
e2e/run-sglang.sh
```

The `e2e/run-*.sh` scripts create a venv with the patched engine in `/tmp/llm-proxy-e2e` (from `e2e/<engine>/uv.lock`), download `HuggingFaceTB/SmolLM2-1.7B-Instruct` into the same directory unless `E2E_MODEL` points to a local copy, build the proxy with `cargo build --release` and run pytest. They pass any extra arguments to pytest, e.g. `e2e/run-vllm.sh -k restart`. They need `uv`, GNU `patch` and about 30% of the GPU's memory.

You can also run `python -m pytest e2e` directly from a venv with a patched engine; `E2E_VLLM_PYTHON` and `E2E_SGLANG_PYTHON` select the Python interpreter for each engine.

## License

Licensed under either of

  * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
  * MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
