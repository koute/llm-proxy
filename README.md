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
  * Optional gateway: API keys in a JSON file with an admin panel, per-key rate limits, Prometheus metrics and an audit log. See [Gateway](#gateway).

It requires a patched engine; this repository includes patches for vLLM 0.30.0 and sglang 0.5.20.

## How it works

```
client ─HTTP(S)─> llm-proxy frontend ─Unix socket─> llm-proxy serve ─HTTP─> vLLM / sglang
                  (TLS, HTTP parsing)               (middle end: QoS)  └─Unix socket per rank─┘
```

  * `llm-proxy frontend` accepts HTTP(S) connections and forwards every request to the middle end. It doesn't contain any logic, so you should rarely need to restart it. (Restarting it closes all of its connections.)
  * `llm-proxy serve` (the middle end) does everything else: API keys, accounts, QoS, the gateway, and the connections to the engine.

The frontend passes the client's IP address to the middle end in `x-llm-proxy-peer`, which the middle end removes before it forwards the request.

Apart from the sglang scheduling change described in [Patching the engine](#patching-the-engine), the patched engine works as usual when the proxy isn't running. When the middle end connects to the engine's control socket the engine suspends and resumes requests on command and reports which requests are generating. When the middle end disconnects the engine resumes all suspended requests.

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

The engine doesn't free a suspended request's KV cache until it needs the memory or the slot for other requests. The exception is sglang with speculative decoding: there the patch moves a suspended request's KV cache into the prefix cache right away, so on resume the engine only recomputes the part which it evicted in the meantime, usually a single token.

The sglang patch also changes how sglang schedules prefills while it works through a long prompt in chunks; this applies whenever `SGLANG_QOS_CONTROL_SOCKET` is set, even without a connected proxy. Unpatched sglang gives the whole `--chunked-prefill-size` budget of every step to the long prompt, so every other request waits for its first token until the long prompt is done. With the patch, a waiting request whose uncached part fits into half of the budget gets prefilled in the same step, and the long prompt gets the rest. Requests with a longer uncached part still wait, and sglang works through long prompts one at a time.

Both patches support tensor parallelism and data parallelism (including MoE waves and DP attention), although there are no tests of data parallelism on real hardware yet. There are also no tests for pipeline parallelism and prefill/decode disaggregation, and the speculative decoding tests only cover sglang with a separate draft model (`STANDALONE`); the patched sglang rejects pipeline parallelism and prefill/decode disaggregation at startup.

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

While there are requests the middle end prints a status line every second, and every 10 seconds one line for every user with requests:

```
status: 6 requests from 4 users: 4 running (4 generating in the engine), 1 suspended, 1 queued, 0 abandoned
status:   high (priority 1): 1 request from 1 user: 1 running (1 generating)
status:   medium (priority 0): 4 requests from 3 users: 3 running (3 generating), 1 suspended
status:   idle (priority -10): 1 request from 1 user: 1 queued
status:   user alice: 2 running (2 generating), 1 suspended; usage 1520.4 s; engine tokens: 1.9M prompt, 1.2M cache hit, 712K prefill in 4 prefills, 40K output
status:   user bob: 1 running (1 generating); usage 85.0 s; engine tokens: 3120 prompt, 0 cache hit, 3120 prefill in 1 prefill, 512 output
status:   user carol: 1 running (1 generating); usage 12.3 s; engine tokens: 950 prompt, 0 cache hit, 950 prefill in 1 prefill, 87 output
status:   user nightly: 1 queued; usage 0.0 s
```

The user lines are sorted by the number of running requests, then by usage. `usage` is the user's usage as described in [Scheduling](#scheduling). The engine tokens are the totals over the user's requests which the engine reported on:

  * `prompt`: prompt tokens.
  * `cache hit`: tokens which the engine read from its prefix cache instead of computing them.
  * `prefill`: tokens which the engine computed during prefills, including recomputation after it lost a request's KV cache.
  * `prefills`: how many times the engine prefilled the requests. More prefills than requests means that the engine recomputed some requests.
  * `output`: generated tokens.

A large `prefill` count shows who sends long prompts which aren't in the prefix cache. sglang doesn't run any decode steps while it can run a prefill, unless you set `--prefill-decode-interval`, so these prompts pause generation for every other request.

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
| `gateway` | `null` | Turns on the [gateway](#gateway). |
| `gateway.keys_file` | required | Path to the keys file. |
| `gateway.audit_log` | `null` | Path to the audit log. |
| `gateway.client_ip_header` | `null` | Header with the client's IP address, e.g. `x-real-ip` or `x-forwarded-for` (first address) from the reverse proxy in front. Without it the audit log has the frontend's peer address. |
| `gateway.keys_file_poll_seconds` | 10 | How often the middle end checks the keys file for changes. |
| `gateway.keys_file_timeout_seconds` | 10 | How long a read or write of the keys file can take before the middle end gives up on it. |

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
    { "name": "alice", "api_key_hash": "589ed636b573aef03d62af555f51c901226ba0bcfe957c8a5f18d1f22f0bcafe", "default_class": "interactive", "allowed_classes": ["batch", "idle"] },
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

Account keys are `name` (can't contain `/`), either `api_key` or `api_key_hash`, `default_class`, `allowed_classes` (classes a client can pick in addition to the default one) and the optional `virtual_user_http_header`.

  * The proxy reads the API key from `Authorization: Bearer <key>` or `x-api-key`. Requests without an API key use the account with an empty `api_key`, if there is one.
  * `api_key_hash` is the SHA-256 of the API key as 64 hex characters. With hashes instead of keys you can make `accounts.json` public. Anyone with the file can try to guess a key offline though, so only use long random keys:

    ```sh
    key="sk-$(openssl rand -hex 32)"
    echo "$key"
    printf '%s' "$key" | sha256sum
    ```

    Don't use `echo "$key" | sha256sum`; `echo` adds a newline, which changes the hash.
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
  3. Requests which the engine started or resumed generating less than `swap_hysteresis_seconds` ago, and requests which the engine hasn't started yet after the proxy started or resumed them.
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
| `engine_running`, `engine_stopped` | The engine started or stopped generating a request, with the request's `tokens` so far. |
| `completed`, `aborted` | The middle end sent the whole response, or the connection closed before that. |
| `finished` | Summary of a request: queue time, generating time, usage, the number of suspensions and the engine's final `tokens`. After a `completed` the middle end waits up to 5 seconds for the engine's last event to get the final token counts. |
| `handed_over`, `adopted` | A request continued across a `SIGUSR1` restart. Queued requests get a new ID. |

For example, here's a batch request which the proxy suspended for an interactive request (without `time`):

```
{"event":"admitted","request":"qos-f50e9f773648ca6f-0000000000000000","user":"bob","class":"batch","priority":0,"path":"/v1/chat/completions","model":"mock-model","stream":false,"max_tokens":100,"body_bytes":101}
{"event":"started","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0,"queued_seconds":0.0}
{"event":"engine_running","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0,"tokens":{"prompt":7,"output":0,"cache_hit":0,"prefill":7,"prefill_count":1}}
{"event":"suspended","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0,"reason":"hard_cap"}
{"event":"engine_stopped","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0,"reason":"suspended","tokens":{"prompt":7,"output":41,"cache_hit":0,"prefill":7,"prefill_count":1}}
{"event":"resumed","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0}
{"event":"engine_running","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0,"tokens":{"prompt":7,"output":41,"cache_hit":0,"prefill":7,"prefill_count":1}}
{"event":"completed","request":"qos-f50e9f773648ca6f-0000000000000000","status":200}
{"event":"engine_stopped","request":"qos-f50e9f773648ca6f-0000000000000000","rank":0,"reason":"finished","tokens":{"prompt":7,"output":100,"cache_hit":0,"prefill":7,"prefill_count":1}}
{"event":"finished","request":"qos-f50e9f773648ca6f-0000000000000000","user":"bob","class":"batch","priority":0,"total_seconds":1.227,"queued_seconds":0.0,"generating_seconds":1.114,"usage_seconds":1.114,"suspensions":1,"tokens":{"prompt":7,"output":100,"cache_hit":0,"prefill":7,"prefill_count":1}}
```

The `tokens` fields are the same as in the status lines: `prompt`, `output`, `cache_hit` and `prefill` tokens, and the `prefill_count`. They are `null` if the engine doesn't report token counts. For example, to see who sent the most prefill work in an hour:

```sh
jq -r 'select(.event == "finished" and .tokens != null and .time >= "2026-10-30T13:00" and .time < "2026-10-30T14:00")
       | [.user, .tokens.prefill] | @tsv' scheduling.log \
  | awk -F'\t' '{prefill[$1] += $2} END {for (user in prefill) printf "%d\t%s\n", prefill[user], user}' | sort -rn
```

To rotate the log either use `logrotate` with `copytruncate`, or move the file and send `SIGUSR1`.

## Gateway

With a `gateway` section in `general.json` the proxy also does what LiteLLM did for us: API keys with a lifecycle, per-key limits, an admin panel, Prometheus metrics and an audit log. Without it nothing changes. With it, `/admin`, `/admin/*` and `/metrics` belong to the proxy, even if `paths` lists them.

### Keys file

The keys file holds the API keys, as SHA-256 hashes, next to the static accounts of `accounts.json`; see `config/keys.example.json`. The classes stay in `accounts.json`.

| Key | Default | Description |
|---|---|---|
| `id` | required | Unique; letters, digits, `-`, `_` and `.`. |
| `user` | required | The user for scheduling and usage; several keys can share a user. |
| `key_hash` | required | SHA-256 of the key as 64 hex characters. |
| `key_prefix` | `""` | The first characters of the key, so that people can tell their keys apart. |
| `description` | `""` | |
| `role` | `"user"` | `"user"` (inference), `"admin"` (inference, admin panel and metrics) or `"metrics"` (only `/metrics`). |
| `default_class`, `allowed_classes` | required, `[]` | As in `accounts.json`. |
| `blocked` | `false` | Blocked keys get a 403. |
| `expires_at` | `null` | RFC 3339 time; afterwards the key gets a 401. |
| `limits.max_parallel_requests` | `null` | Requests of the key in the proxy at once, queued ones included. |
| `limits.rpm_limit` | `null` | Generating requests per minute. |
| `limits.tpm_limit`, `limits.tpd_limit` | `null` | Prompt and output tokens per minute and per 24 hours. |
| `allowed_routes` | `null` | Paths the key can use, with `/prefix/*` wildcards; `null` allows all. |
| `created_at`, `created_by` | `null` | Set by the admin panel and `create-key`. |

The middle end keeps the keys in memory, so a slow or unreachable keys file never delays a request. It checks the file for changes every `keys_file_poll_seconds` and loads them, so you can also edit it by hand; if the new file is invalid it keeps the previous keys and logs the error. Changes from the admin panel read the current file, change it and replace it atomically (a temporary file and `rename`), so they don't lose concurrent edits of other keys. The file can therefore live on another machine, e.g. mounted with sshfs (which needs the `posix-rename` extension of OpenSSH's SFTP server for the atomic replacement). The middle end needs the keys file to start, and `SIGUSR1` reads it again like the other configuration files.

The limits are counted when a request arrives (requests) and when it finishes (tokens, from the engine's token counts), so a single long request can go over a token limit. A request over a limit gets a 429 with `retry-after` and, for the rate limits, `x-ratelimit-limit-*`, `x-ratelimit-remaining-*` and `x-ratelimit-reset-*` headers. The counts survive `SIGUSR1` restarts, but not a full restart of the middle end.

To create the first admin key, or any key from a script:

```sh
llm-proxy create-key --user alice --default-class interactive --allowed-class idle --role admin
```

It prints the key and adds its hash to the keys file, creating the file if necessary; the running middle end picks it up by itself. It takes the same `--general` and `--accounts` options as `serve`.

### Admin panel

`/admin` serves the admin panel, a single page without external dependencies. Sign in with an admin key to create, edit, block, rotate and delete keys and to see their usage, the users with requests and the recent audit events. A new or rotated key is only shown once; the keys file only has its hash. The panel can't delete or demote the last active admin key.

The panel uses a JSON API under `/admin/api/`, which needs an admin key in `Authorization: Bearer <key>` or `x-api-key`:

| Request | Description |
|---|---|
| `GET session` | The user, key ID and role of the key you use. |
| `GET overview` | Classes, backend, scheduler status and the last 200 audit events of this process. |
| `GET keys` | All keys without their hashes, with their `state` and `usage`. |
| `POST keys` | Create a key from a JSON object with the fields of the keys file except `id`, `key_hash`, `key_prefix` and `created_*`; the answer has the new `api_key`. |
| `PATCH keys/<id>` | Change the fields in the JSON object; `limits` are merged. |
| `POST keys/<id>/rotate` | Replace the key and answer with the new `api_key`; the settings stay. |
| `DELETE keys/<id>` | Delete the key. |

### Metrics

`GET /metrics` serves Prometheus metrics to admin and metrics keys:

| Metric | Labels | Description |
|---|---|---|
| `llm_proxy_requests_total` | `user`, `key`, `class`, `path`, `status` | Finished requests; `status` is the HTTP status, `aborted` (the client disconnected) or `error` (the proxy answered with a 502). `path` is the matching pattern of `paths`. |
| `llm_proxy_rejected_requests_total` | `status`, `reason` | Requests which the proxy rejected, e.g. `unknown_api_key`, `rpm_limit`, `queue_full`. |
| `llm_proxy_tokens_total` | `user`, `key`, `class`, `type` | Engine token counts (`prompt`, `output`, `cache_hit`, `prefill`). |
| `llm_proxy_suspensions_total` | `user`, `key`, `class` | Suspensions of finished requests. |
| `llm_proxy_generating_seconds_total` | `user`, `class` | Time finished requests spent generating. |
| `llm_proxy_request_duration_seconds`, `llm_proxy_queue_duration_seconds`, `llm_proxy_time_to_first_token_seconds` | `class` | Histograms of the total time, the time in the queue and the time to the first response data from the engine. |
| `llm_proxy_requests` | `class`, `state` | Requests in the proxy right now (`running`, `generating`, `suspended`, `queued`, `abandoned`). |
| `llm_proxy_active_users`, `llm_proxy_rank_parallel_request_cap` | `rank` | Users with requests and the hard cap per rank. |
| `llm_proxy_keys` | `state` | Keys by state (`active`, `blocked`, `expired`). |
| `llm_proxy_key_in_flight_requests`, `llm_proxy_key_requests_last_minute`, `llm_proxy_key_tokens_last_minute`, `llm_proxy_key_tokens_last_day`, `llm_proxy_key_last_used_timestamp_seconds` | `key`, `user` | Usage per key. |
| `llm_proxy_key_remaining_requests`, `llm_proxy_key_remaining_tokens` | `key`, `user`, `window` | What a key with limits can still use. |
| `llm_proxy_keys_file_reloads_total`, `llm_proxy_keys_file_errors_total` | | Loaded changes and failures of the keys file, e.g. when its mount is gone. |

The counters start from zero when the middle end restarts. The engine's own metrics stay on the engine.

### Audit log

With `gateway.audit_log` set the middle end appends a JSON object to that file for every event below. It never contains prompts, responses, API keys or their hashes; of a rejected unknown key it only has the first 3 and the last 4 characters (the log of the middle end has the whole key).

| Event | Description |
|---|---|
| `key_created`, `key_updated`, `key_rotated`, `key_deleted` | A change of a key: the `actor` (`source` `admin_api` or `cli`, the admin's `user`, `key_id` and `client`) and the key, or for updates the key `before` and `after`. |
| `keys_reloaded` | The keys file changed on disk: the IDs `added`, `removed` and `changed`. |
| `rejected` | A request rejected with 401, 403 or 429: `status`, `reason`, `message`, `method`, `path`, `user`, `key_id`, the masked `api_key` if it's unknown, `client` and `user_agent`. |
| `request` | One per finished request: `request` ID, `key_id`, `user`, `class`, `method`, `path`, `model`, `stream`, `status`, `outcome` (`completed`, `aborted`, or `unknown` if the middle end restarted or stopped before the end of the response), `client`, `total_seconds`, `queued_seconds`, `time_to_first_token_seconds`, `generating_seconds`, `suspensions` and the engine's `tokens`. |

For example, the tokens per user in a day:

```sh
jq -r 'select(.event == "request" and .tokens != null and .time >= "2026-11-06" and .time < "2026-11-07")
       | [.user, .tokens.prompt + .tokens.output] | @tsv' audit.jsonl \
  | awk -F'\t' '{tokens[$1] += $2} END {for (user in tokens) printf "%d\t%s\n", tokens[user], user}' | sort -rn
```

Rotate the audit log like the scheduling log.

## Control protocol

The middle end and the engine exchange JSON messages, one per line, over a Unix socket:

| Proxy → engine | Engine → proxy |
|---|---|
| `{"seq":1,"op":"suspend","ids":["qos-..."]}` | `{"seq":1,"result":null}` |
| `{"seq":2,"op":"resume","ids":["qos-..."]}` | `{"seq":2,"result":null}` |
| `{"seq":3,"op":"info"}` | `{"seq":3,"result":{"engine":"vllm","model":"...","data_parallel_rank":0,"max_running_requests":256,"priority_scheduling":true,"kv_cache_tokens":1048576}}` |
| `{"seq":4,"op":"stats"}` | `{"seq":4,"result":{"requests":[{"id":"qos-...","state":"running","prompt_tokens":12,"output_tokens":40}]}}` |
| | `{"event":"running","id":"qos-...","tokens":{...}}`, `{"event":"stopped","id":"qos-...","reason":"suspended","tokens":{...}}` |

The engine sends a `running` event when a request starts or continues generating, including at the start of a prefill, and a `stopped` event with the `reason` `suspended`, `preempted` or `finished`. Both events carry the request's token counts so far: `{"prompt":...,"output":...,"cache_hit":...,"prefill":...,"prefill_count":...}`. The proxy also accepts events without `tokens` from older patches.

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

The tests also run on macOS. `e2e/sglang-mlx` is a uv project for sglang 0.5.20 on Apple Silicon with its MLX backend: `setup.sh` checks out the sglang source (PyPI only has Linux builds), installs it into `e2e/sglang-mlx/.venv` and applies the patch, and `serve.sh` serves `HuggingFaceTB/SmolLM2-135M-Instruct` on port 8000 with the control socket at `/tmp/llm-proxy-qos.sock`. It's good enough to try the proxy and the gateway locally, but sglang's MLX backend ignores suspensions, and its normal event loop (`--disable-overlap-schedule`, which the patch needs for suspensions) crashes as soon as two requests run at once.

The `e2e/run-*.sh` scripts create a venv with the patched engine in `/tmp/llm-proxy-e2e` (from `e2e/<engine>/uv.lock`), download `HuggingFaceTB/SmolLM2-1.7B-Instruct` into the same directory unless `E2E_MODEL` points to a local copy (and `HuggingFaceTB/SmolLM2-135M-Instruct`, the draft model for the speculative decoding tests, unless `E2E_DRAFT_MODEL` points to one), build the proxy with `cargo build --release` and run pytest. They pass any extra arguments to pytest, e.g. `e2e/run-vllm.sh -k restart`. They need `uv`, GNU `patch` and about 30% of the GPU's memory.

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
