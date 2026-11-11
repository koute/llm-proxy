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
  2. Write `config/general.json` and `config/accounts.json` (see `config/*.example.json`). For the [gateway](#gateway) (drop its section to run without it), create the first admin key: `llm-proxy create-key --user alice --default-class interactive --role admin --new-keys-file`.
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
| `middle_end_socket` | required | Path to the Unix socket between the frontend and the middle end. A `SIGUSR1` restart refuses to change it, since the frontend wouldn't follow. |
| `paths.generating` | chat, completions, responses, messages, `/generate` | QoS-managed paths. |
| `paths.whitelist` | models, embeddings, tokenize, health, ... | Other paths which the proxy forwards as-is. Every other path gets a 404. |
| `limits.max_request_body_bytes` | 64 MiB | Bigger requests get a 413. |
| `limits.request_body_timeout_seconds` | 30 | Requests whose body doesn't make progress for this long get a 408 (a malformed body gets a 400). |
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
| `gateway.client_ip_header` | `null` | Header with the client's address from the reverse proxy in front, e.g. `x-real-ip`, or `x-forwarded-for`, of which the proxy takes the last address; otherwise the audit log has the frontend's peer address. |
| `gateway.keys_file_poll_seconds` | 10 | How often the middle end checks the keys file for changes. |

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

  * The proxy reads the API key from `Authorization: Bearer <key>`, `x-api-key` or `api-key` (Azure OpenAI SDKs); an empty `Authorization` header counts as none. Requests without an API key use the account with an empty `api_key`, if there is one. Logs show API keys masked, unknown ones too.
  * `api_key_hash` is the SHA-256 of the API key as 64 hex characters. With hashes instead of keys you can make `accounts.json` public. Anyone with the file can try to guess a key offline though, so only use long random keys:

    ```sh
    key="sk-$(openssl rand -hex 32)"
    echo "$key"
    printf '%s' "$key" | sha256sum
    ```

    Don't use `echo "$key" | sha256sum`; `echo` adds a newline, which changes the hash.
  * Clients can pick a class with the `x-priority: <class>` header.
  * With `virtual_user_http_header` every value of that header (cut at 128 characters) is a separate user (`<account>/<value>`), with its own metrics series. This is useful e.g. with LiteLLM in front of the proxy, which uses a single API key but can send the name of the end user in a header; whoever has such an account's key can create users at will, so give it only to a trusted gateway.
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

A suspended or queued request's connection stays open. For a streaming request the proxy sends SSE comments every `keepalive_interval_seconds` so that the client doesn't time out; if the engine hasn't sent a response head by the time the first one is due then the proxy sends its own `200` event stream head, and an engine error (or any response which isn't an event stream) which comes after it arrives as an SSE event which clients raise as an error: `data: {"error": ...}`, or for `/v1/messages` Anthropic's `event: error`. A non-streaming request gets no keepalives, so it always gets the engine's status; like through LiteLLM, the client sees nothing until the response is complete.

## Restarting

`kill -USR1 <pid of llm-proxy serve>` restarts the middle end in-place without interrupting any requests:

  1. It validates the configuration files with the binary on disk. If that fails it doesn't restart.
  2. It stops accepting new connections and writes out pending data.
  3. It saves the state of every connection into a memory file and `execve`s the binary on disk, which can be a new version. All of the file descriptors stay open, so streams continue byte for byte.

The saved state doesn't include the QoS configuration, so you can change the classes, the backend or the QoS code while requests are in progress. After the restart the old requests run in their class again (queued ones in the order they arrived) if `accounts.json` still has a class of the same name and priority, and otherwise in a special `handed over (<priority>)` class with their old priority; the usage history starts from zero. If `execve` fails the old process continues with its old configuration.

## Scheduling log

With `scheduling_log` set the middle end appends a JSON object to that file for every scheduling event. Since `SIGUSR1` reloads the configuration you can turn it on or off without stopping the proxy. It doesn't contain any API keys or prompts. If a write fails, the middle end keeps up to 64 MiB of events in memory, drops the rest and tries again every second; the same goes for the audit log.

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

With a `gateway` section in `general.json` the proxy replaces LiteLLM: API keys with a lifecycle, per-key limits, an admin panel, Prometheus metrics and an audit log. `/admin` and `/metrics` then belong to the proxy, and so do LiteLLM's probes `/health/liveliness` (or `/health/liveness`) and `/health/readiness` (which is a 503 while there's no backend), which need no API key. Clients whose base URL lacks the `/v1` work: a path like `/chat/completions` which isn't configured, but whose `/v1` form is, is sent to the latter. Paths must be canonical (no `//`, `.`, `..` or trailing `/`), and generating paths only take `POST`s. Like LiteLLM, the proxy allows browsers to call it from any origin (CORS): it answers preflight requests itself, without an API key, and adds `Access-Control-Allow-Origin: *` to responses to requests with an `Origin`; it only authenticates with keys in headers, never cookies.

### Keys file

The keys file has the API keys as SHA-256 hashes, in addition to the accounts in `accounts.json`; see `config/keys.example.json`. Every key has an `id`, a `user` (several keys can share one), its `key_hash`, a `role` (`user`, `admin`, which can also use the admin panel and `/metrics`, or `metrics`, which can only read `/metrics`), `default_class` and `allowed_classes` like an account, and optionally `blocked`, `expires_at` (RFC 3339), `allowed_routes` (paths like `/v1/models` or `/prefix/*`; they also apply to `/admin` and `/metrics`) and `limits`: `max_parallel_requests` (queued ones included), `rpm_limit` (requests per minute), `tpm_limit` and `tpd_limit` (prompt and output tokens per minute and per 24 hours). A limit of 0 rejects every request; limits can be at most 2^53 - 1. Over a limit a request gets a 429 with `retry-after` (when all of the key's exceeded limits have room again) and `x-ratelimit-*` headers (`...-tokens-day` for `tpd_limit`). While a request runs, the token limits count an estimate of its tokens, like LiteLLM: a token per four characters of the text of its prompt (`messages`, `prompt` or `input`, like LiteLLM, which doesn't count tool definitions, an assistant message's tool calls (OpenAI's `tool_calls` or `function_call`, Anthropic's `tool_use` blocks; elsewhere, e.g. in a user message, these count) or separate instructions; media where the engine takes media (`image_url`, `input_audio`, `file`, Anthropic `source`s) count 85 like an image in LiteLLM, a token ID 1; a body which isn't JSON a token per four bytes) plus the first non-zero of its `max_completion_tokens`, `max_tokens`, `max_output_tokens` or `sampling_params.max_new_tokens` (as vLLM reads them; LiteLLM prefers `max_tokens`) (without one, a quarter of the key's smallest token limit, up to 1024) for each of its `n` (or `best_of`) sequences and each prompt of a completion (which LiteLLM doesn't multiply, but the engine generates for each); when it finishes, its real token counts replace the estimate. The estimates survive `SIGUSR1` restarts, and the key's `usage` in the admin API and `llm_proxy_key_remaining_tokens` include them. So, like in LiteLLM, a request is only admitted if the tokens of the window, those of the running requests and its own estimate fit in the limit (a request whose estimate alone is over the limit never fits, and its 429 says so); its real tokens can still go over it. A request which the proxy rejects because its queue is full doesn't count. The limits apply to the generating paths and, like LiteLLM's to its LLM calls, to `POST`s to the whitelisted paths (e.g. `/v1/embeddings`, `/tokenize`), but not to reads like `GET /v1/models`. A blocked or expired key gets a 401, like in LiteLLM.

The middle end keeps the keys in memory and checks the file every `keys_file_poll_seconds`, so you can also edit it by hand; it keeps the previous keys if the new file is invalid. The admin panel reads the current file, changes it and replaces it with `rename` (of the target, if the file is a symbolic link), so the file can live on another machine, e.g. on an sshfs mount (which needs the `posix-rename` extension of OpenSSH's SFTP server). The admin panel and `create-key` hold a lock on `.<name>.lock` next to the file while they change it, so they don't lose each other's changes (they wait for it up to 8 seconds; a change which hangs releases it when it gives up; the lock file is never opened through a symbolic link, and a regular one of the process's own is made readable for everyone, all a lock needs, so that e.g. one which `sudo create-key` created works for the middle end too; one which still can't be opened is skipped with a warning, and then the admin panel and `create-key` can lose each other's changes); hand edits don't take the lock, so don't edit the file by hand while someone uses the admin panel. A hung mount doesn't delay requests: a change which takes longer than 10 seconds fails with a 503 and isn't applied, unless the file system was already in the middle of replacing the file. Accesses which hang that way keep a thread each; with 4 of them, further accesses fail at once, except for one a minute, and once one of them works, the file is used as usual again (up to 64 hanging threads in all). If the middle end can't read the file when it starts, or it's invalid, only the accounts in `accounts.json` work until it's fixed; `check-config` and `SIGUSR1` still need a readable, valid file. The limit counts survive `SIGUSR1` restarts.

Create the first admin key with `llm-proxy create-key --user alice --default-class interactive --role admin --new-keys-file`, which takes the same `--general` and `--accounts` options as `serve`. It prints the key and adds its hash to the keys file; `--new-keys-file` lets it create the file, without it a missing keys file (e.g. on a mount which is down) is an error. Neither the CLI nor the admin panel replaces a symbolic link whose target is missing.

### Admin panel

`/admin` is a single page to create, edit, block, rotate and delete keys and to see their usage. New keys are only shown once. At least one admin key must stay usable for good: unblocked, without an expiry and with access to all of `/admin/api/*`; a change which would leave none, where there was one, gets a 409. `allowed_routes` only restricts paths, not methods, and an admin can always change keys, including its own: a route-restricted admin key isn't a lesser admin. Its JSON API under `/admin/api/` needs an admin key: `GET session`, `GET keys`, `POST keys`, `PATCH keys/<id>` (`limits` are merged), `POST keys/<id>/rotate` and `DELETE keys/<id>`.

### Metrics

`GET /metrics` needs an admin or metrics key and has:

  * `llm_proxy_requests_total` by `user`, `key`, `class` (empty for the whitelisted paths), `path` and `status` (an HTTP status; `aborted` if the client left, `failed` if the engine broke off the response, `error` without any response and `unknown` if the proxy couldn't tell), and `llm_proxy_tokens_total` and `llm_proxy_suspensions_total` of finished requests.
  * `llm_proxy_rejected_requests_total` by `status` and `reason`, e.g. `unknown_api_key` or `rpm_limit`.
  * Histograms of the total time, the time in the queue and the time to the first data from the engine by `class`.
  * Gauges: requests by `class` (every configured class, also without requests) and `state` (`queued`, `waiting_in_engine`, `generating`, `suspended` or `abandoned`), active users, the cap per rank, keys by state, and per key the requests in flight and what's left of its limits.
  * `llm_proxy_audit_log_failing`, `llm_proxy_audit_log_dropped_records_total` and `llm_proxy_scheduling_log_failing`: logs which can't be written keep their events and retry; a `SIGUSR1` restart hands their events to the new process.
  * `llm_proxy_keys_file_reloads_total` and `llm_proxy_keys_file_errors_total`: failures to read, lock or write the keys file, or an invalid file, e.g. when the mount of the keys file is gone; not mistakes in admin API requests.

### Audit log

With `gateway.audit_log` set the middle end appends one JSON object per event: `key_created`, `key_updated`, `key_rotated` and `key_deleted` with the `actor`; `keys_reloaded` with the IDs which changed on disk (also when the admin panel replaces a file with hand edits it hadn't loaded yet); `rejected` for every 401, 403 and 429 with the `reason`, the user or key if known, the client and the user agent; and `request` for every finished request with the user, key, class, path, model, status, client, times and engine token counts. Requests to the whitelisted paths have no `request` ID, class or token counts, but a `method`. It never contains prompts, responses, API keys or their hashes. Requests which were handed over by a `SIGUSR1` restart keep their class and their times from their arrival; a request whose client went away has the `outcome` `aborted`, one which the engine broke off after the response had started `failed`. The `status` is the engine's, or, if the engine hadn't answered, the one the client had got (`200` for a stream which had its early head). If the file can't be written (or a write takes longer than 5 seconds), the middle end keeps the events (up to 64 MiB) and retries every second, writing each event exactly once (a line it wrote in part, e.g. on a full disk, is removed and written again, also if the file was truncated meanwhile; it writes whole events, at most 4 KiB at a time, and when a restart gives up on a write which hangs, the new process recognizes events of that write which went through after all and doesn't write them again; only a FIFO as the log can't be read back, so there one such write per restart may be duplicated, its first copy possibly cut off (a broken line); a write which makes progress, however slowly, is waited for up to `restart_flush_timeout_seconds`); `llm_proxy_audit_log_failing` shows it, and a `SIGUSR1` restart hands the events to the new process. For an aborted request `generating_seconds` can be longer than `total_seconds`: the engine stops a moment after the client left. To rotate the audit log, rename it and send `SIGUSR1`, or copy and truncate it. With `n` > 1, vLLM reports the output tokens of one of the `n` sequences, so the token counts only count those; the token limits reserve `max_tokens` for each of the `n` while the request runs.

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
