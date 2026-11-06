#!/usr/bin/env python3
"""A/B replica of the LiteLLM deployment (paritytech/lite-llm-proxy) against
llm-proxy's gateway, both in front of the same mock SmolLM2-135M-Instruct.

  A: client -> nginx -> LiteLLM (+Postgres) -> "SSH tunnel" -> llm-proxy A (QoS only) -> mock A
  B: client -> nginx -> llm-proxy B (gateway: keys, limits, admin, metrics, audit) -> mock B

llm-proxy A runs the production general.json/accounts.json from paritytech/vllm-parity
(the `anonymous` account keyed by LiteLLM's x-litellm-user_api_key_hash header).
llm-proxy B runs the same files plus a `gateway` section.

CLI (stdlib only, run from anywhere):
  python3 ab.py nginx       write nginx.conf (gitignored) before `docker compose up -d`
  python3 ab.py up N        start slot N (0-6): mocks, proxies, keys on both sides
  python3 ab.py down N      stop slot N's processes
  python3 ab.py restart N   down + up (e.g. after `cargo build --release`)
  python3 ab.py status

Library:
  import sys; sys.path.insert(0, "<repo>/e2e/litellm-ab"); import ab
  s = ab.Slot(1)
  ra, rb = s.pair("carol", "/v1/chat/completions", {"messages": [...], "max_tokens": 5})
  print(ab.diff(ra, rb))
"""

import concurrent.futures
import http.client
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUN = Path(os.environ.get("AB_RUN_DIR", HERE / "run"))
PROXY = REPO / "target/release/llm-proxy"
MOCK = REPO / "target/release/examples/mock_smollm"
SERVED_MODEL = "SmolLM2-135M-Instruct"
LITELLM_ADMIN = "http://127.0.0.1:21000"

# Production llm-proxy configuration from paritytech/vllm-parity (llm-proxy/general.json
# and llm-proxy/accounts.json); only addresses, paths and the secret differ.
PROD_GENERAL = {
    "frontend": {"listen": None, "tls": None},
    "scheduling_log": None,
    "middle_end_socket": None,
    "paths": {
        "generating": ["/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/generate"],
        "whitelist": [
            "/v1/messages/count_tokens", "/v1/models", "/v1/models/*", "/v1/embeddings",
            "/tokenize", "/detokenize", "/health", "/version",
        ],
    },
    "limits": {
        "max_request_body_bytes": 67108864,
        "request_body_timeout_seconds": 30,
        "max_queued_requests": 1024,
        "keepalive_interval_seconds": 15,
    },
    "quality_of_service": {
        "minimum_peak_throughput": 1.0,
        "peak_throughput_tolerance": 0.05,
        "maximum_parallel_requests": None,
        "swap_hysteresis_seconds": 30,
        "usage_half_life_minutes": 60,
    },
    "process": {"restart_flush_timeout_seconds": 5, "shutdown_drain_timeout_seconds": 30},
}
PROD_ACCOUNTS = {
    "classes": [
        {"name": "high", "priority": 1, "soft_cap": 24, "cost": 1.2},
        {"name": "medium", "priority": 0, "soft_cap": 16, "max_parallel_per_user": 8},
        {"name": "low", "priority": -1, "cost": 0.8},
        {"name": "idle", "priority": -10, "idle_only": True, "cost": 0.1},
    ],
    "accounts": [
        {
            "name": "anonymous",
            "api_key": None,  # VLLM_API_KEY from .env, as LiteLLM's vllm-pod credential
            "default_class": "medium",
            "allowed_classes": ["medium", "low", "idle"],
            "virtual_user_http_header": "x-litellm-user_api_key_hash",
        }
    ],
}

# The users on both sides. `b` is the gateway key record (POST /admin/api/keys),
# `a` the LiteLLM /key/generate body. None means the user doesn't exist on that side.
DEFAULT_CLASSES = {"default_class": "medium", "allowed_classes": ["medium", "low", "idle"]}
USERS = {
    # Production-like key: RUNBOOK's example mints rpm_limit 60 with a budget.
    "alice": {
        "b": {"user": "alice", "description": "laptop", "limits": {"rpm_limit": 60}},
        "a": {"user_id": "alice@parity.io", "rpm_limit": 60, "max_budget": 20, "budget_duration": "30d"},
    },
    # A second key of the same user.
    "alice-ci": {
        "b": {"user": "alice", "description": "ci", "limits": {"rpm_limit": 60}},
        "a": {"user_id": "alice@parity.io", "rpm_limit": 60, "max_budget": 20, "budget_duration": "30d"},
    },
    # Tight limits on everything both sides support.
    "bob": {
        "b": {"user": "bob", "limits": {"rpm_limit": 10, "tpm_limit": 2000, "max_parallel_requests": 2}},
        "a": {"user_id": "bob@parity.io", "rpm_limit": 10, "tpm_limit": 2000, "max_parallel_requests": 2},
    },
    # No limits.
    "carol": {"b": {"user": "carol"}, "a": {"user_id": "carol@parity.io"}},
    "carol2": {"b": {"user": "carol2"}, "a": {"user_id": "carol2@parity.io"}},
    "carol3": {"b": {"user": "carol3"}, "a": {"user_id": "carol3@parity.io"}},
    # Blocked after creation.
    "dave": {"b": {"user": "dave", "blocked": True}, "a": {"user_id": "dave@parity.io"}, "block": True},
    # Expired (LiteLLM: 1s duration; gateway: expires_at in the past).
    "erin": {
        "b": {"user": "erin", "expires_at": "2026-01-01T00:00:00Z"},
        "a": {"user_id": "erin@parity.io", "duration": "1s"},
    },
    # Scoped: LiteLLM by model list, the gateway by routes (it has a single model).
    "frank": {
        "b": {"user": "frank", "allowed_routes": ["/v1/chat/completions", "/v1/models"]},
        "a": {"user_id": "frank@parity.io", "models": ["{parity}"]},
    },
    # Gateway-only features.
    "grace": {"b": {"user": "grace", "limits": {"tpd_limit": 3000}}, "a": None},
    "metrics": {"b": {"user": "prometheus", "role": "metrics"}, "a": None},
    "admin2": {"b": {"user": "admin2", "role": "admin"}, "a": None},
}


def ports(n):
    base = 21000 + n * 100
    return {
        "mock_a": base + 1, "mock_b": base + 2, "proxy_a": base + 3, "proxy_b": base + 4,
        "front_a": base + 10, "front_b": base + 20, "tunnel": 18000 + n,
    }


# compose.yml's nginx: per slot, production's site.conf.j2 `location /` (minus TLS)
# in front of LiteLLM (path A) and llm-proxy B on the host (path B); then the tunnels.
NGINX_HEAD = """\
# Generated for the llm-proxy A/B replica; see compose.yml. The `location /`
# blocks are production's site.conf.j2 verbatim (minus TLS).
worker_processes auto;
events { worker_connections 8192; }
http {
    access_log /dev/stdout;
    error_log /dev/stderr warn;
    resolver 127.0.0.11 valid=10s ipv6=off;
"""
NGINX_SERVER = """\
    server {
        listen %d;
        client_max_body_size 100m;
        location / {
            proxy_pass http://%s;
            proxy_http_version 1.1;
            proxy_set_header Host              $host;
            proxy_set_header X-Real-IP         $remote_addr;
            proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection        "";
            proxy_buffering off;
            proxy_request_buffering off;
            proxy_cache off;
            proxy_read_timeout 900s;
            proxy_send_timeout 900s;
        }
    }
"""


def nginx_conf(slots=range(7)):
    out = [NGINX_HEAD]
    for n in slots:
        p = ports(n)
        out.append(f"    # slot {n}: A = nginx -> LiteLLM, B = nginx -> llm-proxy B on the host\n")
        out.append(NGINX_SERVER % (p["front_a"], "litellm:4000"))
        out.append(NGINX_SERVER % (p["front_b"], f"macos-host:{p['proxy_b']}"))
    out.append("}\nstream {\n    # The reverse SSH tunnels: host.docker.internal:18000+N -> llm-proxy A of slot N.\n")
    for n in slots:
        out.append(f"    server {{ listen {ports(n)['tunnel']}; proxy_pass macos-host:{ports(n)['proxy_a']}; }}\n")
    out.append("    # LiteLLM without nginx, for its admin API.\n    server { listen 4000; proxy_pass litellm:4000; }\n}\n")
    return "".join(out)


def env_file():
    values = {}
    for line in (HERE / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


# --------------------------------------------------------------------------- HTTP


class Resp:
    def __init__(self, status, headers, body, events, elapsed, ttfb, error=None):
        self.status = status
        self.headers = headers  # list of (name lowercased, value)
        self.body = body
        self.events = events  # parsed SSE `data:` payloads (str) for streams
        self.elapsed = elapsed
        self.ttfb = ttfb
        self.error = error  # transport error, e.g. connection reset mid-stream

    def header(self, name):
        name = name.lower()
        return [v for k, v in self.headers if k == name]

    @property
    def json(self):
        try:
            return json.loads(self.body)
        except Exception:
            return None

    def __repr__(self):
        body = self.body[:300]
        return f"<Resp {self.status} err={self.error!r} {len(self.body)}B events={len(self.events)} {body!r}>"


def request(url, method="GET", path="/", body=None, headers=None, stream=None, timeout=60.0):
    """One HTTP/1.1 request. `body` may be bytes, str or a JSON-serializable object.
    Streams (text/event-stream) are read to the end and split into SSE data payloads."""
    parsed = urllib.parse.urlparse(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    if body is not None and not isinstance(body, (bytes, str)):
        body = json.dumps(body)
    if isinstance(body, str):
        body = body.encode()
    hdrs = {"content-type": "application/json"} if body is not None else {}
    hdrs.update(headers or {})
    start = time.monotonic()
    ttfb = None
    try:
        connection.request(method, path, body=body, headers=hdrs)
        response = connection.getresponse()
        ttfb = time.monotonic() - start
        chunks, error = [], None
        try:
            while True:
                chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        except Exception as exc:  # incomplete read, reset, ...
            error = f"{type(exc).__name__}: {exc}"
        data = b"".join(chunks)
        events = []
        content_type = (response.getheader("content-type") or "").lower()
        if "text/event-stream" in content_type:
            for line in data.replace(b"\r\n", b"\n").split(b"\n"):
                if line.startswith(b"data:"):
                    events.append(line[5:].strip().decode(errors="replace"))
        headers_out = [(k.lower(), v) for k, v in response.getheaders()]
        return Resp(response.status, headers_out, data, events, time.monotonic() - start, ttfb, error)
    except Exception as exc:
        return Resp(None, [], b"", [], time.monotonic() - start, ttfb, f"{type(exc).__name__}: {exc}")
    finally:
        connection.close()


def wait_http(url, path="/health", timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if request(url, "GET", path, timeout=2).status is not None:
            return True
        time.sleep(0.2)
    raise RuntimeError(f"{url}{path} didn't come up")


# --------------------------------------------------------------------------- comparison


def stream_text(resp):
    """Concatenated content of an OpenAI stream, finish_reason, usage, [DONE] seen."""
    content, finish, usage, done, bad = [], None, None, False, []
    for event in resp.events:
        if event == "[DONE]":
            done = True
            continue
        try:
            value = json.loads(event)
        except Exception:
            bad.append(event)
            continue
        if isinstance(value, dict) and value.get("usage"):
            usage = {k: value["usage"].get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        for choice in (value.get("choices") or []) if isinstance(value, dict) else []:
            delta = choice.get("delta") or {}
            piece = delta.get("content") if "delta" in choice else choice.get("text")
            if piece:
                content.append(piece)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {"content": "".join(content), "finish_reason": finish, "usage": usage, "done": done, "unparsable": bad}


def essence(resp):
    """What a client relies on, without ids and timestamps."""
    out = {"status": resp.status, "transport_error": resp.error}
    if resp.events:
        out["stream"] = stream_text(resp)
        return out
    value = resp.json
    if isinstance(value, dict) and "choices" in value:
        out["choices"] = [
            {
                "content": (c.get("message") or {}).get("content") if "message" in c else c.get("text"),
                "finish_reason": c.get("finish_reason"),
            }
            for c in value["choices"]
        ]
        usage = value.get("usage") or {}
        out["usage"] = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    elif isinstance(value, dict) and "error" in value:
        out["error"] = value["error"] if isinstance(value["error"], dict) else {"message": value["error"]}
    else:
        out["body"] = resp.body[:500]
    return out


def diff(ra, rb, ignore=()):
    """Differences between the essences of two responses, as a list of strings."""
    ea, eb = essence(ra), essence(rb)
    problems = []
    for key in sorted(set(ea) | set(eb)):
        if key in ignore:
            continue
        if key == "usage" and isinstance(ea.get(key), dict) and isinstance(eb.get(key), dict):
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if ea[key].get(field) != eb[key].get(field):
                    problems.append(f"usage.{field}: A={ea[key].get(field)!r} B={eb[key].get(field)!r}")
        elif key == "error":
            pass  # error shapes differ by design; compare status instead
        elif ea.get(key) != eb.get(key):
            problems.append(f"{key}: A={ea.get(key)!r} B={eb.get(key)!r}")
    return problems


# --------------------------------------------------------------------------- slots


class Slot:
    def __init__(self, n):
        self.n = n
        self.dir = RUN / f"slot{n}"
        self.ports = ports(n)
        self.a_url = f"http://127.0.0.1:{self.ports['front_a']}"
        self.b_url = f"http://127.0.0.1:{self.ports['front_b']}"
        self.proxy_a_url = f"http://127.0.0.1:{self.ports['proxy_a']}"
        self.proxy_b_url = f"http://127.0.0.1:{self.ports['proxy_b']}"
        self.mock_a_url = f"http://127.0.0.1:{self.ports['mock_a']}"
        self.mock_b_url = f"http://127.0.0.1:{self.ports['mock_b']}"
        self.model_a = f"parity/smollm-s{n}"
        self.model_a_auto = f"auto/smollm-s{n}"
        # Mock B serves LiteLLM's public names too, like vLLM with
        # `--served-model-name SmolLM2-135M-Instruct parity/... auto/...`: clients keep
        # their model names when they move from A to B.
        self.model_b = self.model_a
        state = self.dir / "slot.json"
        self.state = json.loads(state.read_text()) if state.exists() else {}
        self.keys = self.state.get("keys", {})  # user -> {"a": key|None, "b": key|None}

    # ---- clients
    def key(self, user, side):
        return self.keys[user][side]

    def _call(self, side, path, body=None, user="carol", method="POST", headers=None, key=None, model=True, **kw):
        if isinstance(body, dict) and model and "model" not in body:
            body = dict(body, model=getattr(self, f"model_{side}"))
        hdrs = {"authorization": f"Bearer {key if key is not None else self.key(user, side)}"}
        hdrs.update(headers or {})
        return request(getattr(self, f"{side}_url"), method, path, body, hdrs, **kw)

    def a(self, *args, **kw):
        return self._call("a", *args, **kw)

    def b(self, *args, **kw):
        return self._call("b", *args, **kw)

    def pair(self, user, path, body=None, method="POST", headers=None, **kw):
        """Sends the same request through A and B at the same time; `model` is set per side
        unless the body already has one."""
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fa = pool.submit(self.a, path, body, user, method, headers, **kw)
            fb = pool.submit(self.b, path, body, user, method, headers, **kw)
            return fa.result(), fb.result()

    # ---- what reached the engines, admin, logs
    def engine_log(self, side, since=0.0):
        path = self.dir / f"mock-{side}.jsonl"
        if not path.exists():
            return []
        entries = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
        return [e for e in entries if e["time"] >= since]

    def audit_log(self, since_line=0):
        path = self.dir / "b-audit.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().split("\n")[since_line:] if line.strip()]

    def admin(self, method, path, body=None, key=None, direct=False):
        url = self.proxy_b_url if direct else self.b_url
        return request(url, method, "/admin/api/" + path.lstrip("/"), body,
                       {"authorization": f"Bearer {key or self.state['admin_key']}"})

    def metrics(self, key=None):
        r = request(self.b_url, "GET", "/metrics", None,
                    {"authorization": f"Bearer {key or self.key('metrics', 'b')}"})
        return r.body.decode(errors="replace")

    def litellm(self, method, path, body=None):
        return request(LITELLM_ADMIN, method, path, body,
                       {"authorization": f"Bearer {env_file()['LITELLM_MASTER_KEY']}"})

    def sigusr1(self, side="b"):
        os.kill(self.state["pids"][f"serve_{side}"], signal.SIGUSR1)


# --------------------------------------------------------------------------- lifecycle


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def succeeded(r, what, statuses=(200,)):
    if r.status not in statuses:
        raise RuntimeError(f"{what} failed: {r}")
    return r.json


def spawn(slot_dir, name, argv):
    log = open(slot_dir / f"{name}.log", "ab")
    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True, cwd=slot_dir)
    return process.pid


def up(n):
    slot_dir = RUN / f"slot{n}"
    if (slot_dir / "slot.json").exists() and json.loads((slot_dir / "slot.json").read_text()).get("pids"):
        down(n)
    slot_dir.mkdir(parents=True, exist_ok=True)
    for stale in slot_dir.glob("*"):
        if stale.name.startswith("backend-") and stale.suffix == ".json":
            continue  # keep benchmarks
        if stale.is_file() or stale.is_socket():
            stale.unlink()
    p = ports(n)
    env = env_file()
    pids = {}

    for side in "ab":
        aliases = ["--alias", f"parity/smollm-s{n}", "--alias", f"auto/smollm-s{n}"] if side == "b" else []
        pids[f"mock_{side}"] = spawn(slot_dir, f"mock-{side}", [
            str(MOCK), "--listen", f"127.0.0.1:{p['mock_' + side]}",
            "--control-socket", str(slot_dir / f"mock-{side}.sock"),
            "--log", str(slot_dir / f"mock-{side}.jsonl"),
        ] + aliases)
    for side in "ab":
        wait_http(f"http://127.0.0.1:{p['mock_' + side]}")

    accounts = json.loads(json.dumps(PROD_ACCOUNTS))
    accounts["accounts"][0]["api_key"] = env["VLLM_API_KEY"]
    write_json(slot_dir / "accounts.json", accounts)
    for side in "ab":
        general = json.loads(json.dumps(PROD_GENERAL))
        general["frontend"]["listen"] = f"127.0.0.1:{p['proxy_' + side]}"
        general["scheduling_log"] = str(slot_dir / f"{side}-scheduling.log")
        general["middle_end_socket"] = str(slot_dir / f"{side}-middle.sock")
        if side == "b":
            general["gateway"] = {
                "keys_file": str(slot_dir / "keys.json"),
                "audit_log": str(slot_dir / "b-audit.jsonl"),
                # nginx sets X-Forwarded-For like production's site config.
                "client_ip_header": "x-forwarded-for",
            }
        write_json(slot_dir / f"general-{side}.json", general)

    # The production launch script benchmarks the engine when backend.json is missing.
    for side in "ab":
        backend = slot_dir / f"backend-{side}.json"
        reference = RUN / "slot0" / f"backend-{side}.json"
        if not backend.exists() and reference.exists() and n != 0:
            # Same mock binary and settings: reuse slot 0's measurements.
            value = json.loads(reference.read_text())
            value["ranks"][0]["upstream"] = f"http://127.0.0.1:{p['mock_' + side]}"
            value["ranks"][0]["control_socket"] = str(slot_dir / f"mock-{side}.sock")
            write_json(backend, value)
        if not backend.exists():
            subprocess.run([
                str(PROXY), "benchmark", "--upstream", f"http://127.0.0.1:{p['mock_' + side]}",
                "--control-socket", str(slot_dir / f"mock-{side}.sock"),
                "--general", str(slot_dir / f"general-{side}.json"),
                "--output", str(backend) + ".partial",
            ], check=True, stdout=open(slot_dir / f"benchmark-{side}.log", "wb"), stderr=subprocess.STDOUT)
            os.rename(str(backend) + ".partial", backend)

    # Gateway: the first admin key from the CLI, as the README says.
    created = subprocess.run([
        str(PROXY), "create-key", "--general", str(slot_dir / "general-b.json"),
        "--accounts", str(slot_dir / "accounts.json"),
        "--user", "admin", "--default-class", "medium", "--role", "admin", "--new-keys-file",
    ], check=True, capture_output=True, text=True)
    admin_key = [w for w in (created.stdout + created.stderr).split() if w.startswith("sk-")][-1]

    for side in "ab":
        pids[f"serve_{side}"] = spawn(slot_dir, f"serve-{side}", [
            str(PROXY), "serve", "--general", str(slot_dir / f"general-{side}.json"),
            "--accounts", str(slot_dir / "accounts.json"), "--backend", str(slot_dir / f"backend-{side}.json"),
        ])
        pids[f"frontend_{side}"] = spawn(slot_dir, f"frontend-{side}", [
            str(PROXY), "frontend", "--general", str(slot_dir / f"general-{side}.json"),
        ])
    state = {"n": n, "pids": pids, "admin_key": admin_key, "keys": {}}
    write_json(slot_dir / "slot.json", state)
    for side in "ab":
        wait_http(f"http://127.0.0.1:{p['proxy_' + side]}", "/health")
    s = Slot(n)

    # LiteLLM: the two pod aliases as in RUNBOOK's model menu (DB rows, credential,
    # explicit zero cost, max parallel 12/20), then one virtual key per user.
    for model in (s.litellm("GET", "/v2/model/info").json or {}).get("data", []):
        if model.get("model_name") in (s.model_a, s.model_a_auto):
            s.litellm("POST", "/model/delete", {"id": model["model_info"]["id"]})
    for name, parallel in ((s.model_a, 12), (s.model_a_auto, 20)):
        r = s.litellm("POST", "/model/new", {
            "model_name": name,
            "litellm_params": {
                "model": f"hosted_vllm/{SERVED_MODEL}",
                "litellm_credential_name": f"vllm-pod-s{n}",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
                "max_parallel_requests": parallel,
            },
            "model_info": {},
        })
        succeeded(r, "LiteLLM /model/new")

    keys = {}
    for user, spec in USERS.items():
        keys[user] = {"a": None, "b": None}
        if spec.get("b") is not None:
            r = s.admin("POST", "keys", dict(DEFAULT_CLASSES, **spec["b"]))
            keys[user]["b"] = succeeded(r, f"gateway POST keys for {user}", (200, 201))["api_key"]
        if spec.get("a") is not None:
            body = dict(spec["a"], key_alias=f"s{n}-{user}-{int(time.time())}")
            if "models" in body:
                body["models"] = [m.replace("{parity}", s.model_a) for m in body["models"]]
            r = s.litellm("POST", "/key/generate", body)
            keys[user]["a"] = succeeded(r, f"LiteLLM /key/generate for {user}")["key"]
            if spec.get("block"):
                succeeded(s.litellm("POST", "/key/block", {"key": keys[user]["a"]}), "LiteLLM /key/block")
    keys["admin"] = {"a": env["LITELLM_MASTER_KEY"], "b": admin_key}
    state["keys"] = keys
    write_json(slot_dir / "slot.json", state)
    time.sleep(1.5)  # let erin's 1s LiteLLM key expire
    print(f"slot {n} up: A={s.a_url} B={s.b_url} ({slot_dir})")


def down(n):
    slot_dir = RUN / f"slot{n}"
    state_path = slot_dir / "slot.json"
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    for name, pid in (state.get("pids") or {}).items():
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 10
    for pid in (state.get("pids") or {}).values():
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                break
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    state["pids"] = {}
    write_json(state_path, state)
    print(f"slot {n} down")


def status():
    for slot_dir in sorted(RUN.glob("slot*")):
        state = json.loads((slot_dir / "slot.json").read_text()) if (slot_dir / "slot.json").exists() else {}
        alive = {}
        for name, pid in (state.get("pids") or {}).items():
            try:
                os.kill(pid, 0)
                alive[name] = "up"
            except ProcessLookupError:
                alive[name] = "DEAD"
        print(slot_dir.name, alive or "down")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "nginx":
        (HERE / "nginx.conf").write_text(nginx_conf())
    elif command == "up":
        up(int(sys.argv[2]))
    elif command == "down":
        down(int(sys.argv[2]))
    elif command == "restart":
        down(int(sys.argv[2]))
        up(int(sys.argv[2]))
    else:
        status()
