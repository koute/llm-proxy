import contextlib
import functools
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from huggingface_hub import snapshot_download

ENGINE_PYTHON = {
    "vllm": os.environ.get("E2E_VLLM_PYTHON", sys.executable),
    "sglang": os.environ.get("E2E_SGLANG_PYTHON", sys.executable),
}
SOCKET_VARIABLE = {"vllm": "VLLM_QOS_CONTROL_SOCKET", "sglang": "SGLANG_QOS_CONTROL_SOCKET"}


@functools.cache
def model() -> str:
    return os.environ.get("E2E_MODEL") or snapshot_download(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        revision="31b70e2e869a7173562077fd711b654946d38674",
        local_dir="/tmp/llm-proxy-e2e/SmolLM2-1.7B-Instruct",
        allow_patterns=["*.json", "model.safetensors"],
    )


@dataclass
class StreamResult:
    token_times: list[float] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    choice_tokens: dict[int, int] = field(default_factory=dict)
    completion_tokens: int = 0
    status: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / (self.end_time - self.start_time)

    @property
    def decode_rate(self) -> float:
        return (len(self.token_times) - 1) / (self.token_times[-1] - self.token_times[0])

    @property
    def time_to_first_token(self) -> float:
        return self.token_times[0] - self.start_time


async def stream(
    client: httpx.AsyncClient, url: str, body: dict, headers: dict, result: StreamResult
) -> StreamResult:
    body = {
        "model": model(),
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        **body,
    }
    result.start_time = time.monotonic()
    async with client.stream("POST", url, json=body, headers=headers) as response:
        result.status = response.status_code
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line.removeprefix("data: "))
            if chunk.get("usage"):
                result.completion_tokens = chunk["usage"]["completion_tokens"]
            for choice in chunk.get("choices", []):
                text = choice.get("text") or choice.get("delta", {}).get("content")
                if text:
                    index = choice["index"]
                    result.choice_tokens[index] = result.choice_tokens.get(index, 0) + 1
                    result.token_times.append(time.monotonic())
                    result.text_parts.append(text)
    result.end_time = time.monotonic()
    return result


def health_ok(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def stop(process: subprocess.Popen) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@contextlib.contextmanager
def running_engine(
    engine: str, port: int, max_running: int, extra_arguments: list[str], directory: Path
):
    python = ENGINE_PYTHON[engine]
    if engine == "vllm":
        command = [
            *(python, "-m", "vllm.entrypoints.cli.main", "serve", model()),
            *("--gpu-memory-utilization", "0.3", "--max-num-seqs", str(max_running)),
        ]
        environment = {"VLLM_BATCH_INVARIANT": "1"}
    else:
        command = [
            *(python, "-m", "sglang.launch_server", "--model-path", model()),
            *("--mem-fraction-static", "0.3", "--max-running-requests", str(max_running)),
            *("--enable-priority-scheduling", "--sleep-on-idle"),
            *("--enable-deterministic-inference", "--attention-backend", "triton"),
        ]
        environment = {}
    socket_template = str(directory / "control-{rank}.sock")
    environment[SOCKET_VARIABLE[engine]] = socket_template
    url = f"http://127.0.0.1:{port}"
    with open(directory / "engine.log", "wb") as log_file:
        process = subprocess.Popen(
            [*command, "--host", "127.0.0.1", "--port", str(port), *extra_arguments],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, **environment},
        )
    try:
        deadline = time.monotonic() + 900
        while not health_ok(url):
            assert process.poll() is None, f"{engine} exited during startup"
            assert time.monotonic() < deadline, f"{engine} startup timeout"
            time.sleep(2)
        yield process, url, socket_template
    finally:
        stop(process)
