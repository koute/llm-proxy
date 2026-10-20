import asyncio
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from engines import StreamResult, model, running_engine, stop, stream

ENGINE_PORT = {"vllm": 8131, "sglang": 8132}
FRONTEND_PORT = {"vllm": 8141, "sglang": 8142}
ENGINE_LIMIT = 64
IDLE_STREAMS = 32
ACCOUNTS = {
    "classes": [
        {"name": "batch", "priority": 0},
        {"name": "idle", "priority": -10, "idle_only": True},
    ],
    "accounts": [
        {"name": "alice", "api_key": "sk-alice", "default_class": "batch"},
        {"name": "heavy", "api_key": "sk-heavy", "default_class": "batch"},
        {"name": "newcomer", "api_key": "sk-newcomer", "default_class": "batch"},
        {"name": "nightly", "api_key": "sk-nightly", "default_class": "idle"},
    ],
}


@dataclass
class Deployment:
    engine: str
    directory: Path
    socket_template: str
    proxy_binary: str
    serve: subprocess.Popen | None = None
    frontend: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{FRONTEND_PORT[self.engine]}"

    def config_arguments(self) -> list[str]:
        return [
            "--general",
            str(self.directory / "general.json"),
            "--accounts",
            str(self.directory / "accounts.json"),
            "--backend",
            str(self.directory / "backend.json"),
        ]

    def write_configuration(self, engine_limit: int, keepalive_seconds: float) -> None:
        general = {
            "frontend": {"listen": f"127.0.0.1:{FRONTEND_PORT[self.engine]}"},
            "middle_end_socket": str(self.directory / "middle-end.sock"),
            "limits": {"keepalive_interval_seconds": keepalive_seconds},
        }
        backend = {
            "engine": self.engine,
            "model": model(),
            "engine_max_parallel_requests_per_rank": engine_limit,
            "priority_scheduling": True,
            "ranks": [
                {
                    "upstream": f"http://127.0.0.1:{ENGINE_PORT[self.engine]}",
                    "control_socket": self.socket_template.replace("{rank}", "0"),
                }
            ],
        }
        (self.directory / "general.json").write_text(json.dumps(general))
        (self.directory / "accounts.json").write_text(json.dumps(ACCOUNTS))
        (self.directory / "backend.json").write_text(json.dumps(backend))

    def start(self, engine_limit: int, keepalive_seconds: float = 15) -> None:
        self.write_configuration(engine_limit, keepalive_seconds)
        self.serve = spawn(
            [self.proxy_binary, "--verbose", "serve", *self.config_arguments()],
            self.directory / "serve.log",
        )
        self.frontend = spawn(
            [self.proxy_binary, "frontend", "--general", str(self.directory / "general.json")],
            self.directory / "frontend.log",
        )
        deadline = time.monotonic() + 30
        while "connected to the engine control socket" not in self.log("serve.log"):
            assert time.monotonic() < deadline
            time.sleep(0.2)

    def stop(self) -> None:
        for process in (self.frontend, self.serve):
            if process is not None:
                stop(process)

    def log(self, name: str) -> str:
        return (self.directory / name).read_text(errors="replace")


def spawn(command: list[str], log_path: Path) -> subprocess.Popen:
    with open(log_path, "ab") as log_file:
        return subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )


@pytest.fixture(scope="session")
def proxy_binary() -> str:
    build = subprocess.run(
        ["cargo", "build", "--release", "--message-format=json-render-diagnostics"],
        cwd=Path(__file__).parent,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    messages = map(json.loads, build.stdout.splitlines())
    return next(message["executable"] for message in messages if message.get("executable"))


@pytest.fixture(scope="module")
def engine_server(engine, tmp_path_factory):
    directory = tmp_path_factory.mktemp(engine)
    arguments = ["--scheduling-policy", "priority"] if engine == "vllm" else []
    with running_engine(engine, ENGINE_PORT[engine], ENGINE_LIMIT, arguments, directory) as (
        _,
        _,
        socket_template,
    ):
        yield socket_template, directory


@pytest.fixture
def deployment(engine, proxy_binary, engine_server, request):
    socket_template, directory = engine_server
    test_directory = directory / request.node.name.replace("/", "_")
    test_directory.mkdir()
    deployment = Deployment(engine, test_directory, socket_template, proxy_binary)
    yield deployment
    deployment.stop()


async def stream_chat(
    client: httpx.AsyncClient, url: str, api_key: str, body: dict, result: StreamResult
) -> StreamResult:
    headers = {"authorization": f"Bearer {api_key}"}
    return await stream(client, f"{url}/v1/chat/completions", body, headers, result)


def chat_body(prompt: str, max_tokens: int) -> dict:
    return {"max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}


def idle_prompts() -> list[str]:
    return [
        f"Write a long story about lighthouse keeper number {index}."
        for index in range(IDLE_STREAMS)
    ]


BATCH_PROMPT = "Explain in detail how a compiler turns source code into machine code."


async def wait_for_tokens(results: list[StreamResult], minimum: int) -> None:
    while any(len(result.token_times) < minimum for result in results):
        await asyncio.sleep(0.02)


def test_idle_suspension_during_batch_request(deployment):
    deployment.start(ENGINE_LIMIT)

    async def scenario():
        async with httpx.AsyncClient(timeout=600) as client:
            url = deployment.url
            batch_alone = await stream_chat(
                client, url, "sk-alice", chat_body(BATCH_PROMPT, 300), StreamResult()
            )
            references = await asyncio.gather(
                *[
                    stream_chat(client, url, "sk-nightly", chat_body(prompt, 400), StreamResult())
                    for prompt in idle_prompts()
                ]
            )
            idle_results = [StreamResult() for _ in range(IDLE_STREAMS)]
            idle_tasks = [
                asyncio.create_task(
                    stream_chat(client, url, "sk-nightly", chat_body(prompt, 400), result)
                )
                for prompt, result in zip(idle_prompts(), idle_results)
            ]
            await wait_for_tokens(idle_results, 30)
            batch = await stream_chat(
                client, url, "sk-alice", chat_body(BATCH_PROMPT, 300), StreamResult()
            )
            await asyncio.gather(*idle_tasks)
        return batch_alone, references, idle_results, batch

    batch_alone, references, idle_results, batch = asyncio.run(scenario())
    idle_token_times = sorted(point for result in idle_results for point in result.token_times)
    last_idle_token_before_batch = max(
        (point for point in idle_token_times if point < batch.token_times[0]),
        default=batch.start_time,
    )
    idle_tokens_during_batch = [
        point for point in idle_token_times if batch.token_times[0] <= point <= batch.end_time
    ]
    print(
        f"\n[{deployment.engine}] batch alone: {batch_alone.decode_rate:.1f} tok/s, "
        f"ttft {batch_alone.time_to_first_token * 1000:.0f} ms; "
        f"batch beside {IDLE_STREAMS} suspended idle: {batch.decode_rate:.1f} tok/s, "
        f"ttft {batch.time_to_first_token * 1000:.0f} ms; last idle token "
        f"{1000 * (last_idle_token_before_batch - batch.start_time):.0f} ms after batch arrival"
    )
    assert batch.status == 200
    assert idle_tokens_during_batch == []
    assert last_idle_token_before_batch - batch.start_time < 0.2
    assert batch.decode_rate > 0.8 * batch_alone.decode_rate
    for reference, result in zip(references, idle_results):
        assert result.status == 200
        assert result.completion_tokens == reference.completion_tokens == 400
        assert result.text == reference.text


def test_guaranteed_request_for_newcomer(deployment):
    deployment.start(4)

    async def scenario():
        async with httpx.AsyncClient(timeout=600) as client:
            url = deployment.url
            heavy_results = [StreamResult() for _ in range(8)]
            heavy_tasks = [
                asyncio.create_task(
                    stream_chat(client, url, "sk-heavy", chat_body(f"Story {index}.", 1500), result)
                )
                for index, result in enumerate(heavy_results)
            ]
            await asyncio.sleep(3)
            newcomer = await stream_chat(
                client, url, "sk-newcomer", chat_body("Say hello.", 50), StreamResult()
            )
            await asyncio.gather(*heavy_tasks)
        return heavy_results, newcomer

    heavy_results, newcomer = asyncio.run(scenario())
    print(f"\n[{deployment.engine}] newcomer ttft {newcomer.time_to_first_token * 1000:.0f} ms")
    assert newcomer.status == 200
    assert newcomer.time_to_first_token < 1.0
    assert all(result.completion_tokens == 1500 for result in heavy_results)


def test_restart_during_streams(deployment):
    deployment.start(ENGINE_LIMIT)
    process_id = deployment.serve.pid

    async def scenario():
        async with httpx.AsyncClient(timeout=600) as client:
            url = deployment.url
            prompts = idle_prompts()[:8]
            references = await asyncio.gather(
                *[
                    stream_chat(client, url, "sk-nightly", chat_body(prompt, 300), StreamResult())
                    for prompt in prompts
                ]
            )
            results = [StreamResult() for _ in prompts]
            tasks = [
                asyncio.create_task(
                    stream_chat(client, url, "sk-nightly", chat_body(prompt, 300), result)
                )
                for prompt, result in zip(prompts, results)
            ]
            batch_task = asyncio.create_task(
                stream_chat(client, url, "sk-alice", chat_body(BATCH_PROMPT, 200), StreamResult())
            )
            await wait_for_tokens(results, 20)
            for _ in range(2):
                os.kill(process_id, signal.SIGUSR1)
                await asyncio.sleep(1.0)
            batch = await batch_task
            await asyncio.gather(*tasks)
        return references, results, batch

    references, results, batch = asyncio.run(scenario())
    assert deployment.serve.poll() is None
    assert deployment.log("serve.log").count("restart complete") == 2
    assert batch.status == 200 and batch.completion_tokens == 200
    for reference, result in zip(references, results):
        assert result.status == 200
        assert result.text == reference.text


def test_non_streaming_keepalive_while_suspended(deployment):
    deployment.start(ENGINE_LIMIT, keepalive_seconds=0.5)

    async def scenario():
        async with httpx.AsyncClient(timeout=600) as client:
            url = deployment.url
            body = {
                "model": model(),
                "temperature": 0,
                "ignore_eos": True,
                **chat_body(idle_prompts()[0], 300),
            }
            idle = asyncio.create_task(
                client.post(
                    f"{url}/v1/chat/completions",
                    json=body,
                    headers={"authorization": "Bearer sk-nightly"},
                )
            )
            await asyncio.sleep(1.0)
            batch = await stream_chat(
                client, url, "sk-alice", chat_body(BATCH_PROMPT, 300), StreamResult()
            )
            return await idle, batch

    idle, batch = asyncio.run(scenario())
    assert batch.status == 200
    assert idle.status_code == 200
    assert idle.headers["content-type"] == "application/json"
    assert idle.content.startswith(b"\n\n")
    completion = json.loads(idle.content)
    assert completion["usage"]["completion_tokens"] == 300
    assert completion["choices"][0]["message"]["content"]


def test_benchmark_backend_file(engine, proxy_binary, engine_server, tmp_path):
    socket_template, _ = engine_server
    output = tmp_path / "backend.json"
    completed = subprocess.run(
        [
            proxy_binary,
            "benchmark",
            "--upstream",
            f"http://127.0.0.1:{ENGINE_PORT[engine]}",
            "--control-socket",
            socket_template,
            "--prompt-tokens",
            "200",
            "--output-tokens",
            "128",
            "--max-parallel-requests-per-rank",
            "16",
            "--level-seconds",
            "5",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    print(completed.stdout, completed.stderr)
    assert completed.returncode == 0
    backend = json.loads(output.read_text())
    assert backend["engine"] == engine
    assert backend["engine_max_parallel_requests_per_rank"] >= 1
    assert abs(backend["benchmark"]["prompt_tokens"] - 200) <= 10
    measurements = backend["benchmark"]["measurements"]
    assert [measurement["parallel_requests_per_rank"] for measurement in measurements][:3] == [
        1,
        2,
        3,
    ]
    assert (
        measurements[-1]["tokens_per_second_per_rank"]
        > measurements[0]["tokens_per_second_per_rank"]
    )
