import asyncio
import itertools
import json
import secrets
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import psutil
import pytest

from engines import StreamResult, health_ok, model, running_engine, stream

PORT = 8111
MAX_RUNNING = 16
BACKGROUND_COUNT = 32
EXTRA_ARGUMENTS = {
    "vllm": ["--max-model-len", "4096", "--num-gpu-blocks-override", "1024"],
    "sglang": [],
}
EXPECTED_INFO = {
    "vllm": {"data_parallel_rank": 0, "priority_scheduling": False},
    "sglang": {"data_parallel_rank": None, "priority_scheduling": True},
}
ID_PREFIX = secrets.token_hex(8)
ID_COUNTER = itertools.count(1)


class ControlError(Exception):
    pass


class ControlClient:
    def __init__(self, socket_path: str, timeout: float = 30.0):
        self.timeout = timeout
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(timeout)
        self.connection.connect(socket_path)
        self.reader = self.connection.makefile("rb")
        self.last_seq = 0
        self.events: list[dict] = []

    def send(self, command: dict) -> int:
        self.last_seq += 1
        line = json.dumps({"seq": self.last_seq, **command}).encode() + b"\n"
        self.connection.sendall(line)
        return self.last_seq

    def receive(self) -> dict:
        line = self.reader.readline()
        if not line:
            raise ConnectionError("the engine closed the control socket")
        message = json.loads(line)
        if "event" in message:
            self.events.append(message)
        return message

    def reply(self) -> dict:
        while "event" in (message := self.receive()):
            pass
        return message

    def call(self, command: dict) -> dict:
        seq = self.send(command)
        reply = self.reply()
        if reply.get("seq") != seq:
            raise ControlError(f"reply {reply} does not answer seq {seq}")
        return reply

    def result(self, command: dict):
        reply = self.call(command)
        if "error" in reply:
            raise ControlError(reply["error"])
        return reply["result"]

    def suspend(self, ids: list[str]) -> None:
        return self.result({"op": "suspend", "ids": list(ids)})

    def resume(self, ids: list[str]) -> None:
        return self.result({"op": "resume", "ids": list(ids)})

    def info(self) -> dict:
        return self.result({"op": "info"})

    def stats(self) -> dict:
        return self.result({"op": "stats"})

    def states(self) -> dict[str, str]:
        return {entry["id"]: entry["state"] for entry in self.stats()["requests"]}

    def wait_for_event(self, timeout: float = 30.0, **fields) -> dict:
        deadline = time.monotonic() + timeout
        try:
            while True:
                for event in self.events:
                    if all(event.get(key) == value for key, value in fields.items()):
                        self.events.remove(event)
                        return event
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no event matching {fields}")
                self.connection.settimeout(remaining)
                self.receive()
        finally:
            self.connection.settimeout(self.timeout)

    def close(self) -> None:
        self.reader.close()
        self.connection.close()

    def __enter__(self) -> "ControlClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


async def wait_for_event(control: ControlClient, timeout: float = 60.0, **fields) -> dict:
    return await asyncio.to_thread(control.wait_for_event, timeout, **fields)


def proxy_id() -> str:
    return f"qos-{ID_PREFIX}-{next(ID_COUNTER):016x}"


@dataclass
class Server:
    engine: str
    url: str
    socket_path: str
    log_path: Path
    process: psutil.Process

    def cpu_seconds(self) -> float:
        tree = [self.process, *self.process.children(recursive=True)]
        return sum(member.cpu_times().user + member.cpu_times().system for member in tree)

    async def stream(
        self,
        client: httpx.AsyncClient,
        path: str,
        request_id: str,
        body: dict,
        result: StreamResult,
        priority: int | None = None,
    ) -> StreamResult:
        if self.engine == "vllm":
            body = {**body, "cache_salt": request_id}
            headers = {"X-Request-Id": request_id}
        else:
            priority_field = {} if priority is None else {"priority": priority}
            body = {**body, "rid": request_id, **priority_field}
            headers = {}
        return await stream(client, f"{self.url}{path}", body, headers, result)

    async def chat(
        self,
        client: httpx.AsyncClient,
        request_id: str,
        prompt: str,
        max_tokens: int,
        result: StreamResult | None = None,
        priority: int | None = None,
    ) -> StreamResult:
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        return await self.stream(
            client,
            "/v1/chat/completions",
            request_id,
            body,
            StreamResult() if result is None else result,
            priority,
        )

    def start_chats(
        self,
        client: httpx.AsyncClient,
        prompts: dict[str, str],
        max_tokens: int,
        priority: int | None = None,
    ) -> tuple[dict[str, StreamResult], list[asyncio.Task]]:
        results = {request_id: StreamResult() for request_id in prompts}
        tasks = [
            asyncio.create_task(
                self.chat(client, request_id, prompt, max_tokens, results[request_id], priority)
            )
            for request_id, prompt in prompts.items()
        ]
        return results, tasks

    async def run_chats(
        self, client: httpx.AsyncClient, prompts: dict[str, str], max_tokens: int
    ) -> dict[str, StreamResult]:
        results, tasks = self.start_chats(client, prompts, max_tokens)
        await asyncio.gather(*tasks)
        return results


@pytest.fixture(scope="module")
def server(engine, tmp_path_factory):
    directory = tmp_path_factory.mktemp(engine)
    with running_engine(engine, PORT, MAX_RUNNING, EXTRA_ARGUMENTS[engine], directory) as (
        process,
        url,
        socket_template,
    ):
        yield Server(
            engine,
            url,
            socket_template.replace("{rank}", "0"),
            directory / "engine.log",
            psutil.Process(process.pid),
        )


async def wait_until(predicate, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.05)


async def wait_for_tokens(
    results: list[StreamResult], minimum: int, stream_count: int | None = None
) -> None:
    stream_count = len(results) if stream_count is None else stream_count
    await wait_until(
        lambda: sum(len(result.token_times) >= minimum for result in results) >= stream_count,
        120,
    )


async def cancel(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def metric_value(url: str, name: str) -> float:
    for line in httpx.get(f"{url}/metrics").text.splitlines():
        if line.startswith(name + "{"):
            return float(line.split()[-1])
    raise KeyError(name)


def test_info_and_stats(server):
    prompts = {
        proxy_id(): f"Request {index}. Describe the ocean in detail."
        for index in range(MAX_RUNNING + 1)
    }

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            with ControlClient(server.socket_path) as control:
                info = control.info()
                assert info.pop("kv_cache_tokens") > 0
                assert info == {
                    "engine": server.engine,
                    "model": model(),
                    "max_running_requests": MAX_RUNNING,
                    **EXPECTED_INFO[server.engine],
                }
                assert control.stats() == {"requests": []}
                results, tasks = server.start_chats(client, prompts, 2000)
                await wait_for_tokens(list(results.values()), 3, MAX_RUNNING)
                requests = control.stats()["requests"]
                assert sorted(entry["id"] for entry in requests) == sorted(prompts)
                states = [entry["state"] for entry in requests]
                assert states.count("running") == MAX_RUNNING
                assert states.count("waiting") == 1
                for entry in requests:
                    assert entry["prompt_tokens"] > 0
                    assert (entry["output_tokens"] > 0) == (entry["state"] == "running")
                await cancel(tasks)
                await wait_until(lambda: not set(prompts) & set(control.states()))

    asyncio.run(scenario())


def test_replies_in_order(server):
    request_id = proxy_id()
    commands = [
        {"op": "suspend", "ids": [request_id]},
        {"op": "info"},
        {"op": "bogus"},
        {"op": "stats"},
        {"op": "resume", "ids": [request_id]},
        {"op": "suspend", "ids": "not a list"},
        {"op": "resume", "ids": [request_id, proxy_id()]},
    ] * 5

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            results, tasks = server.start_chats(
                client, {request_id: "Count to one thousand."}, 2000
            )
            await wait_for_tokens(list(results.values()), 3)
            with ControlClient(server.socket_path) as control:
                sent = [control.send(command) for command in commands]
                replies = [control.reply() for _ in commands]
                assert [reply["seq"] for reply in replies] == sent
                for command, reply in zip(commands, replies):
                    if command["op"] == "bogus" or command.get("ids") == "not a list":
                        assert set(reply) == {"seq", "error"}
                    elif command["op"] in ("suspend", "resume"):
                        assert reply["result"] is None
                    elif command["op"] == "info":
                        assert reply["result"]["engine"] == server.engine
                    else:
                        assert isinstance(reply["result"]["requests"], list)
                count = len(results[request_id].token_times)
                await wait_until(lambda: len(results[request_id].token_times) >= count + 5, 20)
                await cancel(tasks)
                await wait_until(lambda: request_id not in control.states())

    asyncio.run(scenario())


def test_streaming_chat_suspension_and_resumption(server):
    prompts = {
        proxy_id(): f"Write a long story about a robot number {index}." for index in range(8)
    }

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            await server.run_chats(client, {proxy_id(): prompt for prompt in prompts.values()}, 1)
            reference_ids = {request_id: proxy_id() for request_id in prompts}
            reference_results = await server.run_chats(
                client,
                {reference_ids[key]: prompt for key, prompt in prompts.items()},
                200,
            )
            results, tasks = server.start_chats(client, prompts, 200)
            await wait_for_tokens(list(results.values()), 20)
            with ControlClient(server.socket_path) as control:
                for request_id in prompts:
                    await wait_for_event(control, event="running", id=request_id)
                assert control.suspend(list(prompts)) is None
                for request_id in prompts:
                    await wait_for_event(
                        control, event="stopped", id=request_id, reason="suspended"
                    )
                await asyncio.sleep(0.5)
                token_counts = [len(result.token_times) for result in results.values()]
                cpu_before = server.cpu_seconds()
                await asyncio.sleep(2)
                cpu_seconds = server.cpu_seconds() - cpu_before
                assert [len(result.token_times) for result in results.values()] == token_counts
                stats = {entry["id"]: entry for entry in control.stats()["requests"]}
                assert {stats[request_id]["state"] for request_id in prompts} == {"suspended"}
                assert all(
                    stats[request_id]["output_tokens"] >= 20
                    and stats[request_id]["prompt_tokens"] > 0
                    for request_id in prompts
                )
                assert control.resume(list(prompts)) is None
                for request_id in prompts:
                    await wait_for_event(control, event="running", id=request_id)
                await asyncio.gather(*tasks)
                for request_id in prompts:
                    await wait_for_event(control, event="stopped", id=request_id, reason="finished")
            print(f"\nserver CPU while all suspended: {cpu_seconds:.2f} s in 2 s")
            assert cpu_seconds < 0.5
            for request_id, result in results.items():
                assert result.completion_tokens == 200
                assert result.text == reference_results[reference_ids[request_id]].text

    asyncio.run(scenario())


def test_suspension_before_arrival(server):
    request_id = proxy_id()

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            with ControlClient(server.socket_path) as control:
                assert control.suspend([request_id]) is None
                results, tasks = server.start_chats(
                    client, {request_id: "Count to one hundred."}, 50
                )
                await wait_until(lambda: request_id in control.states())
                await asyncio.sleep(2)
                entries = [
                    entry for entry in control.stats()["requests"] if entry["id"] == request_id
                ]
                assert len(entries) == 1
                assert entries[0]["state"] == "suspended"
                assert entries[0]["output_tokens"] == 0
                assert entries[0]["prompt_tokens"] > 0
                assert results[request_id].token_times == []
                assert all(event["id"] != request_id for event in control.events)
                assert control.resume([request_id]) is None
                assert await wait_for_event(control, id=request_id) == {
                    "event": "running",
                    "id": request_id,
                }
                await asyncio.gather(*tasks)
                assert await wait_for_event(control, id=request_id) == {
                    "event": "stopped",
                    "id": request_id,
                    "reason": "finished",
                }
                assert results[request_id].completion_tokens == 50

    asyncio.run(scenario())


def test_memory_preemption_event(server):
    if server.engine != "vllm":
        pytest.skip("the sglang test server has more KV cache than the requests use")
    prompts = {
        proxy_id(): f"Request {index}. " + "Tell me about the history of computing. " * 70
        for index in range(MAX_RUNNING)
    }

    async def scenario():
        async with httpx.AsyncClient(timeout=300) as client:
            with ControlClient(server.socket_path, timeout=300) as control:
                results, tasks = server.start_chats(client, prompts, 1500)
                for request_id in prompts:
                    await wait_for_event(control, event="running", id=request_id)
                preempted = await wait_for_event(
                    control, timeout=240, event="stopped", reason="preempted"
                )
                assert preempted["id"] in prompts
                running = {
                    request_id
                    for request_id, state in control.states().items()
                    if state == "running" and request_id in prompts
                }
                assert running
                control.events.clear()
                await cancel(tasks)
                reasons = [
                    (await wait_for_event(control, event="stopped", id=request_id))["reason"]
                    for request_id in running
                ]
                assert set(reasons) <= {"finished", "preempted"}
                assert "finished" in reasons
                await wait_until(lambda: not control.stats()["requests"])

    asyncio.run(scenario())


def test_priority_preemption_event(server):
    if not EXPECTED_INFO[server.engine]["priority_scheduling"]:
        pytest.skip("the test server runs without priority scheduling")
    low_prompts = {
        proxy_id(): f"Request {index}. Tell me about the history of computing."
        for index in range(MAX_RUNNING)
    }
    high_id = proxy_id()

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            with ControlClient(server.socket_path) as control:
                results, tasks = server.start_chats(client, low_prompts, 2000, 0)
                for request_id in low_prompts:
                    await wait_for_event(control, event="running", id=request_id)
                await wait_for_tokens(list(results.values()), 5)
                high = await asyncio.gather(
                    server.chat(client, high_id, "Say hello.", 20, priority=100),
                    wait_for_event(control, event="stopped", reason="preempted"),
                )
                high_result, preempted = high
                assert preempted["id"] in low_prompts
                await wait_for_event(control, event="running", id=high_id)
                await wait_for_event(control, event="stopped", id=high_id, reason="finished")
                await wait_for_event(control, event="running", id=preempted["id"])
                assert high_result.completion_tokens == 20
                await cancel(tasks)
                for request_id in low_prompts:
                    await wait_for_event(control, event="stopped", id=request_id, reason="finished")
                await wait_until(lambda: not set(low_prompts) & set(control.states()))

    asyncio.run(scenario())


def test_parallel_samples_of_one_request(server):
    if server.engine != "vllm":
        pytest.skip("sglang requires one rid per sample")
    request_id = proxy_id()
    body = {
        "prompt": "Write a list of one thousand animal names:",
        "max_tokens": 300,
        "temperature": 1.0,
        "seed": 7,
        "n": 2,
    }

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            with ControlClient(server.socket_path) as control:
                result = StreamResult()
                task = asyncio.create_task(
                    server.stream(client, "/v1/completions", request_id, body, result)
                )
                await wait_until(
                    lambda: (
                        len(result.choice_tokens) == 2 and min(result.choice_tokens.values()) >= 10
                    )
                )
                await wait_for_event(control, event="running", id=request_id)
                assert control.suspend([request_id]) is None
                await wait_for_event(control, event="stopped", id=request_id, reason="suspended")
                await asyncio.sleep(0.5)
                token_counts = dict(result.choice_tokens)
                await asyncio.sleep(1.5)
                assert result.choice_tokens == token_counts
                entries = [
                    entry for entry in control.stats()["requests"] if entry["id"] == request_id
                ]
                assert [entry["state"] for entry in entries] == ["suspended"] * 2
                assert control.resume([request_id]) is None
                await wait_for_event(control, event="running", id=request_id)
                await task
                assert result.completion_tokens == 600

    asyncio.run(scenario())


def test_suspension_reset_after_last_disconnect(server):
    running_id = proxy_id()
    early_id = proxy_id()

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            results, tasks = server.start_chats(client, {running_id: "Write a long poem."}, 300)
            await wait_for_tokens(list(results.values()), 10)
            first = ControlClient(server.socket_path)
            second = ControlClient(server.socket_path)
            assert second.info()["engine"] == server.engine
            await wait_for_event(first, event="running", id=running_id)
            assert first.suspend([running_id, early_id]) is None
            await wait_for_event(first, event="stopped", id=running_id, reason="suspended")
            first.close()
            await asyncio.sleep(0.5)
            token_count = len(results[running_id].token_times)
            await asyncio.sleep(1.0)
            assert len(results[running_id].token_times) == token_count
            assert second.states()[running_id] == "suspended"
            second.close()
            await asyncio.gather(*tasks)
            assert results[running_id].completion_tokens == 300
            early = await server.chat(client, early_id, "Say hello.", 20)
            assert early.completion_tokens == 20

    asyncio.run(scenario())


def test_new_request_speed_beside_suspended_requests(server):
    background_prompts = {
        proxy_id(): f"Request {index}. " + "Tell me about the history of computing. " * 110
        for index in range(BACKGROUND_COUNT)
    }
    probe_text = "Explain photosynthesis in detail. " * 500

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            alone = await server.chat(client, proxy_id(), "Probe A. " + probe_text, 256)
            results, tasks = server.start_chats(client, background_prompts, 2000)
            await wait_for_tokens(list(results.values()), 5, MAX_RUNNING)
            with ControlClient(server.socket_path) as control:
                assert control.suspend(list(background_prompts)) is None
                await asyncio.sleep(0.5)
                states = control.states()
                assert {states[request_id] for request_id in background_prompts} == {"suspended"}
                contended = StreamResult()
                probe = asyncio.create_task(
                    server.chat(client, proxy_id(), "Probe B. " + probe_text, 256, contended)
                )
                await wait_for_tokens([contended], 10)
                if server.engine == "vllm":
                    preempted = metric_value(server.url, "vllm:num_requests_waiting") - (
                        BACKGROUND_COUNT - MAX_RUNNING
                    )
                    assert preempted >= 2
                await probe
                assert control.resume(list(background_prompts)) is None
                await wait_for_tokens(list(results.values()), 10, MAX_RUNNING)
                await cancel(tasks)
                await wait_until(lambda: not set(background_prompts) & set(control.states()))
        print(
            f"\nprobe alone: {alone.tokens_per_second:.1f} tok/s, "
            f"with {BACKGROUND_COUNT} suspended: "
            f"{contended.tokens_per_second:.1f} tok/s"
        )
        assert contended.completion_tokens == alone.completion_tokens == 256
        assert contended.tokens_per_second > 0.75 * alone.tokens_per_second

    asyncio.run(scenario())


def test_suspended_request_abort(server):
    finished_id = proxy_id()
    parked_id = proxy_id()
    early_id = proxy_id()

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            with ControlClient(server.socket_path) as control:
                await server.chat(client, finished_id, "Say hi.", 5)
                assert control.suspend([finished_id]) is None
                assert control.resume([finished_id]) is None

                results, tasks = server.start_chats(client, {parked_id: "Count to 1000."}, 1000)
                await wait_for_tokens(list(results.values()), 5)
                assert control.suspend([parked_id, early_id]) is None
                await wait_for_event(control, event="stopped", id=parked_id, reason="suspended")
                early_results, early_tasks = server.start_chats(client, {early_id: "Say hi."}, 5)
                await wait_until(lambda: control.states().get(early_id) == "suspended", 10)
                assert control.states()[parked_id] == "suspended"
                await cancel(tasks + early_tasks)
                await wait_until(lambda: not {parked_id, early_id} & set(control.states()), 20)
                assert control.stats() == {"requests": []}
                assert control.resume([parked_id, early_id]) is None
                assert early_results[early_id].token_times == []

            after = await server.chat(client, proxy_id(), "Say hello.", 20)
            assert after.completion_tokens == 20
            assert health_ok(server.url)
            await asyncio.sleep(2)
        log_text = server.log_path.read_text()
        assert "leak" not in log_text
        assert "Traceback" not in log_text

    asyncio.run(scenario())
