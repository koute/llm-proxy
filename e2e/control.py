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

from engines import StreamResult, stream

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
