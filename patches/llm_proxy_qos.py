import json
import os
import queue
import re
import socketserver
import threading
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Optional

ID_PATTERN = re.compile(r"qos-[0-9a-f]{16}-[0-9a-f]{16}")
OPS = ("suspend", "resume", "info", "stats")


@lru_cache(maxsize=1 << 16)
def qos_id(request_id: str) -> Optional[str]:
    match = ID_PATTERN.search(request_id)
    return match.group() if match else None


@dataclass
class Command:
    seq: Any
    op: str
    ids: list[str] = field(default_factory=list)
    error: Optional[str] = None


CLEAR = Command(None, "clear")


def parse_command(line: bytes) -> Command:
    try:
        message = json.loads(line)
        seq, op, ids = message.get("seq"), message.get("op"), message.get("ids", [])
    except (ValueError, AttributeError, RecursionError):
        return Command(None, "invalid", error="invalid JSON object")
    if op not in OPS:
        return Command(seq, "invalid", error=f"unknown op {op!r}")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        return Command(seq, "invalid", error="ids must be a list of strings")
    return Command(seq, op, ids)


class Suspension:
    def __init__(self):
        self.ids: set[str] = set()
        self.is_changed = False

    def apply(self, command: Command) -> None:
        if command.op == "suspend":
            self.ids.update(command.ids)
        elif command.op == "resume":
            self.ids.difference_update(command.ids)
        elif command.op == "clear":
            self.ids.clear()
        else:
            return
        self.is_changed = True

    def contains(self, request_id: str) -> bool:
        return bool(self.ids) and qos_id(request_id) in self.ids


def reply_message(
    command: Command, info: Callable[[], dict], stats: Callable[[], list[dict]]
) -> dict:
    if command.error is not None:
        return {"seq": command.seq, "error": command.error}
    if command.op == "info":
        return {"seq": command.seq, "result": info()}
    if command.op == "stats":
        return {"seq": command.seq, "result": {"requests": stats()}}
    return {"seq": command.seq, "result": None}


class ControlServer:
    def __init__(self, path: str, wake: Callable[[], None]):
        self.wake = wake
        self.received: deque[tuple[Command, Optional[queue.SimpleQueue]]] = deque()
        self.pending: deque[tuple[Command, Optional[queue.SimpleQueue]]] = deque()
        self.clients: set[queue.SimpleQueue] = set()
        self.clients_lock = threading.Lock()
        self.running: dict[str, Any] = {}
        control = self

        class ClientHandler(socketserver.StreamRequestHandler):
            def handle(self):
                outbox = queue.SimpleQueue()
                threading.Thread(target=self.write, args=(outbox,), daemon=True).start()
                with control.clients_lock:
                    control.clients.add(outbox)
                try:
                    for line in self.rfile:
                        control.arrival(parse_command(line), outbox)
                except OSError:
                    pass
                finally:
                    outbox.put(None)
                    with control.clients_lock:
                        control.clients.discard(outbox)
                        if not control.clients:
                            control.arrival(CLEAR, None)

            def write(self, outbox: queue.SimpleQueue):
                try:
                    while (data := outbox.get()) is not None:
                        self.wfile.write(data)
                except OSError:
                    pass

        if os.path.exists(path):
            os.unlink(path)
        server = socketserver.ThreadingUnixStreamServer(path, ClientHandler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def arrival(self, command: Command, outbox: Optional[queue.SimpleQueue]) -> None:
        self.received.append((command, outbox))
        self.wake()

    def take_commands(self) -> list[Command]:
        while self.received:
            self.pending.append(self.received.popleft())
        return [command for command, _ in self.pending]

    def reply(self, message: dict) -> None:
        _, outbox = self.pending.popleft()
        if outbox is not None:
            outbox.put(json.dumps(message).encode() + b"\n")

    def publish(
        self,
        running: dict[str, Any],
        is_finished: Callable[[Any], bool],
        is_suspended: Callable[[Any], bool],
    ) -> None:
        if not self.clients:
            self.running = {}
            return
        previous, self.running = self.running, running
        events = [
            {
                "event": "stopped",
                "id": request_id,
                "reason": "finished"
                if is_finished(request)
                else "suspended"
                if is_suspended(request)
                else "preempted",
            }
            for request_id, request in previous.items()
            if request_id not in running
        ] + [
            {"event": "running", "id": request_id}
            for request_id in running.keys() - previous.keys()
        ]
        if not events:
            return
        data = b"".join(json.dumps(event).encode() + b"\n" for event in events)
        with self.clients_lock:
            for outbox in self.clients:
                outbox.put(data)
