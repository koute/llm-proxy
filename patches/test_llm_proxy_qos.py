import json
import socket
import threading
import time

import pytest

from llm_proxy_qos import (
    Command,
    ControlServer,
    Suspension,
    parse_command,
    reply_message,
)

REQUEST = "qos-00000000000000ab-0000000000000001"
OTHER_REQUEST = "qos-00000000000000ab-0000000000000002"


class Engine:
    def __init__(self, path):
        self.wake = threading.Event()
        self.suspension = Suspension()
        self.server = ControlServer(str(path), self.wake.set)
        self.running = {}
        self.finished = set()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            for command in self.server.take_commands():
                self.suspension.apply(command)
                self.server.reply(
                    reply_message(
                        command,
                        lambda: {"engine": "test"},
                        lambda: [{"id": request_id} for request_id in self.running],
                    )
                )

    def publish(self):
        self.server.publish(
            dict(self.running),
            lambda request: request in self.finished,
            lambda request: self.suspension.contains(request),
        )


class Client:
    def __init__(self, path):
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(5)
        self.connection.connect(str(path))
        self.reader = self.connection.makefile("rb")

    def send(self, *messages):
        self.connection.sendall(
            b"".join(json.dumps(message).encode() + b"\n" for message in messages)
        )

    def receive(self):
        return json.loads(self.reader.readline())

    def close(self):
        self.reader.close()
        self.connection.close()


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.fixture
def engine(tmp_path):
    return Engine(tmp_path / "control.sock")


def test_command_validation():
    assert parse_command(b'{"seq": 1, "op": "suspend", "ids": ["a"]}') == Command(
        1, "suspend", ["a"]
    )
    assert parse_command(b"not json").error is not None
    assert parse_command(b"[1]").error is not None
    assert "bogus" in parse_command(b'{"seq": 2, "op": "bogus"}').error
    assert parse_command(b'{"seq": 3, "op": "resume", "ids": "a"}').error is not None
    assert parse_command(b'{"seq": 4, "op": "clear"}').error is not None


def test_replies_in_command_order(engine, tmp_path):
    client = Client(tmp_path / "control.sock")
    client.send(
        {"seq": 1, "op": "suspend", "ids": [REQUEST]},
        {"seq": 2, "op": "bogus"},
        {"seq": 3, "op": "info"},
        {"seq": 4, "op": "resume", "ids": [REQUEST]},
        {"seq": 5, "op": "stats"},
    )
    replies = [client.receive() for _ in range(5)]
    assert [reply["seq"] for reply in replies] == [1, 2, 3, 4, 5]
    assert replies[0] == {"seq": 1, "result": None}
    assert set(replies[1]) == {"seq", "error"}
    assert replies[2] == {"seq": 3, "result": {"engine": "test"}}
    assert replies[4] == {"seq": 5, "result": {"requests": []}}
    assert engine.suspension.ids == set()
    client.close()


def test_suspension_cleared_after_last_disconnect(engine, tmp_path):
    first = Client(tmp_path / "control.sock")
    second = Client(tmp_path / "control.sock")
    second.send({"seq": 1, "op": "info"})
    second.receive()
    first.send({"seq": 1, "op": "suspend", "ids": [REQUEST]})
    assert first.receive() == {"seq": 1, "result": None}
    first.close()
    second.send({"seq": 2, "op": "info"})
    second.receive()
    assert engine.suspension.ids == {REQUEST}
    second.close()
    wait_until(lambda: not engine.suspension.ids)


def test_events_for_every_client(engine, tmp_path):
    engine.running = {REQUEST: REQUEST, OTHER_REQUEST: OTHER_REQUEST}
    engine.publish()
    clients = [Client(tmp_path / "control.sock") for _ in range(2)]
    clients[0].send({"seq": 1, "op": "info"})
    clients[0].receive()
    clients[1].send({"seq": 1, "op": "info"})
    clients[1].receive()
    engine.publish()
    for client in clients:
        events = [client.receive(), client.receive()]
        assert sorted(event["id"] for event in events) == [REQUEST, OTHER_REQUEST]
        assert {event["event"] for event in events} == {"running"}
    engine.suspension.ids.add(REQUEST)
    engine.finished.add(OTHER_REQUEST)
    engine.running = {}
    engine.publish()
    for client in clients:
        events = [client.receive(), client.receive()]
        reasons = {event["id"]: event["reason"] for event in events}
        assert reasons == {REQUEST: "suspended", OTHER_REQUEST: "finished"}
        client.close()
