import asyncio

import httpx
import psutil
import pytest

from control import ControlClient, Server, proxy_id, wait_for_event, wait_for_tokens
from engines import StreamResult, draft_model, running_engine

PORT = 8112
MAX_RUNNING = 16
CHUNK_TOKENS = 4096
RESUME_PREFILL_TOLERANCE = 4
RESUMED_TEXT_CHECK_CHARACTERS = 100


@pytest.fixture(scope="module")
def server(engine, tmp_path_factory):
    if engine != "sglang":
        pytest.skip("the vLLM patch keeps the KV of suspended requests with speculative decoding")
    arguments = [
        *("--speculative-algorithm", "STANDALONE"),
        *("--speculative-draft-model-path", draft_model()),
        *("--speculative-num-steps", "3", "--speculative-eagle-topk", "1"),
        *("--speculative-num-draft-tokens", "4"),
        *("--chunked-prefill-size", str(CHUNK_TOKENS), "--prefill-decode-interval", "1"),
    ]
    directory = tmp_path_factory.mktemp("speculative")
    with running_engine(engine, PORT, MAX_RUNNING, arguments, directory) as (
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


def test_resume_from_prefix_cache(server):
    prompt = "Write a long story about a lighthouse keeper and the ships she guides home."
    request_id = proxy_id()

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            with ControlClient(server.socket_path) as control:
                result = StreamResult()
                task = asyncio.create_task(server.chat(client, request_id, prompt, 300, result))
                await wait_for_event(control, event="running", id=request_id)
                await wait_for_tokens([result], 50)
                assert control.suspend([request_id]) is None
                suspended = await wait_for_event(
                    control, event="stopped", id=request_id, reason="suspended"
                )
                await asyncio.sleep(0.5)
                checked_length = len(result.text) + RESUMED_TEXT_CHECK_CHARACTERS
                assert control.resume([request_id]) is None
                resumed = await wait_for_event(control, event="running", id=request_id)
                await task
            reference = await server.chat(client, proxy_id(), prompt, 300)
        before, after = suspended["tokens"], resumed["tokens"]
        assert before["output"] >= 50
        assert after["prefill_count"] == 2
        assert after["prefill"] - before["prefill"] <= RESUME_PREFILL_TOLERANCE
        assert (
            after["cache_hit"] - before["cache_hit"]
            >= before["prompt"] + before["output"] - RESUME_PREFILL_TOLERANCE
        )
        assert result.completion_tokens == 300
        assert result.text[:checked_length] == reference.text[:checked_length]

    asyncio.run(scenario())
