import asyncio
import time

import httpx
import psutil
import pytest

from control import ControlClient, Server, cancel, proxy_id, wait_for_event, wait_for_tokens
from engines import StreamResult, draft_model, running_engine

PORT = 8112
MAX_RUNNING = 16
CHUNK_TOKENS = 1024
DETERMINISTIC_CHUNK_ALIGNMENT = 256
STEP_SLEEP_SECONDS = 0.25
RESUME_PREFILL_TOLERANCE = 4
RESUMED_TEXT_CHECK_CHARACTERS = 100
PAUSED_PREFILL_OUTPUT_TOKENS = 40
SUSPENDED_PROMPT_CHUNKS = 8


@pytest.fixture(scope="module")
def server(engine, tmp_path_factory):
    if engine != "sglang":
        pytest.skip("covers the sglang patch only")
    arguments = [
        *("--speculative-algorithm", "STANDALONE"),
        *("--speculative-draft-model-path", draft_model()),
        *("--speculative-num-steps", "3", "--speculative-eagle-topk", "1"),
        *("--speculative-num-draft-tokens", "4"),
        *("--chunked-prefill-size", str(CHUNK_TOKENS), "--prefill-decode-interval", "8"),
        *("--tokenizer-worker-num", "2"),
        *("--enable-hierarchical-cache", "--hicache-size", "4"),
        *("--hicache-storage-backend", "file"),
    ]
    directory = tmp_path_factory.mktemp("production")
    environment = {
        "SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE": str(DETERMINISTIC_CHUNK_ALIGNMENT),
        "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(directory / "kv-cache"),
    }
    with running_engine(engine, PORT, MAX_RUNNING, arguments, directory, environment) as (
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


def test_single_running_event_for_chunked_prefill(server):
    decoding_id = proxy_id()
    long_id = proxy_id()
    long_prompt = "Tell me about the history of computing. " * 850

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            with ControlClient(server.socket_path) as control:
                decoding = StreamResult()
                decoding_task = asyncio.create_task(
                    server.chat(client, decoding_id, "Count to one thousand.", 2000, decoding)
                )
                await wait_for_tokens([decoding], 5)
                await server.chat(client, long_id, long_prompt, 5)
                finished = await wait_for_event(
                    control, event="stopped", id=long_id, reason="finished"
                )
                long_events = [event for event in control.events if event["id"] == long_id]
                await cancel([decoding_task])
        assert [event["event"] for event in long_events] == ["running"]
        tokens = finished["tokens"]
        assert tokens["prompt"] > CHUNK_TOKENS
        assert tokens["prefill_count"] == 1
        assert tokens["cache_hit"] + tokens["prefill"] == tokens["prompt"]
        assert long_events[0]["tokens"] == {**tokens, "output": long_events[0]["tokens"]["output"]}

    asyncio.run(scenario())


def test_short_prompt_during_long_prefill(server):
    long_ids = [proxy_id(), proxy_id()]
    short_id = proxy_id()
    long_prompts = [
        f"Request {request_id}. " + "Describe the history of astronomy in detail. " * 800
        for request_id in long_ids
    ]

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            await client.post(
                f"{server.url}/slow_down", json={"forward_sleep_time": STEP_SLEEP_SECONDS}
            )
            try:
                with ControlClient(server.socket_path) as control:
                    results = [StreamResult() for _ in long_ids]
                    first_task = asyncio.create_task(
                        server.chat(client, long_ids[0], long_prompts[0], 5, results[0])
                    )
                    await wait_for_event(control, event="running", id=long_ids[0])
                    second_task = asyncio.create_task(
                        server.chat(client, long_ids[1], long_prompts[1], 5, results[1])
                    )
                    short = await server.chat(client, short_id, "Say hello.", 5)
                    await asyncio.gather(first_task, second_task)
            finally:
                await client.post(f"{server.url}/slow_down", json={"forward_sleep_time": 0})
        return results, short

    (first, second), short = asyncio.run(scenario())
    assert short.token_times[0] < first.token_times[0] - 2 * STEP_SLEEP_SECONDS
    assert second.token_times[0] > first.token_times[0]
    assert [first.completion_tokens, second.completion_tokens, short.completion_tokens] == [5] * 3


async def slowed_down(client: httpx.AsyncClient, server: Server, scenario):
    await client.post(f"{server.url}/slow_down", json={"forward_sleep_time": STEP_SLEEP_SECONDS})
    try:
        return await scenario()
    finally:
        await client.post(f"{server.url}/slow_down", json={"forward_sleep_time": 0})


async def isolated_chat(
    client: httpx.AsyncClient,
    server: Server,
    request_id: str,
    prompt: str,
    max_tokens: int,
    result: StreamResult | None = None,
) -> StreamResult:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "cache_salt": request_id,
    }
    result = StreamResult() if result is None else result
    return await server.stream(client, "/v1/chat/completions", request_id, body, result)


def test_medium_prompt_during_long_prefill(server):
    """A prompt of a few chunks must not wait for the whole prefill of a longer prompt (sglang
    prefills one chunked prompt at a time): the longer prompt pauses at a chunk boundary,
    continues from its cached prefix afterwards and counts its prefill tokens once."""
    long_id = proxy_id()
    medium_id = proxy_id()
    long_prompt = f"Request {long_id}. " + "Describe the history of astronomy in detail. " * 800
    medium_prompt = f"Request {medium_id}. " + "Summarize the plot of a famous novel. " * 300

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            with ControlClient(server.socket_path) as control:

                async def requests():
                    long = StreamResult()
                    long_task = asyncio.create_task(
                        isolated_chat(
                            client, server, long_id, long_prompt, PAUSED_PREFILL_OUTPUT_TOKENS, long
                        )
                    )
                    await wait_for_event(control, event="running", id=long_id)
                    medium = await isolated_chat(client, server, medium_id, medium_prompt, 5)
                    await long_task
                    return long, medium

                long, medium = await slowed_down(client, server, requests)
                finished = await wait_for_event(
                    control, event="stopped", id=long_id, reason="finished"
                )
                long_events = [event["event"] for event in control.events if event["id"] == long_id]
        return long, medium, finished["tokens"], long_events

    long, medium, tokens, long_events = asyncio.run(scenario())
    assert medium.token_times[0] < long.token_times[0] - 2 * STEP_SLEEP_SECONDS
    assert long_events == ["stopped", "running"]
    assert long.completion_tokens == PAUSED_PREFILL_OUTPUT_TOKENS
    assert tokens["prefill_count"] == 2
    assert tokens["prefill"] <= tokens["prompt"]


def test_suspension_during_chunked_prefill(server):
    """A suspended prompt must stop prefilling at the next chunk boundary instead of finishing its
    prefill: the proxy suspends it to give the engine to other requests. After the resume it
    continues from its cached prefix and produces the same tokens as without the suspension."""
    long_id = proxy_id()
    long_prompt = f"Request {long_id}. " + "Explain in detail how a lighthouse lens works. " * 800

    async def scenario():
        async with httpx.AsyncClient(timeout=120) as client:
            with ControlClient(server.socket_path) as control:

                async def requests():
                    result = StreamResult()
                    task = asyncio.create_task(
                        isolated_chat(
                            client,
                            server,
                            long_id,
                            long_prompt,
                            PAUSED_PREFILL_OUTPUT_TOKENS,
                            result,
                        )
                    )
                    await wait_for_event(control, event="running", id=long_id)
                    assert control.suspend([long_id]) is None
                    suspended = await wait_for_event(
                        control, event="stopped", id=long_id, reason="suspended"
                    )
                    await asyncio.sleep(2 * SUSPENDED_PROMPT_CHUNKS * STEP_SLEEP_SECONDS)
                    resume_time = time.monotonic()
                    assert control.resume([long_id]) is None
                    resumed = await wait_for_event(control, event="running", id=long_id)
                    await task
                    return result, resume_time, suspended["tokens"], resumed["tokens"]

                result, resume_time, before, after = await slowed_down(client, server, requests)
            reference = await isolated_chat(
                client, server, proxy_id(), long_prompt, PAUSED_PREFILL_OUTPUT_TOKENS
            )
        return result, resume_time, before, after, reference

    result, resume_time, before, after, reference = asyncio.run(scenario())
    assert result.token_times[0] - resume_time > 3 * STEP_SLEEP_SECONDS
    assert result.text == reference.text
    assert after["prefill_count"] == 2
    assert after["prefill"] == before["prefill"]
    assert after["cache_hit"] - before["cache_hit"] >= CHUNK_TOKENS
