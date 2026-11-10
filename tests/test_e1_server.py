"""E1 protocol and lifecycle tests. Execute in Colab, never locally."""
import asyncio
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("transformers")
httpx = pytest.importorskip("httpx")

from test_d5_engine import ScriptedModel, engine_for
from tinyserve.server.app import ChatDriver, _TextChunks, create_app
from tinyserve.server.local import serve_local

MODEL_ID = "test-tinyllama"


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        import torch

        assert kwargs["tokenize"] and kwargs["add_generation_prompt"]
        assert kwargs["return_tensors"] == "pt" and not kwargs["return_dict"]
        return torch.tensor([[1, 7]], dtype=torch.int64)

    def decode(self, tokens, **kwargs):
        pieces = {8: "caf", 9: "\u00e9", 10: " ", 11: "\u732b", 12: "\n"}
        return "".join(pieces.get(int(token), "word ") for token in tokens)


def payload(**options):
    return {"model": MODEL_ID, "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4, **options}


def app_for(model=None, max_requests=32):
    engine = engine_for(model, blocks=16, batch_tokens=3)
    return create_app(engine, Tokenizer(), MODEL_ID, max_requests=max_requests), engine


def events(response):
    return [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]


def test_health_models_nonstream_usage_and_stream_shape():
    async def scenario():
        app, engine = app_for()
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                assert (await client.get("/health")).status_code == 200
                assert (await client.get("/v1/models")).json()["data"][0]["id"] == MODEL_ID
                reply = await client.post("/v1/chat/completions", json=payload())
                assert reply.status_code == 200
                result = reply.json()
                assert result["object"] == "chat.completion"
                assert result["choices"][0]["message"] == {"role": "assistant", "content": "caf\u00e9 \u732b"}
                assert result["choices"][0]["finish_reason"] == "length"
                assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}
                streamed = await client.post("/v1/chat/completions", json=payload(stream=True))
                assert streamed.status_code == 200
                assert streamed.headers["content-type"].startswith("text/event-stream")
                records = events(streamed)
                assert records[-1] == "[DONE]" and records.count("[DONE]") == 1
                chunks = [json.loads(record) for record in records[:-1]]
                assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
                assert chunks[-1]["choices"][0]["delta"] == {}
                assert chunks[-1]["choices"][0]["finish_reason"] == "length"
                assert all(chunk["id"] == chunks[0]["id"] for chunk in chunks)
                assert all(chunk["created"] == chunks[0]["created"] for chunk in chunks)
                assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
                text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
                assert text == result["choices"][0]["message"]["content"]
                assert not app.state.driver.jobs
                assert engine.cache.num_free_blocks == engine.cache.num_blocks
        with pytest.raises(RuntimeError, match="closed"):
            engine.step()
    asyncio.run(scenario())


@pytest.mark.parametrize("changes", [
    {"model": "other"}, {"messages": []}, {"messages": "hello"},
    {"messages": [{"role": "tool", "content": "hi"}]},
    {"messages": [{"role": "user", "content": []}]},
    {"messages": [{"role": "user", "content": " "}]},
    {"messages": [{"role": "user", "content": "hi", "name": "extra"}]},
    {"messages": [{"role": "assistant", "content": "hi"}]},
    {"max_tokens": 0}, {"max_tokens": 129}, {"max_tokens": True},
    {"max_tokens": 1.5}, {"temperature": -1}, {"top_p": 0}, {"top_p": True},
    {"stream": "true"}, {"n": 2}, {"n": True}, {"seed": -1},
    {"seed": True}, {"seed": 2**63}, {"stop": ["word"]}, {"tools": []},
])
def test_invalid_or_unsupported_options_raise_before_engine_admission(changes):
    async def scenario():
        app, engine = app_for()
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json=payload(**changes))
                assert response.status_code == 400
                assert response.json()["error"]["type"] == "invalid_request_error"
                assert not engine.scheduler.requests and not engine.has_pending_requests
                assert app.state.driver.accepted == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("body,status", [(b"{bad", 400), (b"\xff", 400),
                                       (b"x" * (64 * 1024 + 1), 413), (b"[]", 400)])
def test_malformed_and_oversized_bodies(body, status):
    async def scenario():
        app, _ = app_for()
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", content=body)
                assert response.status_code == status and "error" in response.json()
    asyncio.run(scenario())


def test_context_overflow_does_not_consume_quota():
    async def scenario():
        app, engine = app_for()
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json=payload(max_tokens=32))
                assert response.status_code == 400
                assert "context" in response.json()["error"]["message"]
                assert app.state.driver.accepted == 0
                assert not engine.scheduler.requests
    asyncio.run(scenario())


def test_lifetime_quota_is_explicit_not_unbounded_history_retention():
    async def scenario():
        app, _ = app_for(max_requests=1)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                assert (await client.post("/v1/chat/completions", json=payload())).status_code == 200
                response = await client.post("/v1/chat/completions", json=payload())
                assert response.status_code == 429
                assert response.json()["error"]["type"] == "rate_limit_error"
    asyncio.run(scenario())


def test_eos_maps_to_openai_stop_and_releases_work():
    async def scenario():
        app, engine = app_for(ScriptedModel(eos=8, constant_token=8))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json=payload(stream=True))
                chunks = [json.loads(record) for record in events(response)[:-1]]
                assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
                assert len(engine.get_output(0).generated_tokens) == 1
                assert engine.cache.num_free_blocks == engine.cache.num_blocks
    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_engine_failure_is_visible_and_driver_fails_closed(stream, caplog):
    async def scenario():
        model = ScriptedModel()
        model.fail = True
        app, engine = app_for(model)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json=payload(stream=stream))
                if stream:
                    records = events(response)
                    assert "[DONE]" not in records
                    assert json.loads(records[-1])["error"]["type"] == "server_error"
                else:
                    assert response.status_code == 503
                    assert response.json()["error"]["type"] == "server_error"
                assert (await client.get("/health")).status_code == 503
                assert (await client.post("/v1/chat/completions", json=payload())).status_code == 503
                assert engine.cache.num_free_blocks == engine.cache.num_blocks
                assert not engine.has_pending_requests
        assert "Inference driver failed" in caplog.text
    asyncio.run(scenario())


def test_driver_shares_one_engine_and_bounds_slow_consumer_queue():
    async def scenario():
        import torch

        engine = engine_for(blocks=8, batch_tokens=4)
        driver = ChatDriver(engine)
        await driver.start()
        try:
            first = driver.submit(torch.tensor([1, 7]), 4, 0.0, 1.0, None)
            second = driver.submit(torch.tensor([1, 12]), 4, 0.0, 1.0, None)
            async def finished():
                while engine.has_pending_requests:
                    await asyncio.sleep(0.001)
            await asyncio.wait_for(finished(), timeout=5)
            for job in (first, second):
                assert job.queue.qsize() == 1
                output = job.queue.get_nowait()
                assert output.finished and len(output.generated_tokens) == 4
                driver.discard(job)
            assert any(len(set(call[2].tolist())) == 2 for call in engine.model.calls)
            assert not driver.jobs
        finally:
            await driver.stop()
    asyncio.run(scenario())


def test_driver_cancellation_and_shutdown_release_waiting_and_pending_work():
    async def scenario():
        import torch

        engine = engine_for(blocks=16)
        driver = ChatDriver(engine)
        await driver.start()
        first = driver.submit(torch.tensor([1, 7]), 20, 0.0, 1.0, None)
        second = driver.submit(torch.tensor([1, 12]), 20, 0.0, 1.0, None)
        driver.discard(second)
        assert engine.get_output(second.request_id).finish_reason == "cancelled"
        await asyncio.sleep(0.002)
        await driver.stop()
        assert engine.get_output(first.request_id).finish_reason == "cancelled"
        assert not engine.has_pending_requests
        assert engine.cache.num_free_blocks == engine.cache.num_blocks
    asyncio.run(scenario())


def test_text_chunks_handle_word_boundaries_unicode_and_flush():
    import torch

    decoder = _TextChunks(Tokenizer())
    decoder.put(torch.tensor([8]))
    assert decoder.take() == ""
    decoder.put(torch.tensor([9]))
    assert decoder.take() == ""
    decoder.put(torch.tensor([10, 11]))
    first = decoder.take()
    decoder.end()
    assert first + decoder.take() == "caf\u00e9 \u732b"


def test_actual_loopback_stream_disconnect_and_shutdown():
    async def scenario():
        app, engine = app_for()
        async with serve_local(app) as url:
            assert url.startswith("http://127.0.0.1:")
            async with httpx.AsyncClient(base_url=url, timeout=10, trust_env=False) as client:
                assert (await client.get("/health")).status_code == 200
                async with client.stream("POST", "/v1/chat/completions",
                                         json=payload(stream=True, max_tokens=20)) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            assert json.loads(line[6:])["choices"][0]["delta"]["role"] == "assistant"
                            break
                async def cancelled():
                    while app.state.driver.jobs:
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(cancelled(), timeout=5)
                assert engine.get_output(0).finish_reason == "cancelled"
                assert engine.cache.num_free_blocks == engine.cache.num_blocks
        assert not engine.has_pending_requests
    asyncio.run(scenario())
