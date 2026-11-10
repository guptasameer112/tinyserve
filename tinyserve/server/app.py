"""A bounded text-only Chat Completions subset over one D5 engine."""
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import TextStreamer

from tinyserve.engine.block_manager import _positive_integer
from tinyserve.sampling import validate_sampling_options

LOGGER = logging.getLogger(__name__)
MAX_BODY_BYTES = 64 * 1024


def error_body(message, kind="invalid_request_error"):
    return {"error": {"message": message, "type": kind, "param": None, "code": None}}


def validate_request(payload, model_id):
    """Reject unsupported options rather than silently pretending to implement them."""
    allowed = {"model", "messages", "max_tokens", "temperature", "top_p", "stream", "seed", "n"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("body must be an object containing only supported chat options")
    if payload.get("model") != model_id:
        raise ValueError("model must match the served model ID")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                or not isinstance(message["content"], str) or not message["content"].strip()):
            raise ValueError("each message needs a supported role and nonempty string content")
    if messages[-1]["role"] != "user":
        raise ValueError("the final message must have role user")
    max_tokens = payload.get("max_tokens", 16)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 128:
        raise ValueError("max_tokens must be an integer from 1 to 128")
    temperature, top_p = payload.get("temperature", 0.0), payload.get("top_p", 1.0)
    validate_sampling_options(temperature, 0, top_p)
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("stream must be a boolean")
    count = payload.get("n", 1)
    if isinstance(count, bool) or not isinstance(count, int) or count != 1:
        raise ValueError("only n=1 is supported")
    seed = payload.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63):
        raise ValueError("seed must be an integer from 0 to 2**63-1")
    return messages, max_tokens, temperature, top_p, stream, seed


@dataclass
class _Job:
    request_id: int
    prompt: torch.Tensor
    queue: asyncio.Queue


class ChatDriver:
    """All engine operations run synchronously on one event-loop thread.

    Queues retain the latest cumulative snapshot, so slow readers do not create
    unbounded token queues. A finite lifetime quota bounds D5's retained histories.
    """
    def __init__(self, engine, max_requests=32):
        self.engine = engine
        self.max_requests = _positive_integer("max_requests", max_requests)
        if engine.has_pending_requests or engine.run():
            raise ValueError("server requires a fresh engine with no retained requests")
        self.jobs = {}
        self.accepted = 0
        self.failure = None
        self._wake = asyncio.Event()
        self._task = None

    @staticmethod
    def _publish(job, value):
        if job.queue.full():
            job.queue.get_nowait()
        job.queue.put_nowait(value)

    async def start(self):
        if self._task is not None:
            raise RuntimeError("driver has already been started")
        self._task = asyncio.create_task(self._run())

    def submit(self, prompt, max_tokens, temperature, top_p, seed):
        if self.failure is not None:
            raise RuntimeError(self.failure)
        if self._task is None or self._task.done():
            raise RuntimeError("inference driver is not running")
        if self.accepted >= self.max_requests:
            raise OverflowError("lesson request quota reached; restart with a fresh engine")
        owner = self.accepted
        generator = None if seed is None else torch.Generator(device=prompt.device).manual_seed(seed)
        self.engine.add_request(owner, prompt, max_tokens, temperature=temperature,
                                top_p=top_p, generator=generator)
        job = _Job(owner, prompt, asyncio.Queue(maxsize=1))
        self.jobs[owner] = job
        self.accepted += 1
        self._wake.set()
        return job

    def discard(self, job):
        if self.jobs.pop(job.request_id, None) is None:
            return
        if self.failure is None and not self.engine.get_output(job.request_id).finished:
            self.engine.cancel(job.request_id)

    async def _run(self):
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                while self.engine.has_pending_requests:
                    for output in self.engine.step():
                        job = self.jobs.get(output.request_id)
                        if job is not None:
                            self._publish(job, output)
                    # Admit arrivals and process disconnects only between complete forwards.
                    await asyncio.sleep(0.001)
        except (RuntimeError, ValueError):
            LOGGER.exception("Inference driver failed; this engine will not accept more requests")
            self.failure = "Inference failed; restart the lesson server with a fresh engine."
        finally:
            self.failure = self.failure or "Server is shutting down."
            self.engine.close()
            for job in self.jobs.values():
                self._publish(job, RuntimeError(self.failure))

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class _TextChunks(TextStreamer):
    """Use HF's word/Unicode buffering instead of decoding each token in isolation."""
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=False, skip_special_tokens=True,
                         clean_up_tokenization_spaces=False)
        self.parts = []

    def on_finalized_text(self, text, stream_end=False):
        if text:
            self.parts.append(text)

    def take(self):
        text = "".join(self.parts)
        self.parts.clear()
        return text


def _chunk(completion_id, created, model_id, delta, finish_reason=None):
    return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def _sse(payload):
    return "data: " + json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n\n"


async def _next_output(job, request):
    while True:
        if await request.is_disconnected():
            return None
        try:
            return await asyncio.wait_for(job.queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue


def create_app(engine, tokenizer, model_id, max_requests=32):
    """Caller owns the loaded model. Lifespan exclusively owns this engine until close."""
    driver = ChatDriver(engine, max_requests)

    @asynccontextmanager
    async def lifespan(app):
        await driver.start()
        try:
            yield
        finally:
            await driver.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.driver = driver

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "error" if driver.failure else "ok"},
                            status_code=503 if driver.failure else 200)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model",
                                           "created": 0, "owned_by": "tinyserve"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if driver.failure:
            return JSONResponse(error_body(driver.failure, "server_error"), status_code=503)
        try:
            body = bytearray()
            async for part in request.stream():
                body.extend(part)
                if len(body) > MAX_BODY_BYTES:
                    return JSONResponse(error_body("request body exceeds 64 KiB"), status_code=413)
            payload = json.loads(body)
            messages, max_tokens, temperature, top_p, stream, seed = validate_request(payload, model_id)
            # return_dict=False keeps this a tensor across supported Transformers versions.
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=False,
            )[0].to(engine.model.get_input_embeddings().weight.device)
            job = driver.submit(prompt, max_tokens, temperature, top_p, seed)
        except (ValueError, UnicodeError) as error:
            return JSONResponse(error_body(str(error)), status_code=400)
        except OverflowError as error:
            return JSONResponse(error_body(str(error), "rate_limit_error"), status_code=429)
        except RuntimeError as error:
            LOGGER.exception("Chat admission failed")
            return JSONResponse(error_body(str(error), "server_error"), status_code=503)
        completion_id = "chatcmpl-tinyserve-%d" % job.request_id
        created = int(time.time())

        if not stream:
            try:
                while True:
                    output = await _next_output(job, request)
                    if output is None:
                        return JSONResponse(error_body("client disconnected"), status_code=499)
                    if isinstance(output, RuntimeError):
                        return JSONResponse(error_body(str(output), "server_error"), status_code=503)
                    if output.finished:
                        text = tokenizer.decode(output.generated_tokens, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
                        reason = "stop" if output.finish_reason == "eos" else output.finish_reason
                        return {"id": completion_id, "object": "chat.completion", "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                             "finish_reason": reason}],
                                "usage": {"prompt_tokens": prompt.numel(),
                                          "completion_tokens": len(output.generated_tokens),
                                          "total_tokens": len(output.token_ids)}}
            finally:
                driver.discard(job)

        async def events():
            decoder, emitted = _TextChunks(tokenizer), 0
            try:
                yield _sse(_chunk(completion_id, created, model_id, {"role": "assistant", "content": ""}))
                while True:
                    output = await _next_output(job, request)
                    if output is None:
                        return
                    if isinstance(output, RuntimeError):
                        yield _sse(error_body(str(output), "server_error"))
                        return
                    new_tokens = output.generated_tokens[emitted:]
                    emitted = len(output.generated_tokens)
                    decoder.put(prompt.new_tensor(new_tokens))
                    if output.finished:
                        decoder.end()
                    text = decoder.take()
                    if text:
                        yield _sse(_chunk(completion_id, created, model_id, {"content": text}))
                    if output.finished:
                        reason = "stop" if output.finish_reason == "eos" else output.finish_reason
                        yield _sse(_chunk(completion_id, created, model_id, {}, reason))
                        yield "data: [DONE]\n\n"
                        return
            finally:
                driver.discard(job)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
