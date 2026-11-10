# 21. Streaming chat over loopback HTTP (E1)

Code: [app.py](../../tinyserve/server/app.py),
[local.py](../../tinyserve/server/local.py).
Tests: [test_e1_server.py](../../tests/test_e1_server.py).
Notebook: [E1_streaming_server.ipynb](../../notebooks/E1_streaming_server.ipynb).

## Status and scope

E1 is prepared, not runtime-validated. D5 remains separately pending while
its full-TinyLlama FP16 numerical comparison is investigated. The E1 demo
uses FP32 and a small explicit KV pool. This is a transport/lifecycle lesson,
not evidence that D5's unresolved FP16 check passes.

The user approved loopback-only exposure: `127.0.0.1` inside Colab.
There is no public tunnel, authentication, TLS, CORS or deployment.
Do not bind this teaching app to `0.0.0.0` or expose it to other machines.
Other processes in the same Colab runtime can access it.

## Where HTTP fits

D5 accepts token IDs and returns cumulative token histories. A chat client
sends JSON messages and expects text. E1 connects those interfaces:

```text
HTTP JSON messages
  -> validate the supported options
  -> tokenizer.apply_chat_template(..., add_generation_prompt=True)
  -> prompt token IDs
  -> one shared D5 engine and D4 scheduler
  -> generated token IDs
  -> decoded text
  -> HTTP JSON or streamed SSE records
```

The chat template inserts the model's role markers and assistant prefix.
It is not the same input as simply concatenating message text.
The notebook's direct reference uses the exact same template and token IDs.

We reuse the engine. We do not load a model or allocate a KV pool for each
HTTP request. A separate engine per request would lose shared scheduling.

## A deliberately limited OpenAI-compatible subset

Endpoints:

| Method and path | Behavior |
|---|---|
| `GET /health` | 200 while healthy; 503 after an inference-driver failure |
| `GET /v1/models` | Lists the single configured model ID |
| `POST /v1/chat/completions` | JSON completion or streamed chat chunks |

Supported request fields:

- `model`: must exactly match the served model ID.
- `messages`: nonempty list of `{role, content}` objects. Roles are `system`,
  `user` and `assistant`; content is a nonempty string. The final role is `user`.
- `max_tokens`: integer 1 through 128, default 16. The model context and
  single-request pool capacity can impose tighter limits.
- `temperature`, `top_p`: C1 options, default greedy temperature zero and
  no nucleus filtering.
- `stream`: boolean, default false.
- `seed`: optional nonnegative integer below `2**63`, creating a per-request
  generator on the model device.
- `n`: only the integer 1.

Unknown options are rejected, not ignored. There is no tool calling,
multimodal content, custom stop sequence, logprobs, streaming usage option,
Responses API or legacy Completions endpoint. This is not a claim of full
OpenAI API compatibility or SDK certification.

Bodies are limited to 64 KiB. Invalid JSON, unsupported options and invalid
prompt/output budgets return an OpenAI-shaped error object with HTTP 400.
Oversized bodies return 413. The finite lesson lifetime quota returns 429.
Inference failure returns 503 when headers have not already been sent.

## What streaming means

Without streaming, the server waits for completion and returns one JSON
object with `choices[0].message`, `finish_reason` and token usage.

With streaming, the response has content type `text/event-stream`.
**Server-sent events (SSE)** are UTF-8 text records separated by a blank line:

```text
data: {"id":"chatcmpl-tinyserve-0","object":"chat.completion.chunk",...}

data: {"id":"chatcmpl-tinyserve-0","object":"chat.completion.chunk",...}

data: [DONE]

```

The abbreviated JSON above is a diagram, not a literal request to send.
Actual records contain:

1. An initial delta declaring `role: "assistant"`.
2. Zero or more deltas containing newly available text.
3. An empty delta with final `finish_reason`.
4. The literal `[DONE]` marker, only after successful completion.

The ID, creation timestamp and model remain stable within one response.
The choice index is always zero. Engine `"eos"` maps to API `"stop"`;
an exhausted output budget maps to `"length"`.

An SSE record is not a network packet or necessarily one model token.
TCP may split or combine records. The client must parse complete lines/events,
not assume every received byte buffer is one JSON object.

## Why not decode each token independently?

A token may represent part of a word or part of a Unicode character.
Decoding token fragments independently can insert incorrect spaces or
replacement characters.

The server uses the existing Transformers `TextStreamer` buffering logic,
overriding its output callback rather than printing to the terminal.
It accumulates token IDs, emits text at supported stable boundaries and
flushes remaining text at completion. Special tokens are skipped and
tokenization-space cleanup is disabled in both streaming and nonstreaming
decoding. This can delay the first text chunk until a word boundary.
It is not a promise of one visible text chunk per generated token.

## One owner for the engine

`ChatDriver` runs one asynchronous task. Within that task an engine forward
is synchronous. No second thread or task calls the engine during that forward.
Between successful steps, a small asynchronous yield allows HTTP handlers
to admit new requests or cancel disconnected ones.

This preserves D4's rule: arrivals and cancellations happen only between
completed forwards. It also permits different clients to share a packed
model forward and the same paged pool.

This teaching choice blocks the event loop during each forward. It is not
a production latency design; it does not make CUDA work nonblocking or
enable cancellation halfway through a kernel.

## Slow clients and retained histories

Each active HTTP job has a queue with capacity one. D5 outputs are cumulative
snapshots, so a newer snapshot can replace an unread older one without
losing generated token IDs. The consumer tracks how many IDs it has decoded
and decodes only the new suffix.

This prevents unbounded per-token queues. It can coalesce several tokens
into one visible chunk, which is correct SSE behavior.

D5 retains completed histories for inspection. Rather than silently retaining
unlimited GPU histories in a long-lived server, E1 accepts at most 32 requests
over one app lifetime by default. After that, it returns an explicit 429.
This is a finite lesson server, not a throughput-oriented service. To start
another session, create a fresh engine and app. The quota counts successful
admissions, including requests later cancelled, not malformed requests.

## Disconnects, failures and shutdown

When a client disconnects, the response generator's `finally` block removes
its job and cancels any unfinished engine request. Resident blocks become
available to other requests. Already completed requests need no cancellation.
Disconnect checks and cancellation cannot interrupt a running forward.

If engine execution raises, the driver logs the error, notifies all waiting
clients, closes the engine and rejects future requests. It does not silently
retry indefinitely or emit a fabricated completion.

After SSE headers are sent, the HTTP status cannot change to 503. Instead,
the server sends an SSE `error` object and closes the stream **without**
`[DONE]` or a successful finish chunk. Clients must treat that as failure.

Engine `close()` is a permanent abort, distinct from cancelling one request.
It releases all leases even when a failed D5 plan is pending, drops retained
pending logits and prevents further admission or stepping. It does not
acknowledge failed computation. The fixed pool's tensors remain allocated
while the engine is referenced.

The app's lifespan starts and stops the driver. The notebook's
`serve_local(app)` context binds an OS-assigned `127.0.0.1` port, waits for
Uvicorn readiness, then shuts down its task and socket in `finally`.
There is no detached server to remember to kill, second model-loading
process, fixed-port conflict or notebook event-loop patch.

## Tests and evidence

Tests cover JSON/SSE schemas, stable IDs, Unicode text reconstruction,
usage, EOS mapping, strict validation, byte limits, quota, shared forwards,
bounded queues, cancellation and fail-closed behavior.

ASGI in-process tests validate protocol behavior, but HTTPX's ASGI transport
can buffer streams. They do not prove real network incremental delivery.
A separate loopback-socket test reads the initial role event, disconnects
early, and checks cancellation and block release.

The notebook uses a fresh T4 runtime and real TinyLlama FP32. It verifies
health and model discovery over actual HTTP, streamed/nonstreamed text
against direct greedy generation, concurrent clients, invalid requests and
early disconnect. It records model-forward ownership and final pool state.
The server must stop before saving the complete result payload.
For finite, predictable overlapping workloads, the demo temporarily disables
EOS stopping and restores the original model setting on exit. Deterministic
tests separately verify the app's EOS-to-`stop` mapping and early release.

Runtime validation remains pending until the user's complete Colab outputs
are available. Local syntax, notebook JSON and whitespace checks are static
only. This lesson makes no throughput, TTFT or latency claim; E2 owns those
measurements.

## Three interview questions

1. Why is streaming not just returning one token per HTTP response?
   One response stays open. It carries a sequence of framed events;
   decoding and network buffering can group multiple tokens together.
2. How can disconnected clients release GPU cache safely?
   Cancel at an engine step boundary and release that request's leases.
   Do not mutate cache ownership while its forward is running.
3. How do you report failure after HTTP 200 has already been sent?
   Send an explicit stream error and terminate without the success marker.
   You cannot retroactively change the response status.

Next: E2, measured throughput, time to first token and latency versus HF.
