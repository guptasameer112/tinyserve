# Stop here: review the core project

E2 is validated. After H packaging is verified, **do not start Parts F/G yet**.
The user requested time to go through the whole core project properly first.
Use this checklist as a discussion guide, not a request to execute models locally.

## A. Start from tokens and hardware

- Explain token IDs, embeddings and why the final logits row predicts the next token.
- Trace one prompt through `(N,) -> (N, 2048) -> (N, 32000)`.
- Explain decode weight traffic, prefill reuse and why batching can improve throughput.
- Distinguish datasheet peaks from measured sustained bandwidth/compute.

## B. Rebuild the model mentally

- Explain RMSNorm, RoPE, causal attention, GQA and SwiGLU in order.
- Draw query/key/value shapes: 32 query heads, four KV heads, head dimension 64.
- Trace both residual additions in one decoder layer and the final model norm.
- Explain strict HF weight loading and what FP32 parity does and does not prove.

## C. Generate without changing semantics

- Explain greedy selection versus sampling and a random draw.
- Trace EOS, zero output budgets, seeded replay and strict context limits.
- Explain why causal K/V can be reused and why a cache does not store final text.
- Derive the difference between `G` returned outputs and `G-1` decode appends.

## D. Own and schedule memory correctly

- Map logical positions to physical blocks and slots, including scrambled tables.
- Calculate FP16 KV bytes/token: `2 * 22 * 4 * 64 * 2 = 22528`.
- Distinguish whole-block budget rounding, unused last-block slots and free blocks.
- Explain reserve, forward commit and scheduler acknowledgement separately.
- Walk through a new arrival, chunked prefill, mixed decode and newest-victim preemption.
- Explain why replay rebuilds K/V without sampling or consuming random draws.
- Trace forward failure/retry versus failure after committed logits.
- Explain why cancelling returns leases without shrinking fixed GPU tensors.

## E. Serve and measure honestly

- Trace messages through the chat template, admission, engine, token IDs and text.
- Explain SSE framing, Unicode/word buffering, errors after HTTP headers and `[DONE]`.
- Explain the disconnect/completion race and why `"length"` can remain correct.
- Define TTFT, ITL, last-ID latency, total duration and output throughput.
- Explain the 128-row forward budget versus total resident KV capacity.
- Interpret the four E2 cases, HF's advantage and the cache-lifecycle asymmetry.
- Distinguish observed token equality from universal numerical equivalence.

## Evidence and limits

For each claim, open the relevant [guide chapter](guide/README.md), source,
test and [complete recorded result](../results/README.md). Explain what the test
actually exercises. Suite counts overlap, and a passing static CI run is not
a new GPU runtime validation.

The loopback app is a finite teaching server, not a public production service.
Prefix caching, Triton attention, CUDA graphs, speculative decoding, another
model family and INT8 KV are not implemented core features.

Proceed to stretch work only after the user finishes this review and explicitly
chooses the next lesson. Do not mark the review complete merely because H built.
