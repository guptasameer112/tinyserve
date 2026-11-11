# Building an inference engine taught me more than winning a benchmark

TinyServe began with a small goal: understand every step between token IDs
and streamed model output. The target was TinyLlama-1.1B-Chat on a Colab T4.
The result is a readable, vLLM-inspired teaching engine, not a replacement
for vLLM or a production service.

## Keep the model interface stable

The model accepts flat token IDs, positions and attention metadata. The same
decoder can run uncached, with a contiguous KV cache, or through a paged-gather
reference path. Request ownership lives in metadata rather than padding.

That made correctness checks concrete. Independent requests must not attend
to each other. Scrambling physical block placement must not change logical
attention. Growing a request must preserve its existing prefix.

TinyLlama also makes memory arithmetic tangible. One FP16 token position
stores keys and values at every layer:

```text
2 * 22 layers * 4 KV heads * 64 dimensions * 2 bytes = 22528 bytes
```

The 32 query heads are not the KV-head count. A 16-slot block costs 360,448
bytes. Returning a block to the free list makes it reusable; it does not
shrink the already allocated GPU tensor.

## Scheduling is a state machine, not a for-loop optimization

The scheduler admits work in FCFS order, chunks prefill and mixes it with
one-token decode. A forward budget counts new model rows, not requests or
total cached context. Under pressure, it can release a newer request's blocks
and later rebuild that prefix from retained token IDs.

Rebuilding is not generating again. Replay must neither select another token
nor consume another sampling random draw. Likewise, reserving slots is not
the same as successfully writing a whole forward or acknowledging its result.
Failure/retry tests made those boundaries explicit.

One useful off-by-one rule is that `P` prompt tokens and `G>0` outputs require
`P+G-1` processed positions. The last selected output is returned but never
fed back through the model.

## Precision bugs need the right reference

An early full-model FP16 check failed against rounded-weight FP32. That alone
did not establish a paging bug. Adding a no-cache FP16 path showed that the
first failing paged and no-cache rows were identical; both differed from FP32.

The corrected contract separates paging correctness, sampled greedy parity
on the workload, and measured cross-precision drift. It does not invent a
universal FP16/FP32 score tolerance or claim that matching argmax proves equal
logits. The full [D5 evidence](../results/D5_engine.json) preserves that distinction.

## Streaming adds another boundary

E1 wraps the engine in a loopback-only Chat Completions subset. A single driver
owns the engine; arrivals and cancellation happen between complete steps.
The app validates unsupported options, buffers token fragments into text,
and reports stream errors without a fabricated success marker.

Even a disconnect test can race: a small model may finish before the client
closes after reading the role event. Completed work must retain `"length"`.
Deterministic socket tests verify both orderings. The real T4 demo cancelled
an unfinished request after three outputs and released all 16 blocks.

## The benchmark did not need a victory

E2 compared direct fixed-work generation against HF using the same copied
weights and prompt IDs. A short FP32 gate preceded four FP16 workloads.
Each had one warmup per mode and three alternating paired trials.
All measured continuations matched, but HF was faster in every case.

Observed output tokens/s, **minimum / median / maximum**:

| Requests | Prompt tokens | TinyServe | HF |
|---|---|---|---|
| 1 | 32 | 12.13 / 14.36 / 14.54 | 33.12 / 38.23 / 40.62 |
| 4 | 32 | 22.20 / 24.68 / 25.13 | 101.97 / 118.16 / 121.91 |
| 1 | 128 | 13.69 / 14.38 / 14.57 | 38.27 / 40.60 / 41.64 |
| 4 | 128 | 19.74 / 21.29 / 22.77 | 101.09 / 130.26 / 137.19 |

These are instrumented token-ID timings, not HTTP text latency. Synchronization
and Python observation cost time. TinyServe's fixed-pool setup is excluded;
HF's DynamicCache setup is included. HF used SDPA; TinyServe used PyTorch
paged gather. Both models remained resident during memory measurements.
Profiling would be needed to attribute the performance gap.

The [complete E2 result](../results/E2_engine_vs_hf.json) records source
`92de867`, a T4, PyTorch `2.14.1+cu130`, Transformers `5.22.1`, 119 passing
D5/E2 tests, every measured trial and full block release. These observations
do not establish production p99 or universal FP16 parity.

## What comes next is a review

The core now connects model math, generation, cache ownership, scheduling,
streaming and measurement. The [study guide](guide/README.md) and
[review checklist](CORE_REVIEW.md) make that path explicit.

Before prefix caching, Triton kernels or CUDA graphs, the next task is to
explain the core properly again. Knowing why a readable engine loses a
carefully defined benchmark is more useful than manufacturing a speedup.
