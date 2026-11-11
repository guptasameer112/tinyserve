# 22. Measure the engine against Hugging Face (E2)

Code: [engine_vs_hf.py](../../bench/engine_vs_hf.py).
Tests: [test_e2_bench.py](../../tests/test_e2_bench.py).
Notebook: [E2_engine_vs_hf.ipynb](../../notebooks/E2_engine_vs_hf.ipynb).

## Status and scope

E2's complete user-supplied T4 run at source `92de867` passed on November 11,
2026: all 119 D5/E2 tests passed, with no skips. Complete evidence is archived
in [E2_engine_vs_hf.json](../../results/E2_engine_vs_hf.json). E1's separate run is
archived in [E1_streaming_server.json](../../results/E1_streaming_server.json).
It does not establish E2 performance.

The user selected a **direct comparison**, not HTTP timing. TinyServe uses
the public D5 engine; HF uses cached greedy `generate()` with DynamicCache.
Both run on the same device, with the same weight precision, prompt IDs and
output count. The notebook copies the HF weights strictly into TinyServe.
The installed HF attention implementation is recorded, not assumed equal
to TinyServe's PyTorch paged-gather implementation.

This is a finite simultaneous-arrival cohort, not a loaded production server.
There is no network, SSE, tokenizer, text decoder, arrival-rate sweep,
public binding or deployment change.

## Define the boundaries before measuring

An **output token** is a selected token ID, including any EOS ID. EOS stopping
is disabled for this benchmark so every request produces exactly `G` outputs.
It is a fixed-work experiment, not ordinary conversational stopping behavior.

All `B` requests share a start time. TinyServe starts timing before admission;
HF starts before `generate()`. The prompt batch is already on the GPU.
CUDA is synchronized before starting and whenever output IDs are observed.
These are instrumented wall-clock measurements, including Python overhead
and synchronization, not isolated GPU kernel times.

| Metric | Definition |
|---|---|
| TTFT | Time from common start until the first output token ID is available |
| Request latency | Time from common start until that request's last output token ID is available |
| Inter-token latency (ITL) | Difference between consecutive output-ID timestamps for one request |
| Total time | Time until the full generation call/engine drain returns, followed by CUDA synchronization |
| Output throughput | `B * G / total_seconds`; prompt tokens are not counted |

Request latency ends at **last ID readiness**, not receipt of an HTTP finish
record. Total time additionally includes final return/drain work. All requests
arrive together; a later TinyServe request can wait behind earlier chunked
prefill, and that wait belongs in its TTFT.

Example: two requests produce three tokens each. Their relative timestamps
are `[0.1, 0.3, 0.5]` and `[0.2, 0.4, 0.7]` seconds; the full run takes 0.8 s.

- TTFT: 100 ms and 200 ms.
- Request latency: 500 ms and 700 ms.
- ITL: `[200, 200]` ms and `[200, 300]` ms.
- Throughput: six outputs / 0.8 s = 7.5 output tokens/s.

For `G=1`, TTFT equals request latency and there is no ITL. We store `null`
for an absent ITL summary rather than dividing by zero or inventing a value.
Invalid, nonfinite or backwards timing raises an error; it is never clamped.

## Observe one full run

A3 estimated decode latency by subtracting a separate one-output run from
a full run. Those runs can have different noise. E2 instead records every
output timestamp **within the same full run**.

TinyServe records the updates returned by `engine.step()`. Partial prefill
does not create an output timestamp. Requests updated in the same step get
the same observation time. HF receives a lightweight streamer: its first
`put()` is the input prompt batch and is skipped; each later `put()` contains
one selected token per batch row. No text decoding runs in that callback.
The returned HF sequence is checked against the stream's recorded IDs.

This observer costs time. Synchronization and Python callbacks can affect
the two implementations differently. Do not compare these numbers directly
to an uninstrumented `generate()` run, claim HTTP TTFT, or interpret a callback
timestamp as the exact instant a CUDA kernel selected a token.

## Keep work and memory explicit

Prompts have equal lengths and no padding. Notebook batch rows are identical,
with no shared prefix cache. Every request still owns independent KV blocks.
Tests also use different rows to verify routing.

With prompt length `P` and output budget `G`, each request processes
`P + G - 1` positions. For block size `S`, the pool has
`B * ceil((P + G - 1) / S)` blocks. This rounds **per request**, not after
adding all requests together. Every request fits simultaneously; any
preemption or leaked block is an error in this no-pressure benchmark.

The notebook measures `P` in `{32, 128}`, `B` in `{1, 4}`, `G=16`, with
16-slot blocks and a 128-row forward budget. The largest pool is 36 FP16
blocks: `36 * 360448 = 12976128` bytes (12.375 MiB). This is only KV storage;
weights, activations, logits, gather temporaries and allocator memory remain.

TinyServe allocates a fresh fixed pool **before** timing each run, as an
already-started serving engine would. HF allocates its dynamic cache inside
`generate()`. This lifecycle asymmetry is explicitly recorded. It is not
a claim of equal cache-allocation work or equal attention kernels.

Both models remain resident while measuring. CUDA records baseline and peak
allocated bytes plus their difference. TinyServe's baseline includes its
preallocated pool; HF's does not. These are allocator allocation observations,
not driver-free memory, reserved bytes or standalone model memory. Freed
allocator blocks can be reused; the benchmark does not flush them per trial.

## Trials, correctness and conclusions

One full-length warmup runs per mode, outside reported measurements.
Three measured trials alternate TinyServe-first and HF-first order.
The result retains every trial and reports minimum, median, maximum and
sample count. TTFT and request-latency summaries pool requests across trials;
ITL summaries pool intervals. Those samples are correlated, not independent
confidence intervals. Three trials do not justify a production p99.

Each trial records `HF total time / TinyServe total time`. Values below one
mean TinyServe is slower. There is **no speedup gate**. This readable Python
engine is not expected to beat optimized HF kernels at every batch size.

The notebook first runs a short full-model FP32 greedy-parity gate, then
converts both models to FP16 for timing. Fixed token counts are always
required. FP16 token equality is recorded per request/trial, not assumed.
If trajectories diverge, compare fixed workload cost, not semantically
identical continuations or universal numerical equivalence. D5's precision
contract still applies; no new FP16 score tolerance is invented.

EOS configuration is restored even on error. HF gets a fresh generation
configuration with no EOS stopping or extra sampling processors. Modern HF
merges unset fields from `model.generation_config`, so the benchmark temporarily
uses the fresh policy as that default source too. It restores the original
object in `finally`, including after generation or timing failure; the original
policy's contents are not edited. No `use_model_defaults` argument is passed.
Tests install conflicting sampling/beam/penalty settings in the original policy
and check isolation, fixed work and restoration.
Engine cleanup closes and releases even a failed pending forward. Errors
propagate; no failed trial becomes a fabricated timing result.

## Recorded Colab result

The FP32 gate and every measured FP16 continuation matched HF. All timed
engine runs recorded zero preemptions and full block release. HF used SDPA.
All summary statistics were recomputed from the supplied raw trials.

Output throughput is shown as **minimum / median / maximum**:

| B | P | TinyServe output tokens/s | HF output tokens/s |
|---|---|---|---|
| 1 | 32 | 12.13 / 14.36 / 14.54 | 33.12 / 38.23 / 40.62 |
| 4 | 32 | 22.20 / 24.68 / 25.13 | 101.97 / 118.16 / 121.91 |
| 1 | 128 | 13.69 / 14.38 / 14.57 | 38.27 / 40.60 / 41.64 |
| 4 | 128 | 19.74 / 21.29 / 22.77 | 101.09 / 130.26 / 137.19 |

HF was faster in all four cases. Median TTFT was 71.79/29.07,
162.22/36.62, 89.22/29.18 and 394.10/72.22 ms (TinyServe/HF), respectively.
For the largest case, TinyServe TTFT spanned 68.32-663.79 ms across requests
and trials, consistent with staggered prefill readiness. Profiling would
be required to attribute the performance gap to individual operations.

The complete archive preserves original timestamp, source and environment.
Local checks remained static/data-only; no benchmark or model ran locally.
These results do not establish HTTP latency, production p99 or universal
FP16 token equality.

## Three interview questions

1. Can higher throughput mean worse TTFT?
   Yes. Batching can increase total outputs per second while later requests
   wait for shared prefill work.
2. Why measure output times within one run?
   It avoids subtracting unrelated noisy runs and preserves individual
   request delays and decode intervals.
3. What makes a fair comparison more than equal model names?
   Match weights, precision, prompts, output budgets and timing boundaries.
   Record backend, cache lifecycle, observer costs and token divergence.

Next: H packaging. After E2 and packaging, pause to review the entire core
project properly before any Parts F/G stretch work.
