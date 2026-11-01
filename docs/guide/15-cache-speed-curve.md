# 15. No-cache versus cached speed curve (C4)

Code: [cache_curve.py](../../bench/cache_curve.py).
Tests: [test_c4_bench.py](../../tests/test_c4_bench.py).
Plot: [plots.py](../../tinyserve/plots.py).
Notebook: [C4_cache_curve.ipynb](../../notebooks/C4_cache_curve.ipynb).

## The question

C3 established correctness. C4 asks how much time the cache saves for one
request, with different prompt lengths and generation budgets.

Compare our own model with itself. Keep weights, dtype, prompt IDs and
new-token count fixed. Change only `use_cache`. This is not an HF comparison,
a serving benchmark or an optimized-attention claim.

## What caching removes

For prompt length P and N new tokens, no-cache forwards process:

```
P, P+1, P+2, ..., P+N-1
```

Cached forwards process:

```
P, 1, 1, ..., 1
```

Total token rows processed by each decoder layer:

```
no cache: N*P + N*(N-1)/2
cached:   P + N - 1
```

For P=128 and N=32, this is 4,592 versus 159 token rows. That removes
repeated old-token projections, MLP work and attention computations.
It does **not** predict a 28.9x wall-clock speedup.

Large matrix operations can process many rows efficiently. One-token
operations have lower arithmetic intensity and still read model weights.
The cached query still reads the growing K/V prefix. Our reference GQA
also expands KV heads temporarily and uses Python validation/synchronization.
Less arithmetic does not translate directly to the same ratio of elapsed time.

## Two controlled sweeps

1. Fix N=32 and vary P across 16, 64, 128, 256 and 512.
2. Fix P=128 and vary N across 1, 8, 16 and 32.

Reuse the shared P=128/N=32 point instead of measuring it twice.
Each unique point uses three trials per mode and one full-length warmup
per mode. The default experiment has eight unique points.

For N=1, both modes process the prompt once. The cache additionally allocates
and writes storage, but has no later forward on which to reuse it.
Cache need not win here. With more generated tokens, the prefill cost is
spread over more decode steps.

## Fixed-length workloads and EOS

A normal chat can stop after eight tokens. Comparing that with a 32-token
continuation would confound speed with work performed.

The user approved temporarily disabling EOS stopping for this benchmark.
`run_case` first validates the original request/configuration, sets EOS to
None, warms up and times both modes, then restores EOS in `finally`.
It restores the original value even on failure.

The model can still select the EOS token ID; the benchmark simply continues
after it. This is synthetic fixed-length generation, not a chat-quality demo.
The normal generation API is unchanged. Do not run concurrent requests on
this temporarily modified benchmark model.

FP16 cached/no-cache token choices can differ through rounding. Record the
observed equality and token IDs; do not demand token-exact FP16 parity.
Both modes must still return exactly P+N IDs. C3's FP32 parity tests remain
the correctness evidence.

## Timing boundaries

Use `time.perf_counter()` around the public generation call. Synchronize
the input's CUDA device immediately before and after the measured call.
Without the ending synchronization, queued GPU work can outlive the timer.

The measurement includes request validation, prompt copying, cache allocation,
all forwards, greedy sampling, Python overhead and output assembly.
It excludes model loading, tokenization, warmup, JSON saving and plotting.
It does not measure server queueing, streaming, HTTP or network latency.

Do not synchronize separately after every token; that would change the
measured workload. Do not subtract independent prefill/full-run timings to
invent a precise decode latency. This lesson reports actual full-call time.

```
generated tokens/s = N / total_seconds
paired speedup    = no_cache_total_ms / cached_total_ms
```

Count new tokens only, not prompt tokens. Above 1 favors cache; below 1
means cache was slower in that paired trial.

## Ranges and trial order

Alternate order: no-cache/cached, cached/no-cache, no-cache/cached.
This reduces systematic order bias from temperature or clock drift; it
does not eliminate noise.

Retain every trial and report min/median/max. Compute speedup within each
trial pair, then summarize those ratios. The median paired ratio need not
equal the ratio of the two independent medians.

Shaded plot bands show observed min/max, not confidence intervals.
No minimum speedup is asserted. A flat curve or regression is a result to
investigate, not a reason to clamp values or loosen a test.

## Memory

The reserved contiguous K/V storage has capacity P+N:

```
cache storage bytes = (P+N) * 2 * layers * KV_heads * head_dim * dtype_bytes
TinyLlama FP16:       (P+N) * 22 KiB
```

The last generated token is not processed, but capacity still reserves its
slot. Report this theoretical storage separately from measured memory.

CUDA allocated peak includes model weights, prompt, outputs and temporary
tensors. Incremental peak subtracts allocated memory just before the call.
Neither is the same as allocator-reserved memory or total GPU process usage.
Cache can add persistent storage while reducing the full-prefix temporaries.
Do not interpret incremental peak as cache bytes alone.

## Evidence and status

Tests cover exact timing arithmetic with a controlled clock, alternating
order, fixed budgets, original EOS restoration on success/failure, invalid
requests/timings, a tiny real-model run, CUDA memory accounting and plots.
They run in Colab, not locally.

The notebook checkpoints each completed point as `status="partial"`, then
marks the final JSON complete only after all points and plots succeed.
An interrupted experiment is not a complete curve. Save the JSON, PNGs and
notebook outputs before replacing the runtime.

**Runtime validation and measured speedups are pending the user's Colab run.**
Local checks are static only. Next: D1, flattened batching and attention metadata.

## Interview questions

1. **Why can a cache be slower for one generated token?** There is no later
   decode forward to benefit; allocation and K/V writes add overhead.
2. **Why does fewer token rows not predict equal wall-clock speedup?**
   Matrix efficiency, weight/KV traffic, Python overhead and prefill still matter.
3. **What makes this comparison fair?** Same model/dtype/prompt and exact
   token counts, warmup, CUDA synchronization, alternating order and raw trials.
