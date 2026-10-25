# 03. The Hugging Face `generate` baseline (Lesson A3)

Everything we build later is compared with these numbers. Setup: TinyLlama-1.1B, FP16, T4, greedy decoding, prompt 128 tokens, 128 new tokens, identical prompts across the batch, 5 trials per point (ranges in [A3_hf_baseline.json](../../results/A3_hf_baseline.json)).

## Definitions (kept for our own engine)

- **TTFT**: time to first token = prefill + one token. Measured with a 1-token `generate`.
- **Step latency**: `(time for N tokens - TTFT) / (N - 1)`. Every sequence in the batch gets one token per step, so this is also the inter-token latency.
- **Decode tokens/s** = `batch / step latency`.
- **MBU**: `steps/s x weight bytes / measured bandwidth` (A2: 2.2 GB, 241 GB/s).

## Results

| Batch | TTFT (ms) | Step latency (ms), min / median / max | Decode tok/s | A2 ceiling tok/s | MBU | Peak GB |
|---|---|---|---|---|---|---|
| 1 | 39.7 | 30.71 / 30.90 / 31.15 | 32 | 110 | 30% | 2.22 |
| 2 | 44.3 | 30.97 / 31.54 / 32.00 | 63 | 219 | 29% | 2.24 |
| 4 | 76.6 | 31.80 / 32.42 / 35.01 | 123 | 438 | 28% | 2.27 |
| 8 | 163.3 | 31.66 / 31.93 / 32.01 | 251 | 876 | 29% | 2.32 |
| 16 | 331.2 | 33.37 / 33.75 / 33.99 | 474 | 1753 | 27% | 2.44 |
| 32 | 694.2 | 52.01 / 52.10 / 52.16 | 614 | 3506 | 18% | 2.67 |

Prefill at batch 1 (TTFT min / median / max): 32 tokens 36.5 / 41.5 / 55.1 ms, 128 tokens 37.6 / 38.3 / 41.4 ms, 512 tokens 119.7 / 120.9 / 121.4 ms, 1024 tokens 324.5 / 326.2 / 328.7 ms. Estimated throughput (2 x 969M matmul parameters x tokens / TTFT) is only 1.5, 6.5, 8.2 and 6.1 TFLOPS, against the roughly 19.4 TFLOPS sustained roof from A2.

With `use_cache=False` at batch 1 the step latency is 42.9 ms instead of 30.9 ms (1.4x slower, only 2 trials).

## What the numbers say

1. **Batch-1 decode reaches 32 tokens/s, about 30% of the 110 tokens/s bandwidth ceiling.** The GPU is not the limit.
2. **The step latency stays almost flat (30.9 to 33.8 ms) from batch 1 to 16,** so throughput scales almost linearly (32 to 474 tokens/s). A step that costs the same whether it moves 2.2 GB for 1 sequence or 16 sequences is mostly a fixed cost. The likely cause is per-step overhead from Python and the many small kernel launches across 22 layers, which is consistent with the data but not proven here (we have not profiled). We will test it in Part F (CUDA graphs).
3. **At batch 32 the step jumps to 52 ms** (MBU 18%); something that does scale with batch (attention over more KV cache, more compute) starts to matter. We will see in Part F.
4. **The KV cache helps only modestly here (1.4x)** because the context is short (128 to 256 tokens) and the cached step is overhead-bound. The cache saves work that grows with context length, so the gap widens for long sequences. We will measure that curve in C4.
5. **TTFT grows roughly linearly with batch x prompt tokens** (batch 8: 163 ms, batch 32: 694 ms). Larger batches improve throughput but make the first token wait: latency and throughput trade off.
6. **Prefill is well below the compute roof** (6 to 8 TFLOPS against about 19). The cause is unknown; short prompts are dominated by the roughly 37 ms floor. This is an open question for our own implementation.

## Targets for our engine

- Batch 1 decode: beat **32 tokens/s** (30% MBU).
- Batch 32 decode: beat **614 tokens/s**.
- Remember HF's static batching: all rows share one length and the batch finishes when the slowest does. Continuous batching (Part D) removes that.

## Code map

- [hf_baseline.py](../../bench/hf_baseline.py): `make_prompt_ids`, `timed_generate`, `measure_ttft_ms`, `run_case`.
- [roofline.py](../../tinyserve/roofline.py): `model_bandwidth_utilization` added.
- [test_a3_bench.py](../../tests/test_a3_bench.py): tiny-model tests (exact output length, fields, ordering); 16 tests pass with the A2 tests.
- [A3 notebook](../../notebooks/A3_hf_baseline.ipynb).

## Interview questions

1. **Define TTFT, inter-token latency and throughput, and how do you measure TTFT separately from decode?**
   TTFT is the wait until the first token (prefill plus one token). Inter-token latency is the time between later tokens. Throughput is total tokens per second across all requests. Measure TTFT with a 1-token generation; get step latency from `(N-token time - TTFT) / (N - 1)`, with a device sync around the timed region.

2. **HF decode runs at 30% of the bandwidth ceiling and its step time barely changes from batch 1 to 16. What does that tell you?**
   The step is dominated by a cost that does not depend on batch size, likely CPU and launch overhead from many small kernels, so the GPU is idle part of the time. More batching is nearly free throughput until the real limits appear (batch 32 here). The fixes are fewer and fused kernels and CUDA graphs, which replay a whole step with one launch.

3. **Why does a bigger batch raise throughput but hurt latency, and what does static batching add to the problem?**
   Weights are read once per step for the whole batch, so tokens per second grows, but a larger batch means more prefill work before anyone's first token (TTFT 40 ms at batch 1 vs 694 ms at batch 32). With static batching all sequences run until the longest finishes and new requests wait for the next batch; continuous batching admits and retires requests at every step.
