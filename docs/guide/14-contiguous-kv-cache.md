# 14. Contiguous KV cache (C3)

Code: [cache.py](../../tinyserve/cache.py),
[llama.py](../../tinyserve/models/llama.py),
[generation.py](../../tinyserve/generation.py).
Tests: [test_c3_cache.py](../../tests/test_c3_cache.py).
Notebook: [C3_kv_cache.ipynb](../../notebooks/C3_kv_cache.ipynb).

## Why the old K/V remain valid

Causal attention prevents a token from seeing later tokens. Appending a token
does not change an earlier token's embedding, hidden states, queries, keys or
values. Reuse the earlier K/V at every layer instead of recomputing them.

This assumes fixed model weights, eval mode, the same request and the same
positions. A cache is not valid across weight changes or unrelated prompts.
It is inference storage, not an autograd graph.

## Shapes and memory

TinyLlama has 22 layers, 32 query heads, 4 KV heads and head dimension 64.
Each of our two storage tensors has shape:

```
(22 layers, capacity tokens, 4 KV heads, 64 dimensions)
```

Store four KV heads, not 32 repeated heads. The reference GQA attention still
expands the visible prefix temporarily; that is not persistent cache storage.

Bytes per token, across both K and V and every layer:

```
2 * 22 * 4 * 64 * bytes_per_element
FP16: 22,528 bytes = 22 KiB
FP32: 45,056 bytes = 44 KiB
```

A 2,048-token FP16 allocation costs 44 MiB. The equivalent FP32 allocation
costs 88 MiB. This counts the two storage tensors, not model weights,
attention temporaries or allocator overhead.

Capacity is reserved in advance; `length` is the committed prefix. Do not
read `[length:capacity]`: it is uninitialized or belongs to an old request.
`reset()` sets length to zero without reallocating or zeroing storage.

## Prefill, then decode

For a three-token prompt and four generated tokens:

| Forward | Input token count | Absolute positions | Committed length after forward |
|---|---:|---|---:|
| Prefill | 3 | 0, 1, 2 | 3 |
| Decode | 1 | 3 | 4 |
| Decode | 1 | 4 | 5 |
| Decode | 1 | 5 | 6 |

Each forward produces the next token. The final generated token is returned,
but does not need another forward unless we want a further prediction.
Therefore cache length can be one less than returned sequence length.

Prefill stores every prompt token's K/V. Later forwards project only the new
token, attend to all visible cached keys and values, and append its K/V.
Queries are not cached. Keys are stored **after RoPE**, once, at their original
absolute positions. Values are not rotated.

Do not restart decode positions at zero. Do not rotate old keys again.

## Rectangular causal mask

The old no-cache mask was square. With a committed prefix of length 3 and
a new two-token chunk at positions 3 and 4:

```
                  key position
                0  1  2  3  4
query at 3      Y  Y  Y  Y  -
query at 4      Y  Y  Y  Y  Y
```

Visibility is `key_position <= query_position`. It is not a fresh
`tril(2, 5)`, which would hide most of the old prefix. For one-token decode,
the mask has shape `(1, prefix_length + 1)`.

## Integration and boundaries

`AttentionMetadata` carries the optional cache without changing the model's
three-argument forward. Each decoder layer has a stable storage index; that
index adds no parameters and does not change checkpoint keys.

`generate(..., use_cache=True)` creates a fresh cache for each request.
The default remains `use_cache=False`. Sampling, output dtype/device,
prompt immutability, EOS and the strict context policy remain unchanged.
Zero-token requests do not allocate a cache or run forward.

Direct cached forwards require eval mode and `torch.no_grad()`. Positions
must append consecutively from `cache.length`; capacity overflow and packed
sequence IDs raise before the decoder stack. The cache is bound to its model.
Recreate it after moving/casting the model; reset or recreate it after a
weight change. Do not share it between concurrent forwards.

Each layer writes an uncommitted suffix. The outer model advances length
only after its full forward succeeds. A failed later layer does not alter
the committed prefix; a retry overwrites the incomplete suffix.

This is a **single-request contiguous cache**. Packed no-cache forwards
still work. Paged storage and multi-request scheduling belong to Part D.

## Verification and current status

Tests cover full prefill, single-token and multi-token chunks, MHA/GQA/MQA,
cached/no-cache/HF logits, greedy tokens, seeded replay, boundaries, reset,
unchanged old entries, hidden unused storage and retry after a layer failure.
One CUDA test compares FP16 with the same rounded weights in FP32.

The notebook runs B8/C1/C2/C3 tests on a fresh Colab T4. It compares real
FP32 TinyLlama cached logits with both no-cache paths at every greedy step.
It separately compares generated sequences with HF's cached generation.

Full-model FP32 comparisons use `rtol=1e-5, atol=1e-4`, as in B8.
Changing GEMM row counts can change rounding; token parity is required
separately. FP16 uses `rtol=atol=1e-2` on a small controlled model, with
no token-exact FP16 claim.

Local checks are static only. Save notebook outputs and the JSON before
replacing a Colab runtime. No speed claim is made until C4 measures the
no-cache/cached curve.

### Recorded Colab run (2026-11-01)

The user-provided [C3_kv_cache.json](../../results/C3_kv_cache.json) records
a Tesla T4 run at commit `7b3969f`, with PyTorch `2.11.0+cu130` and
Transformers `5.21.0`.

- **139 B8/C1/C2/C3 tests passed; one test skipped**, because it requires
  two CUDA devices. The pytest return code is zero and stderr is empty.
- All 20 demonstrated greedy steps matched HF: 12 raw and eight chat.
  Full generated sequences also matched HF's cached generator.
- The raw prompt prefills six tokens, then forwards one token per step.
  It generates 12 tokens and ends with cache length 17 versus 18 returned IDs.
- The chat prompt prefills 41 tokens and generates eight, including EOS.
  It ends with cache length 48 versus 49 returned IDs.
- Cached/no-cache maximum logit differences range from zero to `2.10e-5`.
  Cached/HF maximum differences range from `7.63e-6` to `3.80e-5`.
  All recorded argmax checks passed.
- The FP32 demo reserves eight slots: `360,448` bytes total and `45,056`
  bytes per token (44 KiB). Its committed prefix contains six tokens.
- Reset reuses storage and gives zero logit difference in the demonstrated
  comparison. Zero budget, strict context errors and position errors pass.
- Seed-17 cached sampling replays exactly.

The JSON was archived from the user's pasted result, preserving its original
timestamp and environment. It validates the recorded tests and demos, not
unmeasured performance or every possible input.

## Interview questions

1. **Why can earlier K/V be reused?** Causality makes earlier hidden states
   independent of appended tokens, for fixed weights and positions.
2. **Why is the cache 22 KiB per token in FP16?** Both K and V, all 22
   layers, four KV heads, 64 dimensions and two bytes per element.
3. **Does caching make decode constant-cost?** No. New-token attention still
   reads the growing K/V prefix; model weights are still read each step.
