# 08. Grouped-query attention (B5)

Code: [gqa.py](../../tinyserve/models/gqa.py).
Tests: [test_b5_gqa.py](../../tests/test_b5_gqa.py).
Notebook: [B5_gqa.ipynb](../../notebooks/B5_gqa.ipynb).

## Share K/V, not queries or attention weights

TinyLlama has 32 query heads and 4 KV heads. Each consecutive group of 8
query heads uses one KV head:

```
query heads  0..7  -> KV head 0
query heads  8..15 -> KV head 1
query heads 16..23 -> KV head 2
query heads 24..31 -> KV head 3
```

Each query is different, so scores, probabilities and outputs can differ
even within a group. Sharing a reference book does not mean asking the
same question or getting the same answer.

| Variant | KV heads with 32 query heads | Sharing |
|---|---|---|
| MHA | 32 | One KV head per query head |
| GQA (TinyLlama) | 4 | Eight query heads per KV head |
| MQA | 1 | All query heads share one KV head |

## Implementation and shapes

Input Q is `(Nq, 32, 64)`; K/V are `(Nk, 4, 64)`.
`expand_kv_heads` repeats each KV head consecutively to `(Nk, 32, 64)`,
then `grouped_query_attention` reuses B4's math. Output remains
`(Nq, 32, 64)`, or `(Nq, 2048)` when flattened for the output projection.

`repeat_interleave(..., dim=1)` gives `[0,0,...,1,1,...]`.
A cyclic repetition `[0,1,2,3,0,1,2,3,...]` pairs the wrong heads with the
checkpoint's learned queries.

The query head count must be a positive multiple of the KV head count.
K/V must match before expansion, not just after it. B4 retains the shape,
dtype, device and mask validation. Equal head counts need no expansion.

## Where the memory saving comes from

FP16 KV storage per token is:

```
2 (K and V) * 22 layers * 4 KV heads * 64 dimensions * 2 bytes = 22 KiB
```

Using 32 KV heads would cost 176 KiB: eight times as much.
At 2048 tokens this is 44 MiB versus 352 MiB per sequence.

The reference **temporarily materializes expanded K/V** to reuse B4.
The cache must store the original 4-head tensors. A later optimized
backend reads shared KV heads directly without expanded copies.

GQA also shrinks K/V projection weights, but Q and O projections keep their
original sizes. It does not reduce the number of query attention scores
eightfold: all 32 queries still compare with the visible keys.

## Verification

- Exact head mapping for 32/4, plus HF `repeat_kv` parity.
- MHA, GQA and MQA compared with separate per-query-head FP64 runs.
- Changing one KV head affects only its corresponding query group.
- Shared KV can produce different attention weights for different queries.
- Packed request isolation and cached-decode shape checks.
- FP16/BF16 compared with FP32 on the same quantized inputs, with B4's
  final-rounding tolerances (`5e-4` and `4e-3`).
- Real TinyLlama layers 0, 10 and 21 compared with FP32 HF eager attention,
  `rtol=atol=1e-5`. The probe now uses our GQA, not HF expansion.

The notebook runs B4 and B5 tests together to protect the shared probe.
Runtime results are pending your Colab run; no pass count is claimed here.

## Interview questions

1. **How do MHA, GQA and MQA differ?** They differ in how many query heads
   share a KV head: one, a group, or all of them.
2. **Why does GQA shrink the cache without shrinking the output?** Cache
   storage follows KV head count; output follows query head count.
3. **Does this implementation avoid all expanded KV memory?** No. It is a
   readable reference with temporary copies. Persistent cache storage stays
   compact; a later kernel avoids the temporary expansion.
