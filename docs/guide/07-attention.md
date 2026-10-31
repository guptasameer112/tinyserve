# 07. Attention scores, causal mask and softmax (B4)

Code: [attention.py](../../tinyserve/models/attention.py).
Tests: [test_b4_attention.py](../../tests/test_b4_attention.py).
Notebook: [B4_attention.ipynb](../../notebooks/B4_attention.ipynb).

## The four steps

Each token supplies a query (what it seeks), a key (what it offers) and a
value (the content to mix). These are useful analogies, not labels assigned
to individual dimensions.

```
scores = Q @ K.T / sqrt(head_dim)
scores[not_allowed] = -inf
weights = softmax(scores, dim=keys)
output = weights @ V
```

One head with two tokens: queries `[1, 1]`, keys `[0, ln(3)]`,
values `[10, 20]`, and head dimension 1.

| Query | Allowed scores | Attention weights | Output |
|---|---|---|---|
| Token 0 | `[0, -inf]` | `[1, 0]` | `10` |
| Token 1 | `[0, ln(3)]` | `[0.25, 0.75]` | `17.5` |

Softmax turns scores into nonnegative weights that sum to 1. PyTorch uses a
stable implementation equivalent to subtracting the largest score before
exponentiating. A masked score of `-inf` gets weight zero.

## Why divide by sqrt(head_dim)?

If independent query/key components have roughly unit variance, a dot
product of D components has variance about D. Dividing by sqrt(D) keeps
the variance near 1 rather than letting larger heads make softmax overly
peaked. TinyLlama has D=64, so the divisor is 8.

## Shapes and causality

- Query: `(Nq, H, D)`.
- Key/value: `(Nk, H, D)`.
- Allowed mask: `(Nq, Nk)`, boolean; **True means visible**.
- Scores/weights: `(H, Nq, Nk)`.
- Output: `(Nq, H, D)`.

For one prompt, token i sees positions <= i, including itself. This is
safe: its output predicts token i+1, not token i.

For packed requests, visibility also requires equal sequence IDs.
Restarting positions alone does not isolate requests. The mask combines
`key_position <= query_position` with `query_sequence == key_sequence`.

For single-token cached decode, one query can see all valid cached keys,
including its own newly written key. Unused cache slots and other requests
must still be excluded. B4's rectangular mask supports this shape, but B4
does not implement a cache.

Changing future Q/K/V must not change any earlier outputs. This is the
property that makes old KV entries reusable.

## Precision and validation

The reference uses FP32 for FP16/BF16 scores, softmax and value mixing,
then casts the output back. FP64 inputs remain FP64. This is a deliberately
readable correctness backend, not a fused production kernel. FP16 dot
products can overflow even when all input components are finite.

| Check | Tolerance | Reason |
|---|---|---|
| Hand example | absolute 1e-6 | Small FP32 rounding |
| FP32 vs independent float64 formula | relative 1e-5, absolute 1e-6 | Dot-product and softmax rounding |
| PyTorch MHA/SDPA | relative 1e-5, absolute 1e-6 | Different reduction orders |
| Real HF eager attention, FP32 | relative/absolute 1e-5 | Same input projections and RoPE |
| FP16 vs FP32 on the same quantized inputs | relative/absolute 5e-4 | One final FP16 rounding |
| BF16 vs FP32 on the same quantized inputs | relative/absolute 4e-3 | One final BF16 rounding |

Other tests cover zero future weights, row sums, packed vs separate
requests, one-query decode, RoPE integration and invalid shapes/masks.
A query with no visible keys raises an error instead of returning NaNs.
Input checks synchronize the GPU; that is acceptable for this slow reference.

The real-model probe compares the input to HF's `o_proj` with our flattened
head outputs for layers 0, 10 and 21. It uses HF's `repeat_kv` to expand
TinyLlama's 4 KV heads to 32 query heads. **Our own GQA is B5**, not B4.

Run the notebook top to bottom on a Colab T4. It runs tests before demos,
checks the real model, draws a heatmap, and writes environment information
and measured differences to `results/B4_attention.json`. No results or
test counts are claimed until that run succeeds.

## Interview questions

1. **Why scale attention scores?** Dot-product variance grows with head
   dimension; dividing by sqrt(D) controls the scale entering softmax.
2. **Why is the causal mask applied before softmax?** Disallowed scores
   become `-inf`, so their probabilities are zero and the remaining weights
   still sum to 1.
3. **How do you know attention is correct?** Check hand math, an independent
   float64 formula, PyTorch attention, causality and request isolation, then
   compare real HF activations in FP32. Compare FP16 to FP32 with a justified
   rounding tolerance rather than requiring bit equality.
