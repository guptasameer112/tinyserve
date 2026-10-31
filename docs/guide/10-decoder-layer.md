# 10. Decoder layer and residuals (B7)

Code: [llama.py](../../tinyserve/models/llama.py).
Tests: [test_b7_decoder.py](../../tests/test_b7_decoder.py).
Notebook: [B7_decoder_layer.ipynb](../../notebooks/B7_decoder_layer.ipynb).

## Connect the pieces

```
attention_delta = attention(input_layernorm(x))
after_attention = x + attention_delta
mlp_delta = mlp(post_attention_layernorm(after_attention))
output = after_attention + mlp_delta
```

Both branches read a normalized input. Both residual additions preserve the
**unnormalized** stream. This is pre-norm, not norm-after-add (post-norm).
Neither branch changes the width: input/output are `(N, 2048)`.

Think of a draft and two edits: keep the draft, add the attention edit, then
add the MLP edit to the updated draft. Do not replace the draft with an edit.

One layer contains two RMSNorms, one attention block and one MLP, with two
residual additions. TinyLlama has 22 such layers. The final RMSNorm, embedding
table, LM head and full stack belong to B8, not inside an individual layer.

## Attention branch

Q/K/V projections -> RoPE on Q/K -> grouped-query attention -> O projection.
Q has 32 heads, K/V have 4, each dimension 64. Concatenate the 32 head outputs
back to 2048 features, then apply `o_proj`.

The module reuses B2-B6 operations without duplicating their math.
Weight keys match an HF decoder layer; strict loading rejects missing or
extra weights. This implementation supports the bias-free SiLU Llama shape
used by TinyLlama, and no dropout or KV cache yet.

## Flattened metadata

`AttentionMetadata` carries local positions and optional request IDs.
Ordinary prefill uses positions `0..N-1`. Packed requests restart their
positions and have distinct IDs, so attention cannot cross request boundaries.
The metadata constructs B4's boolean mask. It is the initial no-cache layout;
cache/block fields will arrive with C/D. B7 does not implement paged storage.

## Verification

- Zero both output branches: the decoder is exactly the identity.
- Explicit two-step formula verifies norm/residual order.
- Zero attention alone: MLP adds to the original unnormalized residual.
- Future-token changes cannot change earlier outputs.
- Packed requests match separate runs.
- Checkpoint keys and invalid layouts/configurations are checked.
- Synthetic HF parity covers MHA, GQA and MQA.
- Real-model probe feeds each of 22 real HF layer inputs to our layer.
  It checks **isolated layers**, not accumulated full-stack error.

FP32 HF assertions use relative/absolute `1e-5`. A controlled small CUDA
FP16-vs-FP32 test uses relative/absolute `8e-3` to allow rounding across
norms, projections, activation/products and residual additions. This is
not a universal error bound. B8 will test the complete stack separately.

The notebook runs B4-B7 tests in Colab before demos and saves test output,
environment info and per-layer differences to `results/B7_decoder_layer.json`.
Runtime validation is pending the user's Colab run.

### Recorded Colab run (2026-10-31)

All **94 B4-B7 tests passed**, none skipped. Identity, explicit wiring and
prefix-causality differences were zero. Packed-vs-separate difference was
`1.19e-7`. Real isolated layer differences ranged from `3.73e-9` to
`9.54e-7`, within the FP32 assertions. Results used commit `451a831`.

## Interview questions

1. **What is a residual connection?** Add a branch's output to its input
   stream instead of replacing that stream. It preserves a direct path for
   information and gradients.
2. **Pre-norm versus post-norm?** Pre-norm normalizes inputs to the branch
   before computation; post-norm normalizes after the residual addition.
3. **Does isolated layer parity prove full-model parity?** No. It verifies
   each layer on HF's inputs. B8 checks our own outputs feeding subsequent
   layers, final normalization and logits.
