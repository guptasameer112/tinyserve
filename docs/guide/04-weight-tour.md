# 04. Weight tour (Lesson B1)

Before writing model code we looked at what is inside the checkpoint. Our own model must load these tensors by name, so [weights.py](../../tinyserve/weights.py) holds a pure-Python spec of every name and shape, checked against the real model on the T4 (results: [B1_weights.json](../../results/B1_weights.json)).

## The 201 tensors

`1 (embedding) + 9 per layer x 22 layers + 1 (final norm) + 1 (lm_head) = 201`, all FP16. One decoder layer holds:

| Tensor | Shape | Parameters | Role |
|---|---|---|---|
| `self_attn.q_proj` | (2048, 2048) | 4,194,304 | 32 query heads x 64 |
| `self_attn.k_proj` | (256, 2048) | 524,288 | 4 KV heads x 64 |
| `self_attn.v_proj` | (256, 2048) | 524,288 | 4 KV heads x 64 |
| `self_attn.o_proj` | (2048, 2048) | 4,194,304 | mixes the heads back to 2048 |
| `mlp.gate_proj` | (5632, 2048) | 11,534,336 | SwiGLU gate |
| `mlp.up_proj` | (5632, 2048) | 11,534,336 | SwiGLU up |
| `mlp.down_proj` | (2048, 5632) | 11,534,336 | back to 2048 |
| `input_layernorm` | (2048,) | 2,048 | RMSNorm before attention |
| `post_attention_layernorm` | (2048,) | 2,048 | RMSNorm before the MLP |

Total per layer **44,044,288**; x 22 = 968,974,336. Add the embedding table and `lm_head` (32000 x 2048 = 65,536,000 each, not tied) and the final norm: **1,100,048,384** parameters, exactly what the spec predicts from the config alone.

PyTorch `nn.Linear` stores weights as `(out_features, in_features)`, which is why `k_proj` is `(256, 2048)`.

## Where the bytes are

| Part | Parameters | Bytes | Share |
|---|---|---|---|
| MLP | 761.3 M | 1522.5 MB | **69.2%** |
| Attention | 207.6 M | 415.2 MB | 18.9% |
| Embedding | 65.5 M | 131.1 MB | 6.0% |
| lm_head | 65.5 M | 131.1 MB | 6.0% |
| Norms | 0.1 M | 0.2 MB | 0.0% |

Every decode step reads each weight once (A2), so the MLP alone is about 69% of the weight traffic. One refinement: the embedding table is only a lookup (one row per token), so a decode step reads about 2.07 GB, not 2.20 GB. That moves the batch-1 bandwidth ceiling from 110 to about 116 tokens/s and the measured HF MBU from 30% to about 28%. The A2 and A3 chapters used the full 2.2 GB, which is a slight overestimate of the bytes read.

## GQA and the KV cache

Query heads 32, KV heads 4: 8 queries share each KV head. So `k_proj` and `v_proj` have 256 outputs instead of 2048, and the cache stores only 4 heads per layer:

`KV bytes per token = 2 x 22 layers x 4 KV heads x 64 x 2 bytes = 22,528 bytes = 22 KiB`

Full multi-head attention would need 180,224 bytes (176 KiB) per token, 8x more. Consequences:

- One 2048-token sequence costs 44 MiB of cache, versus 2098 MiB for the weights.
- With 10 GiB free, about 476,000 tokens fit: **232 sequences of 2048 tokens**. This is the budget Part D's block manager has to manage.

## What the weights look like (FP32 statistics)

| Tensor | mean | std | min | max |
|---|---|---|---|---|
| embed_tokens | -0.0000 | 0.0149 | -0.117 | 0.149 |
| layer 0 q_proj | -0.0000 | 0.0164 | -1.586 | 1.031 |
| layer 0 down_proj | 0.0000 | 0.0166 | -0.523 | 0.414 |
| layer 0 input_layernorm | 0.0058 | 0.0460 | -0.582 | 0.770 |
| layer 21 input_layernorm | 0.4637 | 0.0852 | 0.114 | 1.227 |
| final norm | 1.9149 | 0.1365 | 0.508 | 3.188 |
| lm_head | -0.0004 | 0.0247 | -0.471 | 0.395 |

Projection weights are tiny and centred on zero, but have rare outliers far from the typical size (q_proj max magnitude 1.586 against std 0.016, about 100 sigma). Outliers like these matter when we quantize (Part G). Norm scales differ a lot between layers: the first layer's `input_layernorm` is close to zero on average while the last is about 0.46 and the final norm about 1.9. We observe this; we do not yet explain it.

## Code map

- [weights.py](../../tinyserve/weights.py): `expected_shapes`, `expected_param_count`, `kv_cache_bytes_per_token`, `compare_shapes`, `weight_tour`, `bytes_by_category`, `tensor_stats`.
- [test_b1_weights.py](../../tests/test_b1_weights.py): exact arithmetic tests, a state_dict check on a tiny Llama, and the real config check. 13 tests pass with the quiet-logs tests.
- [B1 notebook](../../notebooks/B1_weight_tour.ipynb).

## Interview questions

1. **Where do TinyLlama's 1.1B parameters live?**
   About 969M (88%) are in 22 identical decoder layers of 44M each: the MLP (three 2048 x 5632 matrices) is 69% of all bytes and attention 19%. The rest is the embedding and the separate `lm_head` (65.5M each) and tiny norm vectors.

2. **What is grouped-query attention and what does it save?**
   Many query heads share fewer KV heads (32 queries, 4 KV heads). The K and V projections shrink 8x, and the KV cache is 8x smaller: 22 KiB per token instead of 176 KiB. Quality is close to full multi-head attention, while memory and bandwidth for the cache drop a lot, so more sequences fit in a batch.

3. **How do you budget GPU memory for serving?**
   Weights are fixed (2.1 GiB here); the rest, minus activations and overhead, is for the KV cache at 22 KiB per token. With 10 GiB free that is about 476,000 tokens, e.g. 232 full 2048-token sequences. The block manager (D3) divides this into fixed-size blocks so no sequence reserves more than it uses.
