# 01. Tokens, tensors, shapes, logits (Lesson A1)

## The pipeline

```
text --tokenizer--> ids (N,) --embedding--> (N, 2048) --22 decoder layers--> (N, 2048) --lm_head--> logits (N, 32000)
```

Measured on TinyLlama-1.1B-Chat (T4, FP16), prompt `"The capital of France is"`:

| Stage | Shape | Note |
|---|---|---|
| ids | `(6,)` int64 | `<s> ▁The ▁capital ▁of ▁France ▁is`; `<s>` (id 1) is BOS, added by the tokenizer |
| after embedding | `(6, 2048)` | one 2048-vector per token |
| logits | `(6, 32000)` FP16 | 384,000 bytes = 6 x 32000 x 2 |

Model facts: 1,100,048,384 parameters, 2.20 GB in FP16, 22 layers, hidden 2048, 32 query heads, 4 KV heads, head_dim 64, vocab 32000.

## Ideas to remember

1. **The model only sees integer ids.** The tokenizer maps text to ids and back. `▁` marks a leading space.
2. **Flattened tokens.** We keep ids as one 1-D tensor `(N,)`, not `(batch, seq)`. Later, many requests will share one flat tensor plus metadata saying where each begins.
3. **Row `i` of the logits scores the token after position `i`.** For the prompt above the last row gives `▁Paris` 50.9%, `▁located` 16.5%, `:` 2.2%. Generation uses only the last row; the other rows are wasted work at inference (we will skip them in Part D).
4. **Greedy decoding** is `argmax` of the last row. Sampling (Part C) draws from the softmax instead.
5. **Naive generation** appends the token and re-runs the whole sequence each step. Correct, but every step redoes all earlier work.
6. **Causality.** The attention mask stops position `i` from seeing later tokens, so its logits do not change when tokens are appended. In FP32 on CPU the test shows differences below 1e-5. In FP16 on the T4 we measured a max difference of 0.069. That is most likely rounding noise (FP16 has about 3 decimal digits, logits are order 10, and differently shaped kernels accumulate in a different order across 22 layers), not a causality leak; the FP32 test is the strict check. This is why FP16 is compared against FP32 with a tolerance, not demanded bit-equal.

## Code map

- [basics.py](../../tinyserve/basics.py): `load_model`, `encode`, `forward_logits`, `next_token_greedy`, `generate_greedy_naive`, `prefix_logit_diff`, `shape_report`.
- [test_a1_basics.py](../../tests/test_a1_basics.py): logic tests on a tiny random Llama (CPU) and one `slow` test on the real config.
- [A1 notebook](../../notebooks/A1_tokens_tensors_logits.ipynb) and [results](../../results/A1_shapes.json).

## Interview questions

1. **What shape are the logits, and which row do you use to pick the next token?**
   `(N, vocab)`, one row per input position. Row `i` predicts the token after position `i`, so generation uses the last row. All earlier rows only matter for training or scoring.

2. **Why does appending tokens not change the logits of earlier positions, and why does that matter?**
   Causal masking: a position attends only to itself and earlier positions. So the keys and values of earlier tokens never change as the sequence grows, which is what makes caching them (the KV cache) valid.

3. **Why is the naive generation loop slow, and what does a decode step cost in memory traffic?**
   It recomputes all N positions at every step, so total work grows quadratically in the number of generated tokens. Even with a cache, each decoded token must read all 2.2 GB of FP16 weights, which is why decode is memory-bandwidth-bound (Lesson A2).
