# tinyserve

A small vLLM-style inference engine built from scratch for TinyLlama-1.1B-Chat (FP16 on a Colab T4).

See [MASTERPLAN.md](MASTERPLAN.md) for the roadmap and process rules.

## Layout

- `tinyserve/`: the package (model, generation, engine, server, kernels)
- `tests/`: pytest tests; notebooks import and run them
- `notebooks/`: one Colab notebook per part
- `results/`: benchmark JSON with environment info
- `docs/guide/`: one short study-guide chapter per lesson

## Status

Foundations A1-A3 and lessons B1-B2 are complete. B3's recorded RoPE comparisons
match Hugging Face. [B4: attention](notebooks/B4_attention.ipynb) passed all 30
tests in Colab and matched real HF attention (worst FP32 difference 4.47e-8).

The B4 notebook runs its tests, explains scores/masking/softmax with a hand
example, checks real TinyLlama attention in FP32, and saves
`results/B4_attention.json`.

[B5: GQA](notebooks/B5_gqa.ipynb) implements our own consecutive KV-head
expansion and uses it in the real-model probe. Its Colab run passed all 58
B4/B5 tests, with real HF differences at most 4.47e-8.

[B6: MLP / SwiGLU](notebooks/B6_mlp.ipynb) passed all 74 B4-B6 tests in Colab.
All 22 real TinyLlama MLPs matched HF exactly in FP32.

[B7: decoder layer and residuals](notebooks/B7_decoder_layer.ipynb) passed
all 94 B4-B7 tests in Colab. All 22 isolated layers matched HF within 9.54e-7.

[B8: full model](notebooks/B8_full_model.ipynb) passed 143 tests and all 36
greedy parity steps in Colab. Full-model logits differed from HF by at most
1.67e-5 across the tested prompts/steps.

[C1: sampling](notebooks/C1_sampling.ipynb) is ready for Colab validation:
greedy, temperature, top-k and top-p, checked against HF probability filters.
It needs no model weights and saves `results/C1_sampling.json`.
Next after validation: C2 (generation loop).

Part G is included in the planned scope: Qwen2.5-0.5B, prompt-lookup
speculative decoding, a KV visualizer and INT8 KV cache.
