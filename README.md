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

[B6: MLP / SwiGLU](notebooks/B6_mlp.ipynb) is ready for Colab validation.
It checks hand math, token independence, FP16 vs FP32, and all 22 real
TinyLlama MLPs, then saves `results/B6_mlp.json`.
Next after validation: B7 (decoder layer and residuals).
