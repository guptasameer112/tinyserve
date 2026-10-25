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

Repo skeleton only. Next: Lesson A1 (tokens, tensors, shapes, logits).
