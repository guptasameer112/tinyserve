# How to use this study guide

This is the core TinyServe curriculum: 22 chapters from tokens to a small
inference server and direct benchmarks. Read in numeric order. Each chapter
connects concepts, source code, tests, recorded examples and interview questions.
The [masterplan](../../MASTERPLAN.md) defines the full roadmap.

The chapters preserve the teaching progression: a chapter can describe a later
lesson as future work even though that lesson is now complete. Current runtime
evidence lives in the [results index](../../results/README.md), not in those
historical future-tense passages. Counts from different regression suites overlap;
do not add them into a unique-test total.

## Reading order

| Part | Chapters |
|---|---|
| A: foundations | [01 Tokens](01-tokens-tensors-logits.md), [02 Roofline](02-gpu-roofline.md), [03 HF baseline](03-hf-baseline.md) |
| B: model | [04 Weights](04-weight-tour.md), [05 RMSNorm](05-rmsnorm.md), [06 RoPE](06-rope.md), [07 Attention](07-attention.md), [08 GQA](08-gqa.md), [09 MLP](09-mlp.md), [10 Decoder](10-decoder-layer.md), [11 Full model](11-full-model.md) |
| C: generation | [12 Sampling](12-sampling.md), [13 Generation](13-generation-loop.md), [14 KV cache](14-contiguous-kv-cache.md), [15 Cache timing](15-cache-speed-curve.md) |
| D: batching and paging | [16 Flattening](16-flattened-batching.md), [17 Paging](17-paged-kv-cache.md), [18 Block manager](18-block-manager.md), [19 Scheduler](19-continuous-batching.md), [20 Engine](20-engine-loop.md) |
| E: serving | [21 Streaming server](21-streaming-server.md), [22 HF comparison](22-engine-vs-hf.md) |

Use the [core-review checklist](../CORE_REVIEW.md) after E2 and H. Pause there
before Parts F/G. This review is a user-requested learning checkpoint, not a
claim that later performance features already exist.

## PDF and reproducible build

Read `docs/tinyserve-study-guide.pdf` and its build manifest,
`docs/tinyserve-study-guide.manifest.json`, directly from the repository.
Part H uses a local documentation-only build, not CI or distribution builds.
There is no release or deployment.
The PDF includes this introduction, all numbered chapters, the existing roofline
image and the core-review checklist. Its table of contents is navigable.
Code/result links point to the source commit recorded by the build.

Required documentation tools:

- [Pandoc 3.13](https://github.com/jgm/pandoc/releases/tag/3.13).
- [Typst 0.15.1](https://github.com/typst/typst/releases/tag/v0.15.1).

Use official release archives and verify their published SHA256 digests before
extraction. The build records the link-target commit and source-file/PDF hashes
in `tinyserve-study-guide.manifest.json`.

From a clean checkout, with both executables on PATH:

```text
python scripts/check_static.py
python scripts/build_guide.py --revision FULL_40_CHARACTER_COMMIT_SHA
```

Replace the placeholder with `git rev-parse HEAD`. An explicit SHA fixes the
code/result link targets; it does not prove a dirty tree matches that SHA.
The manifest hashes identify the actual local document inputs. For an exact
committed-source build, use a clean checkout or export the committed source first.
Output defaults to `docs/tinyserve-study-guide.pdf`, alongside its manifest.
`--pandoc`, `--typst` and `--output` accept explicit paths when tools are not on PATH.

The user approved local documentation-only builds for H. This does **not**
permit local pytest, notebook/model execution, dependency setup or weight
downloads. Runtime lessons remain Colab-only. Local static and PDF checks
do not replace the archived T4 runs. No byte-for-byte cross-platform
PDF reproducibility is promised: system fonts can differ.
