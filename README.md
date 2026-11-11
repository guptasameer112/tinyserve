# tinyserve

A small vLLM-style inference engine built from scratch for TinyLlama-1.1B-Chat (FP16 on a Colab T4).

See [MASTERPLAN.md](MASTERPLAN.md) for the roadmap and process rules.

## Start here

The core A-E lessons are implemented and have recorded Colab evidence.
H adds documentation, the compiled study guide and static packaging CI.
This is a learning project: **HF was faster in every measured E2 workload**.
It is not a production server or a claim of vLLM feature/performance parity.

- [Read the short project story](docs/blog.md).
- [Read the 22-chapter guide in order](docs/guide/README.md).
- [Inspect complete runtime evidence](results/README.md), including exact source/environment.
- [Download CI-built PDF and distributions](https://github.com/guptasameer112/tinyserve/actions/workflows/packaging.yml):
  open a successful **Packaging CI** run and download `tinyserve-packaging`.
  Artifacts expire after 30 days; rerun the workflow to rebuild them.
- [Pause for the full core review](docs/CORE_REVIEW.md) before Parts F/G.

### Run a lesson

Open the relevant notebook in Colab, select a **fresh T4 runtime**, and run it
top to bottom. Start with [A1](https://colab.research.google.com/github/guptasameer112/tinyserve/blob/main/notebooks/A1_tokens_tensors_logits.ipynb)
or revisit [E2](https://colab.research.google.com/github/guptasameer112/tinyserve/blob/main/notebooks/E2_engine_vs_hf.ipynb).
Each notebook sets up the checkout and installs its required extras in Colab.
Save the complete JSON and notebook outputs. Pulling the repository does not
refresh already-open notebook cells.

Do not install model dependencies, download weights, run pytest or execute
notebooks/models locally. The approved H exception is documentation-only PDF
building. GitHub CI runs syntax, notebook JSON, local-link, package-content and
document-build checks only; it does not rerun or certify GPU behavior.

### Installable distribution

The wheel contains `tinyserve` and the `bench` helpers used by the notebooks.
The source archive also includes tests, lessons, docs and recorded results.
PyTorch and Transformers remain supplied by the Colab runtime; this is not a
self-contained inference environment. `dev` adds pytest, and `server` adds
FastAPI/Uvicorn/HTTPX. There is no CLI, model weight bundle, PyPI publication,
public deployment or authentication/TLS setup.

The E1 app must stay on `127.0.0.1`. It implements a finite text-only Chat
Completions subset with a default 32-admission lifetime quota, not an unrestricted
OpenAI-compatible production API.

## Layout

- `tinyserve/`: the package (model, generation, engine, server, kernels)
- `tests/`: pytest tests; notebooks import and run them
- `notebooks/`: one Colab notebook per part
- `results/`: benchmark JSON with environment info
- `docs/guide/`: one short study-guide chapter per lesson

## Status

Foundations A1-A3 and model lessons B1-B8 are complete.
[B3: RoPE](notebooks/B3_rope.ipynb) records 16 passed tests and six exact
Hugging Face comparisons; its [guide chapter](docs/guide/06-rope.md) is included.
[B4: attention](notebooks/B4_attention.ipynb) passed all 30
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

[C1: sampling](notebooks/C1_sampling.ipynb) passed all 42 tests in Colab,
including HF probability-filter parity and seeded replay.

[C2: generation loop](notebooks/C2_generation.ipynb) passed 96 B8/C1/C2 tests
in Colab; one two-GPU test was skipped on the single T4. Raw and chat greedy
outputs matched HF token-for-token, seeded sampling replay matched, and
zero-budget and strict context checks passed. Results are saved in
`results/C2_generation.json`.
[C3: contiguous KV cache](notebooks/C3_kv_cache.ipynb) passed 139 B8/C1/C2/C3
tests in Colab on November 1, 2026; one two-GPU test was skipped.
All 20 demonstrated greedy steps matched HF. Cached/no-cache logits differed
by at most 2.10e-5; cached/HF logits differed by at most 3.80e-5.
Reset, seeded replay, zero budget and strict context checks passed.
`generate(..., use_cache=True)` enables it; the default remains no-cache.
The user-provided result is archived in [C3_kv_cache.json](results/C3_kv_cache.json).
Local checks are static only; tests and model demos run in Colab.
[C4: cache speed curve](notebooks/C4_cache_curve.ipynb) passed 101 C2/C3/C4
tests in Colab on November 1, 2026; one two-GPU test was skipped.
The eight unique FP16 cases completed, with three paired trials each.
For 32 new tokens, median paired cache speedup was near break-even for
16-128-token prompts, 1.50x at 256 tokens and 2.95x at 512 tokens.
All measured cached/no-cache token sequences matched in this run.
EOS stopping is disabled only inside the benchmark and restored afterward.
Results are archived in [C4_cache_curve.json](results/C4_cache_curve.json).
These are single-request reference-engine measurements, not serving throughput.
See the [C4 guide](docs/guide/15-cache-speed-curve.md) for timing boundaries
and why cache can lose for a single generated token.
[D1: flattened batching](notebooks/D1_flattened_batching.ipynb) passed all
140 B4/B8/C3/D1 tests in Colab on November 1, 2026, with no skips.
Three prompts of lengths 6, 6 and 9 packed into 21 real rows versus 27 padded
slots. Packed logits matched separate ours/HF within 2.70e-5, with equal
argmax choices on every row. Cross-request changes and request reordering
produced zero differences in the demonstrated isolation comparisons.
This is no-cache packed prefill, not cached batched generation or scheduling.
Results are archived in [D1_flattened_batching.json](results/D1_flattened_batching.json).
No batching speedup is claimed.
See the [D1 guide](docs/guide/16-flattened-batching.md).
[D2: paged KV cache](notebooks/D2_paged_cache.ipynb) passed 197 B8/C2/C3/D1/D2
tests in Colab on November 3, 2026; one two-GPU test was skipped.
Ordered and scrambled pools passed all-row FP32 argmax parity against
separate no-cache/HF references, with maximum logits difference 4.58e-5.
Mixed prefill/decode, pause/resume and reordered batches advanced independent
request lengths correctly. Reset reused storage with zero demonstrated difference.
Existing no-cache and contiguous-cache paths remain supported.
Results are archived in [D2_paged_cache.json](results/D2_paged_cache.json).
The D2 interface still uses manually assigned blocks. Sharing, scheduling and
performance kernels remain deferred.
See the [D2 guide](docs/guide/17-paged-kv-cache.md).

[D3: block manager](notebooks/D3_block_manager.ipynb) passed 253
B8/C2/C3/D1/D2/D3 tests in Colab on November 4, 2026; one two-GPU test
was skipped. The complete run includes atomic exhaustion, growth,
injected later-layer failure, successful retry, block reuse and reset.
The failed forward kept request 0 at length three; retry committed length five.
All fourteen tested logits rows matched no-cache/HF argmax choices, with
maximum HF difference 6.68e-6. All four blocks were free at the end.
Results are archived in [D3_block_manager.json](results/D3_block_manager.json).
See the [D3 guide](docs/guide/18-block-manager.md).

[D4: scheduler](notebooks/D4_scheduler.ipynb) passed 304
B8/C2/C3/D1/D2/D3/D4 tests in Colab on November 4, 2026; one two-GPU test
was skipped. The continuous-batching timeline admitted a new request before
an older one finished and completed in three steps. The small-pool timeline
completed in eight steps with one preemption and two replayed prefix tokens:
ten logical token positions required twelve model rows.
Across both scenarios, all 22 new/replayed logits rows matched no-cache/HF
argmax choices, with maximum difference 2.50e-5. All blocks were free at the end.
Results are archived in [D4_scheduler.json](results/D4_scheduler.json).
See the [D4 guide](docs/guide/19-continuous-batching.md).
These are correctness checks, not serving-throughput measurements.

[D5: engine loop and memory budgeting](notebooks/D5_engine.ipynb) passed
422 B8/C1/C2/C3/D1/D2/D3/D4/D5 tests in Colab on November 10, 2026;
one two-GPU test was skipped. Continuous arrivals, preemption/recomputation,
failure/retry, seeded replay, EOS/cancellation and KV memory checks passed.
All 43 successful FP32 scheduled/replayed rows matched no-cache/HF argmax,
with maximum HF logits difference 3.62e-5.
Five FP16 rows passed paged/no-cache FP16 checks; all four sampled greedy
choices matched the rounded-weight FP32 reference. Cross-precision score
differences reached 0.12117 and are recorded, not claimed numerically equivalent.
All demonstrated pools were fully released. Results are archived in
[D5_engine.json](results/D5_engine.json).
KV-only budgeting still requires separate workspace headroom and does not
guarantee OOM safety. The existing single-request `generate()` remains supported.
See the [D5 guide](docs/guide/20-engine-loop.md). E1 is validated below.

[E1: streaming server](notebooks/E1_streaming_server.ipynb) passed its separate
Colab T4 run on November 11, 2026 at source `cee6f49`: all 114 D5/E1 tests passed.
It serves a documented text-only Chat Completions subset over loopback HTTP,
with SSE, shared-engine scheduling, strict options, cancellation and explicit
failure handling. There is no public tunnel, authentication or deployment.
The finite lesson server accepts at most 32 requests per app lifetime by
default; it is not a production service or a full OpenAI API implementation.
The demo uses FP32 and a small explicit KV pool. D5's supplied run is
validated under its documented precision contract.
The [E1 result](results/E1_streaming_server.json) records JSON/SSE text and
token parity, shared concurrent forwards, cancellation after three tokens,
all 16 blocks released and server shutdown.
See the [E1 guide](docs/guide/21-streaming-server.md).
[E2: engine versus HF](notebooks/E2_engine_vs_hf.ipynb) passed its separate
Colab T4 run on November 11, 2026 at source `92de867`: all 119 D5/E2 tests passed.
It compares direct fixed-work
generation, with same-run token-ID TTFT, inter-token latency, request latency,
output throughput and repeated min/median/max ranges. A short FP32 parity gate
precedes FP16 timing. The comparison records backend/cache-lifecycle differences,
memory, exact token outputs and all trials; it does not claim HTTP latency or
require a speedup. The [complete result](results/E2_engine_vs_hf.json) records
matching FP32 and measured FP16 continuations, no preemption and full block release.
HF was faster in all four FP16 cases; TinyServe median output throughput was
14.36-24.68 tokens/s versus HF's 38.23-130.26 tokens/s across those cases.
See the [E2 guide](docs/guide/22-engine-vs-hf.md) for per-case ranges and limitations.
H packaging adds the [blog](docs/blog.md), [PDF build](docs/guide/README.md)
and static/document/package CI. After packaging verification, **pause for the
[full core-project review](docs/CORE_REVIEW.md)** before stretch work.

Recorded B3-C2 JSON files are now in [results/](results/README.md), recovered
from the saved notebook outputs with their original environment metadata.
Existing notebook outputs are preserved.

Part G is included in the planned scope: Qwen2.5-0.5B, prompt-lookup
speculative decoding, a KV visualizer and INT8 KV cache.
