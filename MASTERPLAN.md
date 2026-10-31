# tinyserve masterplan (restart, from zero)

Goal: build a small vLLM-style inference engine from scratch, understand every line, and be able to defend it in an interview.
Model: TinyLlama-1.1B-Chat (22 layers, hidden 2048, 32 query heads, 4 KV heads, head_dim 64). Hardware: Colab T4 (FP16).

## Deliverables
1. **Repo** (GitHub): tested package `tinyserve/`, `tests/`, `notebooks/`, `results/`, README.
2. **Colab notebooks**: one per part, each runnable top to bottom, with markdown cells that explain the code.
3. **Study guide PDF**: one short chapter per lesson (`docs/guide/NN-name.md`), compiled to PDF at the end. Diagrams are code (matplotlib or Mermaid) and live in the repo.
4. **README and a short blog post.**

## How every lesson works
concept (real numbers and shapes) -> spec -> you write the code -> test against a reference -> 3 interview questions -> guide chapter -> git commit.

## Process rules (lessons learned from the first attempt)
- Code lives in repo files and is written **once**. Never a second `%%writefile` for the same file. Fix by editing that file.
- Every notebook starts with one setup cell: clone or pull the repo, `%cd`, `sys.path`, install pytest.
- Tests are files in `tests/`. Notebooks import and run them, then show a demo.
- No short global names (`h`, `k`, `v`) in notebook cells. Put logic in functions.
- Every tensor created inside a function takes its device and dtype from an input.
- Tolerances are justified: FP32 reference tests are tight (about 1e-5 or exact), FP16 is compared against FP32, not demanded bit-equal.
- Every result is saved as JSON with environment info (GPU, library versions, git SHA). Report ranges, not single numbers.
- Commit after every lesson. Push to GitHub.

## Hardware: only what matters
Three ideas, no deeper:
1. Decode is limited by memory bandwidth (every token reads all the weights). We measure bandwidth once.
2. Prefill is limited by compute. We measure matmul throughput once.
3. Roofline: memory-bound vs compute-bound, and why batching helps.
Clock, power and temperature details are a footnote ("the T4 varies run to run, so we report ranges").

## New design choices (vs. the first attempt)
- **Flattened-token model from day one.** The model takes `input_ids (N,)`, `positions (N,)` and an `AttentionMetadata` object. The same model code then serves no-cache, contiguous-cache and paged runs, with no rewrite in Part D.
- **Pluggable attention backend:** `reference` (PyTorch, slow, obviously correct), `paged` (PyTorch gather), `triton` (fast). Every fast backend is tested against `reference`.
- **Test-first:** each lesson starts with the test the code has to pass.
- **FP32 for parity, FP16 for speed.** One `dtype` switch.
- **Small readable modules**, no framework magic.

## Roadmap

| Part | Lessons | You build | Tier |
|---|---|---|---|
| **A. Foundations** | A1 Tokens, tensors, shapes, logits. A2 GPU basics: bandwidth and matmul measurement, roofline. A3 HF `generate` baseline | repo skeleton, `bench/`, baseline JSON | Core |
| **B. The model** | B1 Weight tour. B2 RMSNorm. B3 RoPE. B4 Attention scores, mask, softmax. B5 GQA. B6 MLP. B7 Decoder layer and residuals. B8 Full model, weight loading, logits and token-exact parity (FP32) | `tinyserve/models/llama.py` | Core |
| **C. Generation** | C1 Sampling (greedy, temperature, top-k, top-p). C2 Generate loop. C3 Contiguous KV cache. C4 No-cache vs cached speed curve | `generate()`, `KVCache` | Core |
| **D. Batching and paging** | D1 Flattened batching and attention metadata. D2 Block table and slot mapping, paged cache, scrambled-block test. D3 Block manager (free list, ref counts) with property tests. D4 Scheduler (continuous batching, preemption). D5 Engine loop and memory budgeting | `engine/` | Core |
| **E. Serving** | E1 Streaming OpenAI-compatible server (driven from the notebook). E2 Benchmark vs HF (throughput, TTFT, latency) | `server/`, `bench/` | Core |
| **F. Performance** | F1 Hash-chained prefix cache. F2 Triton paged-attention decode kernel plus fuzz tests. F3 CUDA graphs. F4 Ablation table and MBU analysis | `kernels/`, `engine/` | Stretch |
| **G. Extras** | Qwen2.5-0.5B support, prompt-lookup speculative decoding, KV visualizer, INT8 KV cache | | Optional |
| **H. Packaging** | README, blog post, PDF, CI | | Core |

Parts A to E and H make a project that can be explained end to end. Part F is what makes it stand out for inference roles.

## Interview map (what each part lets you answer)
- A: why is decode memory-bound, what does batching change.
- B: how RoPE, GQA, RMSNorm and SwiGLU work, and how you verified them.
- C: what is a KV cache, why is it valid (causality), what does it cost per token (22 KiB for TinyLlama).
- D: what paged attention solves, what a block table is, how continuous batching differs from static batching, what happens on preemption.
- E: where TTFT and inter-token latency come from, how streaming works.
- F: how prefix caching hashes blocks, how a decode kernel is structured, what CUDA graphs remove.

## Progress tracker
- [x] Repo created on GitHub
- [x] A1 A2 A3
- [ ] B1 (done) B2 (done) B3 B4 (created; Colab validation pending) B5 B6 B7 B8
- [ ] C1 C2 C3 C4
- [ ] D1 D2 D3 D4 D5
- [ ] E1 E2
- [ ] F1 F2 F3 F4
- [ ] H
