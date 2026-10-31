# 12. Sampling the next token (C1)

Code: [sampling.py](../../tinyserve/sampling.py).
Tests: [test_c1_sampling.py](../../tests/test_c1_sampling.py).
Notebook: [C1_sampling.ipynb](../../notebooks/C1_sampling.ipynb).

## Scores are not the choice

The model returns one score per vocabulary token. Sampling decides which
token to append. It operates on one `(vocab,)` logits row; C2 will call it
on the model's final row.

Toy probabilities: Paris 60%, London 25%, Berlin 10%, Rome 5%.
These are synthetic, not measured TinyLlama predictions.

| Method | Effect |
|---|---|
| Greedy | Always choose the largest score |
| Temperature < 1 | Concentrate probability on high scores |
| Temperature > 1 | Flatten probabilities |
| Top-k | Keep the highest k-score threshold, renormalize |
| Top-p | Keep the high-probability nucleus reaching the chosen mass, renormalize |

With top-k=2 or top-p=0.8 on this example, Paris and London survive:
their probabilities become 60/85 and 25/85, about 70.6% and 29.4%.
Top-p=0.8 does not mean each retained token gets 80% probability.

## Order and conventions

Temperature divides **logits**, not probabilities. Then top-k runs, then
top-p runs on the remaining distribution. Softmax produces final weights
for `torch.multinomial`.

`sample_token` defaults to greedy (`temperature=0`), which consumes no RNG
state. `sampling_probs` defaults to temperature 1 for distribution inspection.
`top_k=0` and `top_p=1` disable their filters. Invalid settings still raise
even if greedy would otherwise ignore filtering.

Top-k uses the kth score as a cutoff, matching HF. Ties at the cutoff can
retain more than k tokens. Greedy ties choose the first (lowest-index) max.

Top-p follows HF's ascending-tail convention: sort ascending, remove tokens
whose cumulative probability is <= 1-p, and always keep at least one token.
For exact cumulative boundaries this can differ from tutorials that shift
a descending `cumsum > p` mask. Tests include the boundary and HF parity.

## Precision, randomness and validation

FP16/BF16 scores upcast to FP32 before scaling and softmax. FP64 remains
FP64. Negative infinity is accepted as an already-masked score. NaN,
positive infinity, all-masked rows and scaling overflow raise clear errors.
The input logits are never modified.

A seeded generator must match the logits device. Repeated seeds reproduce
results on the same device/software, not necessarily across CPU/GPU or
versions. Different seeds change draws, not the probability distribution.

Tests verify hand distributions, temperature, filters/ties, exact HF
warper parity, invalid inputs, reproducibility and GPU generator use.
An empirical 20,000-draw check uses absolute tolerance 0.015: its largest
standard error is about 0.0035, so the bound is over four standard errors.
FP32 probability comparisons use absolute 1e-6.

The notebook needs no model weights. It records probability tables and
frequency ranges across three independent seeded trials, plus pytest output
and environment information, in `results/C1_sampling.json`.
Runtime validation awaits the user's Colab run.

### Recorded Colab run (2026-10-31)

All **42 tests passed**, none skipped. Same-seed replay matched exactly.
Across three 4000-draw trials, observed ranges were Paris 58.98-59.95%,
London 24.13-26.18%, Berlin 9.73-10.75%, Rome 5.10-5.18%, consistent with
the synthetic target distribution and the statistical tolerance.

## Interview questions

1. **Temperature versus top-k/top-p?** Temperature reshapes probabilities;
   filters remove candidates before sampling.
2. **Top-k versus top-p?** K uses a fixed score-rank cutoff; p adapts the
   number of candidates to their probability mass.
3. **Does a seed ensure the same output everywhere?** No. It makes a random
   sequence reproducible within the same device/software conditions.
