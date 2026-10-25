# 05. RMSNorm (Lesson B2)

Code: [norm.py](../../tinyserve/models/norm.py). Tests: [test_b2_rmsnorm.py](../../tests/test_b2_rmsnorm.py) (13 pass). Probe helpers: [probe.py](../../tinyserve/probe.py). Results: [B2_rmsnorm.json](../../results/B2_rmsnorm.json).

## The operation

```
y = x / sqrt(mean(x^2) + eps) * weight        mean over the 2048 hidden values of ONE token
```

Input and output are `(N, 2048)`; every token is normalised independently. TinyLlama has 45 of them: an input norm and a post-attention norm in each of the 22 layers, plus a final norm. `eps = 1e-5`.

Hand example: `x = [3, 4]`, `mean(x^2) = 12.5`, `rms = 3.5355`, `y = [0.8485, 1.1314]`.

**RMSNorm vs LayerNorm.** LayerNorm computes the mean and the variance and subtracts the mean, then adds a bias; RMSNorm skips the mean subtraction and the bias, so it needs one reduction instead of two. The difference is visible: adding 5 to every input leaves LayerNorm's output unchanged (`[1.47, -0.08, ...]`), but changes RMSNorm's (`[1.28, -0.24, ...]` becomes `[1.32, 0.95, ...]`).

## Implementation details that matter

1. **Statistics in FP32.** FP16 tops out at 65504, so squaring any value above 255.9 overflows. An all-FP16 version of `x * rsqrt(mean(x^2) + eps)` on `x = 300` gives `mean = inf`, `rsqrt(inf) = 0`, and the output is **all zeros with no error**. Ours squares in FP32 and returns `1.0` (measured on the T4).
2. **Order of casts.** We normalise in FP32, cast back to the input dtype, and only then multiply by the weight, exactly as Hugging Face does. That makes FP16 outputs identical to HF, not merely close.
3. **Shape-agnostic, dtype-preserving.** Works on `(N, hidden)` or `(batch, seq, hidden)` and returns the input dtype. The module takes `device` and `dtype` arguments (PyTorch convention) and its parameter is named `weight`, matching the checkpoint (B1).

## Test results

| Check | Tolerance | Why |
|---|---|---|
| vs float64 formula (FP32) | atol 1e-6 | FP32 rounding only |
| vs HF `LlamaRMSNorm` (FP32) | 1e-6 | same operations |
| vs HF (FP16) | exact equality | same operation order |
| FP16 vs FP32 computation | rtol 2e-3, atol 1e-3 | two FP16 roundings, each at most 2^-11 relative |
| scale invariance (`eps=0`) | atol 1e-5 | `norm(3x) == norm(x)` |
| FP16 overflow | all zeros (naive), 1.0 (ours) | the trap above |

**Real model.** We recorded the input and output of all 45 norms during one forward pass of "The capital of France is" (FP16, T4) and fed each real input to our module with the same weight: **45 of 45 outputs exactly equal, worst difference 0.0**.

## What the real activations look like

The residual-stream values entering the norms (one 6-token prompt, so treat this as an observation, not a general rule):

| Where | Input RMS | Input max abs |
|---|---|---|
| layers 0 to 2 | 0.010 to 0.028 | 0.1 to 0.9 |
| layers 3 to 7 | 1.589 | **140.6** |
| layers 8 to 17 | 1.58 | 139.5 to 138.8 |
| layers 18 to 21 | 1.59 to 1.62 | 138.4 to 134.1 |
| final norm | 0.727 | 20.7 |

- The RMS jumps from 0.028 to 1.589 between layer 2's post-attention norm and layer 3's input norm, so layer 2's MLP writes a very large value into the residual stream; from then on the maximum stays at about 140 until the last layers.
- That single value makes up most of the energy: the squared RMS over the `6 x 2048` input elements gives a total of about 31,000, and `140.6^2 = 19,775`, about 64% of it. The largest element is about 88 times the RMS.
- This looks like the "massive activations" effect reported for large language models. We did not check which token and dimension it is in, so that link is a hypothesis.
- For FP16: `140.6^2 = 19,775` is below 65,504, but only by a factor of 3.3. Computing the statistic in FP32 is not a theoretical nicety here.

## Code map

- [norm.py](../../tinyserve/models/norm.py): `rms_norm`, `RMSNorm`.
- [probe.py](../../tinyserve/probe.py): `capture_module_io` (hooks) and `compare_rmsnorm` (our module vs HF on real activations). We will reuse this for RoPE, attention, the MLP and the full model.
- [B2 notebook](../../notebooks/B2_rmsnorm.ipynb).

## Interview questions

1. **How does RMSNorm differ from LayerNorm, and why do Llama models use it?**
   RMSNorm divides by the root-mean-square only: no mean subtraction and no bias. It needs one reduction instead of two and has fewer parameters, and in practice trains as well as LayerNorm. A consequence is that it is not invariant to adding a constant to the input, while LayerNorm is.

2. **Why is the RMSNorm statistic computed in FP32 when the model runs in FP16?**
   Squaring overflows FP16 above 255.9 (65504 maximum), and one overflow turns the mean into `inf` and the output into zeros without any error. Real TinyLlama activations reach about 140 (`140^2 = 19,775`, only 3.3x from the limit). We upcast the statistic, then cast back to the input dtype before applying the weight.

3. **How did you verify your RMSNorm is correct?**
   Four levels: a hand-computed example, a float64 formula, the Hugging Face `LlamaRMSNorm` (tight 1e-6 tolerance in FP32, exact equality in FP16 because the operation order matches), and all 45 norms of the real model fed with real captured activations (45 of 45 exactly equal). FP16 versus FP32 is checked with a tolerance derived from FP16 rounding rather than demanded bit-equal.
