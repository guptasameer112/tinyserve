# 11. Full model, loading and FP32 parity (B8)

Code: [llama.py](../../tinyserve/models/llama.py).
Tests: [test_b8_model.py](../../tests/test_b8_model.py).
Notebook: [B8_full_model.ipynb](../../notebooks/B8_full_model.ipynb).

## Complete path

```
input IDs (N,)
 -> embedding lookup (N, 2048)
 -> 22 decoder layers (N, 2048)
 -> final RMSNorm (N, 2048)
 -> lm_head (N, 32000)
```

Each logits row predicts the token after that position. Logits are scores,
not probabilities; greedy selection takes the largest score. TinyLlama
uses separate embedding and LM-head weights. Tied embeddings are also
supported and tested: the two modules then share one parameter.

The model forward requires `input_ids`, explicit `positions`, and an
`AttentionMetadata` object. The positions must agree with the metadata's
B7 position field. Packed requests restart positions and provide request
IDs. Nothing in the model forward calls HF or its attention routines.

## Weight loading

The hierarchy matches checkpoint keys: `model.embed_tokens`, `model.layers`,
`model.norm` and `lm_head`. `from_huggingface` creates our architecture on the
source device/dtype and copies the state dictionary with `strict=True`.
Missing, unexpected or incorrectly shaped weights raise errors.

HF is the weight/tokenizer source and test oracle, not our inference engine.
Only default unscaled RoPE and the bias-free SiLU Llama architecture are
supported. Cache, sampling and serving are later lessons.

## What B8 verifies beyond B7

B7 started every comparison from an HF layer input. B8 instead feeds each
of **our layer outputs into our next layer**, then applies our final norm
and LM head. Error can accumulate across the stack.

The notebook compares all logits and all row argmax IDs for three prompts.
It then checks the two models at every greedy step on the same growing
prefix, stopping on EOS or the configured test length. A token mismatch
raises even if logits are numerically close.

FP32 comparison uses `rtol=1e-5, atol=1e-4`: accumulated rounding through
22 layers needs a larger absolute allowance than isolated operations.
The maximum difference and HF's last-row top-two margin are recorded.
Close top candidates can change argmax despite small errors, which is why
token equality is a separate required check, not implied by tolerance.

Synthetic tests also cover explicit shifted positions, packed isolation,
causality, strict loading, tied weights and greedy parity against HF
`generate(use_cache=False)`. A controlled small CUDA FP16 comparison uses
the same rounded weights in FP32 with `rtol=atol=1e-2`, not token equality.
FP16 logits are returned in FP32, but that cast does not restore lost precision.

## Colab memory and scope

Two FP32 TinyLlama copies occupy about 8.8 GB of weight memory plus
activations and temporary tensors. Close old GPU-heavy notebook kernels.
Use short prompts, and free both models after the run. This notebook is
correctness-focused, not a speed comparison or a public generator.

The notebook saves the test summary, environment, prompt comparisons,
greedy token IDs and error ranges to `results/B8_full_model.json`.
Runtime verification awaits the user's Colab run.

### Recorded Colab run (2026-10-31)

All **143 tests passed**, none skipped. All 36 greedy steps (12 per prompt)
matched HF exactly. Logits differences across prompts/steps ranged from
`7.15e-6` to `1.67e-5`. Packed-vs-separate difference was `4.58e-5`, within
tolerance, with equal token choices. Both models had 1,100,048,384 parameters.

An earlier memory-constrained runtime repeatedly lost its kernel. Model
loading succeeded on a fresh T4 runtime with 11.10 GiB free system RAM and
an almost empty GPU. Resetting a notebook kernel alone is not the same as
replacing the whole Colab runtime. Save results before deleting a runtime.

## Interview questions

1. **Why is full-model parity necessary?** Correct isolated blocks do not
   prove correct wiring, weight loading or accumulated stack behavior.
2. **Why require token parity separately from logits tolerance?** Tiny
   numerical differences can flip the two highest-scoring candidates.
3. **What does HF still do?** Supply trained weights/tokenization and the
   independent reference. Our forward implements the entire model.
