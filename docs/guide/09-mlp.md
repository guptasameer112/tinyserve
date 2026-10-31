# 09. Llama MLP and SwiGLU (B6)

Code: [mlp.py](../../tinyserve/models/mlp.py).
Tests: [test_b6_mlp.py](../../tests/test_b6_mlp.py).
Notebook: [B6_mlp.ipynb](../../notebooks/B6_mlp.ipynb).

## One token at a time

Attention mixes information across tokens. The MLP transforms each token
independently, with the same learned weights applied to every row.

```
gate = gate_proj(x)                   2048 -> 5632
up = up_proj(x)                       2048 -> 5632
mixed = silu(gate) * up               elementwise
output = down_proj(mixed)             5632 -> 2048
```

`silu(z) = z * sigmoid(z)`. This is SwiGLU: a SiLU-activated branch
modulates another branch. The gate is not a probability mask; it can be
negative and can exceed 1. There is no softmax or causal mask here.

An ordinary two-projection feed-forward network has one up projection,
an activation, and one down projection. SwiGLU has two up-sized branches
and one down projection.

## Hand example

Let `x=[1,-1]`, gate weights be identity, up weights be twice identity,
and down weights be identity. Then:

```
gate = [1, -1]
silu(gate) = [0.731059, -0.268941]
up = [2, -2]
output = [1.462117, 0.537883]
```

The second result is positive because two negative numbers multiply.

## Weights and shapes

PyTorch stores linear weights as `(output_features, input_features)`:

| Name | TinyLlama shape | Parameters per layer |
|---|---|---|
| gate_proj.weight | `(5632, 2048)` | 11,534,336 |
| up_proj.weight | `(5632, 2048)` | 11,534,336 |
| down_proj.weight | `(2048, 5632)` | 11,534,336 |

Total: **34,603,008 per layer**, **761,266,176 across 22 layers**,
about 69.2% of TinyLlama's parameters. With uniform FP16 storage that is
also 69.2% of its weight bytes, not necessarily 69.2% of execution time.

The module names match HF checkpoint keys and uses strict state-dict
loading. There are no biases. Input/output preserve leading dimensions,
device and dtype; `(N, 2048)` is the engine's flattened-token convention.

## Verification and numerics

- Hand example and an explicit float64 formula.
- FP32 vs HF `LlamaMLP`, with the same weights.
- Zero gate/up branches, independent rows and strict loading errors.
- FP16/BF16 vs FP32 using the same rounded inputs and weights.
- All 22 real HF MLP inputs fed to our module in FP32; tolerance
  `rtol=atol=1e-5`, enforced by the probe.

The module uses the same operation order as HF, without special FP32
upcasting in its forward pass. Unlike B4's reference attention, multiple
intermediate results are rounded in FP16.

Small synthetic FP16 tests use `rtol=atol=5e-3`, BF16 `4e-2`: several
intermediate roundings plus cancellation near zero require both relative
and absolute allowances. These are bounds for the controlled test inputs,
not universal error guarantees. Real-model parity uses FP32 and tighter
checks instead. Do not loosen a failed tolerance without inspecting why.

Runtime validation is pending the Colab run. The notebook saves test output,
environment metadata and the range of differences across all tested layers
to `results/B6_mlp.json`. This is a correctness lesson, not a speed benchmark.

## Interview questions

1. **What is SwiGLU?** `down(silu(gate(x)) * up(x))`, with separate learned
   gate/up branches and an elementwise product.
2. **Does an MLP mix tokens?** No. It mixes features within each token;
   attention handles cross-token interaction.
3. **Why does the MLP hold most of TinyLlama's weights?** Three large
   hidden-to-intermediate-sized matrices per layer, versus smaller
   attention projections, especially K/V under GQA.
