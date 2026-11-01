# 06. Rotary position embeddings (B3)

Code: [rope.py](../../tinyserve/models/rope.py).
Tests: [test_b3_rope.py](../../tests/test_b3_rope.py).
Notebook: [B3_rope.ipynb](../../notebooks/B3_rope.ipynb).
Recorded results: [B3_rope.json](../../results/B3_rope.json).

## Give attention position information

A query/key dot product does not know token positions by itself. RoPE
rotates queries and keys by position-dependent angles. It does not rotate
values or add a position vector to the embedding.

TinyLlama uses head dimension 64. Hugging Face's Llama convention pairs
dimension `i` with `i + 32`, not adjacent dimensions. There are 32 pairs.

For pair `i`, position `p`, head dimension `D`, and base `theta`:

```
inv_freq[i] = 1 / theta ** (2 * i / D)
angle[i] = p * inv_freq[i]

rotate_half([first_half, second_half]) = [-second_half, first_half]
rotated = x * cos(angle) + rotate_half(x) * sin(angle)
```

Duplicate the 32 angles to obtain 64 cosine/sine entries. Both dimensions
of each pair use the same angle. A two-dimensional rotation preserves
vector length; position zero is the identity.

## Hand example

For `D=4`, `theta=10000`, the two pairs are `(0, 2)` and `(1, 3)`.
Their speeds are 1 and 0.01 radians per position.

```
x at position 1 = [1, 0, 0, 0]
pair (0, 2)    = [1, 0]
after rotation = [cos(1), sin(1)]
full output    = [0.5403, 0, 0.8415, 0]
```

Using adjacent pairs would give a different convention and would not
match these checkpoint weights.

## Fast and slow pairs

With TinyLlama's `D=64` and `theta=10000`:

| Pair | Radians per position | Positions per full turn |
|---|---:|---:|
| 0 | 1 | 6.283 |
| 16 | 0.01 | 628.319 |
| 31 | approximately 0.000133352 | approximately 47,117.24 |

Fast pairs resolve short distances. Slow pairs change gradually across the
2,048-token context. The model combines all frequencies; a fast pair's
full turn is not a full-vector repetition.

## Why the score depends on relative position

For a rotation matrix `R`, with position-scaled angles:

```
dot(R_m q, R_n k) = q.T @ R_m.T @ R_n @ k
                  = q.T @ R_(n-m) @ k
```

The positional part depends on `n-m`, not the absolute locations.
Actual queries and keys still depend on token content.

The notebook repeats one fixed query and one fixed key at many positions.
The score matrix should be constant along diagonals of equal offset.
FP32 angle and arithmetic rounding introduces small differences.

## Shapes, dtype and cache implications

Inputs use flattened layout:

```
positions: (N,)
queries:   (N, 32, 64)
keys:      (N, 4, 64)
cos/sin:   (N, 64), broadcast across heads
```

Positions are explicit. Packed sequences restart their own positions;
cached decode continues from the existing prefix length.
Angles and trigonometry use FP32, then cosine/sine are cast to the input
dtype, matching HF's recipe. `get_rope_theta` supports both the older
configuration field and Transformers 5's `rope_parameters`.

C3 stores keys after this rotation. Reuse each old key at its original
position without rotating it again. Values stay unrotated.

## Recorded validation

The saved October 31, 2026 Colab notebook records **16 passed tests**.
Tests cover hand values, identity, norm preservation, relative scores,
packed/large positions, HF FP32/FP16 comparisons and configuration lookup.

Real TinyLlama comparisons use layers 0, 10 and 21, with ordinary and
packed positions: six comparisons total. Cosine/sine, rotated queries and
rotated keys matched HF exactly in every recorded comparison.

The fixed-vector diagonal spread was `1.7166e-5`, with query/key norm
product approximately `69.2468`. The test bounds spread by `1e-4` times
that scale to account for FP32 angle rounding.

Synthetic FP32 HF comparisons use absolute `1e-6`. Same-recipe FP16
comparisons check exact equality on the tested inputs; this is not a
universal FP16 bit-equality requirement. A separate FP16-versus-FP32 check
uses the same rounded inputs and absolute `2e-2`.

The archived JSON preserves the original T4 environment, timestamp and
git SHA `335035c`. Its payload contains numerical comparisons, not pytest
stdout; the test count remains in the saved notebook output.

## Interview questions

1. **Why rotate queries and keys, but not values?** Position changes the
   similarity used for attention weights; values carry the mixed content.
2. **Why does RoPE encode relative position?** The two rotations combine
   into a rotation by their position difference in the dot product.
3. **What can break checkpoint parity?** Wrong pairing, frequency base,
   positions or dtype recipe. Cached keys must not be rotated twice.
