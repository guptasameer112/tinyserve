"""B4: hand math, independent float64 oracle, PyTorch parity and causality."""
import math

import pytest

torch = pytest.importorskip("torch")

from tinyserve.models.attention import causal_mask, reference_attention  # noqa: E402
from tinyserve.models.rope import apply_rope, rope_cos_sin  # noqa: E402


def inputs(tokens=5, heads=2, head_dim=4, dtype=torch.float32):
    generator = torch.Generator().manual_seed(42)
    return tuple(torch.randn(tokens, heads, head_dim, generator=generator).to(dtype) for _ in range(3))


def scalar_reference(query, key, value, mask):
    """Independent per-head/per-query dot products and weighted sums in float64."""
    output = torch.zeros_like(query, dtype=torch.float64)
    for token in range(query.shape[0]):
        visible = mask[token].nonzero().flatten()
        for head in range(query.shape[1]):
            dots = torch.stack([
                (query[token, head].double() * key[index, head].double()).sum()
                / math.sqrt(query.shape[-1]) for index in visible
            ])
            probabilities = (dots - dots.max()).exp()
            probabilities /= probabilities.sum()
            output[token, head] = sum(
                probabilities[index] * value[key_index, head].double()
                for index, key_index in enumerate(visible)
            )
    return output


def test_hand_example():
    query = torch.ones(2, 1, 1)
    key = torch.tensor([[[0.0]], [[math.log(3.0)]]])
    value = torch.tensor([[[10.0]], [[20.0]]])
    output, weights = reference_attention(query, key, value, causal_mask(torch.arange(2)))
    torch.testing.assert_close(weights[0], torch.tensor([[1.0, 0.0], [0.25, 0.75]]), rtol=0, atol=1e-6)
    torch.testing.assert_close(output[:, 0, 0], torch.tensor([10.0, 17.5]), rtol=0, atol=1e-6)


@pytest.mark.parametrize("tokens,heads,head_dim", [(1, 1, 1), (5, 2, 4), (9, 4, 64)])
def test_fp32_matches_independent_float64(tokens, heads, head_dim):
    query, key, value = inputs(tokens, heads, head_dim)
    mask = causal_mask(torch.arange(tokens))
    output, weights = reference_attention(query, key, value, mask)
    torch.testing.assert_close(output.double(), scalar_reference(query, key, value, mask), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(heads, tokens), rtol=0, atol=1e-6)
    assert torch.count_nonzero(weights[:, ~mask]).item() == 0
    assert output.shape == query.shape and output.dtype == query.dtype


def test_matches_pytorch_multihead_attention():
    query, key, value = inputs()
    tokens, heads, head_dim = query.shape
    hidden = heads * head_dim
    module = torch.nn.MultiheadAttention(hidden, heads, dropout=0.0, bias=False).eval()
    with torch.no_grad():
        module.in_proj_weight.copy_(torch.eye(hidden).repeat(3, 1))
        module.out_proj.weight.copy_(torch.eye(hidden))
    mask = causal_mask(torch.arange(tokens))
    reference, _ = module(query.reshape(tokens, 1, hidden), key.reshape(tokens, 1, hidden),
                          value.reshape(tokens, 1, hidden), attn_mask=~mask)
    output, _ = reference_attention(query, key, value, mask)
    torch.testing.assert_close(output.reshape(tokens, 1, hidden), reference, rtol=1e-5, atol=1e-6)


def test_matches_pytorch_sdpa():
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        pytest.skip("PyTorch 2+ SDPA unavailable; MultiheadAttention parity is tested separately")
    query, key, value = inputs(9, 4, 64)
    mask = causal_mask(torch.arange(9))
    reference = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), attn_mask=mask, dropout_p=0.0,
    )
    output, _ = reference_attention(query, key, value, mask)
    torch.testing.assert_close(output, reference.transpose(0, 1), rtol=1e-5, atol=1e-6)


def test_future_tokens_cannot_change_earlier_outputs():
    query, key, value = inputs()
    mask = causal_mask(torch.arange(5))
    original, _ = reference_attention(query, key, value, mask)
    changed_query, changed_key, changed_value = query.clone(), key.clone(), value.clone()
    for tensor in (changed_query, changed_key, changed_value):
        tensor[3:] += 100
    changed, _ = reference_attention(changed_query, changed_key, changed_value, mask)
    assert torch.equal(original[:3], changed[:3])
    assert not torch.equal(original[3:], changed[3:])


def test_packed_sequences_match_separate_runs():
    query, key, value = inputs(5)
    positions = torch.tensor([0, 1, 2, 0, 1])
    sequence_ids = torch.tensor([7, 7, 7, 9, 9])
    mask = causal_mask(positions, sequence_ids)
    packed, weights = reference_attention(query, key, value, mask)
    separate = []
    for start, end in ((0, 3), (3, 5)):
        output, _ = reference_attention(query[start:end], key[start:end], value[start:end],
                                        causal_mask(positions[start:end]))
        separate.append(output)
    torch.testing.assert_close(packed, torch.cat(separate), rtol=0, atol=1e-6)
    assert torch.count_nonzero(weights[:, :3, 3:]).item() == 0
    assert torch.count_nonzero(weights[:, 3:, :3]).item() == 0


def test_one_decode_query_can_read_all_cached_keys():
    query, key, value = inputs()
    full, _ = reference_attention(query, key, value, causal_mask(torch.arange(5)))
    last, weights = reference_attention(query[-1:], key, value, torch.ones(1, 5, dtype=torch.bool))
    torch.testing.assert_close(last, full[-1:], rtol=1e-5, atol=1e-6)
    assert weights.shape == (2, 1, 5)


def test_rope_then_attention_matches_float64():
    query, key, value = inputs(6, 2, 64)
    positions = torch.arange(6)
    cos, sin = rope_cos_sin(positions, 64, 10000.0, query.dtype)
    query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
    mask = causal_mask(positions)
    output, _ = reference_attention(query, key, value, mask)
    torch.testing.assert_close(output.double(), scalar_reference(query, key, value, mask), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_uses_fp32_math(dtype):
    query, key, value = inputs(9, 2, 64, dtype=dtype)
    mask = causal_mask(torch.arange(9))
    output, weights = reference_attention(query, key, value, mask)
    reference, _ = reference_attention(query.float(), key.float(), value.float(), mask)
    # One final output rounding: 2^-11 (FP16) or 2^-8 (BF16) relative error.
    tolerance = 5e-4 if dtype == torch.float16 else 4e-3
    torch.testing.assert_close(output.float(), reference, rtol=tolerance, atol=tolerance)
    assert output.dtype == dtype and weights.dtype == torch.float32


def test_large_fp16_scores_do_not_overflow():
    query = torch.full((2, 1, 64), 300.0, dtype=torch.float16)
    output, weights = reference_attention(query, query, torch.ones_like(query), causal_mask(torch.arange(2)))
    assert torch.isfinite(output).all() and torch.isfinite(weights).all()
    assert torch.equal(output, torch.ones_like(output))


def test_mask_uses_positions_not_flattened_row_numbers():
    positions = torch.tensor([2, 0, 1])
    assert causal_mask(positions).tolist() == [[True, True, True], [False, True, False], [False, True, True]]


@pytest.mark.parametrize("case", ["rank", "empty", "heads", "value_shape", "dtype", "mask_shape",
                                 "mask_dtype", "fully_masked"])
def test_invalid_attention_inputs_raise(case):
    query, key, value = inputs()
    mask = causal_mask(torch.arange(5))
    if case == "rank":
        query = query[:, 0]
    elif case == "empty":
        query = query[:0]
    elif case == "heads":
        key, value = key[:, :1], value[:, :1]
    elif case == "value_shape":
        value = value[:4]
    elif case == "dtype":
        query = query.long()
    elif case == "mask_shape":
        mask = mask[:1]
    elif case == "mask_dtype":
        mask = mask.float()
    else:
        mask[0] = False
    with pytest.raises(ValueError):
        reference_attention(query, key, value, mask)


@pytest.mark.parametrize("positions,sequence_ids", [
    (torch.empty(0, dtype=torch.int64), None),
    (torch.zeros(2, 2, dtype=torch.int64), None),
    (torch.tensor([0.0, 1.0]), None),
    (torch.arange(2), torch.arange(3)),
    (torch.arange(2), torch.tensor([0.0, 0.0])),
])
def test_invalid_mask_inputs_raise(positions, sequence_ids):
    with pytest.raises(ValueError):
        causal_mask(positions, sequence_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_device_and_fp16_parity():
    query, key, value = (tensor.cuda() for tensor in inputs(9, 2, 64, torch.float16))
    mask = causal_mask(torch.arange(9, device=query.device))
    output, _ = reference_attention(query, key, value, mask)
    reference, _ = reference_attention(query.float(), key.float(), value.float(), mask)
    torch.testing.assert_close(output.float(), reference, rtol=5e-4, atol=5e-4)
    assert output.device == query.device
    with pytest.raises(ValueError, match="device"):
        reference_attention(query, key, value, mask.cpu())


def test_real_attention_probe_on_tiny_huggingface_model():
    transformers = pytest.importorskip("transformers")
    from tinyserve.probe import compare_attention_layers

    config = transformers.LlamaConfig(
        hidden_size=128, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).float().eval()
    rows = compare_attention_layers(model, torch.tensor([1, 7, 9, 2]), layers=(0, 1))
    assert len(rows) == 2
    assert all(row["max_abs_diff"] < 1e-5 and row["future_weight_max"] == 0 for row in rows)


def test_attention_diagram(tmp_path):
    pytest.importorskip("matplotlib")
    from tinyserve.plots import plot_attention

    query, key, value = inputs(2, 1, 4)
    _, weights = reference_attention(query, key, value, causal_mask(torch.arange(2)))
    path = tmp_path / "attention.png"
    assert plot_attention(weights[0], ["first", "second"], path) == path
    assert path.stat().st_size > 0
