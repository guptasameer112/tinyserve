"""B5: head mapping, HF parity, independent per-head math and request isolation."""
import pytest

torch = pytest.importorskip("torch")

from tinyserve.models.attention import causal_mask, reference_attention  # noqa: E402
from tinyserve.models.gqa import expand_kv_heads, grouped_query_attention  # noqa: E402


def inputs(query_tokens=5, key_tokens=5, query_heads=8, kv_heads=2, dimension=16, dtype=torch.float32):
    generator = torch.Generator().manual_seed(17)
    return (
        torch.randn(query_tokens, query_heads, dimension, generator=generator).to(dtype),
        torch.randn(key_tokens, kv_heads, dimension, generator=generator).to(dtype),
        torch.randn(key_tokens, kv_heads, dimension, generator=generator).to(dtype),
    )


def test_tinyllama_head_mapping_is_consecutive_not_cyclic():
    tensor = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    output = expand_kv_heads(tensor, 32)
    assert output[0, :, 0].tolist() == [0.0] * 8 + [1.0] * 8 + [2.0] * 8 + [3.0] * 8
    assert output.shape == (1, 32, 1)
    assert tensor.shape == (1, 4, 1)


@pytest.mark.parametrize("query_heads,kv_heads", [(4, 4), (8, 2), (32, 4), (8, 1)])
def test_expansion_matches_huggingface(query_heads, kv_heads):
    pytest.importorskip("transformers")
    from transformers.models.llama.modeling_llama import repeat_kv

    _, key, _ = inputs(query_heads=query_heads, kv_heads=kv_heads)
    reference = repeat_kv(key.transpose(0, 1)[None], query_heads // kv_heads)[0].transpose(0, 1)
    assert torch.equal(expand_kv_heads(key, query_heads), reference)


def test_equal_head_counts_return_original_tensor():
    _, key, _ = inputs(query_heads=2, kv_heads=2)
    assert expand_kv_heads(key, 2) is key


def test_noncontiguous_tensor_preserves_values_dtype_and_device():
    tensor = torch.arange(24, dtype=torch.float64).reshape(3, 2, 4).transpose(0, 2)
    expanded = expand_kv_heads(tensor, 4)
    assert expanded.dtype == tensor.dtype and expanded.device == tensor.device
    assert torch.equal(expanded[:, 0], tensor[:, 0])
    assert torch.equal(expanded[:, 1], tensor[:, 0])
    assert torch.equal(expanded[:, 2], tensor[:, 1])
    assert torch.equal(expanded[:, 3], tensor[:, 1])


@pytest.mark.parametrize("query_heads,kv_heads", [(4, 4), (8, 2), (32, 4), (8, 1)])
def test_attention_matches_independent_per_head_float64(query_heads, kv_heads):
    query, key, value = inputs(query_heads=query_heads, kv_heads=kv_heads)
    mask = causal_mask(torch.arange(5))
    output, weights = grouped_query_attention(query, key, value, mask)
    repeats = query_heads // kv_heads
    expected_outputs, expected_weights = [], []
    for head in range(query_heads):
        kv_head = head // repeats
        head_output, head_weights = reference_attention(
            query[:, head:head + 1].double(), key[:, kv_head:kv_head + 1].double(),
            value[:, kv_head:kv_head + 1].double(), mask,
        )
        expected_outputs.append(head_output)
        expected_weights.append(head_weights)
    torch.testing.assert_close(output.double(), torch.cat(expected_outputs, dim=1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weights.double(), torch.cat(expected_weights), rtol=1e-5, atol=1e-6)


def test_shared_kv_does_not_mean_shared_attention_weights():
    query = torch.tensor([[[1.0], [-1.0]]])
    key = torch.tensor([[[0.0]], [[2.0]]])
    value = torch.tensor([[[10.0]], [[20.0]]])
    output, weights = grouped_query_attention(query, key, value, torch.ones(1, 2, dtype=torch.bool))
    assert weights[0, 0, 1] > weights[1, 0, 1]
    assert output[0, 0, 0] > output[0, 1, 0]


def test_changing_one_kv_head_only_changes_its_query_group():
    query, key, value = inputs()
    mask = causal_mask(torch.arange(5))
    original, _ = grouped_query_attention(query, key, value, mask)
    changed_value = value.clone()
    changed_value[:, 1] += 10
    changed, _ = grouped_query_attention(query, key, changed_value, mask)
    assert torch.equal(original[:, :4], changed[:, :4])
    torch.testing.assert_close(changed[:, 4:], original[:, 4:] + 10, rtol=1e-5, atol=1e-5)


def test_packed_gqa_matches_separate_requests():
    query, key, value = inputs()
    positions = torch.tensor([0, 1, 2, 0, 1])
    mask = causal_mask(positions, torch.tensor([0, 0, 0, 1, 1]))
    packed, weights = grouped_query_attention(query, key, value, mask)
    separate = []
    for start, end in ((0, 3), (3, 5)):
        output, _ = grouped_query_attention(query[start:end], key[start:end], value[start:end],
                                            causal_mask(positions[start:end]))
        separate.append(output)
    torch.testing.assert_close(packed, torch.cat(separate), rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(weights[:, ~mask]).item() == 0


def test_decode_last_query_matches_full_prefill():
    query, key, value = inputs()
    full, _ = grouped_query_attention(query, key, value, causal_mask(torch.arange(5)))
    last, _ = grouped_query_attention(query[-1:], key, value, torch.ones(1, 5, dtype=torch.bool))
    torch.testing.assert_close(last, full[-1:], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float16, 5e-4), (torch.bfloat16, 4e-3)])
def test_low_precision_matches_fp32(dtype, tolerance):
    query, key, value = inputs(dtype=dtype)
    mask = causal_mask(torch.arange(5))
    output, _ = grouped_query_attention(query, key, value, mask)
    reference, _ = grouped_query_attention(query.float(), key.float(), value.float(), mask)
    torch.testing.assert_close(output.float(), reference, rtol=tolerance, atol=tolerance)
    assert output.dtype == dtype


@pytest.mark.parametrize("tensor,heads", [
    (torch.ones(2, 3, 4), 8), (torch.ones(2, 4, 4), 2),
    (torch.ones(2, 2, 4), 0), (torch.ones(2, 2, 4), -1),
    (torch.ones(2, 2, 4), 4.0), (torch.ones(2, 2, 4), True),
    (torch.ones(2, 4), 8), (torch.ones(0, 2, 4), 8),
])
def test_invalid_head_expansion_raises(tensor, heads):
    with pytest.raises(ValueError):
        expand_kv_heads(tensor, heads)


def test_mismatched_original_kv_heads_raise_even_if_expansion_would_match():
    query, key, value = inputs()
    with pytest.raises(ValueError, match="before head expansion"):
        grouped_query_attention(query, key, value[:, :1], causal_mask(torch.arange(5)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fp16_expansion_and_attention():
    query, key, value = (tensor.cuda() for tensor in inputs(query_heads=32, kv_heads=4, dtype=torch.float16))
    expanded = expand_kv_heads(key, 32)
    assert expanded.device == key.device and expanded.dtype == key.dtype
    mask = causal_mask(torch.arange(5, device=query.device))
    output, _ = grouped_query_attention(query, key, value, mask)
    reference, _ = grouped_query_attention(query.float(), key.float(), value.float(), mask)
    torch.testing.assert_close(output.float(), reference, rtol=5e-4, atol=5e-4)


def test_real_probe_with_our_gqa():
    transformers = pytest.importorskip("transformers")
    from tinyserve.probe import compare_attention_layers

    config = transformers.LlamaConfig(
        hidden_size=128, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).float().eval()
    rows = compare_attention_layers(model, torch.tensor([1, 7, 9, 2]), layers=(0, 1))
    assert len(rows) == 2
    assert all(row["max_abs_diff"] < 1e-5 for row in rows)
