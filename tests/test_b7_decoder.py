"""B7: residual wiring, causality, packed isolation and HF decoder parity."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tinyserve.models.attention import AttentionMetadata  # noqa: E402
from tinyserve.models.llama import DecoderLayer  # noqa: E402


def config(**overrides):
    values = dict(hidden_size=64, intermediate_size=128, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-5,
                  rope_parameters={"rope_theta": 10000.0}, hidden_act="silu",
                  mlp_bias=False, attention_bias=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def layer(dtype=torch.float32, device="cpu"):
    torch.manual_seed(7)
    return DecoderLayer(config(), device=device, dtype=torch.float32).to(dtype).eval()


def metadata(tokens=5, device="cpu"):
    return AttentionMetadata(torch.arange(tokens, device=device))


def test_zero_branches_are_identity_and_do_not_mutate_input():
    module = layer()
    with torch.no_grad():
        module.self_attn.o_proj.weight.zero_()
        module.mlp.down_proj.weight.zero_()
    inputs = torch.randn(5, 64)
    original = inputs.clone()
    assert torch.equal(module(inputs, metadata()), original)
    assert torch.equal(inputs, original)


def test_residual_and_norm_order_matches_explicit_steps():
    module = layer()
    inputs = torch.randn(5, 64)
    layout = metadata()
    attention_delta = module.self_attn(module.input_layernorm(inputs), layout.positions, layout.allowed_mask())
    after_attention = inputs + attention_delta
    expected = after_attention + module.mlp(module.post_attention_layernorm(after_attention))
    assert torch.equal(module(inputs, layout), expected)


def test_mlp_is_added_to_unnormalized_residual():
    module = layer()
    with torch.no_grad():
        module.self_attn.o_proj.weight.zero_()
    inputs = torch.randn(5, 64) * 3
    expected = inputs + module.mlp(module.post_attention_layernorm(inputs))
    assert torch.equal(module(inputs, metadata()), expected)


def test_causality_across_entire_layer():
    module = layer()
    inputs = torch.randn(5, 64)
    changed = inputs.clone()
    changed[3:] += 10
    original_output, changed_output = module(inputs, metadata()), module(changed, metadata())
    assert torch.equal(original_output[:3], changed_output[:3])


def test_packed_sequences_match_separate_runs():
    module = layer()
    inputs = torch.randn(5, 64)
    layout = AttentionMetadata(torch.tensor([0, 1, 2, 0, 1]), torch.tensor([0, 0, 0, 1, 1]))
    packed = module(inputs, layout)
    separate = torch.cat([module(inputs[:3], metadata(3)), module(inputs[3:], metadata(2))])
    torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)


def test_shape_dtype_and_device():
    inputs = torch.randn(5, 64)
    output = layer()(inputs, metadata())
    assert output.shape == inputs.shape
    assert output.dtype == inputs.dtype and output.device == inputs.device


def test_checkpoint_structure():
    names = set(layer().state_dict())
    assert names == {
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    }


@pytest.mark.parametrize("overrides", [
    {"attention_bias": True}, {"mlp_bias": True}, {"hidden_act": "relu"},
    {"num_key_value_heads": 3}, {"head_dim": 15},
])
def test_unsupported_config_raises(overrides):
    with pytest.raises(ValueError):
        DecoderLayer(config(**overrides))


@pytest.mark.parametrize("inputs,layout", [
    (torch.ones(2, 3, 64), metadata(2)),
    (torch.ones(2, 32), metadata(2)),
    (torch.ones(2, 64), metadata(3)),
    (torch.ones(2, 64), AttentionMetadata(torch.tensor([0.0, 1.0]))),
])
def test_invalid_layout_raises(inputs, layout):
    with pytest.raises(ValueError):
        layer()(inputs, layout)


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
def test_fp32_huggingface_layer_parity(kv_heads):
    transformers = pytest.importorskip("transformers")
    from tinyserve.probe import compare_decoder_layers

    hf_config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=kv_heads, vocab_size=32,
    )
    hf_config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(hf_config).float().eval()
    rows = compare_decoder_layers(model, torch.tensor([1, 7, 9, 2]))
    assert len(rows) == 2
    assert all(row["max_abs_diff"] < 1e-5 for row in rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_cuda_against_fp32():
    module = layer(torch.float16, "cuda")
    reference = layer(torch.float32, "cuda")
    reference.load_state_dict({name: weight.float() for name, weight in module.state_dict().items()})
    inputs = torch.randn(5, 64, device="cuda", dtype=torch.float16)
    layout = metadata(5, "cuda")
    output = module(inputs, layout)
    torch.testing.assert_close(output.float(), reference(inputs.float(), layout), rtol=8e-3, atol=8e-3)
    assert output.dtype == torch.float16 and output.device == inputs.device
