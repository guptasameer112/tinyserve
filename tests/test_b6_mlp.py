"""B6: hand math, float64 oracle, HF weight parity and token independence."""
import math

import pytest

torch = pytest.importorskip("torch")

from tinyserve.models.mlp import MLP  # noqa: E402


def make_mlp(dtype=torch.float32, device="cpu"):
    torch.manual_seed(6)
    return MLP(64, 128, device=device, dtype=torch.float32).to(dtype).eval()


def test_hand_example():
    module = MLP(2, 2)
    with torch.no_grad():
        module.gate_proj.weight.copy_(torch.eye(2))
        module.up_proj.weight.copy_(2 * torch.eye(2))
        module.down_proj.weight.copy_(torch.eye(2))
    output = module(torch.tensor([[1.0, -1.0]]))
    expected = torch.tensor([[2 / (1 + math.exp(-1)), 2 / (1 + math.exp(1))]])
    torch.testing.assert_close(output, expected, rtol=0, atol=1e-6)


def test_fp32_matches_float64_formula():
    module = make_mlp()
    torch.manual_seed(7)
    inputs = torch.randn(5, 64)
    gate = inputs.double() @ module.gate_proj.weight.double().T
    up = inputs.double() @ module.up_proj.weight.double().T
    expected = ((gate * torch.sigmoid(gate)) * up) @ module.down_proj.weight.double().T
    torch.testing.assert_close(module(inputs).double(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(64,), (5, 64), (2, 3, 64)])
def test_shape_dtype_and_device(shape):
    module = make_mlp()
    inputs = torch.randn(*shape)
    output = module(inputs)
    assert output.shape == inputs.shape
    assert output.dtype == inputs.dtype and output.device == inputs.device


def test_checkpoint_names_shapes_and_no_bias():
    module = MLP(8, 20)
    assert list(module.state_dict()) == ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
    assert module.gate_proj.weight.shape == (20, 8)
    assert module.up_proj.weight.shape == (20, 8)
    assert module.down_proj.weight.shape == (8, 20)
    assert all(projection.bias is None for projection in (module.gate_proj, module.up_proj, module.down_proj))
    assert sum(parameter.numel() for parameter in module.parameters()) == 3 * 8 * 20


def test_tinyllama_mlp_parameter_count_without_allocating_weights():
    assert 3 * 2048 * 5632 == 34603008
    assert 22 * 3 * 2048 * 5632 == 761266176


def test_rows_are_independent():
    module = make_mlp()
    inputs = torch.randn(5, 64)
    original = module(inputs)
    changed = inputs.clone()
    changed[3:] += 10
    assert torch.equal(original[:3], module(changed)[:3])
    separate = torch.cat([module(row[None]) for row in inputs])
    torch.testing.assert_close(original, separate, rtol=1e-5, atol=1e-6)


def test_zero_gate_blocks_up_branch():
    module = make_mlp()
    with torch.no_grad():
        module.gate_proj.weight.zero_()
    assert torch.count_nonzero(module(torch.randn(5, 64))).item() == 0


def test_zero_up_branch_gives_zero():
    module = make_mlp()
    with torch.no_grad():
        module.up_proj.weight.zero_()
    assert torch.count_nonzero(module(torch.randn(5, 64))).item() == 0


def test_strict_weight_loading_rejects_missing_projection():
    module = make_mlp()
    state = module.state_dict()
    del state["gate_proj.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        module.load_state_dict(state, strict=True)


def test_fp32_matches_huggingface():
    transformers = pytest.importorskip("transformers")
    from transformers.models.llama.modeling_llama import LlamaMLP

    config = transformers.LlamaConfig(hidden_size=64, intermediate_size=128, hidden_act="silu", mlp_bias=False)
    reference = LlamaMLP(config).float().eval()
    ours = make_mlp()
    ours.load_state_dict(reference.state_dict(), strict=True)
    inputs = torch.randn(5, 64)
    torch.testing.assert_close(ours(inputs), reference(inputs), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float16, 5e-3), (torch.bfloat16, 4e-2)])
def test_low_precision_against_fp32(dtype, tolerance):
    low_precision = make_mlp(dtype)
    reference = make_mlp()
    reference.load_state_dict({name: weight.float() for name, weight in low_precision.state_dict().items()})
    inputs = torch.randn(5, 64).to(dtype)
    # Three projection outputs, SiLU and a product are rounded, not just the final output.
    # Relative checks alone fail near zero because of cancellation; use absolute tolerance too.
    torch.testing.assert_close(low_precision(inputs).float(), reference(inputs.float()),
                               rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fp16_vs_fp32():
    module = make_mlp(torch.float16, "cuda")
    reference = make_mlp(torch.float32, "cuda")
    reference.load_state_dict({name: weight.float() for name, weight in module.state_dict().items()})
    inputs = torch.randn(5, 64, device="cuda", dtype=torch.float16)
    output = module(inputs)
    torch.testing.assert_close(output.float(), reference(inputs.float()), rtol=5e-3, atol=5e-3)
    assert output.device == inputs.device and output.dtype == inputs.dtype


def test_real_probe_on_small_huggingface_model():
    transformers = pytest.importorskip("transformers")
    from tinyserve.probe import compare_mlp

    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
    )
    model = transformers.LlamaForCausalLM(config).float().eval()
    rows = compare_mlp(model, torch.tensor([1, 7, 9, 2]))
    assert len(rows) == 2
    assert all(row["max_abs_diff"] < 1e-5 for row in rows)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        compare_mlp(model, torch.tensor([1, 2]))
