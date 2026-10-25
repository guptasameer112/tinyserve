"""Lesson B2 tests. FP32 tests against the reference are tight (1e-6); FP16 is compared against an FP32
computation with a tolerance derived from FP16 rounding, not demanded bit-equal."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.models.llama.modeling_llama import LlamaRMSNorm  # noqa: E402

from tinyserve.models.norm import RMSNorm, rms_norm  # noqa: E402
from tinyserve.probe import capture_module_io, compare_rmsnorm  # noqa: E402

HIDDEN = 64


def random_weight(hidden=HIDDEN, dtype=torch.float32, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(hidden, generator=generator) + 0.5).to(dtype)


def random_input(rows=5, hidden=HIDDEN, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, hidden, generator=generator).to(dtype)


def test_hand_computed_example():
    # x = [3, 4]: mean(x^2) = 12.5, rms = sqrt(12.5) = 3.5355339
    out = rms_norm(torch.tensor([[3.0, 4.0]]), torch.ones(2), eps=0.0)
    torch.testing.assert_close(out, torch.tensor([[3 / 12.5 ** 0.5, 4 / 12.5 ** 0.5]]), rtol=0, atol=1e-6)


def test_matches_float64_formula():
    x, w = random_input(), random_weight()
    expected = x.double() / torch.sqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-5) * w.double()
    torch.testing.assert_close(rms_norm(x, w, 1e-5).double(), expected, rtol=0, atol=1e-6)


def test_fp32_matches_huggingface():
    x, w = random_input(rows=7, hidden=2048), random_weight(hidden=2048)
    reference = LlamaRMSNorm(2048, eps=1e-5)
    reference.weight.data.copy_(w)
    torch.testing.assert_close(rms_norm(x, w, 1e-5), reference(x), rtol=1e-6, atol=1e-6)


def test_fp16_matches_huggingface_exactly():
    # Same operation order as Hugging Face (FP32 statistics, cast, then multiply): identical, not just close.
    x, w = random_input(rows=7, hidden=2048, dtype=torch.float16), random_weight(hidden=2048, dtype=torch.float16)
    reference = LlamaRMSNorm(2048, eps=1e-5).to(torch.float16)
    reference.weight.data.copy_(w)
    assert torch.equal(rms_norm(x, w, 1e-5), reference(x))


def test_fp16_close_to_fp32():
    # Two FP16 roundings (after normalising, after the weight) each cost at most 2^-11 relative error.
    x16, w16 = random_input(dtype=torch.float16), random_weight(dtype=torch.float16)
    fp16_out = rms_norm(x16, w16, 1e-5).float()
    fp32_out = rms_norm(x16.float(), w16.float(), 1e-5)
    torch.testing.assert_close(fp16_out, fp32_out, rtol=2e-3, atol=1e-3)


def test_scale_invariant_without_eps():
    x, w = random_input(), random_weight()
    torch.testing.assert_close(rms_norm(3.0 * x, w, 0.0), rms_norm(x, w, 0.0), rtol=0, atol=1e-5)


def test_unit_weight_gives_unit_rms():
    out = rms_norm(random_input(), torch.ones(HIDDEN), 1e-5)
    torch.testing.assert_close(out.pow(2).mean(-1).sqrt(), torch.ones(5), rtol=0, atol=1e-4)


def test_does_not_subtract_the_mean():
    x, w = random_input(), torch.ones(HIDDEN)
    layer_norm = torch.nn.LayerNorm(HIDDEN, eps=1e-5, elementwise_affine=False)
    torch.testing.assert_close(layer_norm(x + 5.0), layer_norm(x), rtol=0, atol=1e-5)  # LayerNorm ignores the shift
    assert not torch.allclose(rms_norm(x + 5.0, w, 1e-5), rms_norm(x, w, 1e-5), atol=1e-2)  # RMSNorm does not


def test_fp16_overflow_is_avoided_by_fp32_statistics():
    x = torch.full((1, 8), 300.0, dtype=torch.float16)  # 300^2 = 90000 > 65504 (FP16 max)
    naive = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)  # all in FP16
    assert torch.all(naive == 0)  # silent failure: inf mean -> rsqrt 0
    ours = rms_norm(x, torch.ones(8, dtype=torch.float16), 1e-5)
    torch.testing.assert_close(ours.float(), torch.ones(1, 8), rtol=2e-3, atol=1e-3)


def test_works_on_any_leading_shape_and_keeps_dtype():
    x = random_input(rows=6, dtype=torch.float16).reshape(2, 3, HIDDEN)
    out = rms_norm(x, random_weight(dtype=torch.float16), 1e-5)
    assert out.shape == x.shape and out.dtype == torch.float16


def test_module_creates_weight_with_requested_dtype_and_matches_function():
    norm = RMSNorm(HIDDEN, eps=1e-5, device="cpu", dtype=torch.float16)
    assert norm.weight.shape == (HIDDEN,) and norm.weight.dtype == torch.float16
    assert list(norm.state_dict()) == ["weight"]  # same checkpoint name as Hugging Face
    x = random_input(dtype=torch.float16)
    assert torch.equal(norm(x), rms_norm(x, norm.weight, 1e-5))


@pytest.fixture(scope="module")
def tiny_model():
    config = transformers.LlamaConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    for module in model.modules():  # make norm weights non-trivial so the test is meaningful
        if isinstance(module, LlamaRMSNorm):
            module.weight.data.uniform_(0.5, 1.5)
    return model


def test_capture_finds_every_norm_in_the_model(tiny_model):
    records = capture_module_io(tiny_model, torch.tensor([1, 5, 7, 9]), LlamaRMSNorm)
    assert len(records) == 2 * 2 + 1  # input + post-attention norm per layer, plus the final norm
    assert all(inp.shape == (1, 4, 32) for _, inp, _ in records)


def test_our_norm_matches_every_norm_in_a_tiny_model(tiny_model):
    rows = compare_rmsnorm(tiny_model, torch.tensor([1, 5, 7, 9]), LlamaRMSNorm, eps=tiny_model.config.rms_norm_eps)
    assert len(rows) == 5
    assert all(row["max_abs_diff"] < 1e-6 for row in rows)
