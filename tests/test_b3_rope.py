"""Lesson B3 tests. FP32 tests against Hugging Face are tight; the relative-position property is limited only by
FP32 rounding of the angle (position x frequency), which at position ~100 is about 100 * 2^-24 = 6e-6 rad."""
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb  # noqa: E402

from tinyserve.models.rope import (  # noqa: E402
    apply_rope, get_rope_theta, rope_cos_sin, rope_inv_freq, rotate_half,
)
from tinyserve.probe import compare_rope, diagonal_spread, relative_score_matrix  # noqa: E402

THETA = 10000.0
HEAD_DIM = 64


def random_tensor(*shape, dtype=torch.float32, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).to(dtype)


def test_inv_freq_values_for_head_dim_64():
    inv_freq = rope_inv_freq(HEAD_DIM, THETA, "cpu")
    assert inv_freq.shape == (32,) and inv_freq.dtype == torch.float32
    assert inv_freq[0].item() == 1.0  # fastest pair: one radian per position
    assert inv_freq[16].item() == pytest.approx(0.01, rel=1e-5)  # theta^(-0.5)
    assert inv_freq[31].item() == pytest.approx(THETA ** (-62 / 64), rel=1e-5)  # slowest pair
    assert bool(torch.all(inv_freq[1:] < inv_freq[:-1]))


def test_rotate_half_swaps_halves_with_a_sign():
    assert rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0])).tolist() == [-3.0, -4.0, 1.0, 2.0]


def test_hand_computed_example():
    # D=4, theta=10000 -> inv_freq = [1, 0.01]. x = [1,0,0,0] at position 1:
    # pair (0, 2) rotates by 1 rad: [cos 1, 0, sin 1, 0]
    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    cos, sin = rope_cos_sin(torch.tensor([1]), 4, THETA, torch.float32)
    out = apply_rope(x, cos, sin)
    torch.testing.assert_close(out[0, 0], torch.tensor([math.cos(1.0), 0.0, math.sin(1.0), 0.0]), rtol=0, atol=1e-6)


def test_position_zero_is_identity():
    x = random_tensor(4, 8, HEAD_DIM)
    cos, sin = rope_cos_sin(torch.zeros(4, dtype=torch.int64), HEAD_DIM, THETA, torch.float32)
    assert torch.equal(apply_rope(x, cos, sin), x)


def test_rotation_preserves_vector_length():
    x = random_tensor(6, 4, HEAD_DIM)
    cos, sin = rope_cos_sin(torch.tensor([0, 1, 7, 100, 1000, 2047]), HEAD_DIM, THETA, torch.float32)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_rows_depend_only_on_their_own_position():
    # In a flattened batch two sequences both restart at 0, so equal positions must give equal rows.
    cos, sin = rope_cos_sin(torch.tensor([3, 0, 3, 0]), HEAD_DIM, THETA, torch.float32)
    assert torch.equal(cos[0], cos[2]) and torch.equal(sin[1], sin[3])
    assert not torch.equal(cos[0], cos[1])


def test_score_depends_only_on_relative_position():
    q, k = random_tensor(HEAD_DIM, seed=1), random_tensor(HEAD_DIM, seed=2)
    scores = relative_score_matrix(q, k, 128, THETA)
    scale = q.norm().item() * k.norm().item()
    assert diagonal_spread(scores) < 1e-4 * scale  # the same offset gives the same score everywhere
    assert scores[5, 2].item() == pytest.approx(scores[105, 102].item(), abs=1e-4 * scale)
    assert abs(scores[5, 2].item() - scores[5, 4].item()) > 1e-3 * scale  # a different offset does differ


def _config(**overrides):
    values = dict(hidden_size=256, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, vocab_size=100, max_position_embeddings=4096)
    values.update(overrides)
    return transformers.LlamaConfig(**values)


def _hugging_face_rope(q, k, positions, config):
    rotary = LlamaRotaryEmbedding(config=config)
    q_hf, k_hf = q.transpose(0, 1)[None], k.transpose(0, 1)[None]
    cos, sin = rotary(q_hf, positions[None])
    q_ref, k_ref = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)
    return q_ref[0].transpose(0, 1), k_ref[0].transpose(0, 1)


POSITION_CASES = [
    torch.arange(9),
    torch.tensor([0, 1, 2, 3, 0, 1, 2, 5, 100]),  # packed sequences, arbitrary positions
    torch.tensor([0, 1500, 2047, 4000]),
]


@pytest.mark.parametrize("positions", POSITION_CASES)
def test_fp32_matches_huggingface(positions):
    config = _config()
    q, k = random_tensor(len(positions), 4, HEAD_DIM, seed=3), random_tensor(len(positions), 2, HEAD_DIM, seed=4)
    cos, sin = rope_cos_sin(positions, HEAD_DIM, get_rope_theta(config), torch.float32)
    q_ref, k_ref = _hugging_face_rope(q, k, positions, config)
    torch.testing.assert_close(apply_rope(q, cos, sin), q_ref, rtol=0, atol=1e-6)
    torch.testing.assert_close(apply_rope(k, cos, sin), k_ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("positions", POSITION_CASES)
def test_fp16_matches_huggingface_exactly(positions):
    # Same recipe as Hugging Face (FP32 angles, cast cos/sin to FP16, FP16 multiply-add): identical outputs.
    config = _config()
    q = random_tensor(len(positions), 4, HEAD_DIM, dtype=torch.float16, seed=5)
    k = random_tensor(len(positions), 2, HEAD_DIM, dtype=torch.float16, seed=6)
    cos, sin = rope_cos_sin(positions, HEAD_DIM, get_rope_theta(config), torch.float16)
    q_ref, k_ref = _hugging_face_rope(q, k, positions, config)
    assert torch.equal(apply_rope(q, cos, sin), q_ref)
    assert torch.equal(apply_rope(k, cos, sin), k_ref)


def test_fp16_close_to_fp32():
    # cos/sin are rounded to FP16 (relative error 2^-11), and so are the two products and their sum: about five
    # roundings, each at most 2^-11 x |value| (values reach ~5), so the absolute error can approach 1e-2.
    q32 = random_tensor(9, 4, HEAD_DIM)
    q16 = q32.to(torch.float16)
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 5, 100])
    out16 = apply_rope(q16, *rope_cos_sin(positions, HEAD_DIM, THETA, torch.float16)).float()
    out32 = apply_rope(q16.float(), *rope_cos_sin(positions, HEAD_DIM, THETA, torch.float32))
    torch.testing.assert_close(out16, out32, rtol=0, atol=2e-2)


def test_compare_rope_helper_on_a_tiny_model():
    config = _config(hidden_size=128, num_attention_heads=2, num_key_value_heads=1)  # head_dim 64
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    positions = torch.tensor([0, 1, 2, 0, 1, 40])
    q, k = random_tensor(6, 2, HEAD_DIM, seed=7), random_tensor(6, 1, HEAD_DIM, seed=8)
    result = compare_rope(model, q, k, positions)
    assert result["cos_sin_equal"] and result["q_max_abs_diff"] < 1e-6 and result["k_max_abs_diff"] < 1e-6


def test_get_rope_theta_reads_the_config():
    assert get_rope_theta(_config()) == 10000.0
