"""Lesson B1 tests. The arithmetic checks are exact and need no torch. The state_dict check uses a tiny random Llama."""
import types

import pytest

from tinyserve.weights import (
    bytes_by_category, category, compare_shapes, expected_param_count, expected_shapes, head_dim,
    kv_cache_bytes_per_token, weight_tour,
)

TINYLLAMA = types.SimpleNamespace(
    vocab_size=32000, hidden_size=2048, intermediate_size=5632, num_hidden_layers=22,
    num_attention_heads=32, num_key_value_heads=4, head_dim=64, tie_word_embeddings=False,
)


def test_tinyllama_has_201_tensors():
    # 9 per layer x 22 layers + embed + final norm + lm_head
    assert len(expected_shapes(TINYLLAMA)) == 9 * 22 + 3


def test_tinyllama_param_count_matches_measured():
    assert expected_param_count(TINYLLAMA) == 1_100_048_384  # counted from the real model in A1


def test_one_layer_has_44_million_parameters():
    layer0 = {n: s for n, s in expected_shapes(TINYLLAMA).items() if n.startswith("model.layers.0.")}
    assert sum(s[0] * (s[1] if len(s) > 1 else 1) for s in layer0.values()) == 44_044_288


def test_gqa_shapes():
    shapes = expected_shapes(TINYLLAMA)
    assert shapes["model.layers.0.self_attn.q_proj.weight"] == (2048, 2048)
    assert shapes["model.layers.0.self_attn.k_proj.weight"] == (256, 2048)  # 4 KV heads x 64
    assert shapes["model.layers.0.self_attn.o_proj.weight"] == (2048, 2048)
    assert shapes["model.layers.0.mlp.down_proj.weight"] == (2048, 5632)


def test_kv_cache_is_22_kib_per_token_in_fp16():
    assert kv_cache_bytes_per_token(TINYLLAMA, 2) == 22528  # 22 KiB


def test_head_dim_falls_back_to_hidden_over_heads():
    config = types.SimpleNamespace(hidden_size=32, num_attention_heads=4)
    assert head_dim(config) == 8


def test_tied_embeddings_drop_lm_head():
    tied = types.SimpleNamespace(**dict(vars(TINYLLAMA), tie_word_embeddings=True))
    assert "lm_head.weight" not in expected_shapes(tied)


def test_compare_shapes_reports_each_kind_of_difference():
    diff = compare_shapes({"a": (2, 3), "b": (1,), "x": (9,)}, {"a": (2, 4), "b": (1,), "c": (5,)})
    assert diff["missing"] == ["c"] and diff["unexpected"] == ["x"]
    assert diff["wrong_shape"] == [("a", (2, 3), (2, 4))]


def test_category_and_bytes_by_category():
    assert [category(n) for n in (
        "model.embed_tokens.weight", "lm_head.weight", "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.mlp.up_proj.weight", "model.layers.3.input_layernorm.weight", "model.norm.weight",
    )] == ["embedding", "lm_head", "attention", "mlp", "norm", "norm"]
    fake = types.SimpleNamespace
    rows = weight_tour([
        ("model.layers.0.mlp.up_proj.weight", fake(shape=(2, 3), dtype="f16", numel=lambda: 6, element_size=lambda: 2)),
        ("model.layers.0.input_layernorm.weight", fake(shape=(3,), dtype="f16", numel=lambda: 3, element_size=lambda: 2)),
    ])
    totals = bytes_by_category(rows)
    assert totals["mlp"]["nbytes"] == 12 and totals["norm"]["nbytes"] == 6
    assert totals["mlp"]["share"] == pytest.approx(12 / 18)


try:
    import torch
    import transformers
except ImportError:
    torch = transformers = None


@pytest.mark.skipif(torch is None or transformers is None, reason="needs torch and transformers")
def test_expected_shapes_match_a_real_llama_state_dict():
    config = transformers.LlamaConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(config)
    actual = {name: tensor.shape for name, tensor in model.state_dict().items()}
    assert compare_shapes(actual, expected_shapes(config)) == {"missing": [], "unexpected": [], "wrong_shape": []}
    assert expected_param_count(config) == sum(p.numel() for p in model.parameters())


@pytest.mark.slow
@pytest.mark.skipif(torch is None or transformers is None, reason="needs torch and transformers")
def test_real_tinyllama_matches_spec():
    from tinyserve.basics import MODEL_ID

    config = transformers.AutoConfig.from_pretrained(MODEL_ID)
    assert expected_param_count(config) == 1_100_048_384
    assert kv_cache_bytes_per_token(config, 2) == 22528
