"""B8: strict loading, complete-stack parity, packed isolation and greedy tokens."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tinyserve.models.attention import AttentionMetadata  # noqa: E402
from tinyserve.models.llama import LlamaForCausalLM  # noqa: E402
from tinyserve.probe import compare_full_model, compare_greedy_steps  # noqa: E402
from tinyserve.weights import expected_param_count  # noqa: E402


def pair(tied=False, dtype=torch.float32, device="cpu", kv_heads=2):
    torch.manual_seed(8)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=kv_heads, vocab_size=64,
        max_position_embeddings=128, tie_word_embeddings=tied,
        bos_token_id=1, eos_token_id=2, pad_token_id=None,
    )
    config._attn_implementation = "eager"
    reference = transformers.LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
    return LlamaForCausalLM.from_huggingface(reference), reference


def forward(model, ids, positions=None, sequence_ids=None):
    if positions is None:
        positions = torch.arange(ids.numel(), device=ids.device)
    return model(ids, positions, AttentionMetadata(positions, sequence_ids))


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
def test_fp32_full_stack_matches_hf(kv_heads):
    ours, reference = pair(kv_heads=kv_heads)
    result = compare_full_model(ours, reference, torch.tensor([1, 7, 9, 12]))
    assert result["argmax_equal"] and result["max_abs_diff"] < 1e-5


@pytest.mark.parametrize("tied", [False, True])
def test_strict_checkpoint_loading_and_parameter_count(tied):
    ours, reference = pair(tied=tied)
    assert set(ours.state_dict()) == set(reference.state_dict())
    for name, tensor in ours.state_dict().items():
        assert torch.equal(tensor, reference.state_dict()[name])
    assert sum(parameter.numel() for parameter in ours.parameters()) == expected_param_count(reference.config)
    assert (ours.lm_head.weight is ours.model.embed_tokens.weight) == tied


def test_wrong_and_missing_weights_raise():
    ours, reference = pair()
    state = dict(reference.state_dict())
    del state["model.norm.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        ours.load_state_dict(state, strict=True)
    state = dict(reference.state_dict())
    state["lm_head.weight"] = state["lm_head.weight"][:2]
    with pytest.raises(RuntimeError, match="size mismatch"):
        ours.load_state_dict(state, strict=True)


def test_embedding_final_norm_and_lm_head_wiring():
    ours, _ = pair()
    ids = torch.tensor([1, 7, 9])
    layout = AttentionMetadata(torch.arange(3))
    hidden = ours.model.embed_tokens(ids)
    for decoder in ours.model.layers:
        hidden = decoder(hidden, layout)
    expected = ours.lm_head(ours.model.norm(hidden)).float()
    assert torch.equal(ours(ids, layout.positions, layout), expected)
    assert expected.shape == (3, 64)


def test_full_model_causality():
    ours, _ = pair()
    ids = torch.tensor([1, 7, 9, 12, 13])
    changed = ids.clone()
    changed[3:] = torch.tensor([20, 21])
    assert torch.equal(forward(ours, ids)[:3], forward(ours, changed)[:3])


def test_packed_full_model_matches_separate_requests():
    ours, _ = pair()
    ids = torch.tensor([1, 7, 9, 1, 12])
    positions = torch.tensor([0, 1, 2, 0, 1])
    sequence_ids = torch.tensor([0, 0, 0, 1, 1])
    packed = forward(ours, ids, positions, sequence_ids)
    separate = torch.cat([forward(ours, ids[:3]), forward(ours, ids[3:])])
    torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)


def test_explicit_positions_match_hf():
    ours, reference = pair()
    ids = torch.tensor([1, 7, 9])
    positions = torch.tensor([10, 11, 12])
    expected = reference(ids[None], position_ids=positions[None], use_cache=False).logits[0]
    torch.testing.assert_close(forward(ours, ids, positions), expected, rtol=1e-5, atol=1e-6)


def test_token_exact_greedy_matches_hf_generate():
    ours, reference = pair()
    # Disable EOS to require a full multi-step comparison in this synthetic test.
    reference.config.eos_token_id = None
    reference.generation_config.eos_token_id = None
    reference.generation_config.pad_token_id = 0
    ids = torch.tensor([1, 7, 9])
    output, rows = compare_greedy_steps(ours, reference, ids, max_new_tokens=6)
    expected = reference.generate(ids[None], attention_mask=torch.ones_like(ids)[None],
                                  do_sample=False, max_new_tokens=6, use_cache=False)[0]
    assert torch.equal(output, expected)
    assert len(rows) == 6


@pytest.mark.parametrize("ids", [
    torch.empty(0, dtype=torch.int64), torch.tensor([[1, 2]]),
    torch.tensor([1.0, 2.0]), torch.tensor([-1, 2]), torch.tensor([64, 2]),
])
def test_invalid_token_ids_raise(ids):
    ours, _ = pair()
    with pytest.raises(ValueError):
        forward(ours, ids)


def test_mismatched_metadata_positions_raise():
    ours, _ = pair()
    with pytest.raises(ValueError, match="must agree"):
        ours(torch.tensor([1, 2]), torch.arange(2), AttentionMetadata(torch.tensor([1, 2])))


def test_negative_positions_raise():
    ours, _ = pair()
    with pytest.raises(ValueError, match="nonnegative"):
        forward(ours, torch.tensor([1, 2]), torch.tensor([-1, 0]))


def test_unsupported_rope_raises():
    _, reference = pair()
    reference.config.rope_parameters = {"rope_type": "linear", "rope_theta": 10000.0, "factor": 2.0}
    with pytest.raises(ValueError, match="unscaled"):
        LlamaForCausalLM(reference.config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_full_model_vs_same_weights_in_fp32():
    low_precision, reference = pair(dtype=torch.float16, device="cuda")
    float_model = LlamaForCausalLM(reference.config, device="cuda", dtype=torch.float32).eval()
    float_model.load_state_dict({name: weight.float() for name, weight in low_precision.state_dict().items()})
    ids = torch.tensor([1, 7, 9], device="cuda")
    output = forward(low_precision, ids)
    torch.testing.assert_close(output, forward(float_model, ids), rtol=1e-2, atol=1e-2)
    assert output.dtype == torch.float32 and output.device == ids.device
