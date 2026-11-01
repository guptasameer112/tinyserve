"""C3: storage, append positions, causality, parity and generation contracts."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tinyserve.generation import generate  # noqa: E402
from tinyserve.models.attention import AttentionMetadata  # noqa: E402
from tinyserve.models.llama import LlamaForCausalLM  # noqa: E402
from tinyserve.models.rope import apply_rope, rope_cos_sin  # noqa: E402
from tinyserve.probe import compare_cached_steps  # noqa: E402
from tinyserve.weights import kv_cache_bytes_per_token  # noqa: E402


def pair(kv_heads=2, dtype=torch.float32, device="cpu"):
    torch.manual_seed(3)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=kv_heads, vocab_size=64,
        max_position_embeddings=32, eos_token_id=None, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    reference = transformers.LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
    return LlamaForCausalLM.from_huggingface(reference), reference


@torch.no_grad()
def forward(model, ids, cache=None, positions=None, sequence_ids=None):
    if positions is None:
        start = 0 if cache is None else cache.length
        positions = torch.arange(start, start + ids.numel(), device=ids.device)
    return model(ids, positions, AttentionMetadata(positions, sequence_ids, cache))


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("chunks", [(5,), (3, 1, 1), (1, 2, 2)])
def test_every_chunk_logit_matches_no_cache_and_hf(kv_heads, chunks):
    ours, reference = pair(kv_heads=kv_heads)
    ids = torch.tensor([1, 7, 9, 12, 13])
    cache = ours.new_kv_cache(5)
    offset = 0
    with torch.no_grad():
        for size in chunks:
            end = offset + size
            actual = forward(ours, ids[offset:end], cache)
            expected = forward(ours, ids[:end])[offset:]
            hf_logits = reference(ids[None, :end], use_cache=False).logits[0, offset:]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual, hf_logits, rtol=1e-5, atol=1e-6)
            assert torch.equal(actual.argmax(-1), expected.argmax(-1))
            assert cache.length == end
            offset = end


def test_storage_uses_kv_heads_and_exact_memory_cost():
    ours, _ = pair()
    cache = ours.new_kv_cache(8)
    assert cache.key.shape == cache.value.shape == (3, 8, 2, 16)
    assert cache.key.dtype == next(ours.parameters()).dtype
    assert cache.key.device == next(ours.parameters()).device
    assert cache.bytes_per_token == kv_cache_bytes_per_token(ours.config, 4)
    assert cache.allocated_bytes == cache.bytes_per_token * 8
    assert cache.length == 0
    assert not any("cache" in name for name in ours.state_dict())


@torch.no_grad()
def test_keys_are_rotated_once_and_values_are_not_rotated():
    ours, _ = pair()
    ids = torch.tensor([1, 7, 9])
    layer = ours.model.layers[0]
    hidden = layer.input_layernorm(ours.model.embed_tokens(ids))
    attention = layer.self_attn
    key = attention.k_proj(hidden).reshape(3, attention.kv_heads, attention.head_dim)
    value = attention.v_proj(hidden).reshape_as(key)
    cos, sin = rope_cos_sin(torch.arange(3), attention.head_dim, attention.theta, key.dtype)
    cache = ours.new_kv_cache(3)
    forward(ours, ids, cache)
    torch.testing.assert_close(cache.key[0], apply_rope(key, cos, sin), rtol=0, atol=0)
    torch.testing.assert_close(cache.value[0], value, rtol=0, atol=0)


def test_old_entries_are_unchanged_and_unused_storage_is_invisible():
    ours, _ = pair()
    cache = ours.new_kv_cache(8)
    cache.key.fill_(float("nan"))
    cache.value.fill_(float("nan"))
    ids = torch.tensor([1, 7, 9, 12])
    assert torch.isfinite(forward(ours, ids[:3], cache)).all()
    old_key, old_value = cache.key[:, :3].clone(), cache.value[:, :3].clone()
    logits = forward(ours, ids[3:], cache)
    assert torch.isfinite(logits).all()
    assert torch.equal(cache.key[:, :3], old_key)
    assert torch.equal(cache.value[:, :3], old_value)


def test_reset_reuses_storage_without_leaking_previous_request():
    ours, _ = pair()
    cache = ours.new_kv_cache(8)
    forward(ours, torch.tensor([1, 7, 9, 12]), cache)
    key_pointer = cache.key.data_ptr()
    cache.reset()
    assert cache.length == 0 and cache.key.data_ptr() == key_pointer
    ids = torch.tensor([1, 20])
    torch.testing.assert_close(forward(ours, ids, cache), forward(ours, ids), rtol=1e-5, atol=1e-6)
    assert cache.length == 2


def test_prefill_and_chunk_masks_use_absolute_positions():
    ours, _ = pair()
    cache = ours.new_kv_cache(8)
    metadata = AttentionMetadata(torch.arange(3), cache=cache)
    assert torch.equal(metadata.allowed_mask(), torch.ones(3, 3, dtype=torch.bool).tril())
    forward(ours, torch.tensor([1, 7, 9]), cache)
    metadata = AttentionMetadata(torch.tensor([3, 4]), cache=cache)
    assert metadata.allowed_mask().tolist() == [
        [True, True, True, True, False], [True, True, True, True, True],
    ]


@pytest.mark.parametrize("capacity", [0, -1, 33, True, 1.5])
def test_invalid_capacity_raises(capacity):
    ours, _ = pair()
    with pytest.raises(ValueError, match="capacity"):
        ours.new_kv_cache(capacity)


def test_exact_capacity_then_overflow_preserves_committed_prefix():
    ours, _ = pair()
    cache = ours.new_kv_cache(2)
    forward(ours, torch.tensor([1, 7]), cache)
    old_key = cache.key.clone()
    with pytest.raises(ValueError, match="capacity"):
        forward(ours, torch.tensor([9]), cache)
    assert cache.length == 2 and torch.equal(cache.key, old_key)


@pytest.mark.parametrize("positions", [torch.tensor([1, 2]), torch.tensor([0, 2]), torch.tensor([1, 0])])
def test_nonconsecutive_prefill_positions_raise(positions):
    ours, _ = pair()
    cache = ours.new_kv_cache(8)
    with pytest.raises(ValueError, match="consecutively"):
        forward(ours, torch.tensor([1, 7]), cache, positions)
    assert cache.length == 0


def test_decode_cannot_restart_positions_at_zero():
    ours, _ = pair()
    cache = ours.new_kv_cache(8)
    forward(ours, torch.tensor([1, 7]), cache)
    with pytest.raises(ValueError, match="consecutively"):
        forward(ours, torch.tensor([9]), cache, torch.tensor([0]))
    assert cache.length == 2


def test_packed_cached_requests_are_rejected_but_no_cache_still_works():
    ours, _ = pair()
    ids = torch.tensor([1, 7])
    request_ids = torch.tensor([0, 0])
    with pytest.raises(ValueError, match="one request"):
        forward(ours, ids, ours.new_kv_cache(8), sequence_ids=request_ids)
    torch.testing.assert_close(forward(ours, ids, sequence_ids=request_ids), forward(ours, ids))


def test_cache_rejects_other_model_and_changed_dtype():
    ours, _ = pair()
    other, _ = pair()
    cache = ours.new_kv_cache(8)
    with pytest.raises(ValueError, match="different model"):
        forward(other, torch.tensor([1]), cache)
    ours.double()
    with pytest.raises(ValueError, match="device/dtype"):
        forward(ours, torch.tensor([1]), cache)


def test_cached_forward_requires_eval_and_no_grad():
    ours, _ = pair()
    cache = ours.new_kv_cache(8)
    ids, positions = torch.tensor([1]), torch.tensor([0])
    with torch.enable_grad(), pytest.raises(ValueError, match="disabled gradients"):
        ours(ids, positions, AttentionMetadata(positions, cache=cache))
    ours.train()
    with pytest.raises(ValueError, match="eval"):
        forward(ours, ids, cache)
    assert cache.length == 0


def test_failed_later_layer_does_not_commit_and_can_retry():
    ours, _ = pair()
    cache = ours.new_kv_cache(8)
    ids = torch.tensor([1, 7, 9])
    forward(ours, ids[:2], cache)
    old_key = cache.key[:, :2].clone()

    def fail(module, args):
        raise RuntimeError("injected layer failure")

    hook = ours.model.layers[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            forward(ours, ids[2:], cache)
    finally:
        hook.remove()
    assert cache.length == 2 and torch.equal(cache.key[:, :2], old_key)
    torch.testing.assert_close(forward(ours, ids[2:], cache), forward(ours, ids)[-1:],
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bad_kind", ["layer", "shape", "dtype", "overflow"])
def test_invalid_cache_writes_raise(bad_kind):
    ours, _ = pair()
    cache = ours.new_kv_cache(2)
    key = cache.key.new_zeros((1, 2, 16))
    layer = 3 if bad_kind == "layer" else 0
    if bad_kind == "shape":
        key = key[:, :1]
    elif bad_kind == "dtype":
        key = key.double()
    elif bad_kind == "overflow":
        key = key.expand(3, -1, -1)
    with torch.no_grad(), pytest.raises(ValueError):
        cache.write(layer, key, key)
    assert cache.length == 0


@pytest.mark.parametrize("stop_on_first", [False, True])
def test_cached_greedy_matches_no_cache_and_hf(stop_on_first):
    ours, reference = pair()
    ids = torch.tensor([1, 7, 9])
    if stop_on_first:
        eos = forward(ours, ids)[-1].argmax().item()
        ours.config.eos_token_id = eos
        reference.generation_config.eos_token_id = eos
    actual = generate(ours, ids, max_new_tokens=6, use_cache=True)
    assert torch.equal(actual, generate(ours, ids, max_new_tokens=6))
    with torch.no_grad():
        expected = reference.generate(ids[None], attention_mask=torch.ones_like(ids)[None],
                                      do_sample=False, max_new_tokens=6, use_cache=True)[0]
    assert torch.equal(actual, expected)
    assert actual.numel() == (4 if stop_on_first else 9)


def test_lesson_probe_checks_each_step_and_reports_actual_forward_shapes():
    ours, reference = pair()
    ids = torch.tensor([1, 7, 9])
    actual, rows = compare_cached_steps(ours, reference, ids, max_new_tokens=4)
    assert torch.equal(actual, generate(ours, ids, max_new_tokens=4, use_cache=True))
    assert [row["forward_tokens"] for row in rows] == [3, 1, 1, 1]
    assert [row["cache_length"] for row in rows] == [3, 4, 5, 6]
    assert all(row["argmax_equal"] for row in rows)


def test_generation_prefills_once_then_only_forwards_new_token():
    ours, _ = pair()
    ids = torch.tensor([1, 7, 9])
    seen = []

    def record(module, args):
        seen.append((args[0].clone(), args[1].clone(), args[2].cache.length))

    hook = ours.register_forward_pre_hook(record)
    try:
        output = generate(ours, ids, max_new_tokens=4, use_cache=True)
    finally:
        hook.remove()
    assert [row[0].numel() for row in seen] == [3, 1, 1, 1]
    assert [row[1].tolist() for row in seen] == [[0, 1, 2], [3], [4], [5]]
    assert [row[2] for row in seen] == [0, 3, 4, 5]
    assert torch.equal(seen[0][0], ids)
    assert [row[0].item() for row in seen[1:]] == output[3:-1].tolist()
    assert ids.tolist() == [1, 7, 9]


def test_zero_budget_and_context_errors_do_not_allocate_cache():
    ours, _ = pair()
    ids = torch.tensor([1, 7])

    def forbidden(capacity):
        raise AssertionError("cache should not be allocated")

    ours.new_kv_cache = forbidden
    output = generate(ours, ids, max_new_tokens=0, use_cache=True)
    assert torch.equal(output, ids) and output.data_ptr() != ids.data_ptr()
    with pytest.raises(ValueError, match="context limit"):
        generate(ours, ids, max_new_tokens=32, use_cache=True)


def test_exact_context_budget_and_repeated_calls():
    ours, _ = pair()
    ours.config.max_position_embeddings = 6
    ids = torch.tensor([1, 7])
    first = generate(ours, ids, max_new_tokens=4, use_cache=True)
    assert first.numel() == 6
    assert torch.equal(first, generate(ours, ids, max_new_tokens=4, use_cache=True))


def test_seeded_cached_sampling_replays():
    ours, _ = pair()
    ids = torch.tensor([1, 7, 9])
    options = dict(max_new_tokens=6, temperature=0.8, top_k=8, top_p=0.9, use_cache=True)
    first = generate(ours, ids, generator=torch.Generator().manual_seed(17), **options)
    second = generate(ours, ids, generator=torch.Generator().manual_seed(17), **options)
    assert torch.equal(first, second)


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_invalid_cache_switch_raises(value):
    ours, _ = pair()
    with pytest.raises(ValueError, match="boolean"):
        generate(ours, torch.tensor([1]), max_new_tokens=0, use_cache=value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fp16_cache_against_same_rounded_weights_in_fp32():
    ours, reference = pair(dtype=torch.float16, device="cuda")
    float_model = LlamaForCausalLM(reference.config, device="cuda", dtype=torch.float32).eval()
    float_model.load_state_dict({name: weight.float() for name, weight in ours.state_dict().items()})
    ids = torch.tensor([1, 7, 9, 12], device="cuda", dtype=torch.int32)
    cache = ours.new_kv_cache(4)
    forward(ours, ids[:3], cache)
    actual = forward(ours, ids[3:], cache)
    torch.testing.assert_close(actual, forward(float_model, ids)[-1:], rtol=1e-2, atol=1e-2)
    assert cache.key.dtype == torch.float16 and cache.key.device == ids.device
    assert actual.dtype == torch.float32
    generated = generate(ours, ids[:3], max_new_tokens=2, use_cache=True)
    assert generated.dtype == ids.dtype and generated.device == ids.device
