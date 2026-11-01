"""D2: physical addressing, scrambled layouts, mixed batches and parity."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tinyserve.models.attention import AttentionMetadata  # noqa: E402
from tinyserve.paged_cache import PagedKVCache  # noqa: E402


class StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 16)
        self.config = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=1,
                                      num_attention_heads=2, hidden_size=16, max_position_embeddings=16)
        self.eval()

    def get_input_embeddings(self):
        return self.embedding


def test_hand_slot_mapping_and_pool_cost():
    model = StubModel()
    cache = PagedKVCache(model, 6, 2, {7: [4, 1, 5], 9: [2, 0]})
    positions = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    owners = torch.tensor([7] * 6 + [9] * 4)
    assert cache.slot_mapping(positions, owners).tolist() == [8, 9, 2, 3, 10, 11, 4, 5, 0, 1]
    assert cache.key.shape == cache.value.shape == (2, 6, 2, 1, 8)
    assert cache.allocated_bytes == 2 * 2 * 6 * 2 * 1 * 8 * 4
    assert dict(cache.lengths) == {7: 0, 9: 0}


def test_tables_and_committed_lengths_cannot_be_changed_by_callers():
    tables = {0: [2, 0], 1: [1]}
    cache = PagedKVCache(StubModel(), 3, 2, tables)
    tables[0][0] = 1
    assert cache.block_tables[0] == (2, 0)
    with pytest.raises(TypeError):
        cache.block_tables[0] = (1, 2)
    with pytest.raises(TypeError):
        cache.lengths[0] = 7


@pytest.mark.parametrize("num_blocks,block_size,tables", [
    (0, 2, {0: [0]}), (3, 0, {0: [0]}), (True, 2, {0: [0]}),
    (3, 1.5, {0: [0]}), (3, 2, {}), (3, 2, {True: [0]}),
    (3, 2, {-1: [0]}), (3, 2, {0: []}), (3, 2, {0: [3]}),
    (3, 2, {0: [-1]}), (3, 2, {0: [True]}), (3, 2, {0: [0, 0]}),
    (3, 2, {0: [0], 1: [0]}), (3, 2, {0: torch.tensor([0])}),
])
def test_invalid_pool_tables_raise(num_blocks, block_size, tables):
    with pytest.raises(ValueError):
        PagedKVCache(StubModel(), num_blocks, block_size, tables)


@pytest.mark.parametrize("positions,owners", [
    (torch.tensor([-1]), torch.tensor([0])), (torch.tensor([4]), torch.tensor([0])),
    (torch.tensor([0]), torch.tensor([9])), (torch.tensor([0.0]), torch.tensor([0])),
    (torch.tensor([0]), None), (torch.tensor([0]), torch.tensor([0.0])),
    (torch.tensor([[0]]), torch.tensor([[0]])), (torch.tensor([0, 1]), torch.tensor([0])),
])
def test_invalid_slot_layouts_raise(positions, owners):
    cache = PagedKVCache(StubModel(), 2, 2, {0: [1, 0]})
    with pytest.raises(ValueError):
        cache.slot_mapping(positions, owners)


def test_gather_uses_logical_order_and_hides_unwritten_slots():
    model = StubModel()
    cache = PagedKVCache(model, 5, 2, {0: [3, 0], 1: [4]})
    cache.key.fill_(float("nan"))
    cache.value.fill_(float("nan"))
    positions = torch.tensor([0, 1, 2, 0])
    owners = torch.tensor([0, 0, 0, 1])
    query = torch.zeros(4, 2, 8)
    key = torch.zeros(4, 1, 8)
    value = torch.tensor([1., 3., 8., 20.])[:, None, None].expand(-1, 1, 8).contiguous()
    with torch.no_grad():
        output = cache.attend(0, query, key, value, positions, owners)
    expected = torch.tensor([1., 2., 4., 20.])[:, None, None].expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0, atol=1e-6)
    assert dict(cache.lengths) == {0: 0, 1: 0}, "a layer write must not commit model lengths"
    assert torch.equal(cache.value[0, 3, 0], value[0])
    assert torch.equal(cache.value[0, 0, 0], value[2])


def pair(kv_heads=2, device="cpu", dtype=torch.float32, context=16):
    transformers = pytest.importorskip("transformers")
    from tinyserve.models.llama import LlamaForCausalLM

    torch.manual_seed(12)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=kv_heads, vocab_size=64,
        max_position_embeddings=context, eos_token_id=None, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    hf = transformers.LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
    return LlamaForCausalLM.from_huggingface(hf), hf


@torch.no_grad()
def forward(model, ids, positions, owners, cache):
    return model(ids, positions, AttentionMetadata(positions, owners, paged_cache=cache))


@torch.no_grad()
def no_cache(model, ids):
    positions = torch.arange(ids.numel(), device=ids.device)
    return model(ids, positions, AttentionMetadata(positions))


@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
@pytest.mark.parametrize("kv_heads", [4, 2, 1])
def test_scrambled_blocks_prefill_and_decode_match_contiguous_and_hf(block_size, kv_heads):
    ours, hf = pair(kv_heads)
    blocks = (6 + block_size - 1) // block_size
    cache = ours.new_paged_cache(blocks, block_size, {0: list(reversed(range(blocks)))})
    contiguous = ours.new_kv_cache(6)
    ids = torch.tensor([1, 7, 9, 12, 13, 14])
    for start, end in [(0, 3), (3, 4), (4, 6)]:
        positions = torch.arange(start, end)
        actual = forward(ours, ids[start:end], positions, torch.zeros(end - start, dtype=torch.int64), cache)
        with torch.no_grad():
            expected = ours(ids[start:end], positions, AttentionMetadata(positions, cache=contiguous))
            hf_logits = hf(ids[None, :end], use_cache=False).logits[0, start:]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual, hf_logits, rtol=1e-5, atol=1e-6)
        assert cache.lengths[0] == contiguous.length == end


def test_mixed_decode_new_prefill_and_paused_request():
    from tinyserve.probe import compare_paged_batch

    ours, hf = pair()
    cache = ours.new_paged_cache(9, 2, {0: [8, 1, 5], 1: [4, 0, 7], 2: [6, 2, 3]})
    first = {0: torch.tensor([1, 7, 9]), 1: torch.tensor([1, 12])}
    compare_paged_batch(ours, hf, cache, first)
    second = {0: torch.tensor([1, 7, 9, 20]), 2: torch.tensor([1, 21, 22])}
    result = compare_paged_batch(ours, hf, cache, second)
    assert result["positions"] == [3, 0, 1, 2]
    assert result["sequence_ids"] == [0, 2, 2, 2]
    assert dict(cache.lengths) == {0: 4, 1: 2, 2: 3}
    result = compare_paged_batch(ours, hf, cache, {1: torch.tensor([1, 12, 23])})
    assert result["requests"][0]["all_row_argmax_equal"]


def test_interleaved_input_order_is_scattered_back_correctly():
    ours, _ = pair()
    cache = ours.new_paged_cache(4, 2, {7: [3, 0], 9: [1, 2]})
    ids = torch.tensor([1, 1, 7, 12, 9])
    positions = torch.tensor([0, 0, 1, 1, 2])
    owners = torch.tensor([7, 9, 7, 9, 7])
    actual = forward(ours, ids, positions, owners, cache)
    torch.testing.assert_close(actual[owners == 7], no_cache(ours, ids[owners == 7]), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[owners == 9], no_cache(ours, ids[owners == 9]), rtol=1e-5, atol=1e-6)
    assert dict(cache.lengths) == {7: 3, 9: 2}


def test_other_request_storage_cannot_affect_active_decode():
    ours, _ = pair()
    cache = ours.new_paged_cache(5, 2, {0: [4, 0, 2], 1: [1, 3]})
    forward(ours, torch.tensor([1, 7]), torch.arange(2), torch.zeros(2, dtype=torch.int64), cache)
    cache.key[:, [1, 3]] = 1000
    cache.value[:, [1, 3]] = -1000
    actual = forward(ours, torch.tensor([9]), torch.tensor([2]), torch.tensor([0]), cache)
    torch.testing.assert_close(actual, no_cache(ours, torch.tensor([1, 7, 9]))[-1:],
                               rtol=1e-5, atol=1e-6)
    assert cache.lengths[1] == 0


def test_failed_later_layer_preserves_all_lengths_and_can_retry():
    ours, _ = pair()
    cache = ours.new_paged_cache(4, 2, {0: [3, 0], 1: [2, 1]})
    forward(ours, torch.tensor([1, 1]), torch.tensor([0, 0]), torch.tensor([0, 1]), cache)
    original_key = cache.key[:, [3, 2], 0].clone()

    def fail(module, args):
        raise RuntimeError("injected paged failure")

    hook = ours.model.layers[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            forward(ours, torch.tensor([7, 12]), torch.tensor([1, 1]), torch.tensor([0, 1]), cache)
    finally:
        hook.remove()
    assert dict(cache.lengths) == {0: 1, 1: 1}
    assert torch.equal(cache.key[:, [3, 2], 0], original_key)
    actual = forward(ours, torch.tensor([7, 12]), torch.tensor([1, 1]), torch.tensor([0, 1]), cache)
    expected = torch.cat([no_cache(ours, torch.tensor([1, 7]))[-1:],
                          no_cache(ours, torch.tensor([1, 12]))[-1:]])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert dict(cache.lengths) == {0: 2, 1: 2}


def test_reset_one_request_reuses_blocks_and_leaves_other_length():
    ours, _ = pair()
    cache = ours.new_paged_cache(4, 2, {0: [3, 0], 1: [1, 2]})
    forward(ours, torch.tensor([1, 7, 1]), torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1]), cache)
    pointer = cache.key.data_ptr()
    cache.reset(0)
    assert cache.key.data_ptr() == pointer and dict(cache.lengths) == {0: 0, 1: 1}
    actual = forward(ours, torch.tensor([20]), torch.tensor([0]), torch.tensor([0]), cache)
    torch.testing.assert_close(actual, no_cache(ours, torch.tensor([20])), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="unknown"):
        cache.reset(99)


@pytest.mark.parametrize("positions", [torch.tensor([1]), torch.tensor([0, 2]), torch.tensor([0, 0])])
def test_nonappend_positions_raise_before_writes(positions):
    ours, _ = pair()
    cache = ours.new_paged_cache(2, 2, {0: [1, 0]})
    cache.key.fill_(17)
    before = cache.key.clone()
    with pytest.raises(ValueError, match="consecutively"):
        forward(ours, torch.ones_like(positions), positions, torch.zeros_like(positions), cache)
    assert dict(cache.lengths) == {0: 0} and torch.equal(cache.key, before)


def test_pool_capacity_and_model_context_are_separate_limits():
    ours, _ = pair(context=3)
    cache = ours.new_paged_cache(3, 2, {0: [2, 1, 0]})
    with pytest.raises(ValueError, match="context limit"):
        forward(ours, torch.tensor([1, 7, 9, 12]), torch.arange(4), torch.zeros(4, dtype=torch.int64), cache)
    short = ours.new_paged_cache(1, 2, {0: [0]})
    with pytest.raises(ValueError, match="capacity"):
        forward(ours, torch.tensor([1, 7, 9]), torch.arange(3), torch.zeros(3, dtype=torch.int64), short)
    assert cache.lengths[0] == short.lengths[0] == 0


def test_cache_model_dtype_eval_grad_and_backend_boundaries():
    ours, _ = pair()
    other, _ = pair()
    cache = ours.new_paged_cache(2, 2, {0: [1, 0]})
    ids, positions, owners = torch.tensor([1]), torch.tensor([0]), torch.tensor([0])
    with pytest.raises(ValueError, match="different model"):
        forward(other, ids, positions, owners, cache)
    with torch.enable_grad(), pytest.raises(ValueError, match="disabled gradients"):
        ours(ids, positions, AttentionMetadata(positions, owners, paged_cache=cache))
    ours.train()
    with pytest.raises(ValueError, match="eval"):
        forward(ours, ids, positions, owners, cache)
    ours.eval()
    both = AttentionMetadata(positions, owners, cache=ours.new_kv_cache(4), paged_cache=cache)
    with torch.no_grad(), pytest.raises(ValueError, match="not both"):
        ours(ids, positions, both)
    ours.double()
    with pytest.raises(ValueError, match="device/dtype"):
        forward(ours, ids, positions, owners, cache)


@pytest.mark.parametrize("invalid", ["layer", "key_shape", "key_dtype", "query_heads", "query_dtype"])
def test_invalid_attention_inputs_raise_before_pool_writes(invalid):
    model = StubModel()
    cache = PagedKVCache(model, 2, 2, {0: [1, 0]})
    cache.key.fill_(17)
    original = cache.key.clone()
    positions, owners = torch.tensor([0]), torch.tensor([0])
    query, key = torch.zeros(1, 2, 8), torch.zeros(1, 1, 8)
    value = key.clone()
    layer = 3 if invalid == "layer" else 0
    if invalid == "key_shape":
        key = key[:, :, :4]
    elif invalid == "key_dtype":
        key = key.double()
    elif invalid == "query_heads":
        query = query[:, :0]
    elif invalid == "query_dtype":
        query = query.double()
    with torch.no_grad(), pytest.raises(ValueError):
        cache.attend(layer, query, key, value, positions, owners)
    assert torch.equal(original, cache.key) and cache.lengths[0] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fp16_paged_against_same_rounded_weights_in_fp32():
    from tinyserve.models.llama import LlamaForCausalLM

    ours, hf = pair(device="cuda", dtype=torch.float16)
    high = LlamaForCausalLM(hf.config, device="cuda", dtype=torch.float32).eval()
    high.load_state_dict({name: weight.float() for name, weight in ours.state_dict().items()})
    cache = ours.new_paged_cache(4, 2, {0: [3, 0], 1: [2, 1]})
    ids = torch.tensor([1, 7, 1], device="cuda", dtype=torch.int32)
    positions = ids.new_tensor([0, 1, 0])
    owners = ids.new_tensor([0, 0, 1])
    actual = forward(ours, ids, positions, owners, cache)
    expected = torch.cat([no_cache(high, ids[:2]), no_cache(high, ids[2:])])
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    with pytest.raises(ValueError, match="cache device"):
        cache.slot_mapping(positions.cpu(), owners.cpu())
