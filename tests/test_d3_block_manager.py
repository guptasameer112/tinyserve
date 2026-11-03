"""D3 allocator invariants, randomized reference traces and managed-pool parity."""
import random
from types import SimpleNamespace

import pytest

from tinyserve.engine.block_manager import BlockAllocator, BlockPoolExhausted


def allocator_state(allocator):
    return allocator.free_blocks, allocator.ref_counts


def test_reference_lifetime_and_reuse():
    allocator = BlockAllocator(3)
    assert allocator.allocate(2) == (0, 1)
    allocator.retain(0)
    allocator.release(0)
    assert allocator.ref_counts == (1, 1, 0)
    assert allocator.free_blocks == (2,)
    allocator.release(0)
    assert allocator.allocate(1) == (0,)
    allocator.release(0)
    with pytest.raises(ValueError, match="no live reference"):
        allocator.release(0)
    with pytest.raises(ValueError, match="no live reference"):
        allocator.retain(0)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_sizes_do_not_mutate(value):
    with pytest.raises(ValueError):
        BlockAllocator(value)
    allocator = BlockAllocator(3)
    before = allocator_state(allocator)
    with pytest.raises(ValueError):
        allocator.allocate(value)
    assert allocator_state(allocator) == before


@pytest.mark.parametrize("block", [-1, 3, True, 0.5, None, 2])
def test_invalid_reference_operations_are_atomic(block):
    allocator = BlockAllocator(3)
    allocator.allocate(1)
    for operation in (allocator.retain, allocator.release):
        before = allocator_state(allocator)
        with pytest.raises(ValueError):
            operation(block)
        assert allocator_state(allocator) == before


def test_exhaustion_is_atomic_and_live_blocks_are_never_evicted():
    allocator = BlockAllocator(3)
    allocator.allocate(2)
    before = allocator_state(allocator)
    with pytest.raises(BlockPoolExhausted, match="need 2 blocks but only 1"):
        allocator.allocate(2)
    assert allocator_state(allocator) == before
    allocator.allocate(1)
    with pytest.raises(BlockPoolExhausted):
        allocator.allocate(1)


@pytest.mark.parametrize("seed", range(20))
def test_randomized_allocator_properties_against_reference(seed):
    rng = random.Random(seed)
    allocator = BlockAllocator(11)
    reference = [0] * 11
    for _ in range(400):
        live = [block for block, count in enumerate(reference) if count]
        operation = rng.choice(("allocate", "retain", "release")) if live else "allocate"
        if operation == "allocate":
            count = rng.randint(1, 12)
            free = {block for block, refs in enumerate(reference) if refs == 0}
            before = allocator_state(allocator)
            if count > len(free):
                with pytest.raises(BlockPoolExhausted):
                    allocator.allocate(count)
                assert allocator_state(allocator) == before
            else:
                blocks = allocator.allocate(count)
                assert len(set(blocks)) == count and set(blocks) <= free
                for block in blocks:
                    reference[block] = 1
        else:
            block = rng.choice(live)
            if operation == "retain":
                allocator.retain(block)
                reference[block] += 1
            else:
                allocator.release(block)
                reference[block] -= 1
        assert allocator.ref_counts == tuple(reference)
        assert len(set(allocator.free_blocks)) == allocator.num_free_blocks
        assert set(allocator.free_blocks) == {i for i, count in enumerate(reference) if count == 0}
        assert allocator.num_free_blocks + sum(count > 0 for count in reference) == 11
    for block, count in enumerate(reference):
        for _ in range(count):
            allocator.release(block)
    assert allocator.num_free_blocks == 11
    assert allocator.ref_counts == (0,) * 11


def stub_pool(num_blocks=5, block_size=2):
    torch = pytest.importorskip("torch")
    from tinyserve.paged_cache import ManagedPagedKVCache

    class StubModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 16)
            self.config = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=1,
                                          num_attention_heads=2, hidden_size=16,
                                          max_position_embeddings=32)
            self.eval()

        def get_input_embeddings(self):
            return self.embedding

    return ManagedPagedKVCache(StubModel(), num_blocks, block_size)


def pool_state(pool):
    return dict(pool.block_tables), dict(pool.lengths), pool.free_blocks, pool.ref_counts


def test_managed_growth_rounding_reset_release_and_read_only_views():
    pool = stub_pool()
    pointer = pool.key.data_ptr()
    value_pointer = pool.value.data_ptr()
    assert dict(pool.block_tables) == dict(pool.lengths) == {}
    assert pool.reserve(7, 1) == (0,)
    assert pool.reserve(7, 2) == (0,)
    assert pool.reserve(7, 3) == (0, 1)
    assert pool.reserve(9, 2) == (2,)
    with pytest.raises(TypeError):
        pool.block_tables[7] = (4,)
    with pytest.raises(TypeError):
        pool.lengths[7] = 9
    pool.reset(7)
    assert pool.block_tables[7] == (0, 1) and pool.num_free_blocks == 2
    pool.release_request(7)
    assert 7 not in pool.lengths and 7 not in pool.block_tables
    assert pool.reserve(10, 4) == (1, 0)
    assert pool.block_tables[9] == (2,)
    assert pool.key.data_ptr() == pointer
    assert pool.value.data_ptr() == value_pointer
    pool.release_request(9)
    pool.release_request(10)
    assert pool.num_free_blocks == pool.num_blocks


@pytest.mark.parametrize("request_id,tokens", [
    (-1, 1), (True, 1), (1.5, 1), (None, 1),
    (1, 0), (1, -1), (1, True), (1, 1.5), (1, 33),
])
def test_invalid_reservations_do_not_mutate(request_id, tokens):
    pool = stub_pool()
    before = pool_state(pool)
    with pytest.raises(ValueError):
        pool.reserve(request_id, tokens)
    assert pool_state(pool) == before


def test_failed_admission_and_growth_leave_tables_and_lengths_unchanged():
    pool = stub_pool(3)
    pool.reserve(0, 3)
    before = pool_state(pool)
    for request_id, tokens in ((1, 4), (0, 7)):
        with pytest.raises(BlockPoolExhausted):
            pool.reserve(request_id, tokens)
        assert pool_state(pool) == before
    for request_id in (1, True, -1):
        with pytest.raises(ValueError, match="unknown request"):
            pool.release_request(request_id)
    assert pool_state(pool) == before


@pytest.mark.parametrize("seed", range(10))
def test_randomized_request_tables_are_disjoint_and_conserve_blocks(seed):
    rng = random.Random(seed)
    pool = stub_pool(9, 3)
    expected_tables = {}
    for _ in range(200):
        request_id = rng.randrange(7)
        if request_id in expected_tables and rng.random() < 0.3:
            pool.release_request(request_id)
            del expected_tables[request_id]
        else:
            total = rng.randint(1, 27)
            current = expected_tables.get(request_id, ())
            required = (total + 2) // 3
            extra = max(0, required - len(current))
            before = pool_state(pool)
            if extra > pool.num_free_blocks:
                with pytest.raises(BlockPoolExhausted):
                    pool.reserve(request_id, total)
                assert pool_state(pool) == before
            else:
                table = pool.reserve(request_id, total)
                assert table[:len(current)] == current
                assert len(table) == max(len(current), required)
                expected_tables[request_id] = table
        assert dict(pool.block_tables) == expected_tables
        used = [block for table in expected_tables.values() for block in table]
        assert len(used) == len(set(used))
        assert not set(used).intersection(pool.free_blocks)
        assert set(used).union(pool.free_blocks) == set(range(9))
        assert pool.ref_counts == tuple(int(block in used) for block in range(9))
        assert dict(pool.lengths) == {request_id: 0 for request_id in expected_tables}
    for request_id in tuple(pool.block_tables):
        pool.release_request(request_id)
    assert pool.num_free_blocks == 9


def model_pair(device="cpu", dtype=None):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tinyserve.models.llama import LlamaForCausalLM

    torch.manual_seed(23)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=32, eos_token_id=None, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    reference = transformers.LlamaForCausalLM(config).to(
        device=device, dtype=dtype or torch.float32,
    ).eval()
    return LlamaForCausalLM.from_huggingface(reference), reference


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_managed_model_mixed_growth_reuse_and_failure_retry(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tinyserve.models.attention import AttentionMetadata
    from tinyserve.probe import compare_paged_batch

    model, reference = model_pair(device)
    pool = model.new_managed_paged_cache(4, 2)
    pool.key.fill_(float("nan"))
    pool.value.fill_(float("nan"))
    pointer = pool.key.data_ptr()
    value_pointer = pool.value.data_ptr()
    first = torch.tensor([1, 7, 9], device=device)
    other = torch.tensor([1, 12], device=device)
    pool.reserve(0, 3)
    pool.reserve(1, 2)
    compare_paged_batch(model, reference, pool, {0: first, 1: other})
    before = pool_state(pool)
    with pytest.raises(ValueError, match="committed prefix"):
        pool.reserve(0, 2)
    with pytest.raises(BlockPoolExhausted):
        pool.reserve(2, 4)
    assert pool_state(pool) == before
    pool.reserve(0, 5)
    assert pool.num_free_blocks == 0
    prefix_slots = pool.slot_mapping(torch.arange(3, device=device),
                                     torch.zeros(3, dtype=torch.int64, device=device))
    committed = pool.key.reshape(2, -1, 2, 16)[:, prefix_slots].clone()
    committed_value = pool.value.reshape(2, -1, 2, 16)[:, prefix_slots].clone()
    reserved_state = pool_state(pool)

    def fail(module, args):
        raise RuntimeError("injected managed failure")

    hook = model.model.layers[1].register_forward_pre_hook(fail)
    try:
        with torch.no_grad(), pytest.raises(RuntimeError, match="injected"):
            positions = torch.tensor([3, 4], device=device)
            model(torch.tensor([10, 11], device=device), positions,
                  AttentionMetadata(positions, torch.zeros_like(positions), paged_cache=pool))
    finally:
        hook.remove()
    assert pool_state(pool) == reserved_state
    torch.testing.assert_close(pool.key.reshape(2, -1, 2, 16)[:, prefix_slots], committed,
                               rtol=0, atol=0)
    torch.testing.assert_close(pool.value.reshape(2, -1, 2, 16)[:, prefix_slots], committed_value,
                               rtol=0, atol=0)
    compare_paged_batch(model, reference, pool, {
        0: torch.cat((first, first.new_tensor([10, 11]))),
    })
    released = set(pool.block_tables[0])
    pool.release_request(0)
    pool.reserve(2, 4)
    assert set(pool.block_tables[2]) <= released
    compare_paged_batch(model, reference, pool, {
        2: torch.tensor([1, 21, 22, 23], device=device),
    })
    assert pool.lengths[1] == 2
    pool.reserve(1, 3)
    compare_paged_batch(model, reference, pool, {1: torch.cat((other, other.new_tensor([13])))})
    assert pool.key.data_ptr() == pointer
    assert pool.value.data_ptr() == value_pointer
    pool.release_request(1)
    with pytest.raises(ValueError, match="unknown request"):
        pool.reset(1)
    pool.release_request(2)
    assert pool.num_free_blocks == 4 and pool.ref_counts == (0,) * 4
