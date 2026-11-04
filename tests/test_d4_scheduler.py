"""D4 policy traces and real-forward parity; execute on Colab, not locally."""
import random
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from tinyserve.engine.scheduler import Scheduler


def policy_pool(num_blocks=5, block_size=2):
    torch = pytest.importorskip("torch")
    from tinyserve.paged_cache import ManagedPagedKVCache

    embedding = torch.nn.Embedding(64, 4)
    config = SimpleNamespace(
        num_hidden_layers=1, num_key_value_heads=1, num_attention_heads=1,
        hidden_size=4, max_position_embeddings=32,
    )
    model = SimpleNamespace(config=config, get_input_embeddings=lambda: embedding)
    return ManagedPagedKVCache(model, num_blocks, block_size)


def metadata(pool):
    return dict(pool.block_tables), dict(pool.lengths), pool.free_blocks, pool.ref_counts


def simulate_forward(scheduler, plan):
    """Policy-only simulation: advance cache lengths without computing K/V."""
    torch = pytest.importorskip("torch")
    owners = [item.request_id for item in plan.work for _ in range(item.end - item.start)]
    scheduler.cache.commit(torch.tensor(owners, dtype=torch.int64))
    scheduler.complete_step()


def work_rows(plan):
    return [(item.request_id, item.start, item.end, item.phase) for item in plan.work]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_batch_budgets(value):
    with pytest.raises(ValueError):
        Scheduler(policy_pool(), value)


def test_scheduler_requires_exclusive_empty_pool_and_detects_external_changes():
    pool = policy_pool()
    pool.reserve(9, 1)
    with pytest.raises(ValueError, match="empty"):
        Scheduler(pool, 3)
    pool.release_request(9)
    scheduler = Scheduler(pool, 3)
    pool.reserve(9, 1)
    with pytest.raises(RuntimeError, match="outside"):
        scheduler.schedule()


@pytest.mark.parametrize("owner,prompt,decode", [
    (-1, 1, 0), (True, 1, 0), (1.5, 1, 0), (None, 1, 0),
    (0, 0, 0), (0, -1, 0), (0, True, 0), (0, 1.5, 0),
    (0, 1, -1), (0, 1, True), (0, 1, 1.5), (0, 1, None),
    (0, 10, 1), (0, 33, 0),
])
def test_invalid_admission_is_atomic(owner, prompt, decode):
    pool = policy_pool()
    scheduler = Scheduler(pool, 3)
    before = metadata(pool)
    with pytest.raises(ValueError):
        scheduler.add_request(owner, prompt, decode)
    assert dict(scheduler.requests) == {}
    assert metadata(pool) == before


def test_empty_scheduler_read_only_states_and_unique_lifetime_ids():
    scheduler = Scheduler(policy_pool(), 3)
    assert scheduler.schedule() is None
    with pytest.raises(RuntimeError, match="no pending"):
        scheduler.complete_step()
    scheduler.add_request(7, 1)
    with pytest.raises(TypeError):
        scheduler.requests[8] = scheduler.requests[7]
    with pytest.raises(FrozenInstanceError):
        scheduler.requests[7].status = "finished"
    with pytest.raises(ValueError, match="already been used"):
        scheduler.add_request(7, 1)
    plan = scheduler.schedule()
    with pytest.raises(FrozenInstanceError):
        plan.work[0].end = 10
    simulate_forward(scheduler, plan)
    assert scheduler.requests[7].status == "finished"
    with pytest.raises(ValueError, match="already"):
        scheduler.cancel(7)
    with pytest.raises(ValueError, match="already been used"):
        scheduler.add_request(7, 1)


def test_continuous_admission_chunked_prefill_and_mixed_decode():
    pool = policy_pool()
    scheduler = Scheduler(pool, 4)
    scheduler.add_request(0, 3, 2)
    scheduler.add_request(1, 2)
    plan = scheduler.schedule()
    assert work_rows(plan) == [(0, 0, 3, "prefill"), (1, 0, 1, "prefill")]
    assert dict(pool.lengths) == {0: 0, 1: 0}
    simulate_forward(scheduler, plan)
    scheduler.add_request(2, 3)
    plan = scheduler.schedule()
    assert work_rows(plan) == [
        (0, 3, 4, "decode"), (1, 1, 2, "prefill"), (2, 0, 2, "prefill"),
    ]
    assert scheduler.requests[0].status == "running"
    simulate_forward(scheduler, plan)
    assert scheduler.requests[1].status == "finished" and 1 not in pool.block_tables
    plan = scheduler.schedule()
    assert work_rows(plan) == [(0, 4, 5, "decode"), (2, 2, 3, "prefill")]
    simulate_forward(scheduler, plan)
    assert scheduler.schedule() is None
    assert pool.num_free_blocks == pool.num_blocks


def test_newest_victim_recomputes_without_losing_logical_progress():
    pool = policy_pool(3)
    scheduler = Scheduler(pool, 3)
    scheduler.add_request(0, 2, 3)
    scheduler.add_request(1, 2, 3)
    plans = []
    for tick in range(20):
        plan = scheduler.schedule()
        if plan is None:
            break
        plans.append(plan)
        if tick == 3:
            assert plan.preempted == (1,)
            assert work_rows(plan) == [(0, 4, 5, "decode")]
            assert scheduler.requests[1].status == "preempted"
            assert scheduler.requests[1].processed_tokens == 2
            assert 1 not in pool.lengths
        if tick == 4:
            assert work_rows(plan) == [(1, 0, 2, "recompute")]
            assert pool.lengths[1] == 0
        simulate_forward(scheduler, plan)
    else:
        pytest.fail("finite work did not finish")
    assert len(plans) == 8
    assert sum(plan.num_tokens for plan in plans) == 12
    assert sum(item.end - item.start for plan in plans for item in plan.work
               if item.phase == "recompute") == 2
    assert scheduler.requests[1].processed_tokens == 5
    assert scheduler.requests[1].preemptions == 1
    assert all(state.status == "finished" for state in scheduler.requests.values())
    assert pool.num_free_blocks == 3


def test_lower_priority_request_waits_without_unnecessary_eviction():
    pool = policy_pool(4)
    scheduler = Scheduler(pool, 6)
    scheduler.add_request(0, 4, 2)
    scheduler.add_request(1, 2, 2)
    simulate_forward(scheduler, scheduler.schedule())
    table = pool.block_tables[1]
    plan = scheduler.schedule()
    assert work_rows(plan) == [(0, 4, 5, "decode")]
    assert plan.preempted == ()
    assert pool.block_tables[1] == table and pool.lengths[1] == 2
    simulate_forward(scheduler, plan)


def test_victim_selection_prefers_newest_resident_request():
    scheduler = Scheduler(policy_pool(5), 5)
    scheduler.add_request(0, 2, 7)
    scheduler.add_request(1, 2, 6)
    scheduler.add_request(2, 2, 6)
    for _ in range(3):
        simulate_forward(scheduler, scheduler.schedule())
    plan = scheduler.schedule()
    assert plan.preempted == (2,)
    assert scheduler.requests[1].status == "running"
    assert scheduler.requests[2].status == "preempted"
    assert scheduler.requests[2].processed_tokens == 2
    assert 1 in scheduler.cache.block_tables and 2 not in scheduler.cache.block_tables
    simulate_forward(scheduler, plan)


def test_partial_recomputation_and_repeated_preemption_preserve_progress():
    scheduler = Scheduler(policy_pool(5), 3)
    scheduler.add_request(0, 2, 7)
    scheduler.add_request(1, 4, 5)
    partial_replay = False
    for _ in range(40):
        plan = scheduler.schedule()
        if plan is None:
            break
        replay_progress = {}
        for item in plan.work:
            if item.phase == "recompute":
                previous = scheduler.requests[item.request_id].processed_tokens
                partial_replay |= item.end < previous
                replay_progress[item.request_id] = previous
        simulate_forward(scheduler, plan)
        for owner, previous in replay_progress.items():
            assert scheduler.requests[owner].processed_tokens == previous
    else:
        pytest.fail("repeated preemption workload did not finish")
    assert partial_replay and scheduler.requests[1].preemptions >= 2
    assert all(state.status == "finished" for state in scheduler.requests.values())
    assert scheduler.cache.num_free_blocks == scheduler.cache.num_blocks


def test_incomplete_forward_preserves_pending_plan_and_rejects_lifecycle_changes():
    scheduler = Scheduler(policy_pool(), 3)
    scheduler.add_request(0, 2, 1)
    plan = scheduler.schedule()
    before = metadata(scheduler.cache), dict(scheduler.requests)
    with pytest.raises(RuntimeError, match="not committed"):
        scheduler.complete_step()
    assert (metadata(scheduler.cache), dict(scheduler.requests)) == before
    assert scheduler.schedule() is plan
    for operation in (lambda: scheduler.add_request(1, 2), lambda: scheduler.cancel(0)):
        with pytest.raises(RuntimeError, match="pending"):
            operation()
    simulate_forward(scheduler, plan)
    assert scheduler.requests[0].processed_tokens == 2


def test_completion_validates_entire_batch_before_updating_any_request():
    torch = pytest.importorskip("torch")
    scheduler = Scheduler(policy_pool(), 4)
    scheduler.add_request(0, 2)
    scheduler.add_request(1, 2)
    plan = scheduler.schedule()
    scheduler.cache.commit(torch.tensor([0, 0], dtype=torch.int64))
    before = dict(scheduler.requests), metadata(scheduler.cache)
    with pytest.raises(RuntimeError, match="not committed"):
        scheduler.complete_step()
    assert (dict(scheduler.requests), metadata(scheduler.cache)) == before
    scheduler.cache.commit(torch.tensor([1, 1], dtype=torch.int64))
    scheduler.complete_step()
    assert all(state.status == "finished" for state in scheduler.requests.values())


def test_cancel_waiting_running_and_preempted_requests():
    scheduler = Scheduler(policy_pool(3), 3)
    scheduler.add_request(0, 2, 3)
    scheduler.add_request(1, 2, 3)
    scheduler.add_request(2, 2)
    scheduler.cancel(2)
    for _ in range(4):
        simulate_forward(scheduler, scheduler.schedule())
    assert scheduler.requests[1].status == "preempted"
    scheduler.cancel(1)
    assert scheduler.schedule() is None
    assert scheduler.cache.num_free_blocks == 3
    with pytest.raises(ValueError, match="already"):
        scheduler.cancel(1)
    scheduler.add_request(3, 2, 2)
    simulate_forward(scheduler, scheduler.schedule())
    assert scheduler.requests[3].status == "running"
    scheduler.cancel(3)
    assert scheduler.cache.num_free_blocks == 3
    for owner in (-1, True, 1.5, None, 100):
        with pytest.raises(ValueError):
            scheduler.cancel(owner)


@pytest.mark.parametrize("seed", range(20))
def test_randomized_budget_ownership_progress_and_termination(seed):
    rng = random.Random(seed)
    pool = policy_pool(rng.randint(3, 6), rng.randint(1, 3))
    scheduler = Scheduler(pool, rng.randint(1, 6))
    for owner in range(8):
        total = rng.randint(1, min(10, pool.num_blocks * pool.block_size))
        prompt = rng.randint(1, total)
        scheduler.add_request(owner, prompt, total - prompt)
    for _ in range(2000):
        active = [owner for owner, state in scheduler.requests.items()
                  if state.status not in ("finished", "cancelled")]
        if active and rng.random() < 0.1:
            scheduler.cancel(rng.choice(active))
            active = [owner for owner in active if scheduler.requests[owner].status != "cancelled"]
        previous = dict(scheduler.requests)
        plan = scheduler.schedule()
        if plan is None:
            break
        assert 1 <= plan.num_tokens <= scheduler.max_batch_tokens
        assert len({item.request_id for item in plan.work}) == len(plan.work)
        assert not set(plan.preempted).intersection(active[:1])
        for item in plan.work:
            state = previous[item.request_id]
            assert pool.lengths[item.request_id] == item.start
            assert 0 <= item.start < item.end <= state.total_tokens
            if item.phase == "recompute":
                assert item.end <= state.processed_tokens
            elif item.phase == "prefill":
                assert item.start == state.processed_tokens
                assert item.end <= state.prompt_tokens
            else:
                assert item.phase == "decode" and item.end == item.start + 1
                assert item.start == state.processed_tokens >= state.prompt_tokens
        simulate_forward(scheduler, plan)
        used = [block for table in pool.block_tables.values() for block in table]
        assert len(used) == len(set(used))
        assert set(used).isdisjoint(pool.free_blocks)
        assert set(used).union(pool.free_blocks) == set(range(pool.num_blocks))
        assert pool.ref_counts == tuple(int(block in used) for block in range(pool.num_blocks))
        for owner, state in scheduler.requests.items():
            assert previous[owner].processed_tokens <= state.processed_tokens <= state.total_tokens
            if state.status == "finished":
                assert state.processed_tokens == state.total_tokens
    else:
        pytest.fail("finite FCFS workload did not terminate")
    assert all(state.status in ("finished", "cancelled") for state in scheduler.requests.values())
    assert pool.num_free_blocks == pool.num_blocks


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_model_preemption_recompute_parity_and_failed_forward_retry(device):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tinyserve.models.attention import AttentionMetadata
    from tinyserve.models.llama import LlamaForCausalLM
    from tinyserve.probe import compare_paged_batch

    torch.manual_seed(29)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=32, eos_token_id=None, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    reference = transformers.LlamaForCausalLM(config).to(device=device).eval()
    model = LlamaForCausalLM.from_huggingface(reference)
    pool = model.new_managed_paged_cache(3, 2)
    pool.key.fill_(float("nan"))
    pool.value.fill_(float("nan"))
    scheduler = Scheduler(pool, 3)
    histories = {0: torch.tensor([1, 7, 9, 11, 13], device=device),
                 1: torch.tensor([1, 12, 14, 16, 18], device=device)}
    for owner in histories:
        scheduler.add_request(owner, 2, 3)
    phases, victims = [], []
    for tick in range(20):
        plan = scheduler.schedule()
        if plan is None:
            break
        if tick == 1:
            before = metadata(pool), dict(scheduler.requests)
            slots = torch.cat([
                pool.slot_mapping(torch.arange(length, device=device),
                                  torch.full((length,), owner, dtype=torch.int64, device=device))
                for owner, length in pool.lengths.items()
            ])
            old_prefix = [storage.flatten(1, 2)[:, slots].clone()
                          for storage in (pool.key, pool.value)]

            def fail(module, args):
                raise RuntimeError("injected scheduled forward failure")

            hook = model.model.layers[1].register_forward_pre_hook(fail)
            try:
                ids = torch.cat([histories[item.request_id][item.start:item.end] for item in plan.work])
                positions = torch.cat([torch.arange(item.start, item.end, device=device)
                                       for item in plan.work])
                owners = torch.cat([positions.new_full((item.end - item.start,), item.request_id)
                                    for item in plan.work])
                with torch.no_grad(), pytest.raises(RuntimeError, match="injected"):
                    model(ids, positions, AttentionMetadata(positions, owners, paged_cache=pool))
            finally:
                hook.remove()
            assert (metadata(pool), dict(scheduler.requests)) == before
            with pytest.raises(RuntimeError, match="not committed"):
                scheduler.complete_step()
            assert scheduler.schedule() is plan
            for storage, expected in zip((pool.key, pool.value), old_prefix):
                torch.testing.assert_close(storage.flatten(1, 2)[:, slots], expected, rtol=0, atol=0)
        comparison = compare_paged_batch(model, reference, pool, {
            item.request_id: histories[item.request_id][:item.end] for item in plan.work
        })
        assert all(row["all_row_argmax_equal"] for row in comparison["requests"])
        phases.extend(item.phase for item in plan.work)
        victims.extend(plan.preempted)
        scheduler.complete_step()
    else:
        pytest.fail("model workload did not finish")
    assert victims == [1] and "recompute" in phases
    assert all(state.status == "finished" for state in scheduler.requests.values())
    assert pool.num_free_blocks == 3
