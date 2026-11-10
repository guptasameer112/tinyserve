"""D5 generation, pool arithmetic and failure contracts; run only in Colab."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tinyserve.engine.engine import InferenceEngine
from tinyserve.engine.memory import plan_kv_memory
from tinyserve.generation import generate
from tinyserve.paged_cache import ManagedPagedKVCache


class ScriptedModel(torch.nn.Module):
    def __init__(self, eos=None, constant_token=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 4)
        self.config = SimpleNamespace(
            num_hidden_layers=1, num_key_value_heads=1, num_attention_heads=1,
            hidden_size=4, max_position_embeddings=32, eos_token_id=eos,
        )
        self.constant_token = constant_token
        self.calls = []
        self.fail = False
        self.bad_logits = False
        self.eval()

    def get_input_embeddings(self):
        return self.embedding

    def new_managed_paged_cache(self, num_blocks, block_size):
        return ManagedPagedKVCache(self, num_blocks, block_size)

    def forward(self, ids, positions, attention_metadata):
        pool = attention_metadata.paged_cache
        if pool is not None:
            pool.validate_request(self, positions, attention_metadata.sequence_ids)
        self.calls.append((ids.clone(), positions.clone(),
                           None if pool is None else attention_metadata.sequence_ids.clone()))
        if self.fail:
            raise RuntimeError("injected forward failure")
        logits = self.embedding.weight.new_zeros((ids.numel(), 64))
        choices = (ids.long() + 1) % 64
        if self.constant_token is not None:
            choices.fill_(self.constant_token)
        logits.scatter_(1, choices[:, None], 2.0)
        if self.bad_logits:
            logits[-1].fill_(float("nan"))
        if pool is not None:
            pool.commit(attention_metadata.sequence_ids)
        return logits.float()


def engine_for(model=None, blocks=3, block_size=2, batch_tokens=3):
    model = ScriptedModel() if model is None else model
    one_block = plan_kv_memory(model, 10**6, block_size).bytes_per_block
    return InferenceEngine(model, blocks * one_block, block_size, batch_tokens)


def assert_free(engine):
    assert engine.cache.num_free_blocks == engine.cache.num_blocks
    assert not engine.cache.block_tables and not engine.cache.lengths
    assert engine.cache.ref_counts == (0,) * engine.cache.num_blocks


@pytest.mark.parametrize("dtype,bytes_per_token", [(torch.float32, 45056), (torch.float16, 22528)])
def test_tinyllama_memory_uses_kv_heads_and_dtype(dtype, bytes_per_token):
    model = ScriptedModel().to(dtype=dtype)
    model.config = SimpleNamespace(
        hidden_size=2048, num_attention_heads=32, num_key_value_heads=4, num_hidden_layers=22,
    )
    block_bytes = 16 * bytes_per_token
    budget = plan_kv_memory(model, 3 * block_bytes + 123, 16)
    assert budget.bytes_per_token == bytes_per_token
    assert budget.bytes_per_block == block_bytes
    assert budget.num_blocks == 3 and budget.unused_bytes == 123
    assert budget.allocated_bytes + budget.unused_bytes == budget.kv_budget_bytes
    with pytest.raises(FrozenInstanceError):
        setattr(budget, "num_blocks", 4)


@pytest.mark.parametrize("budget,block_size", [
    (0, 2), (-1, 2), (True, 2), (1.5, 2), (None, 2), (1, 2),
    (1000, 0), (1000, -1), (1000, True), (1000, 1.5),
])
def test_invalid_memory_budgets_do_not_allocate(budget, block_size):
    with pytest.raises(ValueError):
        plan_kv_memory(ScriptedModel(), budget, block_size)


def test_actual_storage_matches_budget_and_reuses_one_fixed_pool():
    engine = engine_for()
    assert engine.cache.allocated_bytes == engine.memory_budget.allocated_bytes
    pointers = engine.cache.key.data_ptr(), engine.cache.value.data_ptr()
    for owner in range(2):
        engine.add_request(owner, torch.tensor([1, 7]), max_new_tokens=4)
        engine.run()
        assert_free(engine)
        assert (engine.cache.key.data_ptr(), engine.cache.value.data_ptr()) == pointers


def test_three_outputs_need_prompt_and_two_decode_forwards():
    engine = engine_for(blocks=4, batch_tokens=8)
    prompt = torch.tensor([1, 7, 9], dtype=torch.int32)
    engine.add_request(0, prompt, max_new_tokens=3)
    outputs = engine.run()
    assert outputs[0].token_ids == (1, 7, 9, 10, 11, 12)
    assert outputs[0].generated_tokens == (10, 11, 12)
    assert outputs[0].finish_reason == "length"
    assert [call[0].numel() for call in engine.model.calls] == [3, 1, 1]
    assert [call[1].tolist() for call in engine.model.calls] == [[0, 1, 2], [3], [4]]
    assert engine.scheduler.requests[0].processed_tokens == 5
    assert prompt.tolist() == [1, 7, 9]
    assert engine.step() == ()
    assert_free(engine)


def test_chunked_prefill_only_samples_its_last_row():
    engine = engine_for(blocks=5, batch_tokens=2)
    engine.add_request(0, torch.tensor([1, 2, 3, 4, 5]), max_new_tokens=1)
    assert engine.step() == ()
    assert engine.step() == ()
    result, = engine.step()
    assert result.generated_tokens == (6,) and result.finished
    assert [call[0].numel() for call in engine.model.calls] == [2, 2, 1]
    assert_free(engine)


def test_continuous_arrival_mixes_prefill_decode_and_routes_each_request():
    engine = engine_for(blocks=8, batch_tokens=4)
    engine.add_request(8, torch.tensor([1, 3, 5]), max_new_tokens=3)
    assert [output.request_id for output in engine.step()] == [8]
    engine.add_request(2, torch.tensor([1, 10, 20]), max_new_tokens=1)
    outputs = engine.step()
    assert [output.request_id for output in outputs] == [8, 2]
    assert outputs[0].generated_tokens == (6, 7) and not outputs[0].finished
    assert outputs[1].generated_tokens == (21,) and outputs[1].finished
    assert engine.model.calls[-1][2].tolist() == [8, 2, 2, 2]
    assert engine.model.calls[-1][1].tolist() == [3, 0, 1, 2]
    engine.run()
    assert engine.get_output(8).generated_tokens == (6, 7, 8)
    assert_free(engine)


@pytest.mark.parametrize("eos", [2, [2, 3], (2, 3)])
def test_eos_is_included_and_releases_blocks_early(eos):
    engine = engine_for(ScriptedModel(eos=eos, constant_token=2))
    engine.add_request(0, torch.tensor([2, 7]), max_new_tokens=4)
    result, = engine.step()
    assert result.generated_tokens == (2,) and result.finish_reason == "eos"
    assert engine.scheduler.requests[0].status == "cancelled"
    assert_free(engine)


def test_eos_at_final_budget_and_prompt_eos_semantics():
    engine = engine_for(ScriptedModel(eos=2, constant_token=2))
    engine.add_request(0, torch.tensor([2]), max_new_tokens=1)
    output, = engine.run()
    assert output.token_ids == (2, 2) and output.finish_reason == "eos"
    assert engine.scheduler.requests[0].status == "finished"
    assert_free(engine)


def test_zero_budget_clones_prompt_and_does_not_register_scheduler_work():
    engine = engine_for()
    prompt = torch.tensor([1, 7, 9])
    output = engine.add_request(0, prompt, max_new_tokens=0)
    prompt.fill_(4)
    assert output.token_ids == (1, 7, 9) and output.finished
    assert engine.run() == (output,)
    assert engine.model.calls == [] and not engine.scheduler.requests
    assert_free(engine)
    with pytest.raises(ValueError, match="already"):
        engine.add_request(0, prompt, max_new_tokens=0)


@pytest.mark.parametrize("kwargs", [
    {"max_new_tokens": -1}, {"max_new_tokens": True}, {"max_new_tokens": 1.5},
    {"temperature": -1}, {"temperature": float("nan")}, {"top_k": -1}, {"top_p": 0},
    {"generator": "not a generator"},
])
def test_admission_checks_are_atomic_even_for_zero_output_requests(kwargs):
    engine = engine_for()
    options = {"max_new_tokens": 0, **kwargs}
    with pytest.raises(ValueError):
        engine.add_request(0, torch.tensor([1, 7]), **options)
    assert not engine.scheduler.requests and not engine.has_pending_requests
    assert_free(engine)


@pytest.mark.parametrize("prompt", [
    torch.tensor([], dtype=torch.int64), torch.tensor([[1, 2]]),
    torch.tensor([1.0]), torch.tensor([-1]), torch.tensor([64]),
])
def test_invalid_prompts_raise_before_reservation(prompt):
    engine = engine_for()
    with pytest.raises(ValueError):
        engine.add_request(0, prompt, max_new_tokens=1)
    assert_free(engine)


@pytest.mark.parametrize("owner", [-1, True, 1.5, None, 2**63])
def test_invalid_request_ids_raise(owner):
    engine = engine_for()
    with pytest.raises(ValueError):
        engine.add_request(owner, torch.tensor([1]), max_new_tokens=0)
    assert_free(engine)


@pytest.mark.parametrize("eos", [True, -1, 64, [2, False], "2"])
def test_invalid_eos_raises_before_admission(eos):
    engine = engine_for(ScriptedModel(eos=eos))
    with pytest.raises(ValueError):
        engine.add_request(0, torch.tensor([1]), max_new_tokens=0)
    assert not engine.has_pending_requests


def test_context_and_pool_limits_are_strict_and_exact_boundaries_work():
    engine = engine_for(blocks=16)
    prompt = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="context"):
        engine.add_request(0, prompt, max_new_tokens=31)
    engine.add_request(0, prompt, max_new_tokens=30)
    engine.run()
    assert engine.get_output(0).token_ids[-1] == 32
    small = engine_for(blocks=2)
    with pytest.raises(ValueError, match="entire block pool"):
        small.add_request(0, prompt, max_new_tokens=4)
    small.add_request(0, prompt, max_new_tokens=3)
    small.run()
    assert_free(small)


def test_cancellation_and_unique_lifetime_ids():
    engine = engine_for(blocks=5)
    engine.add_request(0, torch.tensor([1, 7]), max_new_tokens=4)
    engine.add_request(1, torch.tensor([1, 9]), max_new_tokens=3)
    output = engine.cancel(1)
    assert output.generated_tokens == () and output.finish_reason == "cancelled"
    engine.step()
    output = engine.cancel(0)
    assert output.generated_tokens == (8,) and output.finish_reason == "cancelled"
    assert_free(engine)
    for operation in (lambda: engine.cancel(0),
                      lambda: engine.add_request(0, torch.tensor([1]), max_new_tokens=1),
                      lambda: engine.get_output(999)):
        with pytest.raises(ValueError):
            operation()


def test_preempted_request_can_be_cancelled_without_touching_other_blocks():
    engine = engine_for()
    for owner in range(2):
        engine.add_request(owner, torch.tensor([1, 7 + owner]), max_new_tokens=4)
    for _ in range(4):
        engine.step()
    assert engine.scheduler.requests[1].status == "preempted"
    engine.cancel(1)
    assert engine.get_output(1).generated_tokens == (9,)
    assert_free(engine)


@pytest.mark.parametrize("stochastic", [False, True])
def test_preemption_replays_without_duplicate_outputs_or_extra_random_draws(stochastic):
    engine = engine_for()
    expected = {}
    generators = {}
    for owner in range(2):
        prompt = torch.tensor([1, 7 + owner])
        generator = torch.Generator().manual_seed(31 + owner)
        reference_generator = torch.Generator().manual_seed(31 + owner)
        temperature = 0.8 if stochastic else 0.0
        expected[owner] = tuple(generate(engine.model, prompt, 4, temperature=temperature,
                                         generator=reference_generator).tolist())
        engine.add_request(owner, prompt, 4, temperature=temperature, generator=generator)
        generators[owner] = generator, reference_generator
    engine.model.calls.clear()
    replay_steps = 0
    for _ in range(30):
        if not engine.has_pending_requests:
            break
        plan = engine.scheduler.schedule()
        assert plan is not None
        replay = {item.request_id for item in plan.work if item.phase == "recompute"}
        outputs = engine.step()
        assert replay.isdisjoint(output.request_id for output in outputs)
        replay_steps += bool(replay)
    else:
        pytest.fail("finite engine workload did not terminate")
    assert replay_steps > 0 and engine.scheduler.requests[1].preemptions == 1
    for owner in expected:
        assert engine.get_output(owner).token_ids == expected[owner]
        assert torch.equal(generators[owner][0].get_state(), generators[owner][1].get_state())
    assert_free(engine)


def test_forward_failure_keeps_plan_histories_and_rng_for_retry():
    engine = engine_for()
    generator = torch.Generator().manual_seed(31)
    engine.add_request(0, torch.tensor([1, 7]), 4, temperature=0.8, generator=generator)
    plan = engine.scheduler.schedule()
    before = engine.get_output(0), dict(engine.cache.lengths), generator.get_state().clone()
    engine.model.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        engine.step()
    assert engine.scheduler.schedule() is plan
    assert engine.get_output(0) == before[0] and dict(engine.cache.lengths) == before[1]
    assert torch.equal(generator.get_state(), before[2])
    with pytest.raises(RuntimeError, match="pending"):
        engine.add_request(1, torch.tensor([1]), 1)
    with pytest.raises(RuntimeError, match="pending"):
        engine.cancel(0)
    engine.model.fail = False
    engine.run()
    assert len(engine.get_output(0).generated_tokens) == 4
    assert_free(engine)


@pytest.mark.parametrize("use_generator", [False, True])
def test_sampling_failure_keeps_committed_logits_and_restores_all_rng(use_generator, monkeypatch):
    import tinyserve.engine.engine as engine_module

    engine = engine_for(blocks=4, batch_tokens=4)
    generator = torch.Generator().manual_seed(42) if use_generator else None
    for owner in range(2):
        engine.add_request(owner, torch.tensor([1, 7 + owner]), 2,
                           temperature=0.8, generator=generator)
    original = engine_module.sample_token
    state = generator.get_state().clone() if generator is not None else torch.random.get_rng_state().clone()
    calls = []

    def fail_second(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("injected sampling failure")
        return original(*args)

    monkeypatch.setattr(engine_module, "sample_token", fail_second)
    with pytest.raises(ValueError, match="injected sampling"):
        engine.step()
    assert dict(engine.cache.lengths) == {0: 2, 1: 2}
    assert all(engine.get_output(owner).generated_tokens == () for owner in range(2))
    after = generator.get_state() if generator is not None else torch.random.get_rng_state()
    assert torch.equal(state, after)
    monkeypatch.setattr(engine_module, "sample_token", original)
    engine.step()
    assert len(engine.model.calls) == 1
    engine.run()
    assert_free(engine)


def test_invalid_logits_raise_without_success_shaped_output():
    engine = engine_for()
    engine.model.bad_logits = True
    engine.add_request(0, torch.tensor([1, 7]), 4)
    with pytest.raises(ValueError, match="NaN"):
        engine.step()
    assert engine.get_output(0).generated_tokens == ()
    assert engine.has_pending_requests


def test_model_eval_device_dtype_must_not_change_after_pool_creation():
    model = ScriptedModel().train()
    with pytest.raises(ValueError, match="eval"):
        engine_for(model)
    model.eval()
    engine = engine_for(model)
    model.half()
    with pytest.raises(ValueError, match="device/dtype"):
        engine.add_request(0, torch.tensor([1]), 1)


def test_close_aborts_a_failed_pending_forward_and_is_idempotent():
    engine = engine_for()
    engine.add_request(0, torch.tensor([1, 7]), 4)
    engine.model.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        engine.step()
    assert engine.cache.block_tables
    engine.close()
    engine.close()
    assert not engine.has_pending_requests
    assert engine.get_output(0).finish_reason == "cancelled"
    assert_free(engine)
    with pytest.raises(RuntimeError, match="closed"):
        engine.step()
    with pytest.raises(RuntimeError, match="closed"):
        engine.add_request(1, torch.tensor([1]), 1)


def test_available_memory_requires_cuda_and_explicit_valid_headroom():
    model = ScriptedModel()
    with pytest.raises(ValueError, match="CUDA"):
        InferenceEngine.from_available_memory(model, 1024)
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive"):
            InferenceEngine.from_available_memory(model, value)


def model_pair(device, dtype=torch.float32):
    transformers = pytest.importorskip("transformers")
    from tinyserve.models.llama import LlamaForCausalLM

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(29)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=32, eos_token_id=None, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    reference = transformers.LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
    return reference, LlamaForCausalLM.from_huggingface(reference)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_model_preemption_generation_matches_hf_and_c2(device):
    reference, model = model_pair(device)
    engine = engine_for(model)
    prompts = [torch.tensor([1, 7], device=device), torch.tensor([1, 12], device=device)]
    expected = [tuple(generate(model, prompt, 4).tolist()) for prompt in prompts]
    for owner, prompt in enumerate(prompts):
        engine.add_request(owner, prompt, 4)
    for _ in range(30):
        if not engine.has_pending_requests:
            break
        engine.step()
        for owner, prompt in enumerate(prompts):
            output = engine.get_output(owner)
            history = prompt.new_tensor(output.token_ids)
            if not output.finished and output.generated_tokens:
                with torch.no_grad():
                    next_hf = reference(history[None], use_cache=False).logits[0, -1].argmax().item()
                assert expected[owner][len(output.token_ids)] == next_hf
    else:
        pytest.fail("real model engine did not terminate")
    assert engine.scheduler.requests[1].preemptions == 1
    for owner, prompt in enumerate(prompts):
        assert engine.get_output(owner).token_ids == expected[owner]
        with torch.no_grad():
            hf = reference.generate(prompt[None], max_new_tokens=4, do_sample=False,
                                    pad_token_id=0, eos_token_id=None)
        assert engine.get_output(owner).token_ids == tuple(hf[0].tolist())
    assert_free(engine)


def test_real_later_layer_failure_retries_without_changing_resident_prefix():
    _, model = model_pair("cpu")
    engine = engine_for(model, blocks=4)
    prompt = torch.tensor([1, 7])
    expected = tuple(generate(model, prompt, 3).tolist())
    engine.add_request(0, prompt, 3)
    engine.step()
    plan = engine.scheduler.schedule()
    before = engine.get_output(0), dict(engine.cache.lengths)
    old_key, old_value = engine.cache.key.clone(), engine.cache.value.clone()

    def fail(module, args):
        raise RuntimeError("injected later-layer failure")

    hook = model.model.layers[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="later-layer"):
            engine.step()
    finally:
        hook.remove()
    assert engine.scheduler.schedule() is plan
    assert (engine.get_output(0), dict(engine.cache.lengths)) == before
    block = engine.cache.block_tables[0][0]
    for actual, saved in ((engine.cache.key, old_key), (engine.cache.value, old_value)):
        torch.testing.assert_close(actual[:, block], saved[:, block], rtol=0, atol=0)
    engine.run()
    assert engine.get_output(0).token_ids == expected
    assert_free(engine)


def test_cuda_available_memory_rounding_and_fp16_reference(monkeypatch):
    _, model = model_pair("cuda", torch.float16)
    unit = plan_kv_memory(model, 10**6, 2).bytes_per_block
    reserve = 4096
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (reserve + 3 * unit + 7, 10**6))
    engine = InferenceEngine.from_available_memory(model, reserve, block_size=2, max_batch_tokens=3)
    assert engine.memory_budget.num_blocks == 3 and engine.memory_budget.unused_bytes == 7
    with pytest.raises(ValueError, match="leaves no free"):
        InferenceEngine.from_available_memory(model, reserve + 3 * unit + 7)
    from tinyserve.models.attention import AttentionMetadata

    rounded_fp32 = type(model)(model.config, device="cuda", dtype=torch.float32).eval()
    rounded_fp32.load_state_dict(model.state_dict(), strict=True)
    prompt = torch.tensor([1, 7], device="cuda")
    engine.add_request(0, prompt, 4)
    while engine.has_pending_requests:
        plan = engine.scheduler.schedule()
        assert plan is not None
        histories = {item.request_id: prompt.new_tensor(engine.get_output(item.request_id).token_ids)
                     for item in plan.work}
        captured = []
        hook = model.register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
        try:
            engine.step()
        finally:
            hook.remove()
        offset = 0
        for item in plan.work:
            offset += item.end - item.start
            positions = torch.arange(item.end, device=prompt.device)
            with torch.no_grad():
                fp32_logits = rounded_fp32(histories[item.request_id][:item.end], positions,
                                           AttentionMetadata(positions))[item.start:item.end]
            actual = captured[0][offset - (item.end - item.start):offset]
            torch.testing.assert_close(actual, fp32_logits, rtol=1e-2, atol=1e-2)
            if item.phase != "recompute" and item.end >= engine.scheduler.requests[item.request_id].prompt_tokens:
                output = engine.get_output(item.request_id)
                assert output.generated_tokens[-1] == actual[-1].argmax().item()
    assert_free(engine)
