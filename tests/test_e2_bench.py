"""E2 accounting, failure and integration checks. Run only in Colab."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import bench.engine_vs_hf as benchmark
from test_d5_engine import ScriptedModel
from tinyserve.models.llama import LlamaForCausalLM


def test_metrics_use_the_same_run_and_count_all_requests():
    result = benchmark.metrics_from_trace([[0.1, 0.3, 0.5], [0.2, 0.4, 0.7]], 0.8)
    assert result["total_ms"] == 800
    assert result["output_tokens"] == 6
    assert result["output_tokens_per_s"] == 7.5
    first, second = result["requests"]
    assert first["ttft_ms"] == 100 and second["ttft_ms"] == 200
    assert first["latency_ms"] == 500 and second["latency_ms"] == 700
    assert first["inter_token_ms"] == pytest.approx([200, 200])
    assert second["inter_token_ms"] == pytest.approx([200, 300])
    assert second["token_ready_ms"] == [200, 400, 700]


def test_single_output_has_ttft_and_latency_but_no_decode_interval():
    result = benchmark.metrics_from_trace([[0.1]], 0.2)
    assert result["requests"] == [{
        "token_ready_ms": [100], "ttft_ms": 100, "latency_ms": 100, "inter_token_ms": [],
    }]
    assert result["output_tokens_per_s"] == 5


@pytest.mark.parametrize("times,total", [
    ([], 1), ([[]], 1), ([[0.2], [0.2, 0.3]], 1),
    ([[0.2, 0.1]], 1), ([[0]], 1), ([[-0.1]], 1), ([[float("nan")]], 1),
    ([[float("inf")]], 1), ([[1.1]], 1), ([[0.1]], 0),
    ([[0.1]], -1), ([[0.1]], float("nan")), ([[0.1]], float("inf")),
])
def test_invalid_timing_is_rejected_without_clamping(times, total):
    with pytest.raises(ValueError):
        benchmark.metrics_from_trace(times, total)


def test_hf_trace_skips_only_the_prompt_and_timestamps_batch_tokens(monkeypatch):
    ids = torch.tensor([[1, 7], [1, 9]])
    trace = benchmark._HFTrace(ids)
    trace.start = 10
    clock = iter([10.25, 10.5])
    syncs = []
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    monkeypatch.setattr(benchmark, "_sync", lambda device: syncs.append(device))
    trace.put(ids)
    assert trace.times == [[], []]
    trace.put(torch.tensor([8, 10]))
    trace.put(torch.tensor([9, 11]))
    trace.end()
    assert trace.ended
    assert trace.times == [[0.25, 0.5], [0.25, 0.5]]
    assert trace.tokens == [[8, 9], [10, 11]]
    assert syncs == [ids.device, ids.device]
    with pytest.raises(RuntimeError, match="outside"):
        trace.put(torch.tensor([10, 12]))


def test_hf_trace_rejects_missing_prompt_wrong_shape_and_inactive_calls():
    ids = torch.tensor([[1, 7], [1, 9]])
    trace = benchmark._HFTrace(ids)
    with pytest.raises(RuntimeError, match="outside"):
        trace.put(ids)
    trace.start = 0
    with pytest.raises(RuntimeError, match="complete prompt"):
        trace.put(torch.tensor([8, 10]))
    trace.put(ids)
    with pytest.raises(RuntimeError, match="one token"):
        trace.put(ids)


def test_cuda_sync_uses_the_input_device_and_skips_cpu(monkeypatch):
    observed = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: observed.append(device))
    benchmark._sync(torch.device("cpu"))
    benchmark._sync(torch.device("cuda:1"))
    assert observed == [torch.device("cuda:1")]


def fake_measurements(monkeypatch, drift=False, fail_hf=False):
    calls = []

    def measure(mode, model, ids, new_tokens):
        assert not torch.is_grad_enabled()
        calls.append((mode, model.config.eos_token_id))
        if mode == "hf" and fail_hf:
            raise RuntimeError("injected HF failure")
        total = 2 if mode == "tinyserve" else 1
        times = [[(i + 1) * total / (new_tokens + 1) for i in range(new_tokens)] for _ in ids]
        row = benchmark.metrics_from_trace(times, total)
        row.update({
            "generated_token_ids": [[4 if mode == "hf" and drift else 3] * new_tokens for _ in ids],
            "baseline_allocated_bytes": None, "peak_allocated_bytes": None,
            "incremental_peak_allocated_bytes": None,
        })
        return row

    monkeypatch.setattr(benchmark, "_measure_tinyserve",
                        lambda model, ids, new, block, budget: measure("tinyserve", model, ids, new))
    monkeypatch.setattr(benchmark, "_measure_hf",
                        lambda model, ids, new: measure("hf", model, ids, new))
    return calls


def test_warmups_alternation_slowdown_ranges_and_token_drift(monkeypatch):
    model, reference = ScriptedModel(eos=[2, 3]), ScriptedModel(eos=2)
    original = model.config.eos_token_id
    ids = torch.tensor([[1, 7], [1, 9]])
    calls = fake_measurements(monkeypatch, drift=True)
    result = benchmark.run_case(model, reference, ids, new_tokens=3, trials=3)
    assert [mode for mode, _ in calls] == [
        "tinyserve", "hf", "tinyserve", "hf", "hf", "tinyserve", "tinyserve", "hf",
    ]
    assert all(eos is None for mode, eos in calls if mode == "tinyserve")
    assert all(eos == 2 for mode, eos in calls if mode == "hf")
    assert model.config.eos_token_id is original and reference.config.eos_token_id == 2
    assert ids.tolist() == [[1, 7], [1, 9]]
    assert not result["token_ids_equal_in_all_trials"]
    assert result["trial_records"][0]["token_ids_equal_by_request"] == [False, False]
    assert result["hf_over_tinyserve_total_time"] == {"min": 0.5, "median": 0.5, "max": 0.5, "n": 3}
    assert result["summaries"]["tinyserve"]["total_ms"]["n"] == 3
    assert result["summaries"]["tinyserve"]["ttft_ms"]["n"] == 6
    assert result["summaries"]["tinyserve"]["inter_token_ms"]["n"] == 12
    assert result["summaries"]["hf"]["peak_allocated_bytes"] is None
    assert result["pool_setup_excluded"] and result["hf_dynamic_cache_setup_included"]


def test_single_token_case_summary_has_no_itl(monkeypatch):
    fake_measurements(monkeypatch)
    result = benchmark.run_case(ScriptedModel(), ScriptedModel(), torch.tensor([[1]]), 1, trials=1)
    assert result["summaries"]["hf"]["inter_token_ms"] is None
    assert result["token_ids_equal_in_all_trials"]


def test_hf_failure_restores_original_eos_and_propagates(monkeypatch):
    model = ScriptedModel(eos=[2, 3])
    original = model.config.eos_token_id
    fake_measurements(monkeypatch, fail_hf=True)
    with pytest.raises(RuntimeError, match="injected HF"):
        benchmark.run_case(model, ScriptedModel(), torch.tensor([[1]]), 3)
    assert model.config.eos_token_id is original


@pytest.mark.parametrize("options", [
    {"new_tokens": 0}, {"new_tokens": True}, {"new_tokens": 1.5},
    {"new_tokens": 2, "trials": 0}, {"new_tokens": 2, "trials": True},
    {"new_tokens": 2, "block_size": 0}, {"new_tokens": 2, "block_size": 1.5},
    {"new_tokens": 2, "max_batch_tokens": 0},
])
def test_invalid_counts_fail_before_warmup(monkeypatch, options):
    calls = fake_measurements(monkeypatch)
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.run_case(ScriptedModel(), ScriptedModel(), torch.tensor([[1]]), **options)
    assert not calls


@pytest.mark.parametrize("ids", [
    None, torch.tensor([1]), torch.empty((0, 2), dtype=torch.int64),
    torch.empty((2, 0), dtype=torch.int64), torch.tensor([[1.0]]),
    torch.tensor([[64]]), torch.tensor([[-1]]),
])
def test_invalid_prompt_fails_before_warmup(monkeypatch, ids):
    calls = fake_measurements(monkeypatch)
    with pytest.raises(ValueError):
        benchmark.run_case(ScriptedModel(), ScriptedModel(), ids, 2)
    assert not calls


def test_context_dtype_and_eval_validation_precede_warmup(monkeypatch):
    calls = fake_measurements(monkeypatch)
    model, reference = ScriptedModel(), ScriptedModel()
    with pytest.raises(ValueError, match="context limit"):
        benchmark.run_case(model, reference, torch.tensor([[1, 7]]), 31)
    reference.half()
    with pytest.raises(ValueError, match="same device and dtype"):
        benchmark.run_case(model, reference, torch.tensor([[1]]), 2)
    reference.float().train()
    with pytest.raises(ValueError, match="eval mode"):
        benchmark.run_case(model, reference, torch.tensor([[1]]), 2)
    assert not calls


def test_failed_engine_warmup_closes_pool_and_restores_eos(monkeypatch):
    model = ScriptedModel(eos=2)
    model.fail = True
    created = []
    engine_class = benchmark.InferenceEngine

    def capture(*args, **kwargs):
        engine = engine_class(*args, **kwargs)
        created.append(engine)
        return engine

    monkeypatch.setattr(benchmark, "InferenceEngine", capture)
    with pytest.raises(RuntimeError, match="injected forward"):
        benchmark.run_case(model, ScriptedModel(), torch.tensor([[1, 7]]), 3, block_size=2)
    assert model.config.eos_token_id == 2 and len(created) == 1
    engine = created[0]
    assert engine.cache.num_free_blocks == engine.cache.num_blocks
    assert not engine.cache.block_tables and not engine.has_pending_requests
    with pytest.raises(RuntimeError, match="closed"):
        engine.step()


@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32), ("cuda", torch.float32), ("cuda", torch.float16),
])
def test_real_models_exact_work_token_parity_and_memory(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(4)
    config = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
        max_position_embeddings=32, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    reference = transformers.LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
    model = LlamaForCausalLM.from_huggingface(reference)
    before = deepcopy(reference.generation_config.to_dict())
    ids = torch.tensor([[1, 7, 9], [1, 11, 12]], device=device)
    # A two-row budget deliberately chunks prefill and staggers first outputs.
    result = benchmark.run_case(model, reference, ids, 3, trials=2,
                                block_size=2, max_batch_tokens=2)
    assert model.config.eos_token_id == reference.config.eos_token_id == 2
    assert reference.generation_config.to_dict() == before
    if dtype == torch.float32:
        assert result["token_ids_equal_in_all_trials"]
    assert result["dtype"] == str(dtype)
    for pair in result["trial_records"]:
        for mode in ("tinyserve", "hf"):
            record = pair[mode]
            assert record["output_tokens"] == 6
            assert len(record["requests"]) == 2
            assert all(len(row["token_ready_ms"]) == 3 for row in record["requests"])
            assert all(len(row) == 3 for row in record["generated_token_ids"])
            assert record["total_ms"] > 0 and record["output_tokens_per_s"] > 0
            if device == "cuda":
                assert record["peak_allocated_bytes"] >= record["baseline_allocated_bytes"] > 0
                assert record["incremental_peak_allocated_bytes"] >= 0
            else:
                assert record["peak_allocated_bytes"] is None
        assert pair["tinyserve"]["num_blocks"] == pair["tinyserve"]["final_free_blocks"] == 6
        assert pair["tinyserve"]["preemptions"] == 0


def test_short_hf_stream_or_wrong_returned_tokens_are_rejected(monkeypatch):
    reference = ScriptedModel()
    ids = torch.tensor([[1, 7]])

    def fake_generate(input_ids, streamer, **kwargs):
        config = kwargs["generation_config"]
        assert config.eos_token_id is None and config.use_cache and not config.do_sample
        assert config.cache_implementation == "dynamic"
        assert kwargs["use_model_defaults"] is False and kwargs["eos_token_id"] is None
        assert torch.equal(kwargs["attention_mask"], torch.ones_like(input_ids))
        streamer.put(input_ids.cpu())
        streamer.put(torch.tensor([8]))
        streamer.end()
        return torch.tensor([[1, 7, 9]])

    monkeypatch.setattr(reference, "generate", fake_generate, raising=False)
    with pytest.raises(RuntimeError, match="fixed output budget"):
        benchmark._measure_hf(reference, ids, 2)
    with pytest.raises(RuntimeError, match="fixed output budget"):
        benchmark._measure_hf(reference, ids, 1)
