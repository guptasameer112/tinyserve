"""C4 benchmark accounting and failure behavior; run these tests in Colab."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import bench.cache_curve as benchmark  # noqa: E402
from tinyserve.generation import generate  # noqa: E402


class StubModel(torch.nn.Module):
    def __init__(self, eos=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.config = SimpleNamespace(eos_token_id=eos, max_position_embeddings=32)
        self.eval()

    def get_input_embeddings(self):
        return self.embedding


def install_fake_generation(monkeypatch, model, fail=False, short=False, drift=False):
    calls = []

    def fake_generate(current, ids, max_new_tokens=20, *, use_cache=False):
        assert current is model and not torch.is_grad_enabled()
        calls.append((max_new_tokens, use_cache, current.config.eos_token_id))
        if max_new_tokens == 0:
            return ids.clone()
        if fail and use_cache:
            raise RuntimeError("injected benchmark failure")
        if short:
            return ids.clone()
        assert current.config.eos_token_id is None
        token = 4 if drift and use_cache else 3
        return torch.cat((ids, ids.new_full((max_new_tokens,), token)))

    monkeypatch.setattr(benchmark, "generate", fake_generate)
    return calls


def test_exact_accounting_order_ranges_and_eos_restoration(monkeypatch):
    model = StubModel()
    calls = install_fake_generation(monkeypatch, model)
    clock = iter([0, 4, 4, 6, 6, 8, 8, 14, 14, 22, 22, 24])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(benchmark, "kv_cache_bytes_per_token", lambda config, size: 16)
    ids = torch.tensor([1, 7])
    result = benchmark.run_case(model, ids, new_tokens=4, trials=3)
    assert model.config.eos_token_id == 2 and ids.tolist() == [1, 7]
    assert calls[0] == (0, False, 2)
    assert [mode for budget, mode, eos in calls[1:]] == [False, True, False, True, True, False, False, True]
    assert all(budget == 4 and eos is None for budget, mode, eos in calls[1:])
    assert result["no_cache"]["total_ms"] == {"min": 4000, "median": 6000, "max": 8000, "n": 3}
    assert result["cached"]["total_ms"] == {"min": 2000, "median": 2000, "max": 2000, "n": 3}
    assert result["speedup"] == {"min": 2, "median": 3, "max": 4, "n": 3}
    assert result["no_cache"]["end_to_end_tokens_per_s"]["median"] == pytest.approx(4 / 6)
    assert result["cache_reserved_bytes"] == 96
    assert result["no_cache"]["peak_allocated_bytes"] is None
    assert result["token_ids_equal_in_all_trials"]
    assert [pair["order"] for pair in result["trial_records"]] == [
        ["no_cache", "cached"], ["cached", "no_cache"], ["no_cache", "cached"],
    ]


@pytest.mark.parametrize("eos", [2, [2, 4], None])
def test_failure_restores_original_eos(monkeypatch, eos):
    model = StubModel(eos=eos)
    install_fake_generation(monkeypatch, model, fail=True)
    with pytest.raises(RuntimeError, match="injected"):
        benchmark.run_case(model, torch.tensor([1]), new_tokens=2, trials=1)
    assert model.config.eos_token_id is eos


def test_wrong_output_count_raises_and_restores_eos(monkeypatch):
    model = StubModel()
    install_fake_generation(monkeypatch, model, short=True)
    clock = iter([0, 1])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    with pytest.raises(RuntimeError, match="token count"):
        benchmark.run_case(model, torch.tensor([1]), new_tokens=2, trials=1)
    assert model.config.eos_token_id == 2


def test_slow_cache_and_token_drift_are_reported_without_a_success_threshold(monkeypatch):
    model = StubModel()
    install_fake_generation(monkeypatch, model, drift=True)
    monkeypatch.setattr(benchmark, "kv_cache_bytes_per_token", lambda config, size: 16)
    clock = iter([0, 1, 1, 3])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    result = benchmark.run_case(model, torch.tensor([1]), new_tokens=2, trials=1)
    assert result["speedup"]["median"] == 0.5
    assert not result["token_ids_equal_in_all_trials"]
    assert not result["trial_records"][0]["token_ids_equal"]
    assert model.config.eos_token_id == 2


@pytest.mark.parametrize("elapsed", [0, -1, float("nan"), float("inf")])
def test_invalid_timing_raises_without_clamping(monkeypatch, elapsed):
    model = StubModel()
    install_fake_generation(monkeypatch, model)
    clock = iter([0, elapsed])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    with pytest.raises(RuntimeError, match="finite and positive"):
        benchmark.run_case(model, torch.tensor([1]), new_tokens=2, trials=1)
    assert model.config.eos_token_id == 2


@pytest.mark.parametrize("options", [
    {"new_tokens": 0}, {"new_tokens": True}, {"new_tokens": 1.5},
    {"new_tokens": -1}, {"new_tokens": 2, "trials": 0},
    {"new_tokens": 2, "trials": True}, {"new_tokens": 2, "trials": 1.5},
])
def test_invalid_counts_raise_before_generation(monkeypatch, options):
    model = StubModel()
    calls = install_fake_generation(monkeypatch, model)
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.run_case(model, torch.tensor([1]), **options)
    assert calls == [] and model.config.eos_token_id == 2


def test_context_overflow_raises_before_changing_eos(monkeypatch):
    model = StubModel()
    calls = install_fake_generation(monkeypatch, model)
    with pytest.raises(ValueError, match="context limit"):
        benchmark.run_case(model, torch.tensor([1, 7]), new_tokens=31)
    assert calls == [(0, False, 2)] and model.config.eos_token_id == 2


@pytest.mark.parametrize("ids", [torch.tensor([[1]]), torch.tensor([8]), torch.tensor([1.0])])
def test_real_prompt_validation_is_reused(ids):
    model = StubModel()
    with pytest.raises(ValueError):
        benchmark.run_case(model, ids, new_tokens=2)
    assert model.config.eos_token_id == 2


def test_tiny_real_model_fixed_lengths_and_eos_restoration():
    transformers = pytest.importorskip("transformers")
    from tinyserve.models.attention import AttentionMetadata
    from tinyserve.models.llama import LlamaForCausalLM

    torch.manual_seed(4)
    config = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
        max_position_embeddings=32, eos_token_id=None, pad_token_id=0,
    )
    model = LlamaForCausalLM(config).eval()
    ids = torch.tensor([1, 7])
    positions = torch.arange(2)
    with torch.no_grad():
        eos = model(ids, positions, AttentionMetadata(positions))[-1].argmax().item()
    model.config.eos_token_id = eos
    result = benchmark.run_case(model, ids, new_tokens=3, trials=2)
    assert model.config.eos_token_id == eos
    assert generate(model, ids, max_new_tokens=3).numel() == 3
    for pair in result["trial_records"]:
        for mode in ("no_cache", "cached"):
            assert len(pair[mode]["generated_token_ids"]) == pair[mode]["new_tokens"] == 3
            assert pair[mode]["total_ms"] > 0
    assert result["token_ids_equal_in_all_trials"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_timing_and_memory_accounting(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from tinyserve.models.llama import LlamaForCausalLM

    config = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
        max_position_embeddings=32, eos_token_id=None, pad_token_id=0,
    )
    model = LlamaForCausalLM(config, device="cuda", dtype=torch.float16).eval()
    ids = torch.tensor([1, 7], device="cuda")
    synchronized_devices = []
    original_sync = torch.cuda.synchronize

    def sync(device):
        synchronized_devices.append(device)
        original_sync(device)

    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", sync)
    result = benchmark.run_case(model, ids, new_tokens=2, trials=1)
    assert synchronized_devices == [ids.device] * 4
    for mode in ("no_cache", "cached"):
        pair = result["trial_records"][0][mode]
        assert pair["peak_allocated_bytes"] >= model.get_input_embeddings().weight.numel() * 2
        assert pair["incremental_peak_allocated_bytes"] > 0
    assert result["cache_reserved_bytes"] == 2 * 2 * 2 * 8 * 2 * 4


def test_plot_cache_curve_and_axis_validation(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    from tinyserve.plots import plot_cache_curve

    model = StubModel()
    install_fake_generation(monkeypatch, model)
    monkeypatch.setattr(benchmark, "kv_cache_bytes_per_token", lambda config, size: 16)
    clock = iter(range(8))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    rows = [benchmark.run_case(model, torch.tensor([1]), new_tokens=2, trials=1),
            benchmark.run_case(model, torch.tensor([1, 7]), new_tokens=2, trials=1)]
    path = tmp_path / "curve.png"
    assert plot_cache_curve(rows, path) == path and path.stat().st_size > 0
    with pytest.raises(ValueError, match="axis"):
        plot_cache_curve(rows, path, x_key="batch")
    with pytest.raises(ValueError, match="fixed"):
        plot_cache_curve(rows, path, x_key="new_tokens")
