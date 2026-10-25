"""Lesson A3 tests. Run on a tiny random Llama (CPU or GPU) with a stub tokenizer; no download needed."""
import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from bench.hf_baseline import make_prompt_ids, measure_ttft_ms, run_case, timed_generate  # noqa: E402


class StubTokenizer:
    eos_token_id = 2

    def __call__(self, text, return_tensors=None):
        return types.SimpleNamespace(input_ids=torch.arange(3, 1003)[None, :] % 100)


@pytest.fixture(scope="module")
def tiny_model():
    cfg = transformers.LlamaConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
    )
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return transformers.LlamaForCausalLM(cfg).eval().to(device)


def test_make_prompt_ids_shape_and_device():
    ids = make_prompt_ids(StubTokenizer(), 16, 3, "cpu")
    assert ids.shape == (3, 16)
    assert torch.equal(ids[0], ids[2])


def test_make_prompt_ids_rejects_too_long():
    with pytest.raises(ValueError):
        make_prompt_ids(StubTokenizer(), 5000, 1, "cpu")


def test_timed_generate_gives_exact_length(tiny_model):
    device = next(tiny_model.parameters()).device
    ids = make_prompt_ids(StubTokenizer(), 8, 2, device)
    seconds, out = timed_generate(tiny_model, ids, 6, pad_token_id=2)
    assert seconds > 0
    assert out.shape == (2, 14)


def test_run_case_fields(tiny_model):
    result = run_case(tiny_model, StubTokenizer(), batch=2, prompt_len=8, new_tokens=6, trials=2)
    assert result["batch"] == 2 and result["new_tokens"] == 6
    for key in ("ttft_ms", "step_ms", "decode_tokens_per_s", "end_to_end_tokens_per_s"):
        assert result[key]["n"] == 2 and result[key]["min"] <= result[key]["median"] <= result[key]["max"]
    assert result["decode_tokens_per_s"]["median"] > 0


def test_run_case_rejects_single_token(tiny_model):
    with pytest.raises(ValueError):
        run_case(tiny_model, StubTokenizer(), batch=1, prompt_len=8, new_tokens=1)


def test_measure_ttft_returns_one_value_per_trial(tiny_model):
    values = measure_ttft_ms(tiny_model, StubTokenizer(), batch=1, prompt_len=8, trials=3)
    assert len(values) == 3 and all(v > 0 for v in values)
