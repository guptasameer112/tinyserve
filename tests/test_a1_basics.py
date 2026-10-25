"""Lesson A1 tests. Logic tests use a tiny random Llama (no download, run on CPU).
The `slow` test needs the real TinyLlama weights and runs in the Colab notebook."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tinyserve.basics import (  # noqa: E402
    encode, forward_logits, generate_greedy_naive, next_token_greedy, prefix_logit_diff, shape_report,
)

VOCAB = 100


@pytest.fixture(scope="module")
def tiny_model():
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).eval()


def test_logits_shape_is_one_row_per_token(tiny_model):
    ids = torch.tensor([1, 5, 7, 9, 11])
    logits = forward_logits(tiny_model, ids)
    assert logits.shape == (5, VOCAB)


def test_greedy_uses_last_row(tiny_model):
    ids = torch.tensor([1, 5, 7])
    logits = forward_logits(tiny_model, ids)
    assert next_token_greedy(logits) == int(logits[-1].argmax())


def test_causality_later_tokens_do_not_change_earlier_logits(tiny_model):
    # FP32 on CPU, same kernels for the shared prefix: we expect equality up to rounding.
    short = torch.tensor([1, 5, 7, 9])
    longer = torch.tensor([1, 5, 7, 9, 20, 30])
    logits_short = forward_logits(tiny_model, short)
    logits_long = forward_logits(tiny_model, longer)
    torch.testing.assert_close(logits_long[:4], logits_short, rtol=0, atol=1e-5)


def test_prefix_logit_diff_is_tiny(tiny_model):
    diff = prefix_logit_diff(tiny_model, torch.tensor([1, 5, 7, 9]), torch.tensor([20, 30]))
    assert diff < 1e-5


def test_naive_generate_appends_greedy_tokens(tiny_model):
    ids = torch.tensor([1, 5, 7])
    out = generate_greedy_naive(tiny_model, ids, max_new_tokens=4)
    assert out.shape == (7,)
    assert torch.equal(out[:3], ids)
    assert out[3].item() == next_token_greedy(forward_logits(tiny_model, ids))


def test_changing_an_earlier_token_changes_later_logits(tiny_model):
    a = forward_logits(tiny_model, torch.tensor([1, 5, 7, 9]))
    b = forward_logits(tiny_model, torch.tensor([1, 6, 7, 9]))
    assert torch.equal(a[0], b[0])
    assert not torch.allclose(a[3], b[3])


@pytest.mark.slow
def test_real_tinyllama_config_and_shapes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    from tinyserve.basics import load_model

    tokenizer, model = load_model(device=device, dtype=dtype)
    cfg = model.config
    assert (cfg.num_hidden_layers, cfg.hidden_size) == (22, 2048)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads) == (32, 4)
    assert cfg.hidden_size // cfg.num_attention_heads == 64
    assert cfg.vocab_size == 32000

    ids = encode(tokenizer, "Hello world", device)
    assert ids[0].item() == tokenizer.bos_token_id
    assert tokenizer.decode(ids, skip_special_tokens=True) == "Hello world"

    report = shape_report(tokenizer, model, "Hello world")
    assert report["logits_shape"] == (len(ids), 32000)
