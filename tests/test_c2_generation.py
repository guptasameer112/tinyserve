"""C2: append/recompute contract, EOS, boundaries, sampling and HF parity."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tinyserve.generation import generate  # noqa: E402
from tinyserve.sampling import sample_token  # noqa: E402


class ScriptedModel(torch.nn.Module):
    """Deterministic test model with visible inputs and predefined next tokens."""
    def __init__(self, plan=(3, 4, 2), eos=2, context=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.config = SimpleNamespace(eos_token_id=eos, max_position_embeddings=context)
        self.plan = plan
        self.calls = []
        self.grad_enabled = []
        self.eval()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, ids, positions, metadata):
        assert torch.equal(positions, torch.arange(ids.numel(), device=ids.device))
        assert torch.equal(metadata.positions, positions) and metadata.sequence_ids is None
        self.grad_enabled.append(torch.is_grad_enabled())
        token = self.plan[len(self.calls)]
        self.calls.append(ids.clone())
        scores = self.embedding.weight.new_full((ids.numel(), 8), -torch.inf)
        scores[-1, token] = 0
        return scores


def test_recomputes_growing_prefix_and_includes_eos():
    model = ScriptedModel()
    prompt = torch.tensor([1, 5])
    output = generate(model, prompt, max_new_tokens=6)
    assert output.tolist() == [1, 5, 3, 4, 2]
    assert [ids.tolist() for ids in model.calls] == [[1, 5], [1, 5, 3], [1, 5, 3, 4]]
    assert prompt.tolist() == [1, 5]
    assert not any(model.grad_enabled)


def test_new_token_limit_not_total_length():
    model = ScriptedModel(plan=(3, 4, 5), eos=None)
    assert generate(model, torch.tensor([1, 7]), max_new_tokens=2).tolist() == [1, 7, 3, 4]
    assert len(model.calls) == 2


def test_immediate_eos():
    model = ScriptedModel(plan=(2,))
    assert generate(model, torch.tensor([1]), max_new_tokens=4).tolist() == [1, 2]
    assert len(model.calls) == 1


def test_prompt_eos_does_not_stop_a_new_turn():
    model = ScriptedModel(plan=(3, 2))
    assert generate(model, torch.tensor([1, 2]), max_new_tokens=2).tolist() == [1, 2, 3, 2]


@pytest.mark.parametrize("eos", [[2, 4], (2, 4)])
def test_multiple_eos_tokens(eos):
    model = ScriptedModel(plan=(3, 4), eos=eos)
    assert generate(model, torch.tensor([1]), max_new_tokens=3).tolist() == [1, 3, 4]


def test_zero_new_tokens_returns_independent_copy_without_forward():
    model = ScriptedModel()
    prompt = torch.tensor([1, 5])
    output = generate(model, prompt, max_new_tokens=0)
    assert torch.equal(output, prompt) and output.data_ptr() != prompt.data_ptr()
    assert model.calls == []


def test_exact_context_budget_allowed():
    model = ScriptedModel(plan=(3, 4), eos=None, context=4)
    assert generate(model, torch.tensor([1, 5]), max_new_tokens=2).numel() == 4


def test_oversized_request_raises_before_forward_even_if_eos_would_be_early():
    model = ScriptedModel(plan=(2,), context=3)
    with pytest.raises(ValueError, match="context limit"):
        generate(model, torch.tensor([1, 5]), max_new_tokens=2)
    assert model.calls == []


class ConstantModel(ScriptedModel):
    def forward(self, ids, positions, metadata):
        self.calls.append(ids.clone())
        return self.embedding.weight.new_tensor([0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]).expand(ids.numel(), -1)


def test_seeded_sampling_matches_manual_sampler_sequence():
    model = ConstantModel(eos=None)
    prompt = torch.tensor([1])
    actual = generate(model, prompt, max_new_tokens=8, temperature=0.7, top_k=3, top_p=0.9,
                      generator=torch.Generator().manual_seed(17))
    manual_generator = torch.Generator().manual_seed(17)
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    expected = [sample_token(scores, 0.7, 3, 0.9, manual_generator) for _ in range(8)]
    assert actual.tolist() == [1] + expected
    assert [ids.numel() for ids in model.calls] == list(range(1, 9))


@pytest.mark.parametrize("value", [-1, True, 2.5])
def test_invalid_token_budget_raises(value):
    with pytest.raises(ValueError):
        generate(ScriptedModel(), torch.tensor([1]), max_new_tokens=value)


@pytest.mark.parametrize("options", [{"temperature": -1}, {"top_k": -1}, {"top_p": 0}])
def test_invalid_sampling_options_raise_even_for_zero_tokens(options):
    with pytest.raises(ValueError):
        generate(ScriptedModel(), torch.tensor([1]), max_new_tokens=0, **options)


@pytest.mark.parametrize("prompt", [
    torch.empty(0, dtype=torch.int64), torch.tensor([[1, 2]]),
    torch.tensor([1.0, 2.0]), torch.tensor([-1]), torch.tensor([8]),
])
def test_invalid_prompts_raise(prompt):
    with pytest.raises(ValueError):
        generate(ScriptedModel(), prompt, max_new_tokens=0)


@pytest.mark.parametrize("eos", [True, "2", -1, [8], [2.5]])
def test_invalid_eos_configuration_raises(eos):
    with pytest.raises(ValueError, match="EOS|eos_token_id"):
        generate(ScriptedModel(eos=eos), torch.tensor([1]))


def test_training_mode_raises():
    model = ScriptedModel().train()
    with pytest.raises(ValueError, match="eval"):
        generate(model, torch.tensor([1]))


def test_wrong_logits_shape_raises():
    class WrongShape(ScriptedModel):
        def forward(self, ids, positions, metadata):
            return self.embedding.weight.new_zeros(8)

    with pytest.raises(ValueError, match="logits"):
        generate(WrongShape(), torch.tensor([1]))


@pytest.mark.parametrize("stop_on_first", [False, True])
def test_greedy_matches_hf_generate(stop_on_first):
    transformers = pytest.importorskip("transformers")
    from tinyserve.models.attention import AttentionMetadata
    from tinyserve.models.llama import LlamaForCausalLM

    torch.manual_seed(2)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
        max_position_embeddings=32, eos_token_id=None, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    reference = transformers.LlamaForCausalLM(config).float().eval()
    ours = LlamaForCausalLM.from_huggingface(reference)
    prompt = torch.tensor([1, 7, 9])
    if stop_on_first:
        positions = torch.arange(3)
        eos = ours(prompt, positions, AttentionMetadata(positions))[-1].argmax().item()
        reference.config.eos_token_id = eos
        reference.generation_config.eos_token_id = eos
    actual = generate(ours, prompt, max_new_tokens=6)
    expected = reference.generate(prompt[None], attention_mask=torch.ones_like(prompt)[None],
                                  do_sample=False, max_new_tokens=6, use_cache=False)[0]
    assert torch.equal(actual, expected)
    assert actual.numel() == (4 if stop_on_first else 9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_device_dtype_and_generator_validation():
    model = ScriptedModel(plan=(3, 2)).cuda()
    prompt = torch.tensor([1], device="cuda", dtype=torch.int32)
    output = generate(model, prompt, max_new_tokens=2)
    assert output.dtype == prompt.dtype and output.device == prompt.device
    with pytest.raises(ValueError, match="generator"):
        generate(model, prompt, max_new_tokens=1, generator=torch.Generator())
