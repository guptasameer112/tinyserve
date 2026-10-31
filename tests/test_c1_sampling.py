"""C1: exact distributions, HF warper parity, seeded sampling and invalid inputs."""
import math

import pytest

torch = pytest.importorskip("torch")

from tinyserve.sampling import sample_token, sampling_probs  # noqa: E402


def example():
    return torch.tensor([0.6, 0.25, 0.1, 0.05]).log()


def test_plain_softmax_matches_hand_probabilities():
    torch.testing.assert_close(sampling_probs(example()), torch.tensor([0.6, 0.25, 0.1, 0.05]),
                               rtol=0, atol=1e-6)


def test_temperature_squares_probabilities_at_half_temperature():
    raw = torch.tensor([0.6, 0.25, 0.1, 0.05])
    expected = raw.square() / raw.square().sum()
    torch.testing.assert_close(sampling_probs(raw.log(), temperature=0.5), expected, rtol=0, atol=1e-6)


def test_higher_temperature_flattens_distribution():
    scores = example()
    cold, warm, hot = (sampling_probs(scores, temperature=value) for value in (0.5, 1.0, 2.0))
    assert cold[0] > warm[0] > hot[0]
    assert cold[-1] < warm[-1] < hot[-1]


def test_greedy_is_one_hot_lowest_index_on_tie_and_does_not_consume_rng():
    scores = torch.tensor([1.0, 3.0, 3.0])
    generator = torch.Generator().manual_seed(1)
    before = generator.get_state().clone()
    assert sample_token(scores, generator=generator) == 1
    assert torch.equal(sampling_probs(scores, temperature=0), torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(generator.get_state(), before)


def test_top_k_two():
    expected = torch.tensor([0.6 / 0.85, 0.25 / 0.85, 0.0, 0.0])
    torch.testing.assert_close(sampling_probs(example(), top_k=2), expected, rtol=0, atol=1e-6)


def test_top_k_keeps_threshold_ties():
    probabilities = sampling_probs(torch.tensor([3.0, 2.0, 2.0, 1.0]), top_k=2)
    assert torch.count_nonzero(probabilities).item() == 3
    assert probabilities[-1] == 0


def test_top_k_larger_than_vocab_is_unfiltered():
    torch.testing.assert_close(sampling_probs(example(), top_k=100), sampling_probs(example()), rtol=0, atol=0)


def test_top_p_keeps_crossing_token_and_normalizes():
    expected = torch.tensor([0.6 / 0.85, 0.25 / 0.85, 0.0, 0.0])
    torch.testing.assert_close(sampling_probs(example(), top_p=0.8), expected, rtol=0, atol=1e-6)


def test_top_p_tiny_keeps_best_token():
    assert torch.equal(sampling_probs(example(), top_p=0.01), torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_top_p_exact_boundary_uses_hf_ascending_tail_convention():
    # Equal scores -> exactly 0.25 each, avoiding an approximate decimal boundary.
    probabilities = sampling_probs(torch.zeros(4), top_p=0.5)
    assert torch.count_nonzero(probabilities).item() == 2
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0), rtol=0, atol=0)


@pytest.mark.parametrize("temperature,top_k,top_p", [
    (1.0, 0, 1.0), (0.5, 0, 1.0), (2.0, 2, 1.0),
    (1.0, 0, 0.8), (0.7, 3, 0.8), (1.0, 100, 0.01),
])
def test_probabilities_match_hf_warpers(temperature, top_k, top_p):
    pytest.importorskip("transformers")
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
    )

    scores = example()[None]
    ids = torch.tensor([[1]])
    if temperature != 1:
        scores = TemperatureLogitsWarper(temperature)(ids, scores)
    if top_k:
        scores = TopKLogitsWarper(top_k)(ids, scores)
    if top_p != 1:
        scores = TopPLogitsWarper(top_p)(ids, scores)
    torch.testing.assert_close(sampling_probs(example(), temperature, top_k, top_p),
                               scores.softmax(-1)[0], rtol=0, atol=1e-6)


def test_filters_do_not_modify_input():
    scores = example()
    original = scores.clone()
    sampling_probs(scores, temperature=0.7, top_k=3, top_p=0.8)
    assert torch.equal(scores, original)


def test_negative_infinity_is_a_valid_pre_masked_token():
    probabilities = sampling_probs(torch.tensor([0.0, -torch.inf, 0.0]))
    assert torch.equal(probabilities, torch.tensor([0.5, 0.0, 0.5]))


def test_float64_matches_explicit_formula():
    scores = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
    values = (scores / 0.7).exp()
    expected = values / values.sum()
    torch.testing.assert_close(sampling_probs(scores, temperature=0.7), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_scores_upcast_before_softmax(dtype):
    scores = example().to(dtype)
    output = sampling_probs(scores, temperature=0.7, top_k=3, top_p=0.8)
    reference = sampling_probs(scores.float(), temperature=0.7, top_k=3, top_p=0.8)
    assert output.dtype == torch.float32
    assert torch.equal(output, reference)


def test_seeded_sampling_is_reproducible_and_filtered_tokens_never_appear():
    scores = example()
    first = torch.Generator().manual_seed(17)
    second = torch.Generator().manual_seed(17)
    choices = [sample_token(scores, temperature=1, top_k=2, generator=first) for _ in range(40)]
    repeated = [sample_token(scores, temperature=1, top_k=2, generator=second) for _ in range(40)]
    assert choices == repeated
    assert set(choices) == {0, 1}


def test_empirical_multinomial_frequencies_match_probabilities():
    probabilities = sampling_probs(example())
    draws = torch.multinomial(probabilities, 20000, replacement=True,
                             generator=torch.Generator().manual_seed(17))
    frequencies = torch.bincount(draws, minlength=4).float() / draws.numel()
    # Max standard error is ~0.0035 at n=20000; 0.015 is over four standard errors.
    torch.testing.assert_close(frequencies, probabilities, rtol=0, atol=0.015)


@pytest.mark.parametrize("options", [
    {"temperature": -1}, {"temperature": math.inf}, {"temperature": math.nan}, {"temperature": True},
    {"top_k": -1}, {"top_k": 2.5}, {"top_k": True},
    {"top_p": 0}, {"top_p": 1.1}, {"top_p": math.nan}, {"top_p": True},
])
def test_invalid_sampling_options_raise(options):
    with pytest.raises(ValueError):
        sampling_probs(example(), **options)


@pytest.mark.parametrize("scores", [
    torch.empty(0), torch.ones(2, 3), torch.tensor([1, 2]),
    torch.tensor([0.0, torch.nan]), torch.tensor([0.0, torch.inf]),
    torch.tensor([-torch.inf, -torch.inf]),
])
def test_invalid_logits_raise(scores):
    with pytest.raises(ValueError):
        sample_token(scores)


def test_temperature_scaling_overflow_is_reported():
    with pytest.raises(ValueError, match="overflowed"):
        sampling_probs(torch.tensor([3e38, 1e38]), temperature=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_sampling_uses_device_generator():
    scores = example().cuda().half()
    probabilities = sampling_probs(scores)
    assert probabilities.device == scores.device and probabilities.dtype == torch.float32
    first = torch.Generator(device=scores.device).manual_seed(3)
    second = torch.Generator(device=scores.device).manual_seed(3)
    assert [sample_token(scores, temperature=1, generator=first) for _ in range(10)] == [
        sample_token(scores, temperature=1, generator=second) for _ in range(10)
    ]
