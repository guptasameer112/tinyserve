"""C1: greedy, temperature, top-k and nucleus (top-p) sampling for one logits row."""
import math
from numbers import Integral, Real

import torch


def validate_sampling_options(temperature, top_k, top_p):
    """Shared option checks, including generation requests that ask for zero tokens."""
    if (isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError("temperature must be a finite nonnegative number")
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if isinstance(top_p, bool) or not isinstance(top_p, Real) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")


def sampling_probs(logits, temperature=1.0, top_k=0, top_p=1.0):
    """(vocab,) logits -> probabilities, using FP32 for FP16/BF16 inputs.

    Order: temperature -> top-k -> top-p -> softmax. Temperature 0 is
    greedy (one-hot); top_k=0 and top_p=1 disable their filters.
    Top-k retains ties at the kth score, matching HF. Top-p removes the
    ascending low-probability tail, always keeping at least one token.
    """
    if logits.ndim != 1 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("logits must be a nonempty floating-point (vocab,) tensor")
    if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()) or not bool(torch.isfinite(logits).any()):
        raise ValueError("logits must have at least one finite score and no NaN or positive infinity")
    validate_sampling_options(temperature, top_k, top_p)

    scores = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits.clone()
    if temperature == 0:
        probabilities = torch.zeros_like(scores)
        probabilities[scores.argmax()] = 1
        return probabilities
    scores = scores / temperature
    if bool(torch.isposinf(scores).any()) or bool((torch.isfinite(logits) & ~torch.isfinite(scores)).any()):
        raise ValueError("temperature scaling overflowed; increase temperature or reduce logits magnitude")
    if top_k > 0:
        cutoff = torch.topk(scores, min(int(top_k), scores.numel())).values[-1]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=False)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        remove_sorted = cumulative <= (1 - top_p)
        remove_sorted[-1] = False
        remove = torch.zeros_like(remove_sorted).scatter(0, sorted_indices, remove_sorted)
        scores = scores.masked_fill(remove, -torch.inf)
    return torch.softmax(scores, dim=-1)


@torch.no_grad()
def sample_token(logits, temperature=0.0, top_k=0, top_p=1.0, generator=None):
    """Choose one Python token ID from a single logits row.

    Default is greedy. A generator must match the logits device. Greedy does
    not consume RNG state; seeds reproduce samples on the same device/software,
    not necessarily across CPU, GPU or library versions.
    """
    probabilities = sampling_probs(logits, temperature, top_k, top_p)
    if temperature == 0:
        return probabilities.argmax().item()
    return torch.multinomial(probabilities, 1, generator=generator).item()
