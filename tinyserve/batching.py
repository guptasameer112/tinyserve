"""D1: pack independent, variable-length prompts without padding."""
from numbers import Integral

import torch

from tinyserve.models.attention import AttentionMetadata


def pack_prompts(prompts):
    """Return (flat_ids, metadata, lengths) in request order, without modifying inputs.

    Positions restart at zero per request. Sequence IDs are batch-local indices,
    not positions or persistent scheduler IDs. This layout has no KV cache.
    The model remains responsible for validating IDs against its vocabulary.
    """
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError("prompts must be a nonempty list/tuple of token tensors")
    for prompt in prompts:
        if (not isinstance(prompt, torch.Tensor) or prompt.ndim != 1 or prompt.numel() == 0
                or prompt.dtype not in (torch.int32, torch.int64)):
            raise ValueError("each prompt must be a nonempty integer (tokens,) tensor")
        if prompt.device != prompts[0].device or prompt.dtype != prompts[0].dtype:
            raise ValueError("all prompts must have the same device and dtype")
        if bool((prompt < 0).any()):
            raise ValueError("prompt token IDs must be nonnegative")
    lengths = tuple(prompt.numel() for prompt in prompts)
    positions = torch.cat([
        torch.arange(length, device=prompt.device, dtype=prompt.dtype)
        for prompt, length in zip(prompts, lengths)
    ])
    sequence_ids = torch.cat([
        prompt.new_full((length,), index)
        for index, (prompt, length) in enumerate(zip(prompts, lengths))
    ])
    return torch.cat(prompts), AttentionMetadata(positions, sequence_ids), lengths


def split_packed(tensor, lengths):
    """Split a token-major tensor back into request-order views; no padding rows."""
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
        raise ValueError("packed tensor must have a token dimension")
    if (not isinstance(lengths, (list, tuple)) or not lengths
            or any(isinstance(length, bool) or not isinstance(length, Integral) or length <= 0
                   for length in lengths)):
        raise ValueError("lengths must be a nonempty list/tuple of positive integers")
    if sum(lengths) != tensor.shape[0]:
        raise ValueError("lengths must sum to the packed token count")
    return torch.split(tensor, tuple(int(length) for length in lengths), dim=0)
