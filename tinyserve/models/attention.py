"""B4: readable scaled dot-product attention on flattened tokens.

Q, K, V use (tokens, heads, head_dim). This lesson uses equal head counts;
B5 will add GQA. An explicit boolean mask can isolate packed sequences or
describe cached attention without changing the attention math.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AttentionMetadata:
    """No-cache token layout: local positions and optional packed request IDs."""
    positions: torch.Tensor
    sequence_ids: Optional[torch.Tensor] = None

    def allowed_mask(self):
        return causal_mask(self.positions, self.sequence_ids)


def causal_mask(positions, sequence_ids=None):
    """Return (N, N) allowed pairs for prefill. True means a key is visible.

    positions (N,) restart per sequence; sequence_ids (N,) identify requests.
    Without sequence_ids, all tokens belong to one sequence.
    """
    if positions.ndim != 1 or positions.numel() == 0:
        raise ValueError("positions must be a nonempty (N,) tensor")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must contain integer token positions")
    allowed = positions[None, :] <= positions[:, None]
    if sequence_ids is not None:
        if sequence_ids.shape != positions.shape or sequence_ids.device != positions.device:
            raise ValueError("sequence_ids must match positions in shape and device")
        if sequence_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("sequence_ids must contain integer request IDs")
        allowed = allowed & (sequence_ids[:, None] == sequence_ids[None, :])
    return allowed


def reference_attention(query, key, value, allowed_mask):
    """Return output (Nq, H, D) and weights (H, Nq, Nk).

    query (Nq, H, D), key/value (Nk, H, D), allowed_mask (Nq, Nk).
    True means allowed, unlike torch.nn.MultiheadAttention's boolean mask.
    FP16/BF16 math runs in FP32 for a stable reference; output retains the
    input dtype and attention weights retain the computation dtype.
    """
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key and value must have shape (tokens, heads, head_dim)")
    if any(size == 0 for tensor in (query, key, value) for size in tensor.shape):
        raise ValueError("query, key and value dimensions must be nonempty")
    if key.shape != value.shape or query.shape[1:] != key.shape[1:]:
        raise ValueError("key/value shapes and all head counts and head dimensions must match")
    if not query.is_floating_point() or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query, key and value must have the same floating-point dtype")
    if key.device != query.device or value.device != query.device:
        raise ValueError("query, key and value must be on the same device")
    if allowed_mask.shape != (query.shape[0], key.shape[0]) or allowed_mask.dtype != torch.bool:
        raise ValueError("allowed_mask must be a boolean (query_tokens, key_tokens) tensor")
    if allowed_mask.device != query.device:
        raise ValueError("allowed_mask must be on the query device")
    if not bool(allowed_mask.any(dim=-1).all()):
        raise ValueError("every query must have at least one visible key")

    compute_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    query_heads = query.transpose(0, 1).to(compute_dtype)
    key_heads = key.transpose(0, 1).to(compute_dtype)
    value_heads = value.transpose(0, 1).to(compute_dtype)
    scores = (query_heads @ key_heads.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    weights = torch.softmax(scores.masked_fill(~allowed_mask[None, :, :], -torch.inf), dim=-1)
    output = (weights @ value_heads).transpose(0, 1).contiguous().to(query.dtype)
    return output, weights
