"""B5: grouped-query attention using B4's readable attention core."""
import torch

from tinyserve.models.attention import reference_attention


def expand_kv_heads(tensor, query_heads):
    """(N, Hkv, D) -> (N, Hq, D), repeating each KV head consecutively.

    Four KV heads serving 32 query heads map to [0]*8 + [1]*8 + [2]*8 + [3]*8.
    This materializes expanded tensors for the reference, not for cache storage.
    """
    if tensor.ndim != 3 or any(size == 0 for size in tensor.shape):
        raise ValueError("KV tensor must have nonempty shape (tokens, heads, head_dim)")
    if not isinstance(query_heads, int) or isinstance(query_heads, bool) or query_heads <= 0:
        raise ValueError("query_heads must be a positive integer")
    kv_heads = tensor.shape[1]
    if query_heads % kv_heads != 0:
        raise ValueError("query head count must be divisible by KV head count")
    repeats = query_heads // kv_heads
    if repeats == 1:
        return tensor
    return torch.repeat_interleave(tensor, repeats, dim=1)


def grouped_query_attention(query, key, value, allowed_mask):
    """Q (Nq, Hq, D), K/V (Nk, Hkv, D) -> B4 output and attention weights.

    Supports MHA (Hkv=Hq), GQA (1<Hkv<Hq) and MQA (Hkv=1).
    B4 validates dtype/device, head dimension, token counts and mask visibility.
    """
    if query.ndim != 3 or query.shape[1] == 0:
        raise ValueError("query must have shape (tokens, nonempty heads, head_dim)")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match before head expansion")
    return reference_attention(
        query, expand_kv_heads(key, query.shape[1]), expand_kv_heads(value, query.shape[1]), allowed_mask,
    )
