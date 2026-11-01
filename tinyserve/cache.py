"""C3: append-only, single-request contiguous KV storage."""
from numbers import Integral

import torch

from tinyserve.weights import head_dim, kv_cache_bytes_per_token


class KVCache:
    """K/V (layers, capacity, KV heads, head_dim), bound to one model.

    Keys are stored after RoPE; values are not rotated. Only [:length] is
    committed. Each layer writes the same suffix; the model advances length
    once its entire forward succeeds. A failed forward can retry that suffix.
    """
    def __init__(self, model, capacity):
        if (isinstance(capacity, bool) or not isinstance(capacity, Integral)
                or not 0 < capacity <= model.config.max_position_embeddings):
            raise ValueError("cache capacity must be a positive integer within the model context limit")
        self.model = model
        self.capacity = int(capacity)
        self.length = 0
        weight = model.get_input_embeddings().weight
        config = model.config
        shape = (config.num_hidden_layers, self.capacity, config.num_key_value_heads, head_dim(config))
        self.key = weight.new_empty(shape)
        self.value = weight.new_empty(shape)

    @property
    def bytes_per_token(self):
        return kv_cache_bytes_per_token(self.model.config, self.key.element_size())

    @property
    def allocated_bytes(self):
        return 2 * self.key.numel() * self.key.element_size()

    def reset(self):
        """Forget the request without reallocating or exposing old storage."""
        self.length = 0

    def validate_request(self, model, positions, sequence_ids):
        if model is not self.model:
            raise ValueError("cache belongs to a different model")
        weight = model.get_input_embeddings().weight
        if self.key.device != weight.device or self.key.dtype != weight.dtype:
            raise ValueError("cache device/dtype must match the model; recreate it after moving or casting")
        if model.training or torch.is_grad_enabled():
            raise ValueError("cached forward requires eval mode and disabled gradients")
        if sequence_ids is not None:
            raise ValueError("contiguous cache supports one request, not packed sequence_ids")
        if self.length + positions.numel() > self.capacity:
            raise ValueError("cache capacity exceeded")
        expected = torch.arange(self.length, self.length + positions.numel(),
                                device=positions.device, dtype=positions.dtype)
        if not torch.equal(positions, expected):
            raise ValueError("cached positions must append consecutively from cache.length")

    def write(self, layer_index, key, value):
        """Write an uncommitted suffix and return the visible K/V prefix."""
        if (isinstance(layer_index, bool) or not isinstance(layer_index, Integral)
                or not 0 <= layer_index < self.key.shape[0]):
            raise ValueError("cache layer index is out of range")
        if (key.ndim != 3 or key.shape != value.shape or key.shape[0] == 0
                or key.shape[1:] != self.key.shape[2:]):
            raise ValueError("cached K/V must have shape (new_tokens, KV heads, head_dim)")
        if (key.device != self.key.device or value.device != self.key.device
                or key.dtype != self.key.dtype or value.dtype != self.key.dtype):
            raise ValueError("cached K/V device/dtype must match cache storage")
        if torch.is_grad_enabled():
            raise ValueError("cache writes require disabled gradients")
        end = self.length + key.shape[0]
        if end > self.capacity:
            raise ValueError("cache capacity exceeded")
        self.key[layer_index, self.length:end].copy_(key)
        self.value[layer_index, self.length:end].copy_(value)
        return self.key[layer_index, :end], self.value[layer_index, :end]
