"""D2: explicit disjoint block tables and a readable PyTorch gather backend."""
from numbers import Integral
from types import MappingProxyType

import torch

from tinyserve.models.gqa import grouped_query_attention
from tinyserve.weights import head_dim


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return int(value)


class PagedKVCache:
    """Shared K/V pool (layers, physical_blocks, block_size, KV_heads, head_dim).

    Block tables are immutable and disjoint for D2; no allocator or sharing.
    Each request has its own committed length. Writes affect only append slots;
    all active lengths advance together after a successful complete model forward.
    """
    def __init__(self, model, num_blocks, block_size, block_tables):
        self.num_blocks = _positive_integer("num_blocks", num_blocks)
        self.block_size = _positive_integer("block_size", block_size)
        if not isinstance(block_tables, dict) or not block_tables:
            raise ValueError("block_tables must be a nonempty request-ID to block-list dictionary")
        tables = {}
        used = set()
        for request_id, blocks in block_tables.items():
            if isinstance(request_id, bool) or not isinstance(request_id, Integral) or request_id < 0:
                raise ValueError("request IDs must be nonnegative integers")
            if not isinstance(blocks, (list, tuple)) or not blocks:
                raise ValueError("each block table must be a nonempty list/tuple")
            for block in blocks:
                if (isinstance(block, bool) or not isinstance(block, Integral)
                        or not 0 <= block < self.num_blocks):
                    raise ValueError("physical block ID is out of range")
                if block in used:
                    raise ValueError("D2 block tables must be disjoint, without repeated blocks")
                used.add(block)
            tables[int(request_id)] = tuple(int(block) for block in blocks)
        self.model = model
        self.block_tables = MappingProxyType(tables)
        self._lengths = {request_id: 0 for request_id in tables}
        weight = model.get_input_embeddings().weight
        config = model.config
        shape = (config.num_hidden_layers, self.num_blocks, self.block_size,
                 config.num_key_value_heads, head_dim(config))
        self.key = weight.new_empty(shape)
        self.value = weight.new_empty(shape)

    @property
    def lengths(self):
        return MappingProxyType(self._lengths)

    @property
    def allocated_bytes(self):
        return 2 * self.key.numel() * self.key.element_size()

    def reset(self, request_id):
        if (isinstance(request_id, bool) or not isinstance(request_id, Integral)
                or request_id not in self.block_tables):
            raise ValueError("unknown request ID")
        self._lengths[request_id] = 0

    def _validate_layout(self, positions, sequence_ids):
        if (not isinstance(positions, torch.Tensor) or positions.ndim != 1 or positions.numel() == 0
                or positions.dtype not in (torch.int32, torch.int64)):
            raise ValueError("paged positions must be a nonempty integer (tokens,) tensor")
        if (not isinstance(sequence_ids, torch.Tensor) or sequence_ids.shape != positions.shape
                or sequence_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("paged sequence_ids must be an integer tensor matching positions")
        if positions.device != self.key.device or sequence_ids.device != positions.device:
            raise ValueError("paged positions/sequence_ids must be on the cache device")
        if bool((positions < 0).any()):
            raise ValueError("paged positions must be nonnegative")
        groups = {}
        for request_id in sequence_ids.unique().tolist():
            if request_id not in self.block_tables:
                raise ValueError("unknown request ID")
            indices = (sequence_ids == request_id).nonzero().flatten()
            if bool((positions[indices] >= len(self.block_tables[request_id]) * self.block_size).any()):
                raise ValueError("request block-table capacity exceeded")
            groups[request_id] = indices
        return groups

    def _request_slots(self, request_id, positions):
        table = positions.new_tensor(self.block_tables[request_id])
        physical_blocks = table[torch.div(positions, self.block_size, rounding_mode="floor").long()]
        return physical_blocks * self.block_size + positions.remainder(self.block_size)

    def slot_mapping(self, positions, sequence_ids):
        """Physical slot = table[position // block_size] * block_size + position % block_size."""
        groups = self._validate_layout(positions, sequence_ids)
        slots = torch.empty_like(positions)
        for request_id, indices in groups.items():
            slots[indices] = self._request_slots(request_id, positions[indices])
        return slots

    def validate_request(self, model, positions, sequence_ids):
        if model is not self.model:
            raise ValueError("paged cache belongs to a different model")
        weight = model.get_input_embeddings().weight
        if self.key.device != weight.device or self.key.dtype != weight.dtype:
            raise ValueError("paged cache device/dtype must match the model; recreate after moving or casting")
        if model.training or torch.is_grad_enabled():
            raise ValueError("paged forward requires eval mode and disabled gradients")
        groups = self._validate_layout(positions, sequence_ids)
        for request_id, indices in groups.items():
            start = self._lengths[request_id]
            end = start + indices.numel()
            if end > model.config.max_position_embeddings:
                raise ValueError("request exceeds the model context limit")
            expected = torch.arange(start, end, device=positions.device, dtype=positions.dtype)
            if not torch.equal(positions[indices], expected):
                raise ValueError("paged positions must append consecutively from each request length")
        return groups

    def attend(self, layer_index, query, key, value, positions, sequence_ids):
        """Append post-RoPE K/V, gather each logical prefix, and scatter request outputs."""
        groups = self.validate_request(self.model, positions, sequence_ids)
        if (isinstance(layer_index, bool) or not isinstance(layer_index, Integral)
                or not 0 <= layer_index < self.key.shape[0]):
            raise ValueError("paged layer index is out of range")
        expected_shape = (positions.numel(),) + tuple(self.key.shape[-2:])
        if key.shape != expected_shape or value.shape != expected_shape:
            raise ValueError("paged K/V must have shape (new_tokens, KV_heads, head_dim)")
        if (key.device != self.key.device or value.device != self.key.device
                or key.dtype != self.key.dtype or value.dtype != self.key.dtype):
            raise ValueError("paged K/V device/dtype must match storage")
        if (query.ndim != 3 or query.shape[0] != positions.numel()
                or query.shape[-1] != self.key.shape[-1] or query.shape[1] <= 0
                or query.shape[1] % self.key.shape[-2] != 0
                or query.device != key.device or query.dtype != key.dtype):
            raise ValueError("paged query must match token count, GQA heads, head_dim, device and dtype")
        slots = self.slot_mapping(positions, sequence_ids).long()
        layer_key = self.key[layer_index].reshape(-1, *self.key.shape[-2:])
        layer_value = self.value[layer_index].reshape_as(layer_key)
        layer_key.index_copy_(0, slots, key)
        layer_value.index_copy_(0, slots, value)
        output = torch.empty_like(query)
        for request_id, indices in groups.items():
            end = self._lengths[request_id] + indices.numel()
            key_positions = torch.arange(end, device=positions.device, dtype=positions.dtype)
            prefix_slots = self._request_slots(request_id, key_positions).long()
            visible = key_positions[None, :] <= positions[indices, None]
            request_output, _ = grouped_query_attention(
                query[indices], layer_key[prefix_slots], layer_value[prefix_slots], visible,
            )
            output[indices] = request_output
        return output

    def commit(self, sequence_ids):
        """Called by the model only after all layers and logits have succeeded."""
        for request_id in sequence_ids.unique().tolist():
            self._lengths[request_id] += int((sequence_ids == request_id).sum().item())
