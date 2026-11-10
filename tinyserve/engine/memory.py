"""D5: explicit KV-only byte budgets; weights and workspace are separate."""
from dataclasses import dataclass

from tinyserve.engine.block_manager import _positive_integer
from tinyserve.weights import head_dim


@dataclass(frozen=True)
class KVMemoryBudget:
    kv_budget_bytes: int
    block_size: int
    bytes_per_token: int
    num_blocks: int

    @property
    def bytes_per_block(self):
        return self.block_size * self.bytes_per_token

    @property
    def allocated_bytes(self):
        return self.num_blocks * self.bytes_per_block

    @property
    def unused_bytes(self):
        return self.kv_budget_bytes - self.allocated_bytes


def plan_kv_memory(model, kv_budget_bytes, block_size=16) -> KVMemoryBudget:
    """Round down to whole blocks without allocating storage or claiming OOM safety."""
    kv_budget_bytes = _positive_integer("kv_budget_bytes", kv_budget_bytes)
    block_size = _positive_integer("block_size", block_size)
    config = model.config
    layers = _positive_integer("num_hidden_layers", config.num_hidden_layers)
    kv_heads = _positive_integer("num_key_value_heads", config.num_key_value_heads)
    dimension = _positive_integer("head_dim", head_dim(config))
    weight = model.get_input_embeddings().weight
    if not weight.is_floating_point():
        raise ValueError("KV storage requires floating-point model weights")
    bytes_per_token = 2 * layers * kv_heads * dimension * weight.element_size()
    num_blocks = kv_budget_bytes // (block_size * bytes_per_token)
    if num_blocks == 0:
        raise ValueError("KV budget cannot fit one whole block")
    return KVMemoryBudget(kv_budget_bytes, block_size, bytes_per_token, num_blocks)
