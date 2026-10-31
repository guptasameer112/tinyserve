"""B7: one no-cache Llama decoder layer on flattened tokens."""
from torch import nn

from tinyserve.models.gqa import grouped_query_attention
from tinyserve.models.mlp import MLP
from tinyserve.models.norm import RMSNorm
from tinyserve.models.rope import apply_rope, get_rope_theta, rope_cos_sin
from tinyserve.weights import head_dim


class SelfAttention(nn.Module):
    """Q/K/V projections -> RoPE -> GQA attention -> output projection."""
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = head_dim(config)
        self.theta = get_rope_theta(config)
        hidden = config.hidden_size
        if (self.query_heads <= 0 or self.kv_heads <= 0 or self.head_dim <= 0
                or self.query_heads % self.kv_heads != 0 or self.head_dim % 2 != 0
                or self.query_heads * self.head_dim != hidden):
            raise ValueError("attention requires valid GQA heads, an even head_dim, and Hq * D == hidden_size")
        if getattr(config, "attention_bias", False):
            raise ValueError("this Llama attention supports bias-free projections only")
        self.q_proj = nn.Linear(hidden, self.query_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, hidden_states, positions, allowed_mask):
        tokens = hidden_states.shape[0]
        query = self.q_proj(hidden_states).reshape(tokens, self.query_heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape(tokens, self.kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).reshape(tokens, self.kv_heads, self.head_dim)
        cos, sin = rope_cos_sin(positions, self.head_dim, self.theta, query.dtype)
        query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
        output, _ = grouped_query_attention(query, key, value, allowed_mask)
        return self.o_proj(output.reshape(tokens, self.query_heads * self.head_dim))


class DecoderLayer(nn.Module):
    """Pre-norm decoder: x += attention(norm(x)); x += mlp(norm(x)).

    hidden_states (N, hidden), metadata AttentionMetadata -> (N, hidden).
    No cache, dropout or final model norm. State keys match an HF decoder layer.
    """
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        if config.hidden_act != "silu" or getattr(config, "mlp_bias", False):
            raise ValueError("this decoder requires a bias-free SiLU MLP")
        self.self_attn = SelfAttention(config, device=device, dtype=dtype)
        self.mlp = MLP(config.hidden_size, config.intermediate_size, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, hidden_states, metadata):
        hidden_size = self.input_layernorm.weight.numel()
        if hidden_states.ndim != 2 or hidden_states.shape[1] != hidden_size:
            raise ValueError("hidden_states must have shape (tokens, hidden_size)")
        if metadata.positions.shape != (hidden_states.shape[0],) or metadata.positions.device != hidden_states.device:
            raise ValueError("positions must match the hidden-state token count and device")
        mask = metadata.allowed_mask()
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), metadata.positions, mask,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
