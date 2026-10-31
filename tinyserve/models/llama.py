"""B7/B8: a no-cache Llama model on flattened tokens."""
import torch
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


class LlamaModel(nn.Module):
    """Embedding -> decoder stack -> final RMSNorm, with HF-compatible keys."""
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        if config.num_hidden_layers <= 0:
            raise ValueError("the model must contain at least one decoder layer")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                        padding_idx=config.pad_token_id, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            DecoderLayer(config, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, input_ids, positions, attention_metadata):
        if input_ids.ndim != 1 or input_ids.numel() == 0 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be a nonempty integer (N,) tensor")
        if input_ids.device != self.embed_tokens.weight.device:
            raise ValueError("input_ids must be on the model device")
        if bool((input_ids < 0).any()) or bool((input_ids >= self.embed_tokens.num_embeddings).any()):
            raise ValueError("input_ids must be within the model vocabulary")
        if positions.shape != input_ids.shape or positions.device != input_ids.device:
            raise ValueError("positions must match input_ids in shape and device")
        if (positions.dtype not in (torch.int32, torch.int64) or bool((positions < 0).any())):
            raise ValueError("positions must contain nonnegative integers")
        if (attention_metadata.positions.device != positions.device
                or attention_metadata.positions.dtype != positions.dtype
                or not torch.equal(attention_metadata.positions, positions)):
            raise ValueError("explicit positions and attention_metadata.positions must agree")
        hidden_states = self.embed_tokens(input_ids)
        for decoder in self.layers:
            hidden_states = decoder(hidden_states, attention_metadata)
        return self.norm(hidden_states)


class LlamaForCausalLM(nn.Module):
    """B8: full logits (N, vocab), no HF forward calls and no KV cache."""
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        if config.model_type != "llama":
            raise ValueError("only Llama checkpoints are supported")
        rope_params = (getattr(config, "rope_parameters", None)
                       or getattr(config, "rope_scaling", None) or {})
        rope_type = rope_params.get("rope_type", rope_params.get("type", "default"))
        if rope_type != "default":
            raise ValueError("only default, unscaled RoPE is supported")
        self.config = config
        self.model = LlamaModel(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_huggingface(cls, reference):
        """Copy every trained weight strictly; preserve source device/dtype and return eval mode."""
        parameter = next(reference.parameters())
        ours = cls(reference.config, device=parameter.device, dtype=parameter.dtype)
        ours.load_state_dict(reference.state_dict(), strict=True)
        return ours.eval()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, positions, attention_metadata):
        # Return FP32 scores; this does not undo rounding inside an FP16 model.
        return self.lm_head(self.model(input_ids, positions, attention_metadata)).float()
