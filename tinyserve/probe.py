"""Helpers to compare our modules with the Hugging Face model on real activations."""
import torch

from tinyserve.models.norm import RMSNorm
from tinyserve.models.rope import apply_rope, get_rope_theta, rope_cos_sin
from tinyserve.models.attention import causal_mask
from tinyserve.models.gqa import grouped_query_attention
from tinyserve.weights import head_dim


@torch.no_grad()
def capture_module_io(model, input_ids, module_type):
    """Run one forward pass on input_ids (N,) and record (name, input, output) of every module of module_type."""
    records, handles = [], []
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            def hook(_module, args, output, name=name):
                records.append((name, args[0].detach(), output.detach()))

            handles.append(module.register_forward_hook(hook))
    try:
        model(input_ids[None, :])
    finally:
        for handle in handles:
            handle.remove()
    return records


@torch.no_grad()
def compare_rmsnorm(model, input_ids, module_type, eps):
    """For every RMSNorm in the model: feed the real captured input to OUR RMSNorm (with the same weight)
    and report the max |difference| from the Hugging Face output, plus the input's size."""
    modules = dict(model.named_modules())
    rows = []
    for name, norm_input, reference_output in capture_module_io(model, input_ids, module_type):
        ours = RMSNorm(norm_input.shape[-1], eps, device=norm_input.device, dtype=modules[name].weight.dtype)
        ours.weight.copy_(modules[name].weight)
        our_output = ours(norm_input)
        rows.append({
            "name": name,
            "input_rms": norm_input.float().pow(2).mean().sqrt().item(),
            "input_max_abs": norm_input.float().abs().max().item(),
            "max_abs_diff": (our_output.float() - reference_output.float()).abs().max().item(),
            "exactly_equal": bool(torch.equal(our_output, reference_output)),
        })
    return rows


@torch.no_grad()
def compare_rope(model, q, k, positions):
    """Our RoPE vs the Hugging Face rotary embedding on the same tensors.
    q (N, Hq, D), k (N, Hkv, D), positions (N,). Returns max |difference| and exact-equality flags."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    q_hf, k_hf = q.transpose(0, 1)[None], k.transpose(0, 1)[None]  # Hugging Face layout (1, heads, N, D)
    cos_ref, sin_ref = model.model.rotary_emb(q_hf, positions[None])
    q_ref, k_ref = apply_rotary_pos_emb(q_hf, k_hf, cos_ref, sin_ref)
    q_ref, k_ref = q_ref[0].transpose(0, 1), k_ref[0].transpose(0, 1)

    cos, sin = rope_cos_sin(positions, q.shape[-1], get_rope_theta(model.config), q.dtype)
    q_out, k_out = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    return {
        "cos_sin_equal": bool(torch.equal(cos, cos_ref[0]) and torch.equal(sin, sin_ref[0])),
        "q_equal": bool(torch.equal(q_out, q_ref)),
        "k_equal": bool(torch.equal(k_out, k_ref)),
        "q_max_abs_diff": (q_out.float() - q_ref.float()).abs().max().item(),
        "k_max_abs_diff": (k_out.float() - k_ref.float()).abs().max().item(),
    }


@torch.no_grad()
def relative_score_matrix(q_vector, k_vector, num_positions, theta):
    """Put the SAME query vector at every position and the SAME key vector at every position, rotate,
    and return S[m, n] = <R_m q, R_n k> with shape (num_positions, num_positions)."""
    head_dim = q_vector.shape[-1]
    positions = torch.arange(num_positions, device=q_vector.device)
    cos, sin = rope_cos_sin(positions, head_dim, theta, q_vector.dtype)
    q_rot = apply_rope(q_vector[None, None, :].expand(num_positions, 1, head_dim), cos, sin)[:, 0, :]
    k_rot = apply_rope(k_vector[None, None, :].expand(num_positions, 1, head_dim), cos, sin)[:, 0, :]
    return q_rot @ k_rot.T


def diagonal_spread(matrix):
    """Largest (max - min) over every diagonal. Zero means the matrix depends only on the offset n - m."""
    size = matrix.shape[0]
    return max(
        (matrix.diagonal(offset).max() - matrix.diagonal(offset).min()).item() for offset in range(-size + 1, size)
    )


@torch.no_grad()
def compare_attention_layers(model, input_ids, layers=(0, 10, 21)):
    """Compare our B4/B5 attention with real HF inputs to o_proj in FP32 eager mode."""
    if model.training or next(model.parameters()).dtype != torch.float32:
        raise ValueError("attention parity requires an FP32 model in eval mode")
    if model.config._attn_implementation != "eager":
        raise ValueError("attention parity requires Hugging Face eager attention")
    config = model.config
    tokens = input_ids.numel()
    dimension = head_dim(config)
    positions = torch.arange(tokens, device=input_ids.device)
    mask = causal_mask(positions)
    cos, sin = rope_cos_sin(positions, dimension, get_rope_theta(config), torch.float32)
    records = {name: (module_input, output) for name, module_input, output
               in capture_module_io(model, input_ids, torch.nn.Linear)}
    rows = []
    for layer in layers:
        prefix = "model.layers.%d.self_attn." % layer
        query = records[prefix + "q_proj"][1][0].reshape(tokens, config.num_attention_heads, dimension)
        key = records[prefix + "k_proj"][1][0].reshape(tokens, config.num_key_value_heads, dimension)
        value = records[prefix + "v_proj"][1][0].reshape(tokens, config.num_key_value_heads, dimension)
        query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
        output, probabilities = grouped_query_attention(query, key, value, mask)
        output = output.reshape(tokens, config.hidden_size)
        reference = records[prefix + "o_proj"][0][0]
        torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)
        rows.append({
            "layer": layer,
            "max_abs_diff": (output - reference).abs().max().item(),
            "row_sum_max_error": (probabilities.sum(dim=-1) - 1).abs().max().item(),
            "future_weight_max": probabilities.masked_select(~mask[None]).max().item() if tokens > 1 else 0.0,
            "shape": list(output.shape),
        })
    return rows
