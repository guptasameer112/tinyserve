"""Lesson B1: the weight tour. What tensors a Llama checkpoint holds, their shapes, and what they cost.

Pure Python (duck-typed on tensors), so the spec can be tested anywhere. `expected_shapes` is also
the target our own model must match in B8 when it loads the Hugging Face weights.
"""


def head_dim(config):
    return getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


def expected_shapes(config):
    """Every parameter name -> shape for a Llama-architecture model (the HF state_dict names)."""
    hidden, inter, dim = config.hidden_size, config.intermediate_size, head_dim(config)
    q_out = config.num_attention_heads * dim
    kv_out = config.num_key_value_heads * dim
    shapes = {"model.embed_tokens.weight": (config.vocab_size, hidden)}
    for layer in range(config.num_hidden_layers):
        prefix = "model.layers.%d." % layer
        shapes[prefix + "self_attn.q_proj.weight"] = (q_out, hidden)
        shapes[prefix + "self_attn.k_proj.weight"] = (kv_out, hidden)
        shapes[prefix + "self_attn.v_proj.weight"] = (kv_out, hidden)
        shapes[prefix + "self_attn.o_proj.weight"] = (hidden, q_out)
        shapes[prefix + "mlp.gate_proj.weight"] = (inter, hidden)
        shapes[prefix + "mlp.up_proj.weight"] = (inter, hidden)
        shapes[prefix + "mlp.down_proj.weight"] = (hidden, inter)
        shapes[prefix + "input_layernorm.weight"] = (hidden,)
        shapes[prefix + "post_attention_layernorm.weight"] = (hidden,)
    shapes["model.norm.weight"] = (hidden,)
    if not getattr(config, "tie_word_embeddings", False):
        shapes["lm_head.weight"] = (config.vocab_size, hidden)
    return shapes


def numel(shape):
    count = 1
    for size in shape:
        count *= size
    return count


def expected_param_count(config):
    return sum(numel(shape) for shape in expected_shapes(config).values())


def kv_cache_bytes_per_token(config, bytes_per_element):
    """K and V, for every layer, for one token: 2 x layers x kv_heads x head_dim x bytes."""
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim(config) * bytes_per_element


def compare_shapes(actual, expected):
    """actual/expected: name -> shape. Returns what is missing, unexpected or has the wrong shape."""
    actual = {name: tuple(shape) for name, shape in actual.items()}
    expected = {name: tuple(shape) for name, shape in expected.items()}
    return {
        "missing": sorted(set(expected) - set(actual)),
        "unexpected": sorted(set(actual) - set(expected)),
        "wrong_shape": sorted(
            (name, actual[name], expected[name])
            for name in set(actual) & set(expected) if actual[name] != expected[name]
        ),
    }


def category(name):
    """Group a parameter name into embedding / lm_head / attention / mlp / norm."""
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "self_attn" in name:
        return "attention"
    if "mlp" in name:
        return "mlp"
    return "norm"


def weight_tour(named_tensors):
    """named_tensors: iterable of (name, tensor). One row per tensor."""
    return [
        {
            "name": name,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "nbytes": tensor.numel() * tensor.element_size(),
        }
        for name, tensor in named_tensors
    ]


def bytes_by_category(rows):
    """Sum parameters and bytes per category, with each share of the total."""
    totals = {}
    for row in rows:
        entry = totals.setdefault(category(row["name"]), {"numel": 0, "nbytes": 0})
        entry["numel"] += row["numel"]
        entry["nbytes"] += row["nbytes"]
    all_bytes = sum(entry["nbytes"] for entry in totals.values())
    for entry in totals.values():
        entry["share"] = entry["nbytes"] / all_bytes
    return totals


def tensor_stats(tensor):
    """Mean, std, min, max of a weight, computed in FP32 so FP16 does not distort the statistics."""
    values = tensor.detach().float()
    return {
        "mean": values.mean().item(),
        "std": values.std().item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }
