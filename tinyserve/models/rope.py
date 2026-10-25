"""Lesson B3: rotary position embeddings (RoPE), Hugging Face / Llama convention.

head_dim D is split into two halves; dimension i is paired with dimension i + D/2. Pair i is rotated by
the angle  position * inv_freq[i],  inv_freq[i] = 1 / theta^(2i / D). Position 0 is the identity.

    x_rot = x * cos + rotate_half(x) * sin,     rotate_half([x1, x2]) = [-x2, x1]

Applied to queries and keys (not values). A dot product of two rotated vectors depends only on the
DIFFERENCE of their positions, which is how attention sees relative distance.

Positions are an explicit (N,) tensor, not arange(N): in a flattened batch each sequence restarts at 0.
cos/sin are computed in FP32 and cast to the input dtype before use, the same as Hugging Face.
"""
import torch


def get_rope_theta(config):
    """Base of the frequency schedule; transformers 5 keeps it in config.rope_parameters."""
    params = getattr(config, "rope_parameters", None)
    if params and "rope_theta" in params:
        return params["rope_theta"]
    return config.rope_theta


def rope_inv_freq(head_dim, theta, device):
    """(head_dim/2,) FP32 rotation speed in radians per position for each pair."""
    exponents = torch.arange(0, head_dim, 2, device=device, dtype=torch.int64).float() / head_dim
    return 1.0 / (theta ** exponents)


def rope_cos_sin(positions, head_dim, theta, dtype):
    """positions (N,) integers -> cos, sin of shape (N, head_dim) in `dtype` (angles computed in FP32)."""
    inv_freq = rope_inv_freq(head_dim, theta, positions.device)
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x, cos, sin):
    """x (N, heads, head_dim); cos, sin (N, head_dim). Works for queries (32 heads) and keys (4 heads)."""
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]
