"""Lesson B2: RMSNorm.

    y = x / sqrt(mean(x^2) + eps) * weight        (mean over the hidden dimension)

Unlike LayerNorm it does not subtract the mean and has no bias: one reduction instead of two.
The statistic is computed in FP32 even for FP16 inputs: squaring an FP16 value above 255 overflows
(65504 is the FP16 maximum), and a plain FP16 mean of squares would silently turn the output into zeros.
The result is cast back to the input dtype *before* multiplying by the weight, the same order as
Hugging Face, so FP16 outputs match it exactly.
"""
import torch
from torch import nn


def rms_norm(x, weight, eps):
    """x (..., hidden), weight (hidden,). Returns the same shape and dtype as x."""
    stats = x.float()
    stats = stats * torch.rsqrt(stats.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * stats.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)
