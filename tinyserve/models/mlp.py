"""B6: Llama's bias-free SwiGLU MLP, independently applied to each token."""
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """(..., hidden) -> (..., hidden), with checkpoint-compatible projection names.

    down_proj(silu(gate_proj(x)) * up_proj(x))
    Linear weights use PyTorch's (out_features, in_features) layout.
    """
    def __init__(self, hidden_size, intermediate_size, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
