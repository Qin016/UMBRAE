"""Lightweight residual pre-projector adapter for P9."""

import torch
from torch import Tensor, nn


class PreProjectorInterfaceAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 1024, bottleneck_dim: int = 256, gate_logit: float = -2.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_logit)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @property
    def gate(self) -> Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, source: Tensor, return_delta: bool = False):
        if source.ndim != 3 or source.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError(f"Expected [B,T,{self.norm.normalized_shape[0]}], got {tuple(source.shape)}")
        delta = self.up(self.activation(self.down(self.norm(source))))
        adapted = source + self.gate * delta
        return {"z_hat": adapted, "delta": delta, "gate": self.gate} if return_delta else adapted


def adapter_parameter_counts(adapter: PreProjectorInterfaceAdapter):
    gate = adapter.gate_logit.numel()
    total = sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
    return {"adapter_without_gate": total - gate, "gate": gate, "total_trainable": total}
