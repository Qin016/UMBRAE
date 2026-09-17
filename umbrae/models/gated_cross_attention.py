"""Lightweight gated structural-to-semantic cross attention."""

from typing import Dict

import torch
from torch import Tensor, nn


class GatedCrossAttentionFusion(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.0,
        gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.query_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        z_sem: Tensor,
        h_struct: Tensor,
        return_attention: bool = True,
    ) -> Dict[str, Tensor]:
        if z_sem.ndim != 3 or h_struct.ndim != 3:
            raise ValueError("z_sem and h_struct must have shape [B,N,D]")
        if z_sem.shape[0] != h_struct.shape[0]:
            raise ValueError("Semantic and structural batch sizes must match")
        if z_sem.shape[-1] != self.dim or h_struct.shape[-1] != self.dim:
            raise ValueError(f"Both inputs must use D={self.dim}")
        delta_z, attention = self.cross_attention(
            query=self.query_norm(z_sem),
            key=self.kv_norm(h_struct),
            value=self.kv_norm(h_struct),
            need_weights=return_attention,
            average_attn_weights=False,
        )
        gate = torch.sigmoid(self.gate_logit)
        z_cal = z_sem + gate * delta_z
        return {
            "z_cal": z_cal,
            "delta_z": delta_z,
            "fusion_gate": gate,
            "fusion_attention": attention if return_attention else None,
        }
