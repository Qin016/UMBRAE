"""Single-block residual token-mixing interface adapter for P10."""

import math

import torch
from torch import Tensor, nn


class TokenMixingInterfaceAdapter(nn.Module):
    """Remix existing tokens without changing token count or hidden width."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        bottleneck_dim: int = 256,
        gate_logit: float = -2.944,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn_down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.ffn_up = nn.Linear(bottleneck_dim, hidden_dim)
        self.attn_gate_logit = nn.Parameter(torch.tensor(float(gate_logit)))
        self.ffn_gate_logit = nn.Parameter(torch.tensor(float(gate_logit)))

        # Exact identity at initialization. The output projections learn first;
        # Q/K/V and the FFN input projection receive gradients after that.
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.ffn_up.weight)
        nn.init.zeros_(self.ffn_up.bias)

    @property
    def attn_gate(self) -> Tensor:
        return torch.sigmoid(self.attn_gate_logit)

    @property
    def ffn_gate(self) -> Tensor:
        return torch.sigmoid(self.ffn_gate_logit)

    def forward(
        self,
        source: Tensor,
        return_details: bool = False,
        return_attention: bool = False,
    ):
        if source.ndim != 3 or source.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected [B,T,{self.hidden_dim}], got {tuple(source.shape)}")
        normalized = self.norm1(source)
        attn_delta, attention_weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x1 = source + self.attn_gate * attn_delta
        ffn_delta = self.ffn_up(self.activation(self.ffn_down(self.norm2(x1))))
        adapted = x1 + self.ffn_gate * ffn_delta
        if not return_details:
            return adapted
        return {
            "z_hat": adapted,
            "attn_delta": attn_delta,
            "ffn_delta": ffn_delta,
            "attention_weights": attention_weights,
            "attn_gate": self.attn_gate,
            "ffn_gate": self.ffn_gate,
        }


def attention_diagnostics(weights: Tensor) -> dict:
    """Summarize per-head [B,H,T,T] attention without retaining samples."""
    if weights is None or weights.ndim != 4 or weights.shape[-1] != weights.shape[-2]:
        raise ValueError("Expected per-head attention weights with shape [B,H,T,T]")
    probabilities = weights.float().clamp_min(1e-12)
    token_count = probabilities.shape[-1]
    query_entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    diagonal = probabilities.diagonal(dim1=-2, dim2=-1)
    received = probabilities.mean(dim=(0, 2))
    received = received / received.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    usage_entropy = -(received * received.clamp_min(1e-12).log()).sum(dim=-1)
    max_share, max_index = received.max(dim=-1)
    per_head = []
    for head in range(probabilities.shape[1]):
        per_head.append({
            "head": head,
            "mean_attention_entropy": float(query_entropy[:, head].mean()),
            "normalized_attention_entropy": float(query_entropy[:, head].mean() / math.log(token_count)),
            "mean_max_attention_weight": float(probabilities[:, head].max(dim=-1).values.mean()),
            "off_diagonal_attention_mass": float(1.0 - diagonal[:, head].mean()),
            "token_usage_entropy": float(usage_entropy[head]),
            "normalized_token_usage_entropy": float(usage_entropy[head] / math.log(token_count)),
            "most_used_token": int(max_index[head]),
            "most_used_token_share": float(max_share[head]),
        })
    return {
        "token_count": token_count,
        "mean_attention_entropy": float(query_entropy.mean()),
        "normalized_attention_entropy": float(query_entropy.mean() / math.log(token_count)),
        "mean_max_attention_weight": float(probabilities.max(dim=-1).values.mean()),
        "off_diagonal_attention_mass": float(1.0 - diagonal.mean()),
        "per_head": per_head,
    }


def adapter_parameter_counts(adapter: TokenMixingInterfaceAdapter) -> dict:
    gates = adapter.attn_gate_logit.numel() + adapter.ffn_gate_logit.numel()
    total = sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
    return {"block_without_gates": total - gates, "gates": gates, "total_trainable": total}
