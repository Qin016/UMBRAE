"""Fuse global L24 and routed ROI representations into MLLM prefix tokens."""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn


class NeuroRouteMLLMAdapter(nn.Module):
    """Create fMRI-derived visual prefix tokens for an MLLM."""

    SUPPORTED_MODES = (
        "l24_only",
        "routed_only",
        "concat",
        "gated_fusion",
        "cross_attention",
    )

    def __init__(
        self,
        input_dim: int,
        mllm_dim: int,
        fusion_mode: str = "concat",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_attention_heads: int = 8,
    ) -> None:
        super().__init__()
        if fusion_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}; "
                f"expected one of {self.SUPPORTED_MODES}"
            )
        hidden_dim = int(hidden_dim or input_dim)
        self.input_dim = int(input_dim)
        self.mllm_dim = int(mllm_dim)
        self.fusion_mode = fusion_mode
        self.prefix_projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mllm_dim),
        )
        self.gate = (
            nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            if fusion_mode == "gated_fusion"
            else None
        )
        if fusion_mode == "cross_attention":
            if input_dim % num_attention_heads != 0:
                raise ValueError(
                    "input_dim must be divisible by num_attention_heads"
                )
            self.cross_attention = nn.MultiheadAttention(
                input_dim,
                num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.cross_attention = None

    def forward(
        self,
        global_l24_token: Tensor,
        stage1_aligned_roi_tokens: Tensor,
        roi_names: Optional[Sequence[str]] = None,
        routing_weights: Optional[Tensor] = None,
    ) -> Dict[str, object]:
        if global_l24_token.ndim != 2:
            raise ValueError("global_l24_token must have shape [B,D]")
        if stage1_aligned_roi_tokens.ndim != 3:
            raise ValueError(
                "stage1_aligned_roi_tokens must have shape [B,R,D]"
            )
        if global_l24_token.shape[0] != stage1_aligned_roi_tokens.shape[0]:
            raise ValueError("Global and ROI token batch sizes must match")
        if (
            global_l24_token.shape[-1] != self.input_dim
            or stage1_aligned_roi_tokens.shape[-1] != self.input_dim
        ):
            raise ValueError(f"Both inputs must use D={self.input_dim}")

        global_token = global_l24_token.unsqueeze(1)
        gate_values = None
        attention_weights = None
        if self.fusion_mode == "l24_only":
            fused = global_token
        elif self.fusion_mode == "routed_only":
            fused = stage1_aligned_roi_tokens
        elif self.fusion_mode == "concat":
            fused = torch.cat(
                [global_token, stage1_aligned_roi_tokens], dim=1
            )
        elif self.fusion_mode == "gated_fusion":
            expanded_global = global_token.expand_as(
                stage1_aligned_roi_tokens
            )
            gate_values = self.gate(
                torch.cat(
                    [expanded_global, stage1_aligned_roi_tokens], dim=-1
                )
            )
            fused = (
                gate_values * stage1_aligned_roi_tokens
                + (1.0 - gate_values) * expanded_global
            )
        else:
            attended, attention_weights = self.cross_attention(
                query=global_token,
                key=stage1_aligned_roi_tokens,
                value=stage1_aligned_roi_tokens,
                need_weights=True,
            )
            fused = torch.cat([global_token, attended], dim=1)

        prefix = self.prefix_projector(fused)
        diagnostics = {
            "fusion_mode": self.fusion_mode,
            "num_prefix_tokens": int(prefix.shape[1]),
            "global_l24_token_norm": global_l24_token.norm(dim=-1).mean().detach(),
            "stage1_aligned_roi_token_norm": (
                stage1_aligned_roi_tokens.norm(dim=-1).mean().detach()
            ),
            "visual_prefix_token_norm": prefix.norm(dim=-1).mean().detach(),
            "gate_mean": (
                gate_values.mean().detach() if gate_values is not None else None
            ),
            "gate_std": (
                gate_values.std(unbiased=False).detach()
                if gate_values is not None
                else None
            ),
        }
        return {
            "visual_prefix_tokens": prefix,
            "diagnostics": diagnostics,
            "roi_names": list(roi_names) if roi_names is not None else None,
            "routing_weights": routing_weights,
            "cross_attention_weights": attention_weights,
        }
