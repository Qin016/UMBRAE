"""Soft routing from ROI tokens to selected CLIP layer features."""

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn


class ROILayerRouter(nn.Module):
    """Route each ROI token to a weighted mixture of CLIP layer features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        topk: Optional[int] = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if topk is not None and topk <= 0:
            raise ValueError("topk must be positive when provided")

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim or feature_dim)
        self.topk = topk
        self.query_projection = nn.Linear(feature_dim, self.hidden_dim)
        self.key_projection = nn.Linear(feature_dim, self.hidden_dim)

        log_temperature = torch.tensor(math.log(float(temperature)))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer(
                "log_temperature", log_temperature, persistent=True
            )

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp_min(1e-6)

    @staticmethod
    def _validate_inputs(
        roi_tokens: Tensor, clip_layer_features: Tensor
    ) -> None:
        if roi_tokens.ndim != 3:
            raise ValueError(
                "roi_tokens must have shape [B, R, D], "
                f"got {tuple(roi_tokens.shape)}"
            )
        if clip_layer_features.ndim != 3:
            raise ValueError(
                "clip_layer_features must have shape [B, L, D], "
                f"got {tuple(clip_layer_features.shape)}"
            )
        if roi_tokens.shape[0] != clip_layer_features.shape[0]:
            raise ValueError("ROI and CLIP feature batch sizes must match")
        if roi_tokens.shape[-1] != clip_layer_features.shape[-1]:
            raise ValueError("ROI and CLIP feature dimensions must match")
        if clip_layer_features.shape[1] == 0:
            raise ValueError("clip_layer_features must contain at least one layer")

    @staticmethod
    def _apply_topk(weights: Tensor, topk: int) -> Tensor:
        layer_count = weights.shape[-1]
        if topk > layer_count:
            raise ValueError(
                f"topk={topk} exceeds number of CLIP layers L={layer_count}"
            )
        values, indices = torch.topk(weights, k=topk, dim=-1)
        sparse_weights = torch.zeros_like(weights).scatter(-1, indices, values)
        return sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )

    def forward(
        self,
        roi_tokens: Tensor,
        clip_layer_features: Tensor,
        topk: Optional[int] = None,
    ) -> Dict[str, object]:
        """Return routed targets, normalized routing weights, and diagnostics."""
        self._validate_inputs(roi_tokens, clip_layer_features)
        if roi_tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension D={self.feature_dim}, "
                f"got D={roi_tokens.shape[-1]}"
            )

        queries = self.query_projection(roi_tokens)
        keys = self.key_projection(clip_layer_features)
        logits = torch.matmul(queries, keys.transpose(-1, -2))
        logits = logits / math.sqrt(self.hidden_dim)
        logits = logits / self.temperature.to(dtype=logits.dtype)
        routing_weights = torch.softmax(logits, dim=-1)

        effective_topk = self.topk if topk is None else topk
        if effective_topk is not None:
            if effective_topk <= 0:
                raise ValueError("topk must be positive when provided")
            routing_weights = self._apply_topk(
                routing_weights, int(effective_topk)
            )

        routed_targets = torch.matmul(routing_weights, clip_layer_features)
        entropy = -(
            routing_weights
            * routing_weights.clamp_min(
                torch.finfo(routing_weights.dtype).eps
            ).log()
        ).sum(dim=-1)
        diagnostics = {
            "temperature": self.temperature.detach(),
            "mean_entropy": entropy.mean().detach(),
            "mean_max_weight": routing_weights.max(dim=-1).values.mean().detach(),
            "active_layers_per_roi": (
                routing_weights > 0
            ).sum(dim=-1).to(torch.float32).mean().detach(),
            "topk": effective_topk,
        }
        return {
            "routed_targets": routed_targets,
            "routing_weights": routing_weights,
            "routing_logits": logits,
            "diagnostics": diagnostics,
        }
