"""Regularization losses for ROI-to-CLIP-layer routing weights."""

from typing import Dict

import torch
from torch import Tensor


def _validate_routing_weights(routing_weights: Tensor) -> None:
    if routing_weights.ndim != 3:
        raise ValueError(
            "routing_weights must have shape [B, R, L], "
            f"got {tuple(routing_weights.shape)}"
        )
    if routing_weights.shape[-1] == 0:
        raise ValueError("routing_weights must contain at least one layer")
    if not torch.isfinite(routing_weights).all():
        raise ValueError("routing_weights contains non-finite values")


def routing_entropy_loss(
    routing_weights: Tensor, eps: float = 1e-8
) -> Tensor:
    """Mean entropy; minimizing it encourages sharper routing."""
    _validate_routing_weights(routing_weights)
    probabilities = routing_weights.clamp_min(eps)
    return -(routing_weights * probabilities.log()).sum(dim=-1).mean()


def routing_balance_loss(
    routing_weights: Tensor, eps: float = 1e-8
) -> Tensor:
    """KL(mean layer usage || uniform), discouraging global layer collapse."""
    _validate_routing_weights(routing_weights)
    usage = routing_weights.mean(dim=(0, 1)).clamp_min(eps)
    layer_count = routing_weights.shape[-1]
    return (usage * (usage * layer_count).log()).sum()


def routing_smoothness_loss(routing_weights: Tensor) -> Tensor:
    """Mean squared difference between neighboring CLIP-layer weights."""
    _validate_routing_weights(routing_weights)
    if routing_weights.shape[-1] < 2:
        return routing_weights.sum() * 0.0
    differences = routing_weights[..., 1:] - routing_weights[..., :-1]
    return differences.square().mean()


def routing_regularization_loss(
    routing_weights: Tensor,
    entropy_weight: float = 0.0,
    balance_weight: float = 0.0,
    smoothness_weight: float = 0.0,
) -> Dict[str, Tensor]:
    """Return individual routing losses and their weighted total."""
    entropy = routing_entropy_loss(routing_weights)
    balance = routing_balance_loss(routing_weights)
    smoothness = routing_smoothness_loss(routing_weights)
    total = (
        entropy_weight * entropy
        + balance_weight * balance
        + smoothness_weight * smoothness
    )
    return {
        "total": total,
        "entropy": entropy,
        "balance": balance,
        "smoothness": smoothness,
    }
