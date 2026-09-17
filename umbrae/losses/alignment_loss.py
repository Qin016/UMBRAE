"""Representation alignment losses for NeuroRoute Stage-1 training."""

from typing import Dict

import torch.nn.functional as F
from torch import Tensor


def roi_clip_alignment_loss(
    roi_tokens: Tensor,
    routed_targets: Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Combine MSE and cosine-distance losses for tensors ``[B, R, D]``."""
    if roi_tokens.shape != routed_targets.shape:
        raise ValueError(
            f"Alignment shapes must match: {roi_tokens.shape} != "
            f"{routed_targets.shape}"
        )
    if roi_tokens.ndim != 3:
        raise ValueError("Alignment inputs must have shape [B, R, D]")
    mse = F.mse_loss(roi_tokens, routed_targets)
    cosine = (
        1.0 - F.cosine_similarity(roi_tokens, routed_targets, dim=-1)
    ).mean()
    total = mse_weight * mse + cosine_weight * cosine
    return {"total": total, "mse": mse, "cosine": cosine}
