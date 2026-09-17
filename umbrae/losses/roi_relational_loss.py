"""Standardized ROI-specific geometry and weak global contrastive losses."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _upper_triangle(values: Tensor) -> Tensor:
    if values.ndim != 3 or values.shape[-1] != values.shape[-2]:
        raise ValueError("RDM tensor must have shape [K,B,B]")
    if values.shape[-1] < 2:
        raise ValueError("Relational loss requires batch size >= 2")
    indices = torch.triu_indices(values.shape[-1], values.shape[-1], offset=1, device=values.device)
    return values[:, indices[0], indices[1]]


def _standardize_pairs(values: Tensor, eps: float) -> Tensor:
    centered = values - values.mean(dim=-1, keepdim=True)
    return centered / (centered.std(dim=-1, unbiased=False, keepdim=True) + eps)


class ROIRelationalLoss(nn.Module):
    """Match standardized stimulus geometry independently for every ROI."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, roi_tokens: Tensor, clip_layer_pools: Tensor, routing: Tensor) -> Dict[str, Tensor]:
        if roi_tokens.ndim != 3 or clip_layer_pools.ndim != 3:
            raise ValueError("Expected ROI [B,R,D] and CLIP [B,L,D]")
        batch, roi_count, dim = roi_tokens.shape
        if clip_layer_pools.shape[0] != batch or clip_layer_pools.shape[2] != dim:
            raise ValueError("ROI and CLIP batch/feature dimensions must match")
        if routing.shape != (roi_count, clip_layer_pools.shape[1]):
            raise ValueError("Routing prior must have shape [R,L]")
        with torch.autocast(device_type=roi_tokens.device.type, enabled=False):
            brain = F.normalize(roi_tokens.float(), dim=-1).transpose(0, 1)
            visual = F.normalize(clip_layer_pools.float(), dim=-1).transpose(0, 1)
            brain_rdm = 1.0 - brain @ brain.transpose(1, 2)
            layer_rdm = 1.0 - visual @ visual.transpose(1, 2)
            target_rdm = torch.einsum("rl,lij->rij", routing.float(), layer_rdm).detach()
            brain_pairs = _standardize_pairs(_upper_triangle(brain_rdm), self.eps)
            target_pairs = _standardize_pairs(_upper_triangle(target_rdm), self.eps)
            per_roi = (brain_pairs - target_pairs).square().mean(dim=-1)
        return {"loss": per_roi.mean(), "per_roi_loss": per_roi, "brain_rdm": brain_rdm, "target_rdm": target_rdm}


class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, brain: Tensor, visual: Tensor) -> Tensor:
        if brain.shape != visual.shape or brain.ndim != 2:
            raise ValueError("InfoNCE inputs must share shape [B,D]")
        brain = F.normalize(brain.float(), dim=-1)
        visual = F.normalize(visual.float(), dim=-1)
        logits = brain @ visual.transpose(0, 1) / self.temperature
        labels = torch.arange(len(brain), device=brain.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
