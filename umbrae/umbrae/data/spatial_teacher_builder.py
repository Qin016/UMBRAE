"""Parameter-free local visual teacher aggregation for locked P11 mappings."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpatialTeacherBuilder(nn.Module):
    """Apply a fixed KxP affinity to BxPxD CLIP patch tokens."""

    def __init__(self, patch_affinity: Tensor):
        super().__init__()
        affinity = torch.as_tensor(patch_affinity)
        if affinity.ndim != 2 or not torch.isfinite(affinity).all():
            raise ValueError("patch_affinity must be a finite [K,P] tensor")
        if not torch.allclose(affinity.sum(1), torch.ones(affinity.shape[0], device=affinity.device, dtype=affinity.dtype), atol=1e-5):
            raise ValueError("each patch-affinity row must sum to one")
        self.register_buffer("patch_affinity", affinity, persistent=False)

    def forward(self, clip_patch_tokens: Tensor, normalize_patches: bool = False) -> Tensor:
        if clip_patch_tokens.ndim != 3 or clip_patch_tokens.shape[1] != self.patch_affinity.shape[1]:
            raise ValueError("clip_patch_tokens must have shape [B,P,D] matching affinity P")
        values = torch.nn.functional.normalize(clip_patch_tokens, dim=-1) if normalize_patches else clip_patch_tokens
        return torch.einsum("kp,bpd->bkd", self.patch_affinity.to(values), values)
