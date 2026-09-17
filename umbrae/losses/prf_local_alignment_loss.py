"""Locked P11-B local alignment objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PRFLocalAlignmentLoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 0.1,
        relational_weight: float = 0.1,
        nce_weight: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.relational_weight = float(relational_weight)
        self.nce_weight = float(nce_weight)
        self.temperature = float(temperature)

    def forward(self, prediction: Tensor, teacher: Tensor):
        if prediction.shape != teacher.shape or prediction.ndim != 3:
            raise ValueError("prediction and teacher must match [B,K,D]")
        pred = prediction.float()
        target = teacher.detach().float()
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        local_cos = (1.0 - (pred_norm * target_norm).sum(-1)).mean()
        local_mse = F.mse_loss(pred, target)
        pred_relation = pred_norm @ pred_norm.transpose(1, 2)
        target_relation = target_norm @ target_norm.transpose(1, 2)
        mask = ~torch.eye(pred.shape[1], dtype=torch.bool, device=pred.device)
        unit_rel = F.mse_loss(pred_relation[:, mask], target_relation[:, mask])
        # For each unit r, rows are brain samples and columns are teacher samples.
        logits = torch.einsum("brd,jrd->rbj", pred_norm, target_norm) / self.temperature
        labels = torch.arange(pred.shape[0], device=pred.device).expand(pred.shape[1], -1)
        local_nce = F.cross_entropy(logits, labels)
        total = local_cos + self.mse_weight * local_mse + self.relational_weight * unit_rel + self.nce_weight * local_nce
        return {
            "loss": total,
            "local_cos_loss": local_cos,
            "local_mse": local_mse,
            "unit_rel_loss": unit_rel,
            "local_nce": local_nce,
        }
