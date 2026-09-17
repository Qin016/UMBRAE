"""Stage-B global alignment and raw relational-geometry losses."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FusionCalibrationLoss(nn.Module):
    """Calibrate pooled semantic tokens against a frozen visual embedding."""

    def __init__(self, lambda_global: float = 1.0, lambda_rel: float = 1.0) -> None:
        super().__init__()
        self.lambda_global = float(lambda_global)
        self.lambda_rel = float(lambda_rel)

    def forward(self, z_cal: Tensor, q_visual: Tensor):
        if z_cal.ndim != 3 or q_visual.ndim != 2:
            raise ValueError("Expected z_cal [B,N,D] and q_visual [B,D]")
        if z_cal.shape[0] != q_visual.shape[0] or z_cal.shape[2] != q_visual.shape[1]:
            raise ValueError("Semantic and visual shapes are incompatible")
        q_cal = F.normalize(z_cal.float().mean(dim=1), dim=-1)
        q_visual = F.normalize(q_visual.detach().float(), dim=-1)
        global_loss = (1.0 - (q_cal * q_visual).sum(dim=-1)).mean()
        if len(q_cal) < 2:
            relational_loss = q_cal.sum() * 0.0
        else:
            d_cal = 1.0 - q_cal @ q_cal.T
            d_visual = 1.0 - q_visual @ q_visual.T
            mask = ~torch.eye(len(q_cal), dtype=torch.bool, device=q_cal.device)
            relational_loss = (d_cal[mask] - d_visual[mask]).square().mean()
        return {
            "loss": self.lambda_global * global_loss + self.lambda_rel * relational_loss,
            "global_loss": global_loss,
            "rel_loss": relational_loss,
            "q_cal": q_cal,
            "q_visual": q_visual,
        }
