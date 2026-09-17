"""P5 LoRA-only geometry calibration with semantic preservation."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRACalibrationLoss(nn.Module):
    def __init__(self, lambda_global: float = 1.0, lambda_rel: float = 1.0, lambda_preserve: float = 0.1) -> None:
        super().__init__()
        self.lambda_global, self.lambda_rel, self.lambda_preserve = map(float, (lambda_global, lambda_rel, lambda_preserve))

    def forward(self, z_lora: Tensor, q_visual: Tensor, z_base: Tensor):
        if z_lora.ndim != 3 or z_base.shape != z_lora.shape or q_visual.shape != (len(z_lora), z_lora.shape[-1]):
            raise ValueError("Expected z_lora/z_base [B,N,D] and q_visual [B,D]")
        pooled_lora = z_lora.float().mean(1)
        pooled_base = z_base.detach().float().mean(1)
        q_lora = F.normalize(pooled_lora, dim=-1)
        q_base = F.normalize(pooled_base, dim=-1)
        q_visual = F.normalize(q_visual.detach().float(), dim=-1)
        global_loss = (1.0 - (q_lora * q_visual).sum(-1)).mean()
        preserve_loss = (1.0 - (q_lora * q_base).sum(-1)).mean()
        if len(q_lora) < 2:
            rel_loss = q_lora.sum() * 0.0
        else:
            mask = ~torch.eye(len(q_lora), dtype=torch.bool, device=q_lora.device)
            d_lora, d_visual = 1.0 - q_lora @ q_lora.T, 1.0 - q_visual @ q_visual.T
            rel_loss = (d_lora[mask] - d_visual[mask]).square().mean()
        total = self.lambda_global * global_loss + self.lambda_rel * rel_loss + self.lambda_preserve * preserve_loss
        return {"loss": total, "global_loss": global_loss, "rel_loss": rel_loss, "preserve_loss": preserve_loss, "q_lora": q_lora, "q_base": q_base, "q_visual": q_visual, "pooled_lora": pooled_lora, "pooled_base": pooled_base}
