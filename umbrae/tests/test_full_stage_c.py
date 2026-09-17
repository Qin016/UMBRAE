import torch

from losses.lora_calibration_loss import LoRACalibrationLoss
from models.gated_cross_attention import GatedCrossAttentionFusion


def test_final_preserve_loss_and_fusion_gradient_path():
    torch.manual_seed(7)
    z_lora = torch.randn(3, 5, 16, requires_grad=True); h_struct = torch.randn(3, 2, 16); base = torch.randn(3, 5, 16); visual = torch.randn(3, 16)
    fusion = GatedCrossAttentionFusion(16, 4, gate_init=-4.0); fused = fusion(z_lora, h_struct, return_attention=False)
    loss = LoRACalibrationLoss(lambda_preserve=0.1)(fused["z_cal"], visual, base)["loss"]; loss.backward()
    assert z_lora.grad is not None and z_lora.grad.norm() > 0
    assert all(parameter.grad is not None for parameter in fusion.parameters())
    assert h_struct.grad is None
