import torch

from losses.lora_calibration_loss import LoRACalibrationLoss


def test_lora_calibration_loss_preserves_lora_gradient_and_detaches_references():
    torch.manual_seed(3)
    lora = torch.randn(4, 3, 8, requires_grad=True); base = torch.randn(4, 3, 8, requires_grad=True); visual = torch.randn(4, 8, requires_grad=True)
    result = LoRACalibrationLoss()(lora, visual, base); result["loss"].backward()
    assert lora.grad is not None and lora.grad.norm() > 0
    assert base.grad is None and visual.grad is None
    assert torch.isfinite(result["loss"])
