import torch

from losses.fusion_calibration_loss import FusionCalibrationLoss
from models.gated_cross_attention import GatedCrossAttentionFusion


def test_fusion_calibration_loss_is_raw_offdiagonal_and_differentiable():
    torch.manual_seed(1)
    z = torch.randn(4, 3, 8, requires_grad=True)
    visual = z.detach().mean(1) + 0.1 * torch.randn(4, 8)
    result = FusionCalibrationLoss()(z, visual)
    q = torch.nn.functional.normalize(z.float().mean(1), dim=-1)
    v = torch.nn.functional.normalize(visual.float(), dim=-1)
    mask = ~torch.eye(4, dtype=torch.bool)
    expected = ((1 - q @ q.T)[mask] - (1 - v @ v.T)[mask]).square().mean()
    assert torch.allclose(result["rel_loss"], expected)
    result["loss"].backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_stage_b_fusion_shape_gate_and_fusion_only_gradients():
    torch.manual_seed(42)
    fusion = GatedCrossAttentionFusion(dim=16, num_heads=4, gate_init=-4.0)
    z_sem = torch.randn(2, 256, 16)
    h_struct = torch.randn(2, 8, 16)
    output = fusion(z_sem, h_struct, return_attention=False)
    assert output["z_cal"].shape == (2, 256, 16)
    assert output["fusion_attention"] is None
    assert torch.allclose(output["fusion_gate"], torch.tensor(0.01798621), atol=1e-7)
    output["z_cal"].square().mean().backward()
    assert all(p.grad is not None for p in fusion.parameters())
    assert z_sem.grad is None and h_struct.grad is None
