import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.gated_cross_attention import GatedCrossAttentionFusion


def test_gated_cross_attention_shape_gate_and_gradients():
    z_sem = torch.randn(2, 256, 32)
    h_struct = torch.randn(2, 8, 32, requires_grad=True)
    model = GatedCrossAttentionFusion(dim=32, num_heads=4, gate_init=-4.0)
    output = model(z_sem, h_struct)
    assert output["z_cal"].shape == (2, 256, 32)
    assert output["fusion_attention"].shape == (2, 4, 256, 8)
    assert torch.isfinite(output["fusion_attention"]).all()
    assert math.isclose(float(output["fusion_gate"]), 0.017986, rel_tol=1e-4)
    relative_delta = (output["z_cal"] - z_sem).norm() / z_sem.norm()
    assert float(relative_delta) < 0.05
    output["z_cal"].square().mean().backward()
    assert h_struct.grad is not None and float(h_struct.grad.norm()) > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
