import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.semantic_uot_loss import SemanticUOTLoss


@pytest.mark.parametrize("batch", [1, 3])
def test_semantic_uot_finite_and_differentiable(batch):
    torch.manual_seed(5)
    h = torch.randn(batch, 8, 32, requires_grad=True)
    v = torch.randn(batch, 256, 32).detach()
    result = SemanticUOTLoss(epsilon=0.1, tau=1.0, num_iters=80)(h, v)
    assert result["transport"].shape == (batch, 8, 256)
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["transport"]).all()
    assert (result["transport"] >= 0).all()
    assert float(result["transport_mass"]) > 0
    normalized_entropy = result["normalized_transport_entropy_per_sample"]
    assert torch.all(normalized_entropy >= -1e-6)
    assert torch.all(normalized_entropy <= 1.0 + 1e-6)
    assert torch.isfinite(result["effective_support"])
    assert float(result["effective_support"]) >= 1.0
    assert 0.0 < float(result["max_transport_fraction"]) <= 1.0
    assert result["row_mass"].shape == (batch, 8)
    assert result["column_mass"].shape == (batch, 256)
    assert torch.isfinite(result["row_mass_cv"])
    assert torch.isfinite(result["column_mass_cv"])
    result["loss"].backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert float(h.grad.norm()) > 0
    assert v.grad is None


def test_similar_embeddings_have_lower_cost_than_opposites():
    base = torch.randn(2, 1, 32)
    h = base.expand(-1, 8, -1).clone()
    similar = base.expand(-1, 256, -1).clone()
    opposite = (-base).expand(-1, 256, -1).clone()
    criterion = SemanticUOTLoss(epsilon=0.1, tau=1.0, num_iters=40)
    assert criterion(h, similar)["cost"].mean() < criterion(h, opposite)["cost"].mean()


def test_semantic_uot_is_stable_under_autocast():
    h = torch.randn(1, 8, 32, requires_grad=True)
    v = torch.randn(1, 256, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = SemanticUOTLoss(epsilon=0.05, tau=1.0, num_iters=60)(h, v)
    assert result["cost"].dtype == torch.float32
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["transport"]).all()
