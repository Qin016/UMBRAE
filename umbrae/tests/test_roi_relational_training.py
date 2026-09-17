import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.roi_relational_loss import ROIRelationalLoss, SymmetricInfoNCELoss
from models.roi_specific_structural_branch import ROISpecificStructuralBranch, StructuralAttentionPool


def test_roi_specific_branch_uses_true_input_sizes_and_gradients():
    mapping = {name: [2 * i, 2 * i + 1] for i, name in enumerate(("V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA"))}
    model = ROISpecificStructuralBranch(mapping, hidden_dim=4, token_dim=6)
    output = model(torch.randn(3, 16))["h_struct"]
    assert output.shape == (3, 8, 6)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_relational_and_contrastive_losses_are_finite_and_differentiable():
    roi = torch.randn(5, 8, 12, requires_grad=True)
    layers = torch.randn(5, 6, 12)
    routing = torch.full((8, 6), 1 / 6)
    relational = ROIRelationalLoss()(roi, layers, routing)
    pool = StructuralAttentionPool(12)
    q = pool(roi)["pooled"]
    contrastive = SymmetricInfoNCELoss()(q, torch.randn(5, 12))
    total = relational["loss"] + 0.1 * contrastive
    total.backward()
    assert torch.isfinite(total)
    assert roi.grad is not None and pool.query.grad is not None
    assert relational["per_roi_loss"].shape == (8,)
