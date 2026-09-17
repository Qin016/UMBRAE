import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.roi_mapping import MVP_ROI_NAMES
from models.structural_branch import StructuralBranch


def make_indices():
    return {
        name: list(range(index * 3, index * 3 + index + 1))
        for index, name in enumerate(MVP_ROI_NAMES)
    }


def test_structural_branch_flat_and_repeated_inputs():
    model = StructuralBranch(token_dim=32, roi_indices=make_indices())
    flat = model(torch.randn(2, 32))
    repeated = model(torch.randn(2, 3, 32))
    assert flat["h_struct"].shape == (2, 8, 32)
    assert repeated["h_struct"].shape == (2, 8, 32)
    assert flat["roi_names"] == list(MVP_ROI_NAMES)
    flat["h_struct"].square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())
    report = model.parameter_report()
    assert report["total"] == report["trainable"] > 0
    assert report["roi_tokenizer"]["total"] > report["output_norm"]["total"]
