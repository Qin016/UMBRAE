import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dual_branch_umbrae import DualBranchUMBRAE
from models.gated_cross_attention import GatedCrossAttentionFusion
from models.roi_mapping import MVP_ROI_NAMES
from models.structural_branch import StructuralBranch


class TinySemantic(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(32, 32)

    def forward(self, fmri):
        return self.projection(fmri).unsqueeze(1).expand(-1, 256, -1)


def make_model():
    indices = {
        name: list(range(index * 4, index * 4 + 4))
        for index, name in enumerate(MVP_ROI_NAMES)
    }
    return DualBranchUMBRAE(
        TinySemantic(),
        StructuralBranch(token_dim=32, roi_indices=indices),
        GatedCrossAttentionFusion(dim=32, num_heads=4),
    )


def test_complete_dual_branch_shapes():
    model = make_model()
    model.configure_stage("B", verbose=False)
    output = model(torch.randn(2, 3, 32))
    assert output["z_sem"].shape == (2, 256, 32)
    assert output["h_struct"].shape == (2, 8, 32)
    assert output["z_cal"].shape == (2, 256, 32)
    assert output["fusion_attention"].shape == (2, 4, 256, 8)
    assert output["roi_names"] == list(MVP_ROI_NAMES)
