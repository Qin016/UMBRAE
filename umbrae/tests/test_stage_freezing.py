import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_dual_branch_shapes import make_model


def has_gradient(module):
    return any(
        parameter.grad is not None and float(parameter.grad.norm()) > 0
        for parameter in module.parameters()
    )


def test_stage_a_freezing_and_gradients():
    model = make_model()
    report = model.configure_stage("A", verbose=False)
    output = model(
        torch.randn(2, 32),
        enable_semantic=False,
        enable_structural=True,
        enable_fusion=False,
    )
    output["h_struct"].square().mean().backward()
    assert report["semantic_backbone"]["trainable"] == 0
    assert report["fusion"]["trainable"] == 0
    assert not has_gradient(model.semantic_backbone)
    assert has_gradient(model.structural_branch)
    assert not has_gradient(model.fusion)


def test_stage_b_freezing_and_gradients():
    model = make_model()
    report = model.configure_stage("B", verbose=False)
    output = model(torch.randn(2, 32))
    loss = output["h_struct"].square().mean() + output["z_cal"].square().mean()
    loss.backward()
    assert report["semantic_backbone"]["trainable"] == 0
    assert report["fusion"]["trainable"] > 0
    assert not has_gradient(model.semantic_backbone)
    assert has_gradient(model.structural_branch)
    assert has_gradient(model.fusion)
