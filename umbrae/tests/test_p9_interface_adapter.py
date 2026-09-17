import torch

from losses.interface_distillation_loss import InterfaceDistillationLoss
from models.pre_projector_interface_adapter import (
    PreProjectorInterfaceAdapter,
    adapter_parameter_counts,
)


def test_p9_adapter_is_exact_identity_at_initialization():
    torch.manual_seed(0)
    model = PreProjectorInterfaceAdapter()
    source = torch.randn(2, 7, 1024)
    output = model(source, return_delta=True)
    torch.testing.assert_close(output["z_hat"], source, rtol=0, atol=0)
    assert output["delta"].abs().max() == 0


def test_p9_adapter_parameter_count_and_gate_partition():
    assert adapter_parameter_counts(PreProjectorInterfaceAdapter()) == {
        "adapter_without_gate": 527616,
        "gate": 1,
        "total_trainable": 527617,
    }


def test_p9_loss_is_zero_at_exact_teacher_endpoint():
    criterion = InterfaceDistillationLoss()
    source = visual = adapted = torch.randn(2, 4, 8)
    projected = torch.randn(2, 4, 16)
    losses = criterion(source, adapted, visual, projected, projected)
    assert losses["proj_cos"].abs() < 1e-6
    assert losses["proj_mse"] == 0
    assert losses["preproj_cos"].abs() < 1e-6
    assert losses["preserve"].abs() < 1e-6
    assert losses["norm"] == 0
    assert losses["loss"].abs() < 1e-6


def test_frozen_projector_still_propagates_adapter_gradient():
    adapter = PreProjectorInterfaceAdapter()
    projector = torch.nn.Linear(1024, 32).requires_grad_(False)
    source = torch.randn(2, 3, 1024)
    target = torch.randn(2, 3, 32)
    loss = (projector(adapter(source)) - target).square().mean()
    loss.backward()
    assert adapter.up.weight.grad is not None
    assert adapter.up.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in projector.parameters())


def test_gate_receives_gradient_after_nonzero_adapter_delta():
    adapter = PreProjectorInterfaceAdapter()
    torch.nn.init.normal_(adapter.up.weight, std=0.01)
    adapter(torch.randn(2, 3, 1024)).square().mean().backward()
    assert adapter.gate_logit.grad is not None
    assert adapter.gate_logit.grad.abs() > 0
