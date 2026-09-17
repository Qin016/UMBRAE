import torch

from models.token_mixing_interface_adapter import (
    TokenMixingInterfaceAdapter,
    adapter_parameter_counts,
    attention_diagnostics,
)


def test_p10_adapter_is_exact_identity_at_initialization():
    torch.manual_seed(0)
    model = TokenMixingInterfaceAdapter(hidden_dim=32, num_heads=4, bottleneck_dim=8)
    source = torch.randn(2, 7, 32)
    output = model(source, return_details=True, return_attention=True)
    torch.testing.assert_close(output["z_hat"], source, rtol=0, atol=0)
    assert output["attn_delta"].abs().max() == 0
    assert output["ffn_delta"].abs().max() == 0
    assert output["attention_weights"].shape == (2, 4, 7, 7)


def test_p10_parameter_count_matches_one_block_budget():
    assert adapter_parameter_counts(TokenMixingInterfaceAdapter()) == {
        "block_without_gates": 4728064,
        "gates": 2,
        "total_trainable": 4728066,
    }


def test_p10_frozen_projector_propagates_adapter_gradient():
    model = TokenMixingInterfaceAdapter(hidden_dim=32, num_heads=4, bottleneck_dim=8)
    projector = torch.nn.Linear(32, 16).requires_grad_(False)
    source = torch.randn(2, 7, 32)
    target = torch.randn(2, 7, 16)
    (projector(model(source)) - target).square().mean().backward()
    assert model.attention.out_proj.weight.grad.abs().sum() > 0
    assert model.ffn_up.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in projector.parameters())


def test_p10_attention_diagnostics_detect_off_diagonal_mixing():
    weights = torch.full((2, 4, 7, 7), 1 / 7)
    diagnostics = attention_diagnostics(weights)
    assert abs(diagnostics["off_diagonal_attention_mass"] - 6 / 7) < 1e-6
    assert abs(diagnostics["normalized_attention_entropy"] - 1) < 1e-6
    assert len(diagnostics["per_head"]) == 4


def test_p10_same_seed_produces_identical_matched_initialization():
    torch.manual_seed(42)
    left = TokenMixingInterfaceAdapter(hidden_dim=32, num_heads=4, bottleneck_dim=8)
    torch.manual_seed(42)
    right = TokenMixingInterfaceAdapter(hidden_dim=32, num_heads=4, bottleneck_dim=8)
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)
