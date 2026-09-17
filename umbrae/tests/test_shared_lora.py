import copy

import pytest
import torch
from torch import nn

from models.dual_branch_cache import validate_cache_stage
from models.shared_lora import LoRAQLinear, LoRASelectiveVLinear, load_lora_state_dict, lora_named_parameters, lora_state_dict


def test_q_lora_zero_initialization_and_gradient():
    base = nn.Linear(16, 24, bias=False); original = copy.deepcopy(base)
    layer = LoRAQLinear(base, rank=4, alpha=8, dropout=0.0)
    x = torch.randn(3, 5, 16)
    assert torch.equal(layer(x), original(x))
    layer(x).square().mean().backward()
    assert layer.base.weight.grad is None
    assert layer.lora_b.weight.grad is not None and layer.lora_b.weight.grad.norm() > 0


def test_selective_v_lora_never_changes_k_and_both_factors_train():
    base = nn.Linear(16, 48, bias=False); layer = LoRASelectiveVLinear(base, 4, 8, 0.0)
    with torch.no_grad(): layer.lora_b.weight.normal_()
    x = torch.randn(3, 5, 16); base_kv = base(x); adapted = layer(x)
    assert torch.equal(adapted[..., :24], base_kv[..., :24])
    assert not torch.equal(adapted[..., 24:], base_kv[..., 24:])
    adapted.square().mean().backward()
    assert layer.base.weight.grad is None
    assert layer.lora_a.weight.grad.norm() > 0 and layer.lora_b.weight.grad.norm() > 0


def test_lora_only_state_roundtrip_and_optimizer_membership(tmp_path):
    model = nn.Sequential(LoRAQLinear(nn.Linear(8, 8, bias=False), 2, 4, 0.0), LoRASelectiveVLinear(nn.Linear(8, 16, bias=False), 2, 4, 0.0))
    expected = {id(p) for _, p in lora_named_parameters(model)}
    optimizer = torch.optim.AdamW((p for _, p in lora_named_parameters(model)), lr=1e-5)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == expected
    state = lora_state_dict(model); path = tmp_path / "adapter.pth"; torch.save({"lora": state}, path); state = torch.load(path)["lora"]; restored = copy.deepcopy(model)
    with torch.no_grad():
        for _, p in lora_named_parameters(restored): p.normal_()
    load_lora_state_dict(restored, state)
    x = torch.randn(2, 8)
    assert torch.equal(model(x), restored(x))
    assert all("lora_" in key for key in state)


def test_stage_c_semantic_cache_guard():
    with pytest.raises(ValueError, match="forbidden"):
        validate_cache_stage("C")
