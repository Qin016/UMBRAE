"""Shared Q and selective-V LoRA injection for UMBRAE's Perceiver."""

import math
from typing import Dict, Iterable, List, Mapping

import torch
from torch import Tensor, nn


class LoRAQLinear(nn.Module):
    """Frozen Linear plus a standard low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if base.bias is not None or rank <= 0:
            raise ValueError("Q LoRA requires a bias-free Linear and positive rank")
        self.base = base
        self.rank, self.alpha, self.scale = int(rank), float(alpha), float(alpha) / rank
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.base.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * self.lora_b(self.lora_a(self.lora_dropout(x)))


class LoRASelectiveVLinear(nn.Module):
    """Preserve fused K exactly and append a low-rank residual to V only."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if base.bias is not None or base.out_features % 2 or rank <= 0:
            raise ValueError("Selective V LoRA requires an even, bias-free fused KV Linear")
        self.base = base
        self.rank, self.alpha, self.scale = int(rank), float(alpha), float(alpha) / rank
        self.value_dim = base.out_features // 2
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, self.value_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.base.requires_grad_(False)

    def value_delta(self, x: Tensor) -> Tensor:
        return self.scale * self.lora_b(self.lora_a(self.lora_dropout(x)))

    def forward(self, x: Tensor) -> Tensor:
        base_kv = self.base(x)
        delta_v = self.value_delta(x)
        return base_kv + torch.cat((torch.zeros_like(delta_v), delta_v), dim=-1)


def inject_shared_qv_lora(backbone: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> List[str]:
    """Replace Q and fused-KV projections in all shared Perceiver layers."""
    try:
        layers = backbone.encoder.perceiver.perceiver.layers
    except AttributeError as error:
        raise ValueError("Backbone does not expose encoder.perceiver.perceiver.layers") from error
    if len(layers) != 6:
        raise ValueError(f"Expected six shared Perceiver layers, found {len(layers)}")
    targets = []
    for index, layer in enumerate(layers):
        attention = layer[0]
        if not isinstance(attention.to_q, nn.Linear) or attention.to_q.weight.shape != (1536, 1024):
            raise ValueError(f"Unexpected Q projection at Perceiver layer {index}")
        if not isinstance(attention.to_kv, nn.Linear) or attention.to_kv.weight.shape != (3072, 1024):
            raise ValueError(f"Unexpected fused KV projection at Perceiver layer {index}")
        attention.to_q = LoRAQLinear(attention.to_q, rank, alpha, dropout)
        attention.to_kv = LoRASelectiveVLinear(attention.to_kv, rank, alpha, dropout)
        targets.extend([
            f"encoder.perceiver.perceiver.layers.{index}.0.to_q",
            f"encoder.perceiver.perceiver.layers.{index}.0.to_kv[value_only]",
        ])
    return targets


def lora_named_parameters(module: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, parameter in module.named_parameters():
        if ".lora_a." in name or ".lora_b." in name:
            yield name, parameter


def lora_state_dict(module: nn.Module) -> Dict[str, Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in lora_named_parameters(module)}


def load_lora_state_dict(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    parameters = dict(lora_named_parameters(module))
    if set(parameters) != set(state):
        raise ValueError(f"LoRA state mismatch: missing={set(parameters)-set(state)}, unexpected={set(state)-set(parameters)}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f"LoRA tensor shape mismatch for {name}")
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


class SharedLoRAUMBRAEEncoder(nn.Module):
    """Keep the restored UMBRAE base frozen while training shared adapters."""

    def __init__(self, backbone: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> None:
        super().__init__()
        backbone.requires_grad_(False)
        self.backbone = backbone
        self.rank, self.alpha, self.dropout = int(rank), float(alpha), float(dropout)
        self.target_modules = inject_shared_qv_lora(backbone, rank, alpha, dropout)
        for _, parameter in lora_named_parameters(self):
            parameter.requires_grad = True
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        for module in self.modules():
            if isinstance(module, (LoRAQLinear, LoRASelectiveVLinear)):
                module.lora_dropout.train(mode)
        return self

    def forward(self, fmri: Tensor) -> Tensor:
        return self.backbone(fmri)

    def parameter_report(self) -> Dict[str, float]:
        lora = sum(parameter.numel() for _, parameter in lora_named_parameters(self))
        base = sum(parameter.numel() for name, parameter in self.named_parameters() if ".lora_a." not in name and ".lora_b." not in name)
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"brainx_base_parameters": base, "lora_parameters": lora, "total_parameters": base + lora, "trainable_parameters": trainable, "trainable_ratio_percent_of_brainx": 100.0 * trainable / base}
