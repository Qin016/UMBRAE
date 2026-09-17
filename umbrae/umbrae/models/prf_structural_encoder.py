"""Fine-grained pRF structural encoder used by P11-B only."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


class PRFStructuralEncoder(nn.Module):
    """Unit-specific voxel readouts followed by one shared output projection."""

    def __init__(
        self,
        units: Sequence[Mapping[str, object]],
        hidden_dim: int = 256,
        output_dim: int = 1024,
    ) -> None:
        super().__init__()
        if len(units) != 64:
            raise ValueError(f"P11-B v1 requires 64 retinotopic units, got {len(units)}")
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.unit_ids = [str(unit["unit_id"]) for unit in units]
        self.parent_rois = [str(unit["parent_roi"]) for unit in units]
        self.hemispheres = [str(unit["hemisphere"]) for unit in units]
        self.input_projections = nn.ModuleList()
        for position, unit in enumerate(units):
            index = torch.as_tensor(unit["voxel_indices"], dtype=torch.long)
            if index.ndim != 1 or index.numel() == 0 or index.unique().numel() != index.numel():
                raise ValueError(f"invalid voxel indices for unit {self.unit_ids[position]}")
            self.register_buffer(f"_unit_index_{position}", index, persistent=True)
            self.input_projections.append(nn.Linear(index.numel(), self.hidden_dim))
        self.activation = nn.GELU()
        self.normalization = nn.LayerNorm(self.hidden_dim)
        self.shared_projection = nn.Linear(self.hidden_dim, self.output_dim)

    @property
    def unit_size_vector(self):
        return [int(getattr(self, f"_unit_index_{i}").numel()) for i in range(64)]

    def forward(self, fmri: Tensor) -> Tensor:
        if fmri.ndim != 2:
            raise ValueError(f"fmri must have shape [B,V], got {tuple(fmri.shape)}")
        hidden = []
        for position, projection in enumerate(self.input_projections):
            index = getattr(self, f"_unit_index_{position}")
            if int(index.max()) >= fmri.shape[1]:
                raise ValueError("unit voxel index exceeds fMRI dimension")
            hidden.append(projection(fmri.index_select(1, index)))
        values = torch.stack(hidden, dim=1)
        return self.shared_projection(self.normalization(self.activation(values)))

    def parameter_report(self):
        first_layers = [sum(p.numel() for p in layer.parameters()) for layer in self.input_projections]
        shared = sum(p.numel() for p in self.normalization.parameters()) + sum(
            p.numel() for p in self.shared_projection.parameters()
        )
        total = sum(p.numel() for p in self.parameters())
        return {
            "number_of_units": 64,
            "unit_size_vector": self.unit_size_vector,
            "input_linear_shapes": [[256, size] for size in self.unit_size_vector],
            "unit_specific_first_layer_parameters": first_layers,
            "shared_projection_parameters": shared,
            "total_parameters": total,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
