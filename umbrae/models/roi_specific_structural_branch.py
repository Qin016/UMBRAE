"""ROI-specific lightweight structural encoder for P3-R."""

from typing import Dict, Mapping, Sequence

import torch
from torch import Tensor, nn

from .roi_mapping import MVP_ROI_NAMES


class ROIProjector(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )


class ROISpecificStructuralBranch(nn.Module):
    """Map each true-size ROI vector through its own projector."""

    def __init__(
        self,
        roi_indices: Mapping[str, Sequence[int]],
        hidden_dim: int = 256,
        token_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.roi_names = list(MVP_ROI_NAMES)
        missing = [name for name in self.roi_names if name not in roi_indices]
        extra = [name for name in roi_indices if name not in self.roi_names]
        if missing or extra:
            raise ValueError(f"ROI set mismatch; missing={missing}, extra={extra}")
        self.hidden_dim = int(hidden_dim)
        self.token_dim = int(token_dim)
        self.projectors = nn.ModuleDict()
        for position, name in enumerate(self.roi_names):
            index = torch.as_tensor(roi_indices[name], dtype=torch.long)
            if index.ndim != 1 or index.numel() == 0 or index.unique().numel() != index.numel():
                raise ValueError(f"Invalid indices for ROI {name}")
            self.register_buffer(f"_roi_index_{position}", index, persistent=True)
            self.projectors[name] = ROIProjector(
                int(index.numel()), self.hidden_dim, self.token_dim
            )

    def forward(self, fmri: Tensor) -> Dict[str, object]:
        if fmri.ndim != 2:
            raise ValueError(f"Expected preaggregated fMRI [B,V], got {tuple(fmri.shape)}")
        tokens = []
        for position, name in enumerate(self.roi_names):
            index = getattr(self, f"_roi_index_{position}")
            if int(index.max()) >= fmri.shape[1]:
                raise ValueError(f"ROI {name} index exceeds fMRI dimension")
            tokens.append(self.projectors[name](fmri.index_select(1, index)))
        h_struct = torch.stack(tokens, dim=1)
        return {"h_struct": h_struct, "roi_names": list(self.roi_names)}

    def parameter_report(self) -> Dict[str, object]:
        per_roi = {
            name: sum(parameter.numel() for parameter in self.projectors[name].parameters())
            for name in self.roi_names
        }
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"total": total, "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad), "per_roi": per_roi}


class StructuralAttentionPool(nn.Module):
    """Single learned-query attention pool; no Transformer block."""

    def __init__(self, token_dim: int = 1024) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(token_dim))
        nn.init.normal_(self.query, std=token_dim ** -0.5)

    def forward(self, tokens: Tensor) -> Dict[str, Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.query.numel():
            raise ValueError("tokens must have shape [B,R,D] matching the query")
        scores = torch.einsum("brd,d->br", tokens, self.query) / self.query.numel() ** 0.5
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("br,brd->bd", weights, tokens)
        return {"pooled": pooled, "attention_weights": weights}
