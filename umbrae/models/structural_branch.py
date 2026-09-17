"""Minimal 8-ROI structural branch for the dual-branch UMBRAE MVP."""

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

from torch import Tensor, nn

from .roi_mapping import MVP_ROI_NAMES, load_roi_indices
from .roi_tokenizer import ROITokenizer


class StructuralBranch(nn.Module):
    """ROI mapping -> existing ROITokenizer -> optional LayerNorm."""

    def __init__(
        self,
        *,
        token_dim: int = 1024,
        mapping_path: Optional[Union[str, Path]] = None,
        roi_indices: Optional[Mapping[str, Sequence[int]]] = None,
        subject: Optional[object] = None,
        expected_voxel_count: Optional[int] = None,
        tokenizer_type: str = "shared_mlp",
        use_output_norm: bool = True,
    ) -> None:
        super().__init__()
        if (mapping_path is None) == (roi_indices is None):
            raise ValueError("Provide exactly one of mapping_path or roi_indices")
        self.roi_names = list(MVP_ROI_NAMES)
        if mapping_path is not None:
            roi_indices, self.mapping_summary = load_roi_indices(
                str(mapping_path),
                expected_voxel_count=expected_voxel_count,
                expected_subject=subject,
                roi_order=self.roi_names,
                strict=True,
            )
        else:
            missing = [name for name in self.roi_names if name not in roi_indices]
            extra = [name for name in roi_indices if name not in self.roi_names]
            if missing or extra:
                raise ValueError(f"MVP ROI set mismatch; missing={missing}, extra={extra}")
            roi_indices = {name: roi_indices[name] for name in self.roi_names}
            self.mapping_summary = None
        self.token_dim = int(token_dim)
        self.roi_tokenizer = ROITokenizer(
            roi_names=self.roi_names,
            roi_indices=roi_indices,
            token_dim=self.token_dim,
            tokenizer_type=tokenizer_type,
            use_roi_embeddings=True,
        )
        self.output_norm = nn.LayerNorm(self.token_dim) if use_output_norm else nn.Identity()

    def forward(self, fmri: Tensor) -> Dict[str, object]:
        output = self.roi_tokenizer(fmri)
        h_struct = self.output_norm(output["roi_tokens"])
        expected = (fmri.shape[0], len(self.roi_names), self.token_dim)
        if tuple(h_struct.shape) != expected:
            raise RuntimeError(
                f"Unexpected structural shape {tuple(h_struct.shape)}; expected {expected}"
            )
        return {"h_struct": h_struct, "roi_names": list(self.roi_names)}

    def parameter_report(self) -> Dict[str, object]:
        def counts(module: nn.Module) -> Dict[str, int]:
            return {
                "total": sum(parameter.numel() for parameter in module.parameters()),
                "trainable": sum(
                    parameter.numel()
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ),
            }

        return {
            "total": counts(self)["total"],
            "trainable": counts(self)["trainable"],
            "roi_tokenizer": counts(self.roi_tokenizer),
            "output_norm": counts(self.output_norm),
        }
