"""ROI-wise tokenization for flattened fMRI voxel vectors."""

from typing import Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor, nn


ROIIndices = Union[Mapping[str, Sequence[int]], Sequence[Sequence[int]]]


class ROITokenizer(nn.Module):
    """Convert ROI-specific voxel groups into a stable sequence of tokens.

    The tokenizer accepts either flattened fMRI tensors ``[B, V]`` or the
    repository's repeated-trial format ``[B, repeats, V]``. Repeated trials are
    averaged before ROI extraction.
    """

    SUPPORTED_TYPES = ("shared_mlp", "roi_specific_mlp")

    def __init__(
        self,
        roi_names: Sequence[str],
        token_dim: int,
        roi_indices: Optional[ROIIndices] = None,
        roi_masks: Optional[Tensor] = None,
        tokenizer_type: str = "shared_mlp",
        use_roi_embeddings: bool = True,
        use_subject_embeddings: bool = False,
        num_subjects: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.roi_names = list(roi_names)
        self.token_dim = token_dim
        self.tokenizer_type = tokenizer_type
        self.use_roi_embeddings = use_roi_embeddings
        self.use_subject_embeddings = use_subject_embeddings

        if not self.roi_names:
            raise ValueError("roi_names must contain at least one ROI")
        if len(set(self.roi_names)) != len(self.roi_names):
            raise ValueError(f"roi_names contains duplicates: {self.roi_names}")
        if tokenizer_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported tokenizer_type={tokenizer_type!r}; "
                f"expected one of {self.SUPPORTED_TYPES}"
            )
        if (roi_indices is None) == (roi_masks is None):
            raise ValueError("Provide exactly one of roi_indices or roi_masks")

        indices = self._normalize_roi_indices(roi_indices, roi_masks)
        self.roi_sizes = [int(index.numel()) for index in indices]
        self.max_roi_size = max(self.roi_sizes)
        for roi_idx, index in enumerate(indices):
            self.register_buffer(f"_roi_index_{roi_idx}", index, persistent=True)

        mlp_hidden_dim = hidden_dim or token_dim
        if tokenizer_type == "shared_mlp":
            self.shared_mlp = self._make_mlp(
                self.max_roi_size, mlp_hidden_dim, token_dim
            )
            self.roi_mlps = None
        else:
            self.shared_mlp = None
            self.roi_mlps = nn.ModuleList(
                [
                    self._make_mlp(roi_size, mlp_hidden_dim, token_dim)
                    for roi_size in self.roi_sizes
                ]
            )

        self.roi_embeddings = (
            nn.Embedding(len(self.roi_names), token_dim)
            if use_roi_embeddings
            else None
        )

        if use_subject_embeddings:
            if num_subjects is None or num_subjects <= 0:
                raise ValueError(
                    "num_subjects must be a positive integer when "
                    "use_subject_embeddings=True"
                )
            self.subject_embeddings = nn.Embedding(num_subjects, token_dim)
        else:
            self.subject_embeddings = None

    @staticmethod
    def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _normalize_roi_indices(
        self,
        roi_indices: Optional[ROIIndices],
        roi_masks: Optional[Tensor],
    ) -> List[Tensor]:
        if roi_masks is not None:
            masks = torch.as_tensor(roi_masks, dtype=torch.bool)
            if masks.ndim != 2 or masks.shape[0] != len(self.roi_names):
                raise ValueError(
                    "roi_masks must have shape [R, V], where R matches roi_names"
                )
            indices = [
                torch.nonzero(mask, as_tuple=False).flatten() for mask in masks
            ]
        elif isinstance(roi_indices, Mapping):
            missing = [name for name in self.roi_names if name not in roi_indices]
            if missing:
                raise ValueError(f"Missing ROI indices for: {missing}")
            indices = [
                torch.as_tensor(roi_indices[name], dtype=torch.long)
                for name in self.roi_names
            ]
        else:
            if roi_indices is None or len(roi_indices) != len(self.roi_names):
                raise ValueError("roi_indices must contain one index list per ROI")
            indices = [
                torch.as_tensor(index, dtype=torch.long) for index in roi_indices
            ]

        for name, index in zip(self.roi_names, indices):
            if index.ndim != 1 or index.numel() == 0:
                raise ValueError(f"ROI {name!r} must have a non-empty 1D index list")
            if torch.any(index < 0):
                raise ValueError(f"ROI {name!r} contains negative voxel indices")
            if torch.unique(index).numel() != index.numel():
                raise ValueError(f"ROI {name!r} contains duplicate voxel indices")
        return indices

    def _get_roi_index(self, roi_idx: int) -> Tensor:
        return getattr(self, f"_roi_index_{roi_idx}")

    def _prepare_fmri(self, fmri: Tensor) -> Tensor:
        if fmri.ndim == 3:
            fmri = fmri.mean(dim=1)
        if fmri.ndim != 2:
            raise ValueError(
                f"fmri must have shape [B, V] or [B, repeats, V], got {tuple(fmri.shape)}"
            )

        max_index = max(
            int(self._get_roi_index(i).max()) for i in range(len(self.roi_names))
        )
        if max_index >= fmri.shape[-1]:
            raise ValueError(
                f"ROI index {max_index} exceeds fMRI voxel dimension {fmri.shape[-1]}"
            )
        return fmri

    def forward(
        self,
        fmri: Tensor,
        subject_ids: Optional[Tensor] = None,
    ) -> Dict[str, object]:
        """Return ROI tokens ``[B, R, D]`` and their stable ROI-name order."""
        fmri = self._prepare_fmri(fmri)

        roi_values = [
            fmri.index_select(dim=-1, index=self._get_roi_index(roi_idx))
            for roi_idx in range(len(self.roi_names))
        ]

        if self.tokenizer_type == "shared_mlp":
            padded_values = fmri.new_zeros(
                fmri.shape[0], len(self.roi_names), self.max_roi_size
            )
            for roi_idx, values in enumerate(roi_values):
                padded_values[:, roi_idx, : values.shape[-1]] = values
            roi_tokens = self.shared_mlp(padded_values)
        else:
            roi_tokens = torch.stack(
                [
                    roi_mlp(values)
                    for roi_mlp, values in zip(self.roi_mlps, roi_values)
                ],
                dim=1,
            )

        if self.roi_embeddings is not None:
            roi_ids = torch.arange(
                len(self.roi_names), device=fmri.device, dtype=torch.long
            )
            roi_tokens = roi_tokens + self.roi_embeddings(roi_ids).unsqueeze(0)

        if self.subject_embeddings is not None:
            if subject_ids is None:
                raise ValueError(
                    "subject_ids is required when use_subject_embeddings=True"
                )
            subject_ids = torch.as_tensor(
                subject_ids, device=fmri.device, dtype=torch.long
            )
            if subject_ids.ndim == 0:
                subject_ids = subject_ids.expand(fmri.shape[0])
            if subject_ids.shape != (fmri.shape[0],):
                raise ValueError(
                    f"subject_ids must have shape [B], got {tuple(subject_ids.shape)}"
                )
            roi_tokens = roi_tokens + self.subject_embeddings(subject_ids).unsqueeze(1)

        return {
            "roi_tokens": roi_tokens,
            "roi_names": list(self.roi_names),
        }
