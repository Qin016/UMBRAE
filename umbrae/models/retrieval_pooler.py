"""Pooling heads for Stage-1 global retrieval embeddings."""

import torch
from torch import Tensor, nn


class RetrievalPooler(nn.Module):
    """Pool ROI tokens ``[B,R,D]`` into one embedding ``[B,D]``."""

    SUPPORTED_MODES = ("mean", "attention", "roi_flatten_mlp")

    def __init__(self, mode: str, num_rois: int, feature_dim: int) -> None:
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported pooling mode: {mode}")
        self.mode = mode
        self.num_rois = int(num_rois)
        self.feature_dim = int(feature_dim)
        self.attention_score = (
            nn.Linear(feature_dim, 1, bias=False)
            if mode == "attention"
            else None
        )
        self.flatten_mlp = (
            nn.Sequential(
                nn.LayerNorm(num_rois * feature_dim),
                nn.Linear(num_rois * feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )
            if mode == "roi_flatten_mlp"
            else None
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,R,D]")
        if tokens.shape[1:] != (self.num_rois, self.feature_dim):
            raise ValueError(
                f"Expected [B,{self.num_rois},{self.feature_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        if self.mode == "mean":
            return tokens.mean(dim=1)
        if self.mode == "attention":
            weights = torch.softmax(self.attention_score(tokens), dim=1)
            return (weights * tokens).sum(dim=1)
        return self.flatten_mlp(tokens.flatten(start_dim=1))
