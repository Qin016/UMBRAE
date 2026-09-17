"""fMRI-derived global semantic token for the Stage-2 L24 pathway."""

from typing import Optional

from torch import Tensor, nn


class GlobalL24Projector(nn.Module):
    """Pool ROI tokens and predict one global CLIP-L24-style token."""

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        pooling: str = "attention",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        output_dim = int(output_dim or input_dim)
        hidden_dim = int(hidden_dim or input_dim)
        if pooling not in ("mean", "attention"):
            raise ValueError("pooling must be 'mean' or 'attention'")
        self.input_dim = int(input_dim)
        self.output_dim = output_dim
        self.pooling = pooling
        self.input_norm = nn.LayerNorm(input_dim)
        self.attention_score = (
            nn.Linear(input_dim, 1, bias=False)
            if pooling == "attention"
            else None
        )
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, roi_tokens: Tensor) -> Tensor:
        if roi_tokens.ndim == 2:
            pooled = roi_tokens
        elif roi_tokens.ndim == 3:
            normalized = self.input_norm(roi_tokens)
            if self.pooling == "mean":
                pooled = normalized.mean(dim=1)
            else:
                weights = self.attention_score(normalized).softmax(dim=1)
                pooled = (weights * normalized).sum(dim=1)
        else:
            raise ValueError(
                "roi_tokens must have shape [B,D] or [B,R,D], "
                f"got {tuple(roi_tokens.shape)}"
            )
        if pooled.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {pooled.shape[-1]}"
            )
        return self.projector(pooled)
