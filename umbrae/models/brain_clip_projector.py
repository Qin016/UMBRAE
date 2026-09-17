"""Projection heads from ROI-token space into CLIP feature space."""

from torch import Tensor, nn


class BrainToCLIPProjector(nn.Module):
    """Project ROI tokens ``[B, R, D_in]`` to ``[B, R, D_out]``."""

    SUPPORTED_TYPES = ("linear", "mlp", "residual_mlp")

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        projector_type: str = "mlp",
        hidden_dim: int = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if projector_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported projector_type={projector_type!r}; "
                f"expected one of {self.SUPPORTED_TYPES}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        hidden_dim = int(hidden_dim or input_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if projector_type == "residual_mlp" and input_dim != output_dim:
            raise ValueError(
                "residual_mlp requires input_dim == output_dim"
            )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.projector_type = projector_type
        self.input_norm = (
            nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        )

        if projector_type == "linear":
            self.projector = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Dropout(dropout),
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(dropout),
            )

    def forward(self, roi_tokens: Tensor) -> Tensor:
        if roi_tokens.ndim != 3:
            raise ValueError(
                "roi_tokens must have shape [B, R, D], "
                f"got {tuple(roi_tokens.shape)}"
            )
        if roi_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {roi_tokens.shape[-1]}"
            )
        projected = self.projector(self.input_norm(roi_tokens))
        if self.projector_type == "residual_mlp":
            projected = roi_tokens + projected
        return projected
