"""Reusable multi-layer feature extraction for frozen CLIP vision encoders."""

from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn
from torchvision import transforms
from transformers import CLIPVisionModel


class CLIPLayerBank(nn.Module):
    """Extract and project patch tokens from selected CLIP vision layers.

    Layer indices are one-based Transformer block indices. For example,
    ``[4, 8, 12, 16, 20, 24]`` selects six blocks from CLIP ViT-L/14.

    The returned patch-token tensor excludes the CLS token so its token count
    stays compatible with UMBRAE's existing 16x16 patch grid (T=256).
    """

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-large-patch14",
        selected_layers: Sequence[int] = (4, 8, 12, 16, 20, 24),
        target_dim: int = 1024,
        freeze_clip: bool = True,
        clip_model: Optional[CLIPVisionModel] = None,
    ) -> None:
        super().__init__()
        self.clip = clip_model or CLIPVisionModel.from_pretrained(model_name_or_path)
        self.selected_layers = self._validate_layers(selected_layers)
        self.target_dim = target_dim
        self.freeze_clip = freeze_clip

        hidden_dim = self.clip.config.hidden_size
        self.projection_heads = nn.ModuleDict(
            {
                str(layer): self._make_projection(hidden_dim, target_dim)
                for layer in self.selected_layers
            }
        )

        image_size = self.clip.config.image_size
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(image_size),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        if self.freeze_clip:
            self.clip.requires_grad_(False)
            self.clip.eval()

    def _validate_layers(self, selected_layers: Sequence[int]) -> List[int]:
        layers = list(selected_layers)
        if not layers:
            raise ValueError("selected_layers must contain at least one layer")
        if len(set(layers)) != len(layers):
            raise ValueError(f"selected_layers contains duplicates: {layers}")

        num_layers = self.clip.config.num_hidden_layers
        invalid = [layer for layer in layers if layer < 1 or layer > num_layers]
        if invalid:
            raise ValueError(
                f"Invalid CLIP layer indices {invalid}; expected values in [1, {num_layers}]"
            )
        return layers

    @staticmethod
    def _make_projection(input_dim: int, target_dim: int) -> nn.Linear:
        projection = nn.Linear(input_dim, target_dim)
        if input_dim == target_dim:
            nn.init.eye_(projection.weight)
            nn.init.zeros_(projection.bias)
        return projection

    def train(self, mode: bool = True) -> "CLIPLayerBank":
        super().train(mode)
        if self.freeze_clip:
            self.clip.eval()
        return self

    def forward(self, images: Tensor) -> Dict[str, object]:
        """Return projected tokens for all configured layers.

        Args:
            images: Float tensor ``[B, 3, H, W]`` in the ``[0, 1]`` range.

        Returns:
            ``layer_tokens``: ``[B, L, T, target_dim]``.
            ``pooled_tokens``: ``[B, L, target_dim]`` (mean over T).
            ``selected_layers``: the configured one-based layer indices.
        """
        pixel_values = self.preprocess(images)
        clip_context = torch.no_grad() if self.freeze_clip else nullcontext()
        with clip_context:
            outputs = self.clip(pixel_values=pixel_values, output_hidden_states=True)

        projected_layers = []
        for layer in self.selected_layers:
            # hidden_states[0] is the embedding output; index N is block N.
            patch_tokens = outputs.hidden_states[layer][:, 1:, :]
            projected_layers.append(self.projection_heads[str(layer)](patch_tokens))

        layer_tokens = torch.stack(projected_layers, dim=1)
        pooled_tokens = layer_tokens.mean(dim=2)
        return {
            "layer_tokens": layer_tokens,
            "pooled_tokens": pooled_tokens,
            "selected_layers": list(self.selected_layers),
        }

    def aggregate_layer_tokens(self, images: Tensor) -> Tensor:
        """Return layer-mean patch tokens ``[B, T, target_dim]`` for training."""
        return self(images)["layer_tokens"].mean(dim=1)
