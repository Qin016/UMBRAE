"""Frozen single-layer CLIP patch teacher matching original UMBRAE exactly."""

from typing import Optional

import torch
from torch import Tensor, nn
from torchvision import transforms
from transformers import CLIPVisionModel


class FixedCLIPPatchTeacher(nn.Module):
    """Return ``hidden_states[-2][:, 1:, :]`` from CLIP ViT-L/14."""

    MODEL_NAME = "openai/clip-vit-large-patch14"
    LAYER_DEFINITION = "vision_encoder.hidden_states[-2][:,1:,:]"

    def __init__(
        self,
        model_name_or_path: str = MODEL_NAME,
        clip_model: Optional[CLIPVisionModel] = None,
    ) -> None:
        super().__init__()
        self.clip = clip_model or CLIPVisionModel.from_pretrained(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        image_size = int(self.clip.config.image_size)
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
        self.hidden_dim = int(self.clip.config.hidden_size)
        patch_size = int(self.clip.config.patch_size)
        self.num_patch_tokens = (image_size // patch_size) ** 2
        self.clip.requires_grad_(False)
        self.clip.eval()

    def train(self, mode: bool = True) -> "FixedCLIPPatchTeacher":
        super().train(False)
        self.clip.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"images must have shape [B,3,H,W], got {tuple(images.shape)}"
            )
        with torch.no_grad():
            pixels = self.preprocess(images)
            patch_grid = self.clip.vision_model.embeddings.patch_embedding(pixels)
            tokens = patch_grid.flatten(2).transpose(1, 2)
            class_token = self.clip.vision_model.embeddings.class_embedding.to(
                device=tokens.device, dtype=tokens.dtype
            )
            class_token = class_token.expand(tokens.shape[0], 1, -1)
            tokens = torch.cat([class_token, tokens], dim=1)
            position_ids = torch.arange(
                tokens.shape[1], device=tokens.device, dtype=torch.long
            ).unsqueeze(0)
            tokens = tokens + self.clip.vision_model.embeddings.position_embedding(
                position_ids
            )
            tokens = self.clip.vision_model.pre_layrnorm(tokens)
            encoded = self.clip.vision_model.encoder(
                tokens, output_hidden_states=True
            )
            patch_tokens = encoded.hidden_states[-2][:, 1:, :]
        expected = (images.shape[0], self.num_patch_tokens, self.hidden_dim)
        if tuple(patch_tokens.shape) != expected:
            raise RuntimeError(
                f"Unexpected CLIP patch shape {tuple(patch_tokens.shape)}; expected {expected}"
            )
        return patch_tokens
