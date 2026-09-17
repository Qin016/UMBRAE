import sys
from pathlib import Path

import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.clip_layer_bank import CLIPLayerBank


def main() -> None:
    selected_layers = [4, 8, 12, 16, 20, 24]
    tiny_clip = CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=24,
            num_attention_heads=4,
            image_size=32,
            patch_size=2,
        )
    )
    model = CLIPLayerBank(
        selected_layers=selected_layers,
        target_dim=16,
        freeze_clip=True,
        clip_model=tiny_clip,
    ).eval()

    images = torch.rand(2, 3, 32, 32)
    outputs = model(images)

    assert outputs["layer_tokens"].shape == (2, 6, 256, 16)
    assert outputs["pooled_tokens"].shape == (2, 6, 16)
    assert outputs["selected_layers"] == selected_layers
    assert not any(parameter.requires_grad for parameter in model.clip.parameters())
    assert all(
        parameter.requires_grad for parameter in model.projection_heads.parameters()
    )
    outputs["layer_tokens"].mean().backward()
    assert all(parameter.grad is None for parameter in model.clip.parameters())
    assert any(
        parameter.grad is not None for parameter in model.projection_heads.parameters()
    )
    print("CLIPLayerBank smoke test passed")
    print("layer_tokens:", tuple(outputs["layer_tokens"].shape))
    print("pooled_tokens:", tuple(outputs["pooled_tokens"].shape))
    print("selected_layers:", outputs["selected_layers"])


if __name__ == "__main__":
    main()
