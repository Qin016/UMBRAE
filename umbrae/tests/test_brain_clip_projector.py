import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.brain_clip_projector import BrainToCLIPProjector


def main() -> None:
    inputs = torch.randn(2, 8, 64)
    for projector_type in ("linear", "mlp", "residual_mlp"):
        projector = BrainToCLIPProjector(
            input_dim=64,
            output_dim=64,
            projector_type=projector_type,
            hidden_dim=64,
            dropout=0.1,
        )
        outputs = projector(inputs)
        assert outputs.shape == (2, 8, 64)
        outputs.mean().backward()
        assert any(
            parameter.grad is not None for parameter in projector.parameters()
        )
        print(projector_type, tuple(outputs.shape))

    cross_dimensional = BrainToCLIPProjector(
        input_dim=32,
        output_dim=64,
        projector_type="mlp",
    )
    assert cross_dimensional(torch.randn(2, 8, 32)).shape == (2, 8, 64)
    print("BrainToCLIPProjector smoke test passed")


if __name__ == "__main__":
    main()
