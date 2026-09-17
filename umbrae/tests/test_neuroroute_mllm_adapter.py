import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.global_l24_projector import GlobalL24Projector
from models.neuroroute_mllm_adapter import NeuroRouteMLLMAdapter


def main() -> None:
    global_token = torch.randn(2, 64)
    roi_tokens = torch.randn(2, 8, 64)
    expected_lengths = {
        "l24_only": 1,
        "routed_only": 8,
        "concat": 9,
        "gated_fusion": 8,
        "cross_attention": 2,
    }
    for mode, token_count in expected_lengths.items():
        adapter = NeuroRouteMLLMAdapter(
            input_dim=64,
            mllm_dim=128,
            fusion_mode=mode,
            num_attention_heads=8,
        )
        output = adapter(global_token, roi_tokens)
        assert output["visual_prefix_tokens"].shape == (
            2,
            token_count,
            128,
        )
        assert torch.isfinite(output["visual_prefix_tokens"]).all()
        output["visual_prefix_tokens"].mean().backward()
        assert any(
            parameter.grad is not None for parameter in adapter.parameters()
        )
        print(mode, tuple(output["visual_prefix_tokens"].shape))

    projector = GlobalL24Projector(64, pooling="attention")
    assert projector(roi_tokens).shape == (2, 64)
    assert projector(roi_tokens.mean(dim=1)).shape == (2, 64)
    print("NeuroRouteMLLMAdapter smoke test passed")


if __name__ == "__main__":
    main()
