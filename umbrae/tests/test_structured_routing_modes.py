import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.umbrae_neuroroute_adapter import UMBRAENeuroRouteAdapter
from scripts.train_umbrae_neuroroute import (
    build_effective_routing,
    run_training,
)


def main():
    rng = np.random.default_rng(42)
    soft = rng.random((8, 6))
    soft /= soft.sum(axis=-1, keepdims=True)
    uniform = np.full((8, 6), 1 / 6)

    alpha_zero = build_effective_routing(
        soft, "umbrae_plus_uniform_residual_soft", 0.0, 2.0
    )
    alpha_one = build_effective_routing(
        soft, "umbrae_plus_uniform_residual_soft", 1.0, 2.0
    )
    alpha_three = build_effective_routing(
        soft, "umbrae_plus_uniform_residual_soft", 0.3, 2.0
    )
    temp = build_effective_routing(
        soft, "umbrae_plus_temperature_soft", 0.3, 2.0
    )
    np.testing.assert_allclose(alpha_zero, uniform, atol=1e-7)
    np.testing.assert_allclose(alpha_one, soft, atol=1e-7)
    np.testing.assert_allclose(alpha_three.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(temp.sum(axis=-1), 1.0, atol=1e-6)
    assert np.isfinite(alpha_three).all()
    assert np.isfinite(temp).all()

    umbrae = torch.randn(1, 256, 32)
    roi = torch.randn(1, 8, 32)
    for mode, routing in (
        ("umbrae_plus_uniform_residual_soft", alpha_three),
        ("umbrae_plus_temperature_soft", temp),
    ):
        adapter = UMBRAENeuroRouteAdapter(
            umbrae_dim=32,
            neuroroute_dim=32,
            mllm_dim=4096,
            fusion_dim=32,
            num_visual_tokens=256,
            fusion_mode=mode,
            num_attention_heads=8,
            perceiver_depth=1,
            num_routing_layers=6,
        )
        output = adapter(
            umbrae,
            roi,
            routing_weights=torch.from_numpy(routing).unsqueeze(0),
        )
        assert output["visual_tokens"].shape == (1, 256, 4096)
        assert torch.isfinite(output["visual_tokens"]).all()

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        for filename in (
            "checkpoint_last.pt",
            "metrics.json",
            "generated_captions.jsonl",
            "adapter_config.json",
        ):
            (output / filename).write_text(
                json.dumps({
                    "uses_image_clip_tokens_at_eval": False,
                    "oracle_image_token_mode": False,
                    "bridge_type": "shikra_patch",
                })
            )
        result = run_training(
            Namespace(
                output_dir=str(output),
                overwrite=False,
                resume_if_incomplete=False,
                seed=42,
            )
        )
        assert result["skipped"] is True
        config = json.loads((output / "adapter_config.json").read_text())
        assert config["uses_image_clip_tokens_at_eval"] is False
        assert config["oracle_image_token_mode"] is False
        assert config["bridge_type"] == "shikra_patch"
    print("Structured routing modes and completed-run protection passed")


if __name__ == "__main__":
    main()
