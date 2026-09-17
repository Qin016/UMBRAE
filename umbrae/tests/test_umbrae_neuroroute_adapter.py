import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.umbrae_neuroroute_adapter import UMBRAENeuroRouteAdapter


def main() -> None:
    umbrae = torch.randn(2, 256, 64)
    roi = torch.randn(2, 8, 32)
    routing = torch.softmax(torch.randn(2, 8, 6), dim=-1)
    reliability = torch.rand(2, 8)
    for mode in UMBRAENeuroRouteAdapter.SUPPORTED_MODES:
        for fusion_type in UMBRAENeuroRouteAdapter.SUPPORTED_FUSIONS:
            if mode.endswith("prior") and fusion_type == "concat_then_project":
                continue
            adapter = UMBRAENeuroRouteAdapter(
                umbrae_dim=64,
                neuroroute_dim=32,
                mllm_dim=128,
                num_visual_tokens=256,
                fusion_mode=mode,
                fusion_type=fusion_type,
                fusion_dim=64,
                roi_token_expansion=4,
                num_attention_heads=8,
                perceiver_depth=1,
                fgw_transport_plan=(
                    torch.full((8, 6), 1.0 / 48.0)
                    if mode in {"fgw_prior", "row_shuffled_fgw_prior"}
                    else None
                ),
            )
            output = adapter(
                None if mode == "neuroroute_only" else umbrae,
                None if mode == "umbrae_only" else roi,
                routing_weights=(
                    routing
                    if mode
                    in {
                        "umbrae_plus_uniform_residual_soft",
                        "umbrae_plus_temperature_soft",
                        "uniform_prior",
                        "fgw_prior",
                        "row_shuffled_fgw_prior",
                    }
                    else routing
                ),
                reliability=(
                    None if mode == "umbrae_only" else reliability
                ),
            )
            assert output["visual_tokens"].shape == (2, 256, 128)
            assert torch.isfinite(output["visual_tokens"]).all()
            assert output["diagnostics"]["num_visual_tokens"] == 256
            if mode != "umbrae_only":
                expected = (
                    32 * 6
                    if mode
                    in {
                        "umbrae_plus_uniform_residual_soft",
                        "umbrae_plus_temperature_soft",
                        "uniform_prior",
                        "fgw_prior",
                        "row_shuffled_fgw_prior",
                    }
                    else 32
                )
                assert output["diagnostics"][
                    "num_expanded_roi_tokens"
                ] == expected
            output["visual_tokens"].mean().backward()
            assert any(
                parameter.grad is not None
                for parameter in adapter.parameters()
            )
    print("All UMBRAE-NeuroRoute adapter fusion modes passed")


if __name__ == "__main__":
    main()
