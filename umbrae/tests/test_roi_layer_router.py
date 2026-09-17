import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.routing_loss import routing_regularization_loss
from models.roi_layer_router import ROILayerRouter


def run_device(device: torch.device) -> None:
    torch.manual_seed(7)
    roi_tokens = torch.randn(2, 8, 64, device=device)
    clip_layer_features = torch.randn(2, 6, 64, device=device)
    router = ROILayerRouter(
        feature_dim=64,
        hidden_dim=32,
        temperature=0.7,
    ).to(device)

    output = router(roi_tokens, clip_layer_features)
    routed_targets = output["routed_targets"]
    routing_weights = output["routing_weights"]
    assert routed_targets.shape == (2, 8, 64)
    assert routing_weights.shape == (2, 8, 6)
    assert torch.allclose(
        routing_weights.sum(dim=-1),
        torch.ones(2, 8, device=device),
        atol=1e-6,
    )

    losses = routing_regularization_loss(
        routing_weights,
        entropy_weight=0.01,
        balance_weight=0.02,
        smoothness_weight=0.03,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    (routed_targets.mean() + losses["total"]).backward()
    assert any(parameter.grad is not None for parameter in router.parameters())

    topk_output = router(roi_tokens, clip_layer_features, topk=2)
    topk_weights = topk_output["routing_weights"]
    assert topk_weights.shape == (2, 8, 6)
    assert torch.all((topk_weights > 0).sum(dim=-1) == 2)
    assert torch.allclose(
        topk_weights.sum(dim=-1),
        torch.ones(2, 8, device=device),
        atol=1e-6,
    )
    print(
        f"{device.type}: routed={tuple(routed_targets.shape)}, "
        f"weights={tuple(routing_weights.shape)}, topk=2"
    )


def main() -> None:
    run_device(torch.device("cpu"))
    if torch.cuda.is_available():
        run_device(torch.device("cuda"))
    else:
        print("cuda: unavailable, compatibility test skipped")
    print("ROILayerRouter smoke test passed")


if __name__ == "__main__":
    main()
