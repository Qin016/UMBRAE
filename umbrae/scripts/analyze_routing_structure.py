#!/usr/bin/env python
"""Analyze Stage-1 routing matrices used by UMBRAE-NeuroRoute runs."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-png", action="store_true")
    return parser.parse_args()


def load_routing(run_dir: Path):
    config = json.loads((run_dir / "adapter_config.json").read_text())
    summary_path = run_dir / "routing_summary.json"
    summary = (
        json.loads(summary_path.read_text())
        if summary_path.is_file()
        else {}
    )
    matrix = summary.get("routing_matrix")
    if matrix is None:
        checkpoint = Path(config["neuroroute_checkpoint"])
        matrix_path = checkpoint.parent / "val_routing_weights_mean.npy"
        if not matrix_path.is_file():
            raise FileNotFoundError(matrix_path)
        matrix = np.load(matrix_path, allow_pickle=False)
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.clip(matrix, 0.0, None)
    matrix /= np.maximum(matrix.sum(axis=-1, keepdims=True), 1e-12)
    roi_names = list(config.get("roi_names", []))
    if not roi_names:
        roi_mapping = json.loads(
            Path(config["roi_indices_path"]).read_text()
        )
        roi_names = list(roi_mapping["roi_names"])
    layers = summary.get("selected_clip_layers")
    if not layers:
        checkpoint = Path(config["neuroroute_checkpoint"])
        import torch

        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        layers = payload.get("config", {}).get(
            "selected_clip_layers", list(range(matrix.shape[-1]))
        )
    return config, matrix, roi_names, np.asarray(layers, dtype=np.float64)


def main():
    args = parse_args()
    run_dir = Path(args.run).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config, matrix, roi_names, layers = load_routing(run_dir)
    if matrix.shape != (len(roi_names), len(layers)):
        raise ValueError(
            f"Routing shape {matrix.shape} does not match "
            f"{len(roi_names)} ROIs and {len(layers)} layers"
        )

    entropy = -(matrix * np.log(matrix + 1e-12)).sum(axis=-1)
    normalized_entropy = entropy / np.log(matrix.shape[-1])
    max_weight = matrix.max(axis=-1)
    expected_depth = matrix @ layers
    layer_usage = matrix.mean(axis=0)
    early_names = {"V1", "V2", "V3", "hV4"}
    high_names = {"FFA", "EBA", "PPA", "OPA"}
    early = [
        expected_depth[index]
        for index, name in enumerate(roi_names)
        if name in early_names
    ]
    high = [
        expected_depth[index]
        for index, name in enumerate(roi_names)
        if name in high_names
    ]
    pairwise_tv = [
        0.5 * np.abs(matrix[left] - matrix[right]).sum()
        for left, right in combinations(range(len(roi_names)), 2)
    ]
    # 1 means all layers receive equal aggregate usage; 1/L means one-layer
    # collapse. ROI diversity is mean pairwise total-variation distance.
    usage_entropy = -np.sum(layer_usage * np.log(layer_usage + 1e-12))
    coverage = float(np.exp(usage_entropy) / matrix.shape[-1])
    diversity = float(np.mean(pairwise_tv)) if pairwise_tv else 0.0

    payload = {
        "run": str(run_dir),
        "fusion_mode": config.get("fusion_mode"),
        "structured_routing_alpha": config.get(
            "structured_routing_alpha"
        ),
        "routing_temperature": config.get("routing_temperature"),
        "selected_clip_layers": layers.astype(int).tolist(),
        "roi_names": roi_names,
        "mean_layer_usage": {
            str(int(layer)): float(value)
            for layer, value in zip(layers, layer_usage)
        },
        "per_roi_layer_usage": {
            name: {
                str(int(layer)): float(value)
                for layer, value in zip(layers, row)
            }
            for name, row in zip(roi_names, matrix)
        },
        "routing_entropy_per_roi": {
            name: float(value)
            for name, value in zip(roi_names, entropy)
        },
        "normalized_entropy_per_roi": {
            name: float(value)
            for name, value in zip(roi_names, normalized_entropy)
        },
        "average_entropy": float(entropy.mean()),
        "average_normalized_entropy": float(normalized_entropy.mean()),
        "max_layer_weight_per_roi": {
            name: float(value)
            for name, value in zip(roi_names, max_weight)
        },
        "expected_clip_depth_per_roi": {
            name: float(value)
            for name, value in zip(roi_names, expected_depth)
        },
        "early_roi_expected_depth": float(np.mean(early)),
        "high_level_roi_expected_depth": float(np.mean(high)),
        "early_vs_high_depth_gap": float(np.mean(high) - np.mean(early)),
        "layer_coverage_score": coverage,
        "roi_diversity_score": diversity,
    }
    (output_dir / "routing_structure.json").write_text(
        json.dumps(payload, indent=2)
    )

    with (output_dir / "routing_matrix.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["roi", *[f"L{int(layer)}" for layer in layers]])
        for name, row in zip(roi_names, matrix):
            writer.writerow([name, *row.tolist()])
    for filename, label, values in (
        ("entropy_by_roi.csv", "entropy", entropy),
        ("depth_by_roi.csv", "expected_depth", expected_depth),
    ):
        with (output_dir / filename).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["roi", label])
            writer.writerows(zip(roi_names, values.tolist()))

    if not args.no_png:
        try:
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(8, 4.5))
            image = axis.imshow(matrix, aspect="auto", vmin=0.0)
            axis.set_xticks(range(len(layers)), [f"L{int(x)}" for x in layers])
            axis.set_yticks(range(len(roi_names)), roi_names)
            axis.set_title(config.get("fusion_mode", run_dir.name))
            figure.colorbar(image, ax=axis, label="routing weight")
            figure.tight_layout()
            figure.savefig(
                output_dir / "routing_matrix.png",
                dpi=180,
            )
            plt.close(figure)
        except ImportError:
            pass

    print(json.dumps({
        "fusion_mode": payload["fusion_mode"],
        "average_entropy": payload["average_entropy"],
        "layer_coverage_score": coverage,
        "roi_diversity_score": diversity,
        "early_roi_expected_depth": payload["early_roi_expected_depth"],
        "high_level_roi_expected_depth": payload[
            "high_level_roi_expected_depth"
        ],
        "early_vs_high_depth_gap": payload["early_vs_high_depth_gap"],
    }, indent=2))
    print(f"Saved routing analysis to {output_dir}")


if __name__ == "__main__":
    main()
