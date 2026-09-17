#!/usr/bin/env python
"""Inspect and compare NeuroRoute Stage-1 routing outputs."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


LOSS_KEYS = [
    "total_loss",
    "mse_loss",
    "cosine_loss",
    "entropy_loss",
    "balance_loss",
    "smoothness_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one or more NeuroRoute Stage-1 output directories"
    )
    parser.add_argument("--output-dirs", nargs="+", required=True)
    parser.add_argument(
        "--names",
        nargs="+",
        help="Display names matching --output-dirs; defaults to directory names",
    )
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument(
        "--csv-output-dir",
        default=".",
        help="Directory for routing_matrix_<name>.csv and comparison.csv",
    )
    return parser.parse_args()


def require_file(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage-1 artifact: {path}")
    return path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"Failed to read JSON {path}: {exc}") from exc


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path} line {line_number}: {exc}"
            ) from exc
    if not records:
        raise ValueError(f"No metric records found in {path}")
    return records


def format_value(value, precision: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}"
    return str(value)


def print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    float_precision: int = 4,
) -> None:
    text_rows = [
        [format_value(value, float_precision) for value in row] for row in rows
    ]
    widths = [
        max(
            len(str(header)),
            *(len(row[index]) for row in text_rows),
        )
        for index, header in enumerate(headers)
    ]
    print(
        " | ".join(
            str(header).ljust(widths[index])
            for index, header in enumerate(headers)
        )
    )
    print("-+-".join("-" * width for width in widths))
    for row in text_rows:
        print(
            " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


def routing_diagnostics(
    routing_mean: np.ndarray, selected_layers: Sequence[int]
) -> Dict[str, object]:
    eps = np.finfo(np.float64).eps
    probabilities = np.asarray(routing_mean, dtype=np.float64)
    row_sums = probabilities.sum(axis=-1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Routing matrix contains a row with non-positive sum")
    probabilities = probabilities / row_sums
    layer_count = probabilities.shape[-1]
    entropy_per_roi = -np.sum(
        probabilities * np.log(np.clip(probabilities, eps, None)), axis=-1
    )
    entropy_mean = float(entropy_per_roi.mean())
    normalized_entropy = (
        entropy_mean / math.log(layer_count) if layer_count > 1 else 0.0
    )
    max_weight_mean = float(probabilities.max(axis=-1).mean())
    uniform_weight = 1.0 / layer_count
    max_uniform_deviation = float(
        np.abs(probabilities - uniform_weight).max()
    )
    layer_usage = probabilities.mean(axis=0)
    dominant_position = int(layer_usage.argmax())
    dominant_usage = float(layer_usage[dominant_position])
    near_uniform = bool(
        normalized_entropy >= 0.95
        and max_weight_mean <= uniform_weight + 0.10
        and max_uniform_deviation <= 0.15
    )
    collapsed = bool(dominant_usage >= 0.80)
    top_positions = probabilities.argmax(axis=-1)
    return {
        "routing_entropy": entropy_mean,
        "normalized_entropy": normalized_entropy,
        "max_weight_mean": max_weight_mean,
        "near_uniform": near_uniform,
        "collapsed_to_single_layer": collapsed,
        "dominant_layer": int(selected_layers[dominant_position]),
        "dominant_layer_usage": dominant_usage,
        "top1_layers": [int(selected_layers[position]) for position in top_positions],
        "normalized_matrix": probabilities,
    }


def inspect_output(
    output_dir: Path, name: str
) -> Dict[str, object]:
    config = load_json(require_file(output_dir, "config.json"))
    roi_names = load_json(require_file(output_dir, "roi_names.json"))
    selected_layers = load_json(
        require_file(output_dir, "selected_clip_layers.json")
    )
    train_metrics = load_jsonl(
        require_file(output_dir, "metrics_train.jsonl")
    )
    val_metrics = load_jsonl(require_file(output_dir, "metrics_val.jsonl"))
    routing_mean = np.load(
        require_file(output_dir, "val_routing_weights_mean.npy"),
        allow_pickle=False,
    )
    routing_std = np.load(
        require_file(output_dir, "val_routing_weights_std.npy"),
        allow_pickle=False,
    )

    expected_shape = (len(roi_names), len(selected_layers))
    if routing_mean.shape != expected_shape:
        raise ValueError(
            f"{output_dir}: routing mean shape {routing_mean.shape} "
            f"!= expected {expected_shape}"
        )
    if routing_std.shape != expected_shape:
        raise ValueError(
            f"{output_dir}: routing std shape {routing_std.shape} "
            f"!= expected {expected_shape}"
        )
    if not np.isfinite(routing_mean).all() or not np.isfinite(routing_std).all():
        raise ValueError(f"{output_dir}: routing matrices contain NaN/Inf")

    best_val = min(val_metrics, key=lambda record: record["total_loss"])
    final_val = val_metrics[-1]
    diagnostics = routing_diagnostics(routing_mean, selected_layers)

    print(f"\n=== {name} ===")
    print(f"Output directory: {output_dir}")
    print(
        f"Subject: {config.get('subject', '-')} | "
        f"router_type: {config.get('router_type', '-')} | "
        f"epochs recorded: {len(val_metrics)}"
    )
    print("\nValidation losses")
    print_table(
        ["record", "epoch", *LOSS_KEYS],
        [
            ["best", best_val.get("epoch"), *[best_val.get(key) for key in LOSS_KEYS]],
            ["final", final_val.get("epoch"), *[final_val.get(key) for key in LOSS_KEYS]],
        ],
        float_precision=6,
    )

    print("\nMean routing matrix")
    print_table(
        ["ROI", *[f"L{layer}" for layer in selected_layers]],
        [
            [roi_name, *routing_mean[index]]
            for index, roi_name in enumerate(roi_names)
        ],
        float_precision=4,
    )
    print("\nRouting standard deviation")
    print_table(
        ["ROI", *[f"L{layer}" for layer in selected_layers]],
        [
            [roi_name, *routing_std[index]]
            for index, roi_name in enumerate(roi_names)
        ],
        float_precision=4,
    )

    print("\nRouting diagnostics")
    print(f"  mean routing entropy: {diagnostics['routing_entropy']:.6f}")
    print(f"  normalized entropy: {diagnostics['normalized_entropy']:.6f}")
    print(f"  mean max routing weight: {diagnostics['max_weight_mean']:.6f}")
    print(
        "  near-uniform: "
        f"{diagnostics['near_uniform']} "
        "(normalized entropy >= 0.95 and close to 1/L)"
    )
    print(
        "  collapsed to one layer: "
        f"{diagnostics['collapsed_to_single_layer']} "
        "(dominant mean layer usage >= 0.80)"
    )
    print(
        f"  dominant mean layer: L{diagnostics['dominant_layer']} "
        f"(usage={diagnostics['dominant_layer_usage']:.4f})"
    )
    print("  per-ROI top-1 layer:")
    for roi_name, layer in zip(roi_names, diagnostics["top1_layers"]):
        print(f"    {roi_name}: L{layer}")

    return {
        "name": name,
        "output_dir": str(output_dir),
        "router_type": config.get("router_type", name),
        "roi_names": roi_names,
        "selected_layers": selected_layers,
        "routing_mean": routing_mean,
        "routing_std": routing_std,
        "best_val": best_val,
        "final_val": final_val,
        "diagnostics": diagnostics,
        "train_metrics": train_metrics,
    }


def safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return safe or "run"


def save_routing_csv(result: Mapping[str, object], output_dir: Path) -> Path:
    path = output_dir / f"routing_matrix_{safe_filename(result['name'])}.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["roi", *[f"layer_{layer}" for layer in result["selected_layers"]]]
        )
        for roi_name, row in zip(result["roi_names"], result["routing_mean"]):
            writer.writerow([roi_name, *[float(value) for value in row]])
    return path


def comparison_rows(results: Sequence[Mapping[str, object]]) -> List[List[object]]:
    return [
        [
            result["name"],
            result["router_type"],
            result["best_val"]["total_loss"],
            result["final_val"]["total_loss"],
            result["diagnostics"]["routing_entropy"],
            result["diagnostics"]["max_weight_mean"],
            result["diagnostics"]["near_uniform"],
            result["diagnostics"]["collapsed_to_single_layer"],
        ]
        for result in results
    ]


def save_comparison_csv(
    results: Sequence[Mapping[str, object]], output_dir: Path
) -> Path:
    path = output_dir / "comparison.csv"
    headers = [
        "name",
        "router_type",
        "best_val_loss",
        "final_val_loss",
        "routing_entropy",
        "max_weight_mean",
        "near_uniform",
        "collapsed_to_single_layer",
    ]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(comparison_rows(results))
    return path


def main() -> None:
    args = parse_args()
    output_dirs = [
        Path(value).expanduser().resolve() for value in args.output_dirs
    ]
    names = args.names or [path.name for path in output_dirs]
    if len(names) != len(output_dirs):
        raise ValueError(
            f"--names has {len(names)} values but --output-dirs has "
            f"{len(output_dirs)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("--names must be unique")

    results = [
        inspect_output(output_dir, name)
        for output_dir, name in zip(output_dirs, names)
    ]

    if len(results) > 1:
        print("\n=== Comparison ===")
        print_table(
            [
                "name",
                "router_type",
                "best_val_loss",
                "final_val_loss",
                "routing_entropy",
                "max_weight_mean",
                "near_uniform",
                "collapsed",
            ],
            comparison_rows(results),
            float_precision=6,
        )

    if args.save_csv:
        csv_output_dir = Path(args.csv_output_dir).expanduser().resolve()
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            print(f"Saved CSV: {save_routing_csv(result, csv_output_dir)}")
        print(f"Saved CSV: {save_comparison_csv(results, csv_output_dir)}")


if __name__ == "__main__":
    main()
