#!/usr/bin/env python
"""Summarize projector-enabled Stage-1 baselines across subjects."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


EARLY_ROIS = ("V1", "V2", "V3", "hV4")
HIGH_LEVEL_ROIS = ("FFA", "EBA", "PPA", "OPA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NeuroRoute Stage-1 runs across subjects"
    )
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("SUBJECT", "METHOD", "OUTPUT_DIR"),
        required=True,
        help="Repeat for each subject/method/output directory",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["soft", "uniform", "single_L24"],
        help="Method display/order used for comparisons",
    )
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--output-dir", default="stage1_outputs/cross_subject_summary")
    return parser.parse_args()


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact: {path}")
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    text_rows = [
        [
            f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        for row in rows
    ]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in text_rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def normalize_routing(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    sums = matrix.sum(axis=-1, keepdims=True)
    if np.any(sums <= 0) or not np.isfinite(matrix).all():
        raise ValueError("Invalid routing matrix")
    return matrix / sums


def load_run(subject: str, method: str, directory: str) -> Dict[str, object]:
    output_dir = Path(directory).expanduser().resolve()
    config = load_json(output_dir / "config.json")
    roi_names = load_json(output_dir / "roi_names.json")
    layers = load_json(output_dir / "selected_clip_layers.json")
    val_metrics = load_jsonl(output_dir / "metrics_val.jsonl")
    matrix = normalize_routing(
        np.load(output_dir / "val_routing_weights_mean.npy", allow_pickle=False)
    )
    if matrix.shape != (len(roi_names), len(layers)):
        raise ValueError(
            f"{output_dir}: routing matrix {matrix.shape} does not match "
            f"{len(roi_names)} ROIs and {len(layers)} layers"
        )
    best = min(val_metrics, key=lambda item: item["total_loss"])
    final = val_metrics[-1]
    layer_values = np.asarray(layers, dtype=np.float64)
    expected_depth = matrix @ layer_values
    depth_by_roi = {
        roi_name: float(expected_depth[index])
        for index, roi_name in enumerate(roi_names)
    }
    missing = [
        roi for roi in (*EARLY_ROIS, *HIGH_LEVEL_ROIS) if roi not in depth_by_roi
    ]
    if missing:
        raise ValueError(f"{output_dir}: missing required ROIs {missing}")
    l24_usage = (
        float(matrix[:, layers.index(24)].mean()) if 24 in layers else float("nan")
    )
    return {
        "subject": subject,
        "method": method,
        "output_dir": str(output_dir),
        "router_type": config.get("router_type"),
        "best_val_loss": float(best["total_loss"]),
        "final_val_loss": float(final["total_loss"]),
        "final_mse_loss": float(final["mse_loss"]),
        "final_cosine_loss": float(final["cosine_loss"]),
        "l24_usage": l24_usage,
        "roi_names": roi_names,
        "selected_layers": layers,
        "routing_matrix": matrix,
        "expected_depth_by_roi": depth_by_roi,
        "early_visual_mean_depth": float(
            np.mean([depth_by_roi[roi] for roi in EARLY_ROIS])
        ),
        "high_level_mean_depth": float(
            np.mean([depth_by_roi[roi] for roi in HIGH_LEVEL_ROIS])
        ),
    }


def relative_improvement(soft_loss: float, baseline_loss: float) -> float:
    if baseline_loss == 0:
        return float("nan")
    return 100.0 * (baseline_loss - soft_loss) / baseline_loss


def validate_grid(
    runs: Sequence[Mapping[str, object]], methods: Sequence[str]
) -> List[str]:
    subjects = sorted({run["subject"] for run in runs})
    indexed = {(run["subject"], run["method"]) for run in runs}
    missing = [
        f"{subject}:{method}"
        for subject in subjects
        for method in methods
        if (subject, method) not in indexed
    ]
    if missing:
        raise ValueError(f"Missing subject/method runs: {missing}")
    if len(indexed) != len(runs):
        raise ValueError("Duplicate subject/method run supplied")
    return subjects


def print_summary(
    runs: Sequence[Mapping[str, object]],
    subjects: Sequence[str],
    methods: Sequence[str],
) -> None:
    indexed = {(run["subject"], run["method"]): run for run in runs}
    print("\nPer-subject method comparison")
    print_table(
        [
            "subject",
            "method",
            "best_val",
            "final_val",
            "mse",
            "cosine",
            "L24_usage",
            "early_depth",
            "high_depth",
        ],
        [
            [
                subject,
                method,
                indexed[(subject, method)]["best_val_loss"],
                indexed[(subject, method)]["final_val_loss"],
                indexed[(subject, method)]["final_mse_loss"],
                indexed[(subject, method)]["final_cosine_loss"],
                indexed[(subject, method)]["l24_usage"],
                indexed[(subject, method)]["early_visual_mean_depth"],
                indexed[(subject, method)]["high_level_mean_depth"],
            ]
            for subject in subjects
            for method in methods
        ],
    )

    print("\nSoft relative improvement")
    print_table(
        [
            "subject",
            "soft_vs_uniform_%",
            "soft_vs_single_L24_%",
        ],
        [
            [
                subject,
                relative_improvement(
                    indexed[(subject, "soft")]["best_val_loss"],
                    indexed[(subject, "uniform")]["best_val_loss"],
                ),
                relative_improvement(
                    indexed[(subject, "soft")]["best_val_loss"],
                    indexed[(subject, "single_L24")]["best_val_loss"],
                ),
            ]
            for subject in subjects
        ],
    )

    print("\nAverage across subjects")
    print_table(
        ["method", "mean_best_val", "mean_final_val", "mean_L24_usage"],
        [
            [
                method,
                float(np.mean([
                    indexed[(subject, method)]["best_val_loss"]
                    for subject in subjects
                ])),
                float(np.mean([
                    indexed[(subject, method)]["final_val_loss"]
                    for subject in subjects
                ])),
                float(np.mean([
                    indexed[(subject, method)]["l24_usage"]
                    for subject in subjects
                ])),
            ]
            for method in methods
        ],
    )

    print("\nExpected CLIP-layer depth per ROI")
    roi_names = indexed[(subjects[0], methods[0])]["roi_names"]
    print_table(
        ["subject", "method", *roi_names],
        [
            [
                subject,
                method,
                *[
                    indexed[(subject, method)]["expected_depth_by_roi"][roi]
                    for roi in roi_names
                ],
            ]
            for subject in subjects
            for method in methods
        ],
    )


def save_csvs(
    runs: Sequence[Mapping[str, object]],
    subjects: Sequence[str],
    methods: Sequence[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    indexed = {(run["subject"], run["method"]): run for run in runs}
    comparison_path = output_dir / "cross_subject_comparison.csv"
    with comparison_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "subject", "method", "best_val_loss", "final_val_loss",
            "mse_loss", "cosine_loss", "l24_usage",
            "early_visual_mean_depth", "high_level_mean_depth",
            "soft_vs_uniform_percent", "soft_vs_single_l24_percent",
        ])
        for subject in subjects:
            soft = indexed[(subject, "soft")]["best_val_loss"]
            improvement_uniform = relative_improvement(
                soft, indexed[(subject, "uniform")]["best_val_loss"]
            )
            improvement_single = relative_improvement(
                soft, indexed[(subject, "single_L24")]["best_val_loss"]
            )
            for method in methods:
                run = indexed[(subject, method)]
                writer.writerow([
                    subject, method, run["best_val_loss"], run["final_val_loss"],
                    run["final_mse_loss"], run["final_cosine_loss"],
                    run["l24_usage"], run["early_visual_mean_depth"],
                    run["high_level_mean_depth"],
                    improvement_uniform if method == "soft" else "",
                    improvement_single if method == "soft" else "",
                ])

    depth_path = output_dir / "expected_depth_per_roi.csv"
    roi_names = runs[0]["roi_names"]
    with depth_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["subject", "method", *roi_names])
        for subject in subjects:
            for method in methods:
                run = indexed[(subject, method)]
                writer.writerow([
                    subject,
                    method,
                    *[run["expected_depth_by_roi"][roi] for roi in roi_names],
                ])

    for run in runs:
        path = output_dir / f"routing_{run['subject']}_{run['method']}.csv"
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "roi", *[f"layer_{layer}" for layer in run["selected_layers"]]
            ])
            for roi_name, row in zip(run["roi_names"], run["routing_matrix"]):
                writer.writerow([roi_name, *[float(value) for value in row]])
    print(f"Saved CSV summary to {output_dir}")


def main() -> None:
    args = parse_args()
    runs = [load_run(*run_spec) for run_spec in args.run]
    subjects = validate_grid(runs, args.methods)
    print_summary(runs, subjects, args.methods)
    if args.save_csv:
        save_csvs(
            runs,
            subjects,
            args.methods,
            Path(args.output_dir).expanduser().resolve(),
        )


if __name__ == "__main__":
    main()
