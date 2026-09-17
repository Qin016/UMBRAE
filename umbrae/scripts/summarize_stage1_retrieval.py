#!/usr/bin/env python
"""Summarize Stage-1 retrieval runs across subjects and methods."""

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


METRIC_KEYS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "median_rank",
    "mean_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("SUBJECT", "METHOD", "OUTPUT_DIR"),
        required=True,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default="stage1_outputs/retrieval_summary",
    )
    parser.add_argument("--save-csv", action="store_true")
    return parser.parse_args()


def load_run(subject: str, method: str, directory: str):
    path = Path(directory).expanduser().resolve()
    metrics_path = path / "metrics_retrieval.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    payload = json.loads(metrics_path.read_text())
    return {
        "subject": subject,
        "method": method,
        "output_dir": str(path),
        "num_samples": int(payload["num_samples"]),
        "retrieval_protocol": payload.get(
            "retrieval_protocol", "legacy_diagonal"
        ),
        "num_unique_coco_ids": int(
            payload.get("num_unique_coco_ids", payload["num_samples"])
        ),
        "average_positives_per_query": float(
            payload.get("average_positives_per_brain_query", 1.0)
        ),
        "brain_to_image": payload["brain_to_image"],
        "image_to_brain": payload["image_to_brain"],
    }


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]):
    text = [
        [
            f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        for row in rows
    ]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in text))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in text:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def relative_improvement(soft: float, baseline: float) -> float:
    return float("nan") if baseline == 0 else 100.0 * (soft - baseline) / baseline


def summarize(runs, methods):
    indexed = {(run["subject"], run["method"]): run for run in runs}
    if len(indexed) != len(runs):
        raise ValueError("Duplicate subject/method run")
    subjects = sorted({run["subject"] for run in runs})
    missing = [
        f"{subject}:{method}"
        for subject in subjects
        for method in methods
        if (subject, method) not in indexed
    ]
    if missing:
        raise ValueError(f"Missing runs: {missing}")

    print("\nPer-subject brain-to-image retrieval")
    headers = [
        "subject",
        "method",
        "protocol",
        "unique_ids",
        "avg_pos",
        "R@1",
        "R@5",
        "R@10",
        "MedR",
        "MeanR",
    ]
    rows = []
    for subject in subjects:
        for method in methods:
            values = indexed[(subject, method)]["brain_to_image"]
            rows.append([
                subject,
                method,
                indexed[(subject, method)]["retrieval_protocol"],
                indexed[(subject, method)]["num_unique_coco_ids"],
                indexed[(subject, method)]["average_positives_per_query"],
                values["recall_at_1"],
                values["recall_at_5"],
                values["recall_at_10"],
                values["median_rank"],
                values["mean_rank"],
            ])
    print_table(headers, rows)

    print("\nAverage retrieval across subjects")
    average_rows = []
    for method in methods:
        for direction in ("brain_to_image", "image_to_brain"):
            average_rows.append([
                method,
                direction,
                *[
                    float(np.mean([
                        indexed[(subject, method)][direction][key]
                        for subject in subjects
                    ]))
                    for key in METRIC_KEYS
                ],
            ])
    print_table(["method", "direction", *METRIC_KEYS], average_rows)

    improvement_rows = []
    if {"soft", "uniform", "single_L24"}.issubset(methods):
        for subject in subjects:
            soft = indexed[(subject, "soft")]["brain_to_image"]["recall_at_1"]
            improvement_rows.append([
                subject,
                relative_improvement(
                    soft,
                    indexed[(subject, "uniform")]["brain_to_image"]["recall_at_1"],
                ),
                relative_improvement(
                    soft,
                    indexed[(subject, "single_L24")]["brain_to_image"]["recall_at_1"],
                ),
            ])
        print("\nSoft R@1 relative improvement")
        print_table(
            ["subject", "vs_uniform_%", "vs_single_L24_%"],
            improvement_rows,
        )
    return subjects, indexed, rows, average_rows, improvement_rows


def save_csv(output_dir: Path, rows, average_rows, improvement_rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_subject_retrieval.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "subject",
                "method",
                "protocol",
                "unique_ids",
                "average_positives_per_query",
                "R@1",
                "R@5",
                "R@10",
                "median_rank",
                "mean_rank",
            ]
        )
        writer.writerows(rows)
    with (output_dir / "average_retrieval.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["method", "direction", *METRIC_KEYS])
        writer.writerows(average_rows)
    with (output_dir / "soft_improvement.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["subject", "soft_vs_uniform_percent", "soft_vs_single_l24_percent"]
        )
        writer.writerows(improvement_rows)
    print(f"Saved CSV files to {output_dir}")


def main():
    args = parse_args()
    runs = [load_run(*spec) for spec in args.run]
    methods = args.methods or list(dict.fromkeys(
        run["method"] for run in runs
    ))
    _, _, rows, averages, improvements = summarize(runs, methods)
    if args.save_csv:
        save_csv(
            Path(args.output_dir).expanduser().resolve(),
            rows,
            averages,
            improvements,
        )


if __name__ == "__main__":
    main()
