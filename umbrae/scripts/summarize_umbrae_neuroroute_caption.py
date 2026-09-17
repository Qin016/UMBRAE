#!/usr/bin/env python
"""Summarize real UMBRAE-NeuroRoute caption ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch


METRICS = (
    "best_val_loss",
    "val_loss",
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "ROUGE-L",
    "CIDEr",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_run(name: str, raw_path: str) -> Dict[str, object]:
    path = Path(raw_path).expanduser().resolve()
    metrics_path = path / "metrics.json"
    config_path = path / "adapter_config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        return {
            "name": name,
            "path": str(path),
            "status": "missing",
            "missing_reason": (
                "missing metrics.json or adapter_config.json"
            ),
        }
    metrics = json.loads(metrics_path.read_text())
    config = json.loads(config_path.read_text())
    caption_metrics = metrics.get("caption_metrics", {})
    generated_path = path / "generated_captions.jsonl"
    generated_count = (
        sum(1 for line in generated_path.read_text().splitlines() if line)
        if generated_path.is_file()
        else int(metrics.get("num_generated_captions", 0))
    )
    leaked = bool(
        config.get("uses_image_clip_tokens_at_eval", False)
        or config.get("ground_truth_image_features_used", False)
        or metrics.get("uses_image_clip_tokens_at_eval", False)
    )
    best_checkpoint = path / "checkpoint_best.pt"
    best_val_loss = (
        float(
            torch.load(
                best_checkpoint,
                map_location="cpu",
                weights_only=False,
            )["val_loss"]
        )
        if best_checkpoint.is_file()
        else float(metrics["val_loss"])
    )
    return {
        "name": name,
        "path": str(path),
        "status": "complete",
        "fusion_mode": config.get("fusion_mode"),
        "bridge_type": config.get("bridge_type"),
        "structured_routing_alpha": config.get(
            "structured_routing_alpha"
        ),
        "routing_temperature": config.get("routing_temperature"),
        "best_val_loss": best_val_loss,
        "val_loss": float(metrics["val_loss"]),
        **{
            key: (
                float(caption_metrics[key])
                if caption_metrics.get(key) is not None
                else None
            )
            for key in METRICS
            if key not in ("best_val_loss", "val_loss")
        },
        "num_generated_captions": generated_count,
        "uses_image_tokens_at_eval": leaked,
        "oracle_image_token_mode": bool(
            config.get("oracle_image_token_mode", False)
        ),
    }


def relative_improvement(
    candidate: Dict[str, object],
    baseline: Dict[str, object],
    metric: str,
) -> float | None:
    left, right = candidate.get(metric), baseline.get(metric)
    if left is None or right in (None, 0):
        return None
    if metric in ("best_val_loss", "val_loss"):
        return 100.0 * (right - left) / abs(right)
    return 100.0 * (left - right) / abs(right)


def main():
    args = parse_args()
    runs = [load_run(name, path) for name, path in args.run]
    complete_runs = [
        run for run in runs if run.get("status") == "complete"
    ]
    if any(run["uses_image_tokens_at_eval"] for run in complete_runs):
        raise RuntimeError("Image-token leakage detected; summary aborted")
    by_name = {run["name"]: run for run in complete_runs}
    soft = by_name.get("soft", by_name.get("umbrae_plus_soft"))
    uniform = by_name.get("uniform", by_name.get("umbrae_plus_uniform"))
    umbrae_only = by_name.get("umbrae_only")
    single_l24 = by_name.get(
        "single_l24", by_name.get("umbrae_plus_single_l24")
    )
    comparisons = {}
    if soft is not None:
        for baseline_name in (
            "umbrae_only",
            "umbrae_plus_uniform",
            "umbrae_plus_single_l24",
        ):
            if baseline_name not in by_name:
                continue
            comparisons[f"umbrae_plus_soft_vs_{baseline_name}"] = {
                metric: relative_improvement(
                    soft,
                    by_name[baseline_name],
                    metric,
                )
                for metric in METRICS
            }
    structured = [
        run
        for run in complete_runs
        if run.get("fusion_mode")
        in {
            "umbrae_plus_uniform_residual_soft",
            "umbrae_plus_temperature_soft",
        }
    ]
    for run in structured:
        for baseline_name, baseline in (
            ("uniform", uniform),
            ("soft", soft),
        ):
            if baseline is not None:
                comparisons[f"{run['name']}_vs_{baseline_name}"] = {
                    metric: relative_improvement(run, baseline, metric)
                    for metric in METRICS
                }
    best_structured = (
        max(
            structured,
            key=lambda run: (
                run.get("CIDEr") or float("-inf"),
                run.get("BLEU-4") or float("-inf"),
            ),
        )
        if structured
        else None
    )
    if best_structured is not None:
        for baseline_name, baseline in (
            ("umbrae_only", umbrae_only),
            ("single_l24", single_l24),
        ):
            if baseline is not None:
                comparisons[
                    f"best_structured_{best_structured['name']}_vs_{baseline_name}"
                ] = {
                    metric: relative_improvement(
                        best_structured, baseline, metric
                    )
                    for metric in METRICS
                }

    best_method = {}
    for metric in METRICS:
        valid = [
            run for run in complete_runs if run.get(metric) is not None
        ]
        if valid:
            best_method[metric] = (
                min(valid, key=lambda run: run[metric])["name"]
                if metric in ("best_val_loss", "val_loss")
                else max(valid, key=lambda run: run[metric])["name"]
            )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": runs,
        "missing_runs": [
            run["name"]
            for run in runs
            if run.get("status") != "complete"
        ],
        "best_structured_method": (
            best_structured["name"] if best_structured else None
        ),
        "best_method_per_metric": best_method,
        "relative_improvement_percent": comparisons,
        "leakage_check_passed": True,
    }
    (output_dir / "umbrae_neuroroute_caption_summary.json").write_text(
        json.dumps(payload, indent=2)
    )
    fields = [
        "name",
        "fusion_mode",
        "bridge_type",
        "structured_routing_alpha",
        "routing_temperature",
        *METRICS,
        "num_generated_captions",
        "uses_image_tokens_at_eval",
        "path",
    ]
    with (
        output_dir / "umbrae_neuroroute_caption_summary.csv"
    ).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: run.get(field) for field in fields} for run in runs
        )

    print(
        "method".ljust(26),
        "best_val".rjust(10),
        "val_loss".rjust(10),
        "BLEU-4".rjust(10),
        "CIDEr".rjust(10),
        "ROUGE-L".rjust(10),
        "captions".rjust(10),
        "bridge".rjust(15),
    )
    for run in runs:
        if run.get("status") != "complete":
            print(run["name"].ljust(26), "MISSING")
            continue
        print(
            run["name"].ljust(26),
            f"{run['best_val_loss']:.6f}".rjust(10),
            f"{run['val_loss']:.6f}".rjust(10),
            f"{run['BLEU-4']:.6f}".rjust(10),
            f"{run['CIDEr']:.6f}".rjust(10),
            f"{run['ROUGE-L']:.6f}".rjust(10),
            str(run["num_generated_captions"]).rjust(10),
            str(run["bridge_type"]).rjust(15),
        )
    print("\nBest method per metric:")
    print(json.dumps(best_method, indent=2))
    print("\nRelative improvement (%):")
    print(json.dumps(comparisons, indent=2))
    print(f"\nSaved summary to {output_dir}")


if __name__ == "__main__":
    main()
