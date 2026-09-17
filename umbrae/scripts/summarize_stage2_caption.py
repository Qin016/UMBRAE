#!/usr/bin/env python
"""Compare NeuroRoute Stage-2 caption runs."""

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("NAME", "OUTPUT_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-csv", action="store_true")
    return parser.parse_args()


def load_run(name, directory):
    path = Path(directory).expanduser().resolve()
    metrics = json.loads((path / "metrics.json").read_text())
    config = json.loads((path / "adapter_config.json").read_text())
    best = torch.load(
        path / "checkpoint_best.pt", map_location="cpu", mmap=True
    )
    predictions = [
        json.loads(line)
        for line in (path / "generated_captions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    caption_metrics = metrics.get("caption_metrics", {})
    return {
        "name": name,
        "output_dir": str(path),
        "best_val_loss": float(best["val_loss"]),
        "final_val_loss": float(metrics["val_loss"]),
        "bleu_1": caption_metrics.get("BLEU-1"),
        "bleu_2": caption_metrics.get("BLEU-2"),
        "bleu_3": caption_metrics.get("BLEU-3"),
        "bleu_4": caption_metrics.get("BLEU-4"),
        "cider": caption_metrics.get("CIDEr"),
        "meteor": caption_metrics.get("METEOR"),
        "rouge_l": caption_metrics.get("ROUGE-L"),
        "spice": caption_metrics.get("SPICE"),
        "num_generated_captions": len(predictions),
        "uses_image_tokens_at_eval": bool(
            config.get("uses_image_clip_tokens_at_eval")
            or metrics.get("ground_truth_image_features_used")
        ),
        "fusion_mode": config.get("fusion_mode"),
        "mllm_bridge_type": config.get("mllm_bridge_type"),
    }


def relative_loss_improvement(candidate, baseline):
    return (
        None if baseline == 0
        else 100.0 * (baseline - candidate) / baseline
    )


def main():
    args = parse_args()
    runs = [load_run(*spec) for spec in args.run]
    indexed = {run["name"]: run for run in runs}
    headers = [
        "name", "best_val_loss", "final_val_loss", "bleu_1", "bleu_2",
        "bleu_3", "bleu_4", "cider", "rouge_l",
        "num_generated_captions", "uses_image_tokens_at_eval",
    ]
    print(" | ".join(headers))
    for run in runs:
        print(" | ".join(str(run[key]) for key in headers))
    comparisons = {}
    if "concat_soft" in indexed:
        candidate = indexed["concat_soft"]["best_val_loss"]
        for baseline in (
            "l24_only", "concat_uniform", "concat_single_l24"
        ):
            if baseline in indexed:
                comparisons[
                    f"concat_soft_vs_{baseline}_percent"
                ] = relative_loss_improvement(
                    candidate, indexed[baseline]["best_val_loss"]
                )
    print(json.dumps(comparisons, indent=2))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage2_caption_summary.json").write_text(
        json.dumps({"runs": runs, "comparisons": comparisons}, indent=2)
    )
    if args.save_csv:
        with (output_dir / "stage2_caption_summary.csv").open(
            "w", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
    if any(run["uses_image_tokens_at_eval"] for run in runs):
        raise RuntimeError("A run used image tokens during validation/test")
    print(f"Saved summary to {output_dir}")


if __name__ == "__main__":
    main()
