#!/usr/bin/env python
"""Summarize the two locked P3 Stage-A runs without additional evaluation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dual_branch_cache import sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--random-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_run(path):
    root = Path(path).expanduser().resolve()
    return {
        "root": root,
        "initial": json.loads((root / "initial_val_metrics.json").read_text()),
        "final": json.loads((root / "final_val_metrics.json").read_text()),
        "epochs": [json.loads(line) for line in (root / "metrics_per_epoch.jsonl").read_text().splitlines()],
        "provenance": json.loads((root / "provenance.json").read_text()),
    }


def compact(run):
    final = run["final"]
    retrieval = final["retrieval"]["brain_to_image"]
    return {
        "best_epoch": final["best_epoch"],
        "best_val_uot": final["best_val_uot"],
        "spearman_rsa": final["rsa"]["spearman_rsa"],
        "pearson_rsa": final["rsa"]["pearson_rsa"],
        "recall_at_1": retrieval["recall_at_1"],
        "recall_at_5": retrieval["recall_at_5"],
        "recall_at_10": retrieval["recall_at_10"],
        "median_rank": retrieval["median_rank"],
        "mean_rank": retrieval["mean_rank"],
        "transport_mass": final["transport_mass"],
        "normalized_transport_entropy": final["normalized_transport_entropy"],
        "max_transport_fraction": final["max_transport_fraction"],
        "mean_roi_token_norm": final["mean_roi_token_norm"],
        "mean_pairwise_roi_cosine": final["mean_pairwise_roi_cosine"],
        "best_checkpoint": str(run["root"] / "best.pth"),
        "best_checkpoint_sha256": sha256_file(str(run["root"] / "best.pth")),
    }


def run(args):
    real, random = load_run(args.real_dir), load_run(args.random_dir)
    same_init = real["provenance"]["initial_parameter_sha256"] == random["provenance"]["initial_parameter_sha256"]
    if not same_init:
        raise RuntimeError("Real and random controls did not share initialization")
    result = {
        "protocol_version": "protocol_v1",
        "same_learned_parameter_initialization": same_init,
        "initial_parameter_sha256": real["provenance"]["initial_parameter_sha256"],
        "real_roi": compact(real),
        "random_structure": compact(random),
        "initialization": {
            "real": compact({**real, "final": {**real["initial"], "best_epoch": 0, "best_val_uot": real["initial"]["uot_loss"]}}),
            "random": compact({**random, "final": {**random["initial"], "best_epoch": 0, "best_val_uot": random["initial"]["uot_loss"]}}),
        },
        "interpretation": {
            "OPTIMIZATION": "PASS",
            "REPRESENTATION_LEARNING": "POSITIVE",
            "STRUCTURE_SIGNAL": "NEGATIVE",
            "P3_STATUS": "NEEDS_ADJUSTMENT",
        },
    }
    # Initialization has no checkpoint; remove synthetic checkpoint fields.
    for value in result["initialization"].values():
        value.pop("best_checkpoint")
        value.pop("best_checkpoint_sha256")
    target = Path(args.output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    (target / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())
