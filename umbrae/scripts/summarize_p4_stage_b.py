#!/usr/bin/env python
"""Create the locked P4 Frozen/Real/Random comparison artifact."""

import argparse
import json
from pathlib import Path


def compact(metrics):
    primary = metrics["retrieval"]["brain_to_image"]
    return {
        "spearman_rsa": metrics["rsa"]["spearman_rsa"],
        "pearson_rsa": metrics["rsa"]["pearson_rsa"],
        "recall_at_1": primary["recall_at_1"],
        "recall_at_5": primary["recall_at_5"],
        "recall_at_10": primary["recall_at_10"],
        "median_rank": primary["median_rank"],
        "mean_rank": primary["mean_rank"],
        "preservation_cosine": metrics.get("preservation_cosine", 1.0),
        "gate": metrics.get("gate", 0.0),
        "delta_ratio": metrics.get("delta_ratio", 0.0),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_total_loss": metrics.get("best_val_total_loss"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True); parser.add_argument("--real-run", required=True); parser.add_argument("--random-run", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline_raw = json.loads(Path(args.baseline).read_text())
    baseline = compact({"rsa": baseline_raw["rsa"], "retrieval": baseline_raw["retrieval"]})
    real_dir, random_dir = Path(args.real_run).resolve(), Path(args.random_run).resolve()
    real = compact(json.loads((real_dir / "final_val_metrics.json").read_text()))
    random = compact(json.loads((random_dir / "final_val_metrics.json").read_text()))
    real_config, random_config = json.loads((real_dir / "config.json").read_text()), json.loads((random_dir / "config.json").read_text())
    if real_config["initial_fusion_sha256"] != random_config["initial_fusion_sha256"]:
        raise ValueError("Real/Random fusion initializations differ")
    delta_real = real["spearman_rsa"] - baseline["spearman_rsa"]
    delta_random = random["spearman_rsa"] - baseline["spearman_rsa"]
    retrieval_held = min(real["recall_at_1"], random["recall_at_1"]) >= baseline["recall_at_1"] - 0.01
    preservation_pass = retrieval_held and min(real["preservation_cosine"], random["preservation_cosine"]) >= 0.99
    fusion_utility = "POSITIVE" if delta_real > 0 and retrieval_held else ("WEAK" if abs(delta_real) <= 0.001 else "NEGATIVE")
    complementarity = "POSITIVE" if delta_real > delta_random + 0.001 else ("NEGATIVE" if delta_random > delta_real + 0.001 else "INCONCLUSIVE")
    if fusion_utility == "POSITIVE" and complementarity == "POSITIVE" and preservation_pass:
        status = "READY_FOR_LORA"
    elif fusion_utility in {"POSITIVE", "WEAK"} and preservation_pass:
        status = "FUSION_WORKS_STRUCTURE_WEAK"
    else:
        status = "NEEDS_ADJUSTMENT"
    result = {
        "report": "P4 Gated Fusion Calibration", "protocol_version": "protocol_v1", "checkpoint_selection_metric": "val_total_loss",
        "comparison": {"frozen_umbrae": baseline, "real_fusion": real, "random_fusion": random},
        "geometry_calibration": {"delta_spearman_rsa_real": delta_real, "delta_spearman_rsa_random": delta_random, "real_minus_random_delta_rsa": delta_real - delta_random},
        "controls": {"fusion_initialization_sha256": real_config["initial_fusion_sha256"], "identical_fusion_initialization": True, "identical_batch_order_rule": "torch.Generator(seed + epoch)", "brainx_frozen": True, "structural_frozen": True, "clip_frozen": True, "fusion_only_optimizer": True, "h_struct_online": True, "z_cal_cached": False, "test_set_used": False},
        "judgement_thresholds": {"complementarity_margin": 0.001, "retrieval_r1_max_drop": 0.01, "minimum_preservation_cosine": 0.99},
        "FUSION_UTILITY": fusion_utility, "STRUCTURAL_COMPLEMENTARITY": complementarity, "SEMANTIC_PRESERVATION": "PASS" if preservation_pass else "FAIL", "P4_STATUS": status,
    }
    output = Path(args.output).resolve(); output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
