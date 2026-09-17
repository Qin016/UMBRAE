#!/usr/bin/env python
"""Create the complete P6 Stage-C ablation comparison."""

import argparse
import json
from pathlib import Path


def compact(metrics):
    retrieval = metrics["retrieval"]["brain_to_image"]
    result = {"spearman_rsa": metrics["rsa"]["spearman_rsa"], "pearson_rsa": metrics["rsa"]["pearson_rsa"], **retrieval}
    for key in ("base_lora_cosine", "base_calibrated_cosine", "lora_relative_delta", "fusion_delta_ratio", "final_relative_delta", "final_pooled_linear_cka", "gate", "best_epoch", "best_val_total_loss"): result[key] = metrics.get(key)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--baseline", required=True); parser.add_argument("--p4", required=True); parser.add_argument("--p5", required=True); parser.add_argument("--real-run", required=True); parser.add_argument("--random-run", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    baseline_raw = json.loads(Path(args.baseline).read_text()); p4 = json.loads(Path(args.p4).read_text()); p5 = json.loads(Path(args.p5).read_text()); real = compact(json.loads((Path(args.real_run) / "final_val_metrics.json").read_text())); random = compact(json.loads((Path(args.random_run) / "final_val_metrics.json").read_text()))
    baseline = {"spearman_rsa": baseline_raw["rsa"]["spearman_rsa"], "pearson_rsa": baseline_raw["rsa"]["pearson_rsa"], **baseline_raw["retrieval"]["brain_to_image"]}; lora = p5["lora_only"]; real_fusion, random_fusion = p4["comparison"]["real_fusion"], p4["comparison"]["random_fusion"]
    real_gain, random_gain = real["spearman_rsa"] - lora["spearman_rsa"], random["spearman_rsa"] - lora["spearman_rsa"]
    real_base, random_base = real["spearman_rsa"] - baseline["spearman_rsa"], random["spearman_rsa"] - baseline["spearman_rsa"]
    real_interaction = real_gain - (real_fusion["spearman_rsa"] - baseline["spearman_rsa"]); random_interaction = random_gain - (random_fusion["spearman_rsa"] - baseline["spearman_rsa"])
    preservation = min(real["recall_at_1"], random["recall_at_1"]) >= baseline["recall_at_1"] - 0.01 and min(real["base_calibrated_cosine"], random["base_calibrated_cosine"]) >= 0.99 and max(real["final_relative_delta"], random["final_relative_delta"]) <= 0.20
    full_gain = "POSITIVE" if real_base > 0.001 else ("WEAK" if real_base >= -0.001 else "NEGATIVE"); synergy = "POSITIVE" if real_gain > 0.001 else ("WEAK" if real_gain >= -0.001 else "NEGATIVE")
    difference = real["spearman_rsa"] - random["spearman_rsa"]; structure = "POSITIVE" if difference > 0.001 else ("NEGATIVE" if difference < -0.001 else "INCONCLUSIVE")
    status = "FULL_MODEL_VALIDATED" if full_gain == "POSITIVE" and synergy == "POSITIVE" and preservation else ("LORA_DOMINATES_STRUCTURE_WEAK" if preservation and synergy == "WEAK" else "FULL_MODEL_NEEDS_ADJUSTMENT")
    result = {"report": "P6 Full Stage-C", "protocol_version": "protocol_v1", "comparison": {"frozen_umbrae": baseline, "lora_only": lora, "real_fusion": real_fusion, "random_fusion": random_fusion, "full_real": real, "full_random": random}, "structural_gain_on_lora": {"real": real_gain, "random": random_gain, "full_real_minus_full_random": difference}, "combined_gain_vs_base": {"real": real_base, "random": random_base}, "interaction": {"real": real_interaction, "random": random_interaction}, "judgement_thresholds": {"material_rsa_margin": 0.001, "maximum_r1_drop": 0.01, "minimum_base_calibrated_cosine": 0.99, "maximum_final_relative_delta": 0.20}, "FULL_MODEL_GAIN": full_gain, "LORA_FUSION_SYNERGY": synergy, "COARSE_STRUCTURE_SIGNAL": structure, "SEMANTIC_PRESERVATION": "PASS" if preservation else "FAIL", "P6_STATUS": status}
    output = Path(args.output).resolve(); output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
