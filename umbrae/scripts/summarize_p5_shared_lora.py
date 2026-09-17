#!/usr/bin/env python
"""Create the P5 frozen/LoRA/P4 comparison artifact."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--baseline", required=True); parser.add_argument("--p4", required=True); parser.add_argument("--p5-run", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text()); p4 = json.loads(Path(args.p4).read_text()); run = Path(args.p5_run).resolve(); lora = json.loads((run / "final_val_metrics.json").read_text()); config = json.loads((run / "config.json").read_text())
    brain_to_image = lora["retrieval"]["brain_to_image"]; frozen_retrieval = baseline["retrieval"]["brain_to_image"]; delta = lora["rsa"]["spearman_rsa"] - baseline["rsa"]["spearman_rsa"]
    preservation = brain_to_image["recall_at_1"] >= frozen_retrieval["recall_at_1"] - 0.01 and lora["base_lora_cosine"] >= 0.99 and lora["relative_feature_delta"] <= 0.20
    calibration = "POSITIVE" if delta > 0.001 else ("WEAK" if delta >= -0.001 else "NEGATIVE")
    efficient = config["parameter_report"]["trainable_ratio_percent_of_brainx"] < 1.0
    status = "READY_FOR_FULL_STAGE_C" if calibration == "POSITIVE" and preservation and efficient else "LORA_NEEDS_ADJUSTMENT"
    result = {
        "report": "P5 Shared LoRA-Only Calibration", "protocol_version": "protocol_v1",
        "frozen_umbrae": {"spearman_rsa": baseline["rsa"]["spearman_rsa"], "pearson_rsa": baseline["rsa"]["pearson_rsa"], **frozen_retrieval},
        "lora_only": {"spearman_rsa": lora["rsa"]["spearman_rsa"], "pearson_rsa": lora["rsa"]["pearson_rsa"], **brain_to_image, "base_lora_cosine": lora["base_lora_cosine"], "relative_feature_delta": lora["relative_feature_delta"], "pooled_relative_delta": lora["pooled_relative_delta"], "pooled_linear_cka": lora["pooled_linear_cka"], "best_epoch": lora["best_epoch"], "best_val_total_loss": lora["best_val_total_loss"]},
        "p4": {"real_fusion": p4["comparison"]["real_fusion"], "random_fusion": p4["comparison"]["random_fusion"]},
        "delta_spearman_rsa_lora": delta, "parameter_report": config["parameter_report"],
        "controls": {"structural_used": False, "fusion_used": False, "brainx_base_frozen": True, "z_lora_online": True, "base_reference_cache_detached_only": True, "test_set_used": False},
        "judgement_thresholds": {"positive_delta_rsa": 0.001, "maximum_r1_drop": 0.01, "minimum_base_lora_cosine": 0.99, "maximum_relative_feature_delta": 0.20, "maximum_trainable_ratio_percent": 1.0},
        "LORA_CALIBRATION": calibration, "SEMANTIC_PRESERVATION": "PASS" if preservation else "FAIL", "PARAMETER_EFFICIENT": "YES" if efficient else "NO", "P5_STATUS": status,
    }
    output = Path(args.output).resolve(); output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
