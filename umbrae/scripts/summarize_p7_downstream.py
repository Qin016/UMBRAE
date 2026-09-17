#!/usr/bin/env python
"""Create the deterministic P7 comparison, integrity audit, and examples.

This script only reads completed P7 artifacts.  It does not load or update a
model, and it deliberately performs no checkpoint or hyperparameter selection.
"""

import argparse
import json
from pathlib import Path


MODES = ("umbrae", "lora", "full_real", "full_random")
MODEL_LABELS = {
    "umbrae": {"model": "UMBRAE", "structural_type": "None", "role": "BASELINE"},
    "lora": {"model": "LoRA-only", "structural_type": "None", "role": "ABLATION"},
    "full_real": {"model": "Full Real", "structural_type": "Real ROI", "role": "MAIN METHOD"},
    "full_random": {"model": "Full Random", "structural_type": "Random partition", "role": "RANDOM STRUCTURAL CONTROL"},
}
REPRESENTATION = {
    "umbrae": {"spearman_rsa": 0.8527783548, "pearson_rsa": 0.8659444924, "recall_at_1": 0.9933333333},
    "lora": {"spearman_rsa": 0.8675653394, "pearson_rsa": 0.8796118013, "recall_at_1": 0.9966666667},
    "full_real": {"spearman_rsa": 0.8745309277, "pearson_rsa": 0.8881918487, "recall_at_1": 0.9966666667},
    "full_random": {"spearman_rsa": 0.8751833739, "pearson_rsa": 0.8877349792, "recall_at_1": 0.9966666667},
}
PRIMARY_METRICS = ("CIDEr", "BLEU-4", "ROUGE-L", "mean_iou", "grounding_accuracy")


def read_json(path):
    return json.loads(path.read_text())


def flatten(raw, mode):
    caption = raw["caption_metrics"][mode]
    grounding = raw["grounding_metrics"][mode]
    return {
        "CIDEr": caption["CIDEr"],
        "BLEU-4": caption["BLEU-4"],
        "ROUGE-L": caption["ROUGE-L"],
        "mean_iou": grounding["mean_iou"],
        "grounding_accuracy": grounding["grounding_accuracy"],
    }


def subtract(values, left, right):
    return {metric: values[left][metric] - values[right][metric] for metric in PRIMARY_METRICS}


def integrity(output, raw):
    report = {}
    caption_ids = {}
    grounding_keys = {}
    for mode in MODES:
        captions = read_json(output / mode / "caption_predictions.json")
        grounding = read_json(output / mode / "grounding_predictions.json")
        ids = [int(row["sample_id"]) for row in captions]
        keys = [(int(row["sample_id"]), row["expression"]) for row in grounding]
        caption_ids[mode] = ids
        grounding_keys[mode] = keys
        caption_metric = raw["caption_metrics"][mode]
        grounding_metric = raw["grounding_metrics"][mode]
        report[mode] = {
            "caption_rows": len(captions),
            "caption_unique_sample_ids": len(set(ids)),
            "caption_ids_exactly_0_to_981": ids == list(range(982)),
            "caption_status_counts": {
                status: sum(row["status"] == status for row in captions)
                for status in ("success", "empty", "failed")
            },
            "caption_denominator": caption_metric["denominator"],
            "grounding_rows": len(grounding),
            "grounding_unique_query_keys": len(set(keys)),
            "grounding_parse_failures": grounding_metric["parse_failures"],
            "grounding_target_failures": grounding_metric["target_failures"],
            "provenance_present": (output / mode / "provenance.json").is_file(),
        }
    report["cross_model"] = {
        "same_caption_sample_order": all(caption_ids[mode] == caption_ids["umbrae"] for mode in MODES[1:]),
        "same_grounding_query_order": all(grounding_keys[mode] == grounding_keys["umbrae"] for mode in MODES[1:]),
        "all_expected_outputs_present": all(
            (output / mode / name).is_file()
            for mode in MODES
            for name in ("caption_predictions.json", "caption_metrics.json", "grounding_predictions.json", "grounding_metrics.json", "provenance.json")
        ),
    }
    return report


def qualitative(output, count=10):
    fixed_ids = list(range(count))
    captions = {mode: read_json(output / mode / "caption_predictions.json") for mode in MODES}
    grounding = {mode: read_json(output / mode / "grounding_predictions.json") for mode in MODES}
    caption_lookup = {mode: {int(row["sample_id"]): row for row in rows} for mode, rows in captions.items()}
    ground_lookup = {}
    for mode, rows in grounding.items():
        by_sample = {}
        for row in rows:
            by_sample.setdefault(int(row["sample_id"]), []).append(row)
        ground_lookup[mode] = by_sample
    examples = []
    for sample_id in fixed_ids:
        base_caption = caption_lookup["umbrae"][sample_id]
        base_ground = ground_lookup["umbrae"][sample_id][0]
        expression = base_ground["expression"]
        ground_rows = {
            mode: next(row for row in ground_lookup[mode][sample_id] if row["expression"] == expression)
            for mode in MODES
        }
        examples.append({
            "sample_id": sample_id,
            "selection": "fixed first 10 sample IDs; first annotation-order grounding expression",
            "caption_references": base_caption["references"],
            "captions": {
                mode: {"prediction": caption_lookup[mode][sample_id]["caption"], "status": caption_lookup[mode][sample_id]["status"]}
                for mode in MODES
            },
            "grounding": {
                "expression": expression,
                "ground_truth_boxes": base_ground["ground_truth_boxes"],
                "predictions": {
                    mode: {"boxes": ground_rows[mode]["boxes"], "status": ground_rows[mode]["status"]}
                    for mode in MODES
                },
            },
        })
    return {"selection_policy": "deterministic, no result-based selection", "fixed_sample_ids": fixed_ids, "examples": examples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dual_branch_outputs/p7_downstream")
    args = parser.parse_args()
    output = Path(args.output_dir)
    raw = read_json(output / "p7_raw_metrics.json")
    feature_metadata = read_json(output / "feature_cache_metadata.json")
    values = {mode: flatten(raw, mode) for mode in MODES}
    rows = []
    for mode in MODES:
        rows.append({
            **MODEL_LABELS[mode],
            **REPRESENTATION[mode],
            **values[mode],
            "caption_integrity": {
                key: raw["caption_metrics"][mode][key]
                for key in ("sample_count", "successful_generations", "empty_outputs", "failed_generations", "denominator", "failure_policy")
            },
            "grounding_integrity": {
                key: raw["grounding_metrics"][mode][key]
                for key in ("query_count", "parse_failures", "target_failures", "threshold", "failure_policy")
            },
        })
    gains = {
        "lora_minus_umbrae": subtract(values, "lora", "umbrae"),
        "full_real_minus_lora": subtract(values, "full_real", "lora"),
        "full_random_minus_lora": subtract(values, "full_random", "lora"),
        "full_real_minus_full_random": subtract(values, "full_real", "full_random"),
    }
    audit = integrity(output, raw)
    all_integrity_pass = (
        audit["cross_model"]["same_caption_sample_order"]
        and audit["cross_model"]["same_grounding_query_order"]
        and audit["cross_model"]["all_expected_outputs_present"]
        and all(audit[mode]["caption_rows"] == 982 and audit[mode]["grounding_rows"] == 2419 for mode in MODES)
    )
    comparison = {
        "report": "P7 Downstream Validation",
        "protocol_version": "protocol_v1",
        "split": "test",
        "test_evaluation_started_after_model_selection": True,
        "main_method": "full_real",
        "random_structural_control": "full_random",
        "unified_results": rows,
        "downstream_gains": gains,
        "pre_evaluation_equivalence": feature_metadata["equivalence"],
        "projected_token_diagnostics": feature_metadata["projected_token_diagnostics"],
        "compute": raw["compute"],
        "parameter_efficiency": {
            "brainx_base_parameters": 146500608,
            "umbrae_calibration_trainable": 0,
            "lora_only_trainable": 245760,
            "lora_only_percent_of_brainx": 0.1677549356,
            "full_stage_c_lora_trainable": 245760,
            "full_stage_c_fusion_trainable": 4202497,
            "full_stage_c_total_trainable": 4448257,
            "full_stage_c_percent_of_brainx": 3.0363440504,
            "structural_branch_stage_c": "present but frozen; excluded from Stage-C trainable ratio",
        },
        "integrity_audit": audit,
        "integrity_pass": all_integrity_pass,
        "scientific_interpretation": {
            "DOWNSTREAM_UTILITY": "MIXED",
            "LORA_DOWNSTREAM_EFFECT": "NEGATIVE",
            "FUSION_DOWNSTREAM_EFFECT": "WEAK",
            "REAL_VS_RANDOM_CAPTION": "REAL_BETTER",
            "REAL_VS_RANDOM_GROUNDING": "REAL_BETTER",
            "ANATOMICAL_DOWNSTREAM_SIGNAL": "TASK_SPECIFIC",
            "P7_STATUS": "STRUCTURE_SIGNAL_EMERGES_IN_DOWNSTREAM",
            "scope_note": "Directional test-set comparison only; effect sizes are small and no uncertainty/significance estimate was requested or computed.",
        },
    }
    (output / "p7_downstream_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (output / "qualitative_examples.json").write_text(json.dumps(qualitative(output), indent=2) + "\n")
    print(json.dumps({"integrity_pass": all_integrity_pass, "gains": gains, "scientific_interpretation": comparison["scientific_interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
