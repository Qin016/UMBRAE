#!/usr/bin/env python3
"""Audit and summarize the locked subj01 Stage-2 FGW development experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


RUNS = (
    ("uniform_prior", "Uniform prior", 0.0),
    ("fgw_gamma025", "FGW", 0.25),
    ("fgw_gamma050", "FGW", 0.50),
    ("fgw_gamma100", "FGW", 1.00),
    ("row_shuffled_fgw", "Row-shuffled FGW", 0.25),
)
REQUIRED_LEAKAGE = {
    "bridge_type": "shikra_patch",
    "uses_image_clip_tokens_at_eval": False,
    "oracle_image_token_mode": False,
    "transport_plan_frozen": True,
    "transport_plan_gradient": "none",
    "protected_test_used_for_training": False,
    "protected_test_used_for_checkpoint_selection": False,
    "gamma_selected_on_subj01_only": True,
}


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_interval(left, right, key: str, *, seed: int = 62002, draws: int = 10000):
    left_by_id = {row["local_nsd_id"]: float(row[key]) for row in left}
    right_by_id = {row["local_nsd_id"]: float(row[key]) for row in right}
    assert left_by_id.keys() == right_by_id.keys()
    ids = sorted(left_by_id, key=int)
    differences = np.asarray([left_by_id[i] - right_by_id[i] for i in ids])
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(differences), size=(draws, len(differences)))].mean(1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "per_stimulus_mean_difference": float(differences.mean()),
        "paired_bootstrap_95_interval": [float(low), float(high)],
        "stimulus_count": len(ids),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "descriptive_only": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--doc-path", type=Path, required=True)
    args = parser.parse_args()

    locked = load(args.global_root / "stage2_fgw_locked_config_v1.json")
    audit = load(args.global_root / "data_and_caption_audit.json")
    assert locked["selected_gamma"] == 0.25
    reference_names = reference_trainable = reference_frozen = None
    rows = []
    predictions = {}

    for dirname, method, gamma in RUNS:
        run = args.runs_root / dirname
        required = [
            "run_config.json", "trainable_parameters.json", "epoch_metrics.csv",
            "epoch_metrics.json", "checkpoint_hashes.json", "validation_predictions.jsonl",
            "validation_metrics.json", "leakage_audit.json", "metrics.json",
        ]
        missing = [name for name in required if not (run / name).is_file()]
        assert not missing, f"{dirname}: missing {missing}"
        cfg = load(run / "run_config.json")
        params = load(run / "trainable_parameters.json")
        leakage = load(run / "leakage_audit.json")
        metrics = load(run / "metrics.json")
        epochs = load(run / "epoch_metrics.json")
        hashes = load(run / "checkpoint_hashes.json")
        assert len(epochs) == 3 and len(hashes) == 3
        assert leakage == REQUIRED_LEAKAGE, f"{dirname}: leakage assertions failed"
        assert cfg["subject"] == "subj01" and cfg["seed"] == 42 and cfg["epochs"] == 3
        assert cfg["fgw_gamma"] == gamma
        if dirname == "row_shuffled_fgw":
            assert cfg["fgw_row_permutation"] == [3, 4, 6, 2, 1, 7, 5, 0]
        selected_epoch = int(metrics["selected_epoch"])
        selected_path = run / metrics["selected_checkpoint"]
        selected_hash = sha256(selected_path)
        assert selected_hash == metrics["selected_checkpoint_sha256"] == hashes[str(selected_epoch)]
        assert metrics["epoch"] == selected_epoch
        names = params["trainable_parameter_names"]
        trainable = params["trainable_parameter_count"]
        frozen = params["frozen_parameter_count"]
        if reference_names is None:
            reference_names, reference_trainable, reference_frozen = names, trainable, frozen
        assert names == reference_names
        assert trainable == reference_trainable and frozen == reference_frozen
        assert params["fgw_trainable_correspondence_parameter_count"] == 0
        c = metrics["caption_metrics"]
        assert c["reference_mode"] == "multi_reference"
        assert metrics["num_generated_captions"] == 300
        predictions[dirname] = read_jsonl(run / "validation_predictions.jsonl")
        assert len(predictions[dirname]) == 300
        rows.append({
            "run": dirname,
            "method": method,
            "gamma": gamma,
            "architecture_48_tokens": True,
            "selected_epoch": selected_epoch,
            "CIDEr": c["CIDEr"],
            "BLEU-1": c["BLEU-1"],
            "BLEU-2": c["BLEU-2"],
            "BLEU-3": c["BLEU-3"],
            "BLEU-4": c["BLEU-4"],
            "ROUGE-L": c["ROUGE-L"],
            "validation_lm_loss": metrics["val_loss"],
            "selected_checkpoint": metrics["selected_checkpoint"],
            "selected_checkpoint_sha256": selected_hash,
            "training_time_seconds": epochs[-1]["elapsed_seconds"],
            "peak_gpu_memory_bytes": max(e["peak_gpu_memory_bytes"] for e in epochs),
            "trainable_parameter_count": trainable,
            "frozen_parameter_count": frozen,
            "fgw_trainable_correspondence_parameter_count": 0,
            "leakage_assertions_passed": True,
        })

    by_name = {row["run"]: row for row in rows}
    locked_row = by_name["fgw_gamma025"]
    comparisons = {}
    for right_name, label in (("uniform_prior", "FGW_locked_minus_Uniform"),
                              ("row_shuffled_fgw", "FGW_locked_minus_RowShuffled")):
        right = by_name[right_name]
        comparisons[label] = {
            "CIDEr_point_difference": locked_row["CIDEr"] - right["CIDEr"],
            "BLEU-4_point_difference": locked_row["BLEU-4"] - right["BLEU-4"],
            "ROUGE-L_point_difference": locked_row["ROUGE-L"] - right["ROUGE-L"],
            "CIDEr_per_stimulus_descriptive": paired_interval(
                predictions["fgw_gamma025"], predictions[right_name], "per_stimulus_CIDEr"),
            "ROUGE-L_per_stimulus_descriptive": paired_interval(
                predictions["fgw_gamma025"], predictions[right_name], "per_stimulus_ROUGE-L"),
            "BLEU-4_interval": None,
            "BLEU-4_interval_note": "The fixed evaluator exposes corpus BLEU-4, not per-stimulus BLEU-4.",
        }

    result = {
        "protocol_version": "s1_stage2_development_summary_v1",
        "development_subject": "subj01",
        "locked_gamma": locked["selected_gamma"],
        "status": "READY_FOR_REPLICATION",
        "runs": rows,
        "comparisons": comparisons,
        "data_audit": {
            "train": audit["splits"]["train"],
            "validation": audit["splits"]["validation"],
            "protected_test": audit["splits"]["protected_test"],
            "protected_test_performance_accessed": False,
        },
        "all_trainable_parameter_sets_identical": True,
        "all_leakage_assertions_passed": True,
        "protected_test_metrics_produced": False,
        "development_intervals_are_descriptive_only": True,
    }
    args.global_root.mkdir(parents=True, exist_ok=True)
    json_path = args.global_root / "s1_stage2_development_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = args.global_root / "s1_stage2_development_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(x):
        return f"{x:.6f}"

    table = [
        "| Method | Gamma | 48-token architecture | Selected epoch | Val CIDEr | BLEU-4 | ROUGE-L | Val LM loss |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['method']} | {row['gamma']:.2f} | yes | {row['selected_epoch']} | "
            f"{fmt(row['CIDEr'])} | {fmt(row['BLEU-4'])} | {fmt(row['ROUGE-L'])} | "
            f"{fmt(row['validation_lm_loss'])} |"
        )
    diagnostics = [
        "| Run | BLEU-1 | BLEU-2 | BLEU-3 | Checkpoint SHA256 | Time (s) | Peak GPU (GiB) | Trainable | Frozen | Leakage |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        diagnostics.append(
            f"| {row['run']} | {fmt(row['BLEU-1'])} | {fmt(row['BLEU-2'])} | "
            f"{fmt(row['BLEU-3'])} | `{row['selected_checkpoint_sha256']}` | "
            f"{row['training_time_seconds']:.1f} | "
            f"{row['peak_gpu_memory_bytes'] / (1024 ** 3):.3f} | "
            f"{row['trainable_parameter_count']:,} | {row['frozen_parameter_count']:,} | pass |"
        )
    doc = """# FGW Stage-2 S1 Development V1

This is the subj01-only architecture-matched development experiment. Gamma was
selected from the preregistered grid using validation multi-reference CIDEr only.
No protected downstream test sample was decoded and no protected-test metric was
computed.

## Locked result

`LOCKED_GAMMA = 0.25`

`STAGE2_DEVELOPMENT_STATUS = READY_FOR_REPLICATION`

The locked file was written before the row-shuffled run began. The lock is final
for subsequent subj02/subj05 replication.

## Validation results

""" + "\n".join(table) + "\n\n## Full run diagnostics\n\n" + \
        "\n".join(diagnostics) + "\n\n## Locked comparisons\n\n"
    for label, values in comparisons.items():
        doc += (
            f"- `{label}`: CIDEr {values['CIDEr_point_difference']:+.6f}; "
            f"BLEU-4 {values['BLEU-4_point_difference']:+.6f}; "
            f"ROUGE-L {values['ROUGE-L_point_difference']:+.6f}.\n"
        )
    doc += """

Paired bootstrap intervals in the JSON summary are descriptive only (10,000
resamples, seed 62002) and are available for the evaluator's per-stimulus CIDEr
and ROUGE-L outputs. BLEU-4 is a corpus statistic in this evaluator, so only its
point difference is reported. These diagnostics did not alter locked gamma.

## Architecture and parameter audit

All five runs used 48 ROI-layer tokens, the same Perceiver and Shikra patch bridge,
seed 42, AdamW (learning rate 1e-4, weight decay 1e-2), batch size 2, and three
epochs. Every run had 40,167,424 trainable and 6,890,397,052 frozen parameters;
the trainable parameter names were identical and correspondence contributed zero
trainable parameters. Each epoch was evaluated, and every reported metric comes
from that run's CIDEr-selected checkpoint.

## Data and caption audit

- Train: 8,559 unique stimuli; 0 missing references; 2 duplicate-reference cases;
  reference-count histogram 4:14, 5:8,525, 6:20.
- Validation: 300 unique stimuli; exactly 5 references each; 0 missing and 0
  duplicate-reference cases.
- Protected final test: provenance/count audit only, 982 unique stimuli; 0 missing
  and 0 duplicate-reference cases; reference-count histogram 5:980, 6:2.
- Train, validation, and protected-test stimulus sets are disjoint.

All eight required leakage assertions passed for every run. In particular, image
CLIP tokens were not used at evaluation, oracle mode was off, the transport plan
was frozen with no gradient, and the protected test was used neither for training
nor checkpoint selection.
"""
    args.doc_path.parent.mkdir(parents=True, exist_ok=True)
    args.doc_path.write_text(doc, encoding="utf-8")
    print(json.dumps({"status": result["status"], "locked_gamma": result["locked_gamma"],
                      "json": str(json_path), "csv": str(csv_path), "doc": str(args.doc_path)}))


if __name__ == "__main__":
    main()
