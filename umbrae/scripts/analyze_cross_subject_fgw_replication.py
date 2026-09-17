#!/usr/bin/env python3
"""Frozen S1/S2/S5 plan comparison for Prompt-5B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ROIS = ["V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA"]
LAYERS = [4, 8, 12, 16, 20, 24]
SUBJECTS = ["subj01", "subj02", "subj05"]
PAIRS = [("subj01", "subj02"), ("subj01", "subj05"), ("subj02", "subj05")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for subject in SUBJECTS:
        parser.add_argument(f"--{subject}-result-dir", required=True)
        parser.add_argument(f"--{subject}-validation-summary", required=True)
        parser.add_argument(f"--{subject}-manifest", required=True)
    parser.add_argument("--subj07-validation-root", required=True)
    parser.add_argument("--pretest-protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    """Match the pre-test ``find ... | sort -z | sha256sum | sha256sum`` audit.

    The preregistration recorded paths relative to the repository working
    directory (including the validation-root prefix), so hashing paths merely
    relative to ``root`` would report a false mutation even when every byte is
    unchanged.
    """
    digest = hashlib.sha256()
    cwd = Path.cwd().resolve()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            displayed_path = path.resolve().relative_to(cwd)
        except ValueError as error:
            raise RuntimeError(
                "subj07 validation root must be below the repository working directory"
            ) from error
        digest.update(f"{sha256(path)}  {displayed_path}\n".encode())
    return digest.hexdigest()


def js(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)
    mask_left = left > 0
    mask_right = right > 0
    return float(
        0.5 * np.sum(left[mask_left] * np.log(left[mask_left] / middle[mask_left]))
        + 0.5 * np.sum(right[mask_right] * np.log(right[mask_right] / middle[mask_right]))
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else float("nan")


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value)


def upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), k=1)]


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "ci_2_5": float(np.percentile(array, 2.5)),
        "ci_97_5": float(np.percentile(array, 97.5)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def manifest_test_ids(path: Path) -> set[str]:
    manifest = json.loads(path.read_text())
    entry = manifest["offline_splits"]["offline_test"]
    indices = np.load(path.parent / entry["index_file"], allow_pickle=False)
    table = {int(row["global_cache_row"]): row for row in manifest["cache_row_table"]}
    result = {str(table[int(index)]["stable_stimulus_id"]) for index in indices}
    if len(result) != int(entry["unique_stimulus_count"]):
        raise RuntimeError("offline_test ID count mismatch")
    return result


def ambiguity_by_roi(validation: Mapping[str, Any]) -> dict[str, str]:
    selected = validation["selected_structural_config"]
    local = selected["local_minimum_identifiability"]
    reference = (
        local["near_optimal"]
        if local.get("near_optimal_count", 0) >= 2
        else local["top_k_analysis"]
    )
    result = {}
    for roi, item in zip(ROIS, reference["per_roi_row_correlation"]):
        median = item.get("median")
        result[roi] = "ambiguous" if median is None or float(median) < 1.0 else "stable"
    return result


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    prereg_path = Path(args.pretest_protocol).resolve()
    if output != prereg_path.parent:
        raise ValueError("Pretest protocol must live in output directory")
    existing = {path.name for path in output.iterdir()}
    if existing != {prereg_path.name}:
        raise FileExistsError(f"Unexpected pre-analysis output files: {existing}")
    prereg_hash = sha256(prereg_path)
    prereg = json.loads(prereg_path.read_text())
    if not prereg["written_before_subj02_or_subj05_offline_test_index_or_feature_access"]:
        raise RuntimeError("Cross-subject protocol was not preregistered")

    result_dirs = {
        subject: Path(getattr(args, f"{subject}_result_dir")).resolve()
        for subject in SUBJECTS
    }
    validation_paths = {
        subject: Path(getattr(args, f"{subject}_validation_summary")).resolve()
        for subject in SUBJECTS
    }
    manifest_paths = {
        subject: Path(getattr(args, f"{subject}_manifest")).resolve()
        for subject in SUBJECTS
    }
    plans = {}
    summaries = {}
    validations = {}
    for subject in SUBJECTS:
        plan_path = result_dirs[subject] / "final_transport_plan_pretest.npy"
        plans[subject] = np.load(plan_path, allow_pickle=False) * 8.0
        summaries[subject] = json.loads(
            (result_dirs[subject] / "offline_test_summary.json").read_text()
        )
        validations[subject] = json.loads(validation_paths[subject].read_text())
        if summaries[subject]["OFFLINE_TEST_STATUS"] != "EXPLORATORY_PASS":
            raise RuntimeError(f"{subject} final plan is not from EXPLORATORY_PASS")
        if sha256(plan_path) != summaries[subject]["FINAL_PLAN_HASH"]:
            raise RuntimeError(f"{subject} final plan hash mismatch")
        if not np.allclose(plans[subject].sum(axis=1), 1.0, atol=1e-12):
            raise RuntimeError(f"{subject} normalized rows do not sum to one")

    whole_rows = []
    roi_pair_rows = []
    for left, right in PAIRS:
        a, b = plans[left], plans[right]
        whole_rows.append(
            {
                "subject_pair": f"{left}-{right}",
                "flattened_spearman": spearman(a.ravel(), b.ravel()),
                "flattened_pearson": pearson(a.ravel(), b.ravel()),
                "frobenius_distance": float(np.linalg.norm(a - b)),
                "cosine_similarity": cosine(a.ravel(), b.ravel()),
                "target_marginal_js": js(a.sum(axis=0), b.sum(axis=0)),
                "preferred_layer_agreement": float(
                    np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1))
                ),
            }
        )
        for roi_index, roi in enumerate(ROIS):
            roi_pair_rows.append(
                {
                    "subject_pair": f"{left}-{right}",
                    "roi": roi,
                    "cosine_similarity": cosine(a[roi_index], b[roi_index]),
                    "js_divergence": js(a[roi_index], b[roi_index]),
                    "pearson": pearson(a[roi_index], b[roi_index]),
                    "spearman": spearman(a[roi_index], b[roi_index]),
                    "preferred_layer_agreement": int(
                        np.argmax(a[roi_index]) == np.argmax(b[roi_index])
                    ),
                    "left_preferred_layer": f"L{LAYERS[int(np.argmax(a[roi_index]))]}",
                    "right_preferred_layer": f"L{LAYERS[int(np.argmax(b[roi_index]))]}",
                }
            )
    write_csv(output / "cross_subject_plan_similarity.csv", whole_rows)

    ambiguity = {
        subject: ambiguity_by_roi(validations[subject]) for subject in SUBJECTS
    }
    roi_rows = []
    for roi in ROIS:
        pair_items = [row for row in roi_pair_rows if row["roi"] == roi]
        ambiguous_subjects = [s for s in SUBJECTS if ambiguity[s][roi] == "ambiguous"]
        mean_cos = float(np.mean([row["cosine_similarity"] for row in pair_items]))
        mean_js = float(np.mean([row["js_divergence"] for row in pair_items]))
        agreement = float(np.mean([row["preferred_layer_agreement"] for row in pair_items]))
        if mean_cos >= 0.9 and mean_js <= 0.1 and agreement == 1.0:
            interpretation = "high cross-subject stability"
        elif mean_cos >= 0.7 and agreement >= 2 / 3:
            interpretation = "moderate cross-subject stability"
        else:
            interpretation = "low or ambiguous cross-subject stability"
        roi_rows.append(
            {
                "roi": roi,
                "mean_pairwise_cosine_similarity": mean_cos,
                "mean_pairwise_js": mean_js,
                "preferred_layer_agreement": agreement,
                "within_subject_initialization_ambiguity": (
                    ",".join(ambiguous_subjects) if ambiguous_subjects else "none flagged"
                ),
                "prior_subj01_ambiguity_flag": roi in {"FFA", "PPA", "OPA"},
                "cross_subject_stability_interpretation": interpretation,
            }
        )
    write_csv(output / "cross_subject_roi_similarity.csv", roi_rows)

    real_cos = float(np.mean([row["cosine_similarity"] for row in roi_pair_rows]))
    real_js = float(np.mean([row["js_divergence"] for row in roi_pair_rows]))
    draws = int(prereg["cross_subject_analysis"]["roi_row_permutation_null"]["draws"])
    rng = np.random.default_rng(
        int(prereg["cross_subject_analysis"]["roi_row_permutation_null"]["seed"])
    )
    null_cos = np.empty(draws, dtype=np.float64)
    null_js = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        permuted = {
            subject: plans[subject][rng.permutation(8)] for subject in SUBJECTS
        }
        cos_values, js_values = [], []
        for left, right in PAIRS:
            for roi in range(8):
                cos_values.append(cosine(permuted[left][roi], permuted[right][roi]))
                js_values.append(js(permuted[left][roi], permuted[right][roi]))
        null_cos[draw] = np.mean(cos_values)
        null_js[draw] = np.mean(js_values)
    cosine_p = float((1 + np.sum(null_cos >= real_cos)) / (draws + 1))
    js_p = float((1 + np.sum(null_js <= real_js)) / (draws + 1))
    np.savez_compressed(
        output / "cross_subject_plan_null_values.npz",
        mean_corresponding_roi_cosine=null_cos,
        mean_corresponding_roi_js=null_js,
    )
    null_summary = {
        "construction": "independent ROI row-label permutation within each frozen subject plan",
        "draws": draws,
        "seed": prereg["cross_subject_analysis"]["roi_row_permutation_null"]["seed"],
        "clip_layer_order_permuted": False,
        "real_mean_corresponding_roi_cosine": real_cos,
        "null_mean_corresponding_roi_cosine": distribution(null_cos),
        "one_sided_p_real_cosine_higher": cosine_p,
        "real_mean_corresponding_roi_js": real_js,
        "null_mean_corresponding_roi_js": distribution(null_js),
        "one_sided_p_real_js_lower": js_p,
        "corresponding_named_rows_exceed_null": cosine_p < 0.05 and js_p < 0.05,
    }
    write_json(output / "cross_subject_plan_null_summary.json", null_summary)

    subject_rows = []
    geometry_rows = []
    roles = {
        "subj01": "development subject",
        "subj02": "replication subject",
        "subj05": "replication subject",
    }
    for subject in SUBJECTS:
        summary_data = summaries[subject]
        metrics = summary_data["offline_test_metrics"]
        differences = summary_data["fgw_minus_baseline"]
        p_values = summary_data["geometry_null_p_values"]
        subject_rows.append(
            {
                "subject": subject,
                "role": roles[subject],
                "offline_test_status": summary_data["OFFLINE_TEST_STATUS"],
                "final_plan_sha256": summary_data["FINAL_PLAN_HASH"],
                "offline_test_composite": metrics["offline_test_composite"],
                "delta_feature_normalized": differences["feature_normalized"],
                "delta_gw_normalized": differences["gw_normalized"],
                "delta_composite": differences["composite"],
                "maximum_structural_null_p": max(p_values.values()),
                "feature_pairing_p": summary_data["feature_pairing_null"][
                    "one_sided_p_real_feature_lower"
                ],
            }
        )
        train_brain = np.load(
            result_dirs[subject] / "final_train_brain_geometry.npy", allow_pickle=False
        )
        test_brain = np.load(
            result_dirs[subject] / "offline_test_brain_geometry.npy", allow_pickle=False
        )
        train_clip = np.load(
            result_dirs[subject] / "final_train_clip_geometry.npy", allow_pickle=False
        )
        test_clip = np.load(
            result_dirs[subject] / "offline_test_clip_geometry.npy", allow_pickle=False
        )
        geometry_rows.append(
            {
                "subject": subject,
                "role": roles[subject],
                "brain_final_train_vs_test_spearman": spearman(
                    upper(train_brain), upper(test_brain)
                ),
                "clip_final_train_vs_test_spearman": spearman(
                    upper(train_clip), upper(test_clip)
                ),
                "brain_geometry_frobenius_distance": float(
                    np.linalg.norm(train_brain - test_brain)
                ),
                "clip_geometry_frobenius_distance": float(
                    np.linalg.norm(train_clip - test_clip)
                ),
                "delta_gw_normalized": differences["gw_normalized"],
                "delta_composite": differences["composite"],
                "maximum_structural_null_p": max(p_values.values()),
                "offline_test_status": summary_data["OFFLINE_TEST_STATUS"],
            }
        )
    write_csv(output / "subject_replication_summary.csv", subject_rows)
    write_json(output / "subject_replication_summary.json", {"subjects": subject_rows})
    write_csv(output / "cross_subject_geometry_summary.csv", geometry_rows)

    test_id_sets = {
        subject: manifest_test_ids(manifest_paths[subject]) for subject in SUBJECTS
    }
    shared = set.intersection(*(test_id_sets[subject] for subject in SUBJECTS))
    pairwise_shared = {
        f"{left}-{right}": len(test_id_sets[left] & test_id_sets[right])
        for left, right in PAIRS
    }
    if len(shared) >= int(prereg["cross_subject_analysis"]["shared_stimulus_minimum"]):
        shared_status = "AVAILABLE_BUT_NOT_IMPLEMENTED"
        raise RuntimeError(
            "Shared held-out count reached threshold; common-stimulus feature analysis is required"
        )
    else:
        shared_status = "INSUFFICIENT_HELDOUT_SHARED_STIMULI"
    write_json(
        output / "shared_stimulus_analysis.json",
        {
            "SHARED_STIMULUS_ANALYSIS": shared_status,
            "minimum_required": prereg["cross_subject_analysis"][
                "shared_stimulus_minimum"
            ],
            "all_three_shared_count": len(shared),
            "pairwise_shared_counts": pairwise_shared,
            "subject_offline_test_counts": {
                subject: len(values) for subject, values in test_id_sets.items()
            },
            "subj07_included": False,
            "can_change_cross_subject_status": False,
        },
    )

    replication_rows = [row for row in subject_rows if row["subject"] in {"subj02", "subj05"}]
    both_pass = all(row["offline_test_status"] == "EXPLORATORY_PASS" for row in replication_rows)
    both_negative = all(row["delta_gw_normalized"] < 0 for row in replication_rows)
    correspondence_pass = null_summary["corresponding_named_rows_exceed_null"]
    if both_pass and both_negative and correspondence_pass:
        cross_status = "REPLICATION_SUPPORTED"
    elif any(row["offline_test_status"] == "EXPLORATORY_PASS" for row in replication_rows):
        cross_status = "MIXED_REPLICATION"
    else:
        cross_status = "NO_REPLICATION"
    replication_effects = [row["delta_gw_normalized"] for row in replication_rows]
    status_payload = {
        "CROSS_SUBJECT_STATUS": cross_status,
        "both_eligible_replication_subjects_exploratory_pass": both_pass,
        "both_replication_delta_gw_negative": both_negative,
        "named_roi_correspondence_exceeds_row_permutation_null": correspondence_pass,
        "replication_delta_gw_median": float(np.median(replication_effects)),
        "replication_delta_gw_range": [min(replication_effects), max(replication_effects)],
        "replication_delta_gw_sign_consistent": both_negative,
        "subj07_role": "validation-inconclusive; no offline test and no final plan",
        "subj07_counted_as_pass_or_fail": False,
        "scientific_hyperparameter_retuning": False,
        "inferential_t_test_over_three_subjects": False,
    }
    write_json(output / "cross_subject_replication_status.json", status_payload)

    subj07_root = Path(args.subj07_validation_root).resolve()
    subj07_hash_after = tree_hash(subj07_root)
    expected_subj07_hash = prereg["ineligible_subjects"]["subj07"][
        "tree_sha256_before_prompt5b"
    ]
    if subj07_hash_after != expected_subj07_hash:
        raise RuntimeError("subj07 validation tree changed during Prompt-5B")
    write_json(
        output / "cross_subject_provenance.json",
        {
            "pretest_protocol_sha256": prereg_hash,
            "pretest_protocol_unchanged": sha256(prereg_path) == prereg_hash,
            "plan_hashes": {
                subject: summaries[subject]["FINAL_PLAN_HASH"] for subject in SUBJECTS
            },
            "result_summary_hashes": {
                subject: sha256(result_dirs[subject] / "offline_test_summary.json")
                for subject in SUBJECTS
            },
            "validation_summary_hashes": {
                subject: sha256(validation_paths[subject]) for subject in SUBJECTS
            },
            "manifest_hashes": {
                subject: sha256(manifest_paths[subject]) for subject in SUBJECTS
            },
            "subj07_tree_sha256_before": expected_subj07_hash,
            "subj07_tree_sha256_after": subj07_hash_after,
            "subj07_offline_test_loaded": False,
            "subj07_final_plan_generated": False,
            "native_clip_layer_order": LAYERS,
            "layer_order_optimized_or_permuted": False,
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "stage2_run": False,
            "perceiver_modified": False,
            "caption_evaluation_run": False,
        },
    )
    print(f"CROSS_SUBJECT_STATUS = {cross_status}")
    print(f"ROI null p(cosine)={cosine_p:.6g} p(JS)={js_p:.6g}")
    print(f"SHARED_STIMULUS_ANALYSIS = {shared_status} (N={len(shared)})")


if __name__ == "__main__":
    main()
