#!/usr/bin/env python3
"""Locked final refit followed by a one-shot offline-test evaluation.

The program enforces a hard two-phase boundary.  It first fits and persists the
final real, baseline, and null plans using discovery+validation only.  The
offline-test index/features are not opened until the final freeze marker has
been durably written and its hashes have been verified.
"""

from __future__ import annotations

# This is an executable protocol driver, not a pytest module.  The repository's
# ``*_test.py`` collection pattern otherwise mistakes its helper functions for
# fixture-based tests.
__test__ = False

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_representation_geometry import (
    cosine_distance_matrices,
    relation_matrix_from_rdms,
)
from scripts.analyze_representation_geometry_v2 import gather_cache_rows, sha256_ids
from scripts.build_fgw_feature_cost_v2 import (
    build_unique_id_folds,
    cosine_costs,
    fit_ridge_predict_multioutput,
)
from scripts.validate_fgw_correspondence import (
    LossCalibration,
    best_discovery_result,
    clean_json,
    coupling_diagnostics,
    feature_pairing_null_cost,
    fit_plan_initializations,
    independently_permuted_brain_geometry,
    initialization_plan,
    one_sided_lower_p,
    sha256_file,
    solve_projected_simplex,
    summary,
    unique_permutations,
    validation_metrics,
    write_dict_rows,
    write_json,
    write_matrix_csv,
)


EXPECTED_LOCK = {
    "method": "sr_fgw",
    "structure_weight": 0.5,
    "coverage_weight": 0.0,
    "entropy_weight": 0.0,
    "VALIDATION_STATUS": "READY_FOR_OFFLINE_TEST",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked refit and one-shot FGW offline test")
    parser.add_argument("--stimulus-manifest", required=True)
    parser.add_argument("--locked-configuration", required=True)
    parser.add_argument("--validation-summary", required=True)
    parser.add_argument("--feature-cost-dir", required=True)
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--solver-device", default="cpu")
    return parser.parse_args()


def ordered_ids(
    manifest: Mapping[str, Any], indices: np.ndarray
) -> List[str]:
    table = {int(row["global_cache_row"]): row for row in manifest["cache_row_table"]}
    return [str(table[int(index)]["stable_stimulus_id"]) for index in indices]


def load_non_test_split(
    manifest: Mapping[str, Any], manifest_path: Path, split: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if split == "offline_test":
        raise RuntimeError("offline_test is locked before final_plan_frozen.marker")
    entry = manifest["offline_splits"][split]
    indices = np.load(manifest_path.parent / entry["index_file"], allow_pickle=False)
    ids = ordered_ids(manifest, indices)
    cache_root = Path(manifest["representation_cache"])
    brain, _ = gather_cache_rows(
        cache_root, manifest["cache_row_table"], indices, "brain_roi_features.npy"
    )
    clip, _ = gather_cache_rows(
        cache_root, manifest["cache_row_table"], indices, "clip_layer_features.npy"
    )
    return indices, brain, clip, ids


def assert_id_hash(ids: Sequence[str], expected: str, label: str) -> None:
    actual = sha256_ids(sorted(ids))
    if actual != expected:
        raise ValueError(f"{label} ID hash mismatch: {actual} != {expected}")


def verify_locked_inputs(
    manifest_path: Path,
    locked_path: Path,
    validation_summary_path: Path,
    feature_dir: Path,
    geometry_dir: Path,
    validation_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    locked = json.loads(locked_path.read_text())
    validation_summary = json.loads(validation_summary_path.read_text())
    feature_config = json.loads((feature_dir / "feature_cost_config.json").read_text())
    geometry_config = json.loads((geometry_dir / "geometry_config.json").read_text())
    for field, expected in EXPECTED_LOCK.items():
        if locked.get(field) != expected:
            raise RuntimeError(
                f"OFFLINE_TEST_STATUS = BLOCKED_BY_CONFIG_MISMATCH: {field}="
                f"{locked.get(field)!r}, expected {expected!r}"
            )
    selected = validation_summary["selected_structural_config"]
    if selected.get("best_discovery_initialization") != locked.get(
        "best_discovery_initialization"
    ):
        raise RuntimeError(
            "OFFLINE_TEST_STATUS = BLOCKED_BY_FROZEN_ASSET_MISMATCH: "
            "selected initialization disagrees across frozen Prompt-5A assets"
        )
    if validation_summary.get("VALIDATION_STATUS") != "READY_FOR_OFFLINE_TEST":
        raise RuntimeError(
            "OFFLINE_TEST_STATUS = BLOCKED_BY_FROZEN_ASSET_MISMATCH: "
            "validation summary is not READY_FOR_OFFLINE_TEST"
        )
    if validation_summary.get("data_usage", {}).get("offline_test_accessed") is not False:
        raise RuntimeError(
            "OFFLINE_TEST_STATUS = BLOCKED_BY_FROZEN_ASSET_MISMATCH: "
            "Prompt-5A offline_test_count_loaded is not zero"
        )
    baseline = validation_summary["strongest_nonstructural_baseline"]
    expected_baseline = {
        "method": "source_constrained_feature_coverage",
        "structure_weight": 0.0,
        "coverage_weight": 1.0,
    }
    for field, expected in expected_baseline.items():
        if baseline.get(field) != expected:
            raise RuntimeError(
                f"OFFLINE_TEST_STATUS = BLOCKED_BY_CONFIG_MISMATCH: baseline {field}"
            )
    hashes = locked["matrix_hashes"]
    checks = {
        manifest_path: hashes["stimulus_manifest_sha256"],
        feature_dir / "feature_cost_oof.npy": hashes["feature_cost_discovery_sha256"],
        geometry_dir / "brain_geometry.npy": hashes["brain_geometry_discovery_sha256"],
        geometry_dir / "clip_geometry.npy": hashes["clip_geometry_discovery_sha256"],
        validation_dir / "feature_cost_validation.npy": hashes["feature_cost_validation_sha256"],
        validation_dir / "brain_geometry_validation.npy": hashes["brain_geometry_validation_sha256"],
        validation_dir / "clip_geometry_validation.npy": hashes["clip_geometry_validation_sha256"],
        Path(__file__).resolve().parent / "validate_fgw_correspondence.py": hashes[
            "validation_script_sha256"
        ],
    }
    for path, expected in checks.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"OFFLINE_TEST_STATUS = BLOCKED_BY_CONFIG_MISMATCH: hash {path}"
            )
    if feature_config["analysis"] != "cross_fitted_roi_to_clip_feature_compatibility_v2":
        raise RuntimeError("OFFLINE_TEST_STATUS = BLOCKED_BY_CONFIG_MISMATCH: feature protocol")
    if geometry_config["analysis_version"] != "geometry_v2_corrected":
        raise RuntimeError("OFFLINE_TEST_STATUS = BLOCKED_BY_CONFIG_MISMATCH: geometry protocol")
    if geometry_config["primary_brain_representation"] != "roi_tokens_before_projector":
        raise RuntimeError("OFFLINE_TEST_STATUS = BLOCKED_BY_CONFIG_MISMATCH: brain representation")
    fixed = locked["fixed_loss_calibration"]
    for key in ("s_feature", "s_gw", "eps_scale"):
        if key not in fixed or not np.isfinite(float(fixed[key])) or float(fixed[key]) <= 0:
            raise RuntimeError(
                "OFFLINE_TEST_STATUS = BLOCKED_BY_FROZEN_ASSET_MISMATCH: loss scales"
            )
    if float(fixed["eps_scale"]) != 1e-12:
        raise RuntimeError(
            "OFFLINE_TEST_STATUS = BLOCKED_BY_FROZEN_ASSET_MISMATCH: eps_scale"
        )
    for split in ("offline_discovery", "offline_validation", "offline_test"):
        entry = manifest["offline_splits"][split]
        if (
            int(entry["unique_stimulus_count"]) <= 0
            or entry["cache_row_count"] != entry["unique_stimulus_count"]
        ):
            raise RuntimeError(f"OFFLINE_TEST_STATUS = BLOCKED_BY_CONFIG_MISMATCH: {split} count")
        if not entry.get("sorted_id_sha256"):
            raise RuntimeError(f"OFFLINE_TEST_STATUS = BLOCKED_BY_CONFIG_MISMATCH: {split} ID hash")
    overlaps = manifest["overlap_checks"]
    if any(overlaps["offline_unique_stimulus_overlaps"].values()):
        raise RuntimeError("OFFLINE_TEST_STATUS = BLOCKED_BY_CONFIG_MISMATCH: offline overlap")
    if any(
        value
        for split_values in overlaps["protected_set_overlaps"].values()
        for value in split_values.values()
    ):
        raise RuntimeError("OFFLINE_TEST_STATUS = BLOCKED_BY_CONFIG_MISMATCH: protected overlap")
    return manifest, locked, validation_summary, feature_config, geometry_config


def crossfit_cost(
    brain: np.ndarray,
    clip: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    std_epsilon: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(brain)
    per_stimulus = np.empty((n, 8, 6), dtype=np.float32)
    covered = np.zeros(n, dtype=np.int64)
    for fold in range(int(assignments.max()) + 1):
        heldout = np.flatnonzero(assignments == fold)
        training = np.flatnonzero(assignments != fold)
        y_train = np.ascontiguousarray(clip[training].reshape(len(training), -1))
        for roi in range(8):
            prediction = fit_ridge_predict_multioutput(
                brain[training, roi],
                y_train,
                brain[heldout, roi],
                alpha,
                std_epsilon,
                device,
            ).reshape(len(heldout), 6, 1024)
            per_stimulus[heldout, roi] = cosine_costs(prediction, clip[heldout])
        covered[heldout] += 1
        print(f"final-train OOF fold {fold + 1}/{int(assignments.max()) + 1}", flush=True)
    if not np.all(covered == 1):
        raise RuntimeError("Final OOF feature cost did not cover every stimulus exactly once")
    return per_stimulus.mean(axis=0, dtype=np.float64), per_stimulus


def fit_full_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    alpha: float,
    std_epsilon: float,
    device: str,
) -> Dict[str, np.ndarray]:
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, unbiased=False, keepdim=True)
    x_std = torch.where(x_std > std_epsilon, x_std, torch.ones_like(x_std))
    y_mean = y.mean(dim=0, keepdim=True)
    standardized = (x - x_mean) / x_std
    centered_y = y - y_mean
    gram = standardized.T @ standardized
    gram.diagonal().add_(float(alpha))
    right = standardized.T @ centered_y
    try:
        factor = torch.linalg.cholesky(gram)
        weights = torch.cholesky_solve(right, factor)
    except RuntimeError:
        weights = torch.linalg.solve(gram, right)
    return {
        "x_mean": x_mean.cpu().numpy().astype(np.float32, copy=False),
        "x_std": x_std.cpu().numpy().astype(np.float32, copy=False),
        "y_mean": y_mean.cpu().numpy().astype(np.float32, copy=False),
        "weights": weights.cpu().numpy().astype(np.float32, copy=False),
    }


def predict_full_probe(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(x, dtype=np.float32) - model["x_mean"]) / model["x_std"]
    return standardized @ model["weights"] + model["y_mean"]


def hash_probe_models(models: Sequence[Mapping[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for roi, model in enumerate(models):
        digest.update(f"roi={roi}\n".encode())
        for key in ("x_mean", "x_std", "y_mean", "weights"):
            array = np.ascontiguousarray(model[key])
            digest.update(key.encode())
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def test_feature_cost(
    models: Sequence[Mapping[str, np.ndarray]],
    brain: np.ndarray,
    clip: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    costs = np.empty((len(brain), 8, 6), dtype=np.float32)
    for roi, model in enumerate(models):
        prediction = predict_full_probe(model, brain[:, roi]).reshape(len(brain), 6, 1024)
        costs[:, roi] = cosine_costs(prediction, clip)
    return costs.mean(axis=0, dtype=np.float64), costs


def fixed_anchor_positions(sample_count: int, anchor_count: int, seed: int) -> np.ndarray:
    if anchor_count > sample_count:
        raise ValueError("Anchor count exceeds samples")
    anchor_stream = np.random.SeedSequence(seed).spawn(5)[0]
    return np.sort(
        np.random.default_rng(anchor_stream).choice(
            sample_count, size=anchor_count, replace=False
        )
    )


def compute_geometry(
    brain: np.ndarray,
    clip: np.ndarray,
    anchor_positions: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    brain_rdms = cosine_distance_matrices(brain, anchor_positions, device)
    clip_rdms = cosine_distance_matrices(clip, anchor_positions, device)
    positions = np.arange(len(anchor_positions), dtype=np.int64)
    return (
        relation_matrix_from_rdms(brain_rdms, positions),
        relation_matrix_from_rdms(clip_rdms, positions),
        brain_rdms,
        clip_rdms,
    )


def fit_one_plan(
    feature: np.ndarray,
    brain_geometry: np.ndarray,
    clip_geometry: np.ndarray,
    calibration: LossCalibration,
    beta: float,
    coverage: float,
    init_kind: str,
    init_seed: int,
    solver: Mapping[str, Any],
    device: str,
) -> Dict[str, Any]:
    result = solve_projected_simplex(
        feature,
        brain_geometry,
        clip_geometry,
        calibration,
        beta,
        coverage,
        initialization_plan(init_kind, feature, init_seed),
        entropy_weight=float(solver["entropy_weight"]),
        learning_rate=float(solver["learning_rate"]),
        max_iterations=int(solver["max_iterations"]),
        tolerance=float(solver["tolerance"]),
        patience=int(solver["patience"]),
        eps_cov=float(solver["eps_coverage"]),
        device=device,
    )
    if not result["converged"]:
        raise RuntimeError(
            f"Locked initialization {init_kind}/{init_seed} failed numerically; no replacement allowed"
        )
    plan = result["plan"]
    if not np.isfinite(plan).all() or np.min(plan) < -1e-12:
        raise RuntimeError("Final plan has NaN/Inf or negative mass")
    if not np.allclose(plan.sum(axis=1), 1.0 / 8.0, atol=1e-10):
        raise RuntimeError("Final plan violates source marginal")
    if not np.isclose(plan.sum(), 1.0, atol=1e-10):
        raise RuntimeError("Final plan total mass is not one")
    return result


def fit_null(
    feature: np.ndarray,
    brain_geometry: np.ndarray,
    clip_geometry: np.ndarray,
    calibration: LossCalibration,
    null_specs: Sequence[Sequence[Any]],
    solver: Mapping[str, Any],
    device: str,
) -> Dict[str, Any]:
    results = fit_plan_initializations(
        feature,
        brain_geometry,
        clip_geometry,
        calibration,
        0.5,
        0.0,
        null_specs,
        solver,
        validation_inputs=None,
        device=device,
    )
    index, winner = best_discovery_result(results)
    return {"winner_index": index, **winner}


def save_plan_csv(
    path: Path, plan: np.ndarray, roi_names: Sequence[str], layer_labels: Sequence[str]
) -> None:
    write_matrix_csv(path, plan, roi_names, layer_labels)


def durable_marker(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(clean_json(payload), indent=2, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def unlock_offline_test(
    marker_path: Path,
    plan_path: Path,
    expected_plan_hash: str,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if not marker_path.is_file():
        raise RuntimeError("offline_test remains locked: final marker missing")
    marker = json.loads(marker_path.read_text())
    if marker["final_plan_sha256"] != expected_plan_hash:
        raise RuntimeError("offline_test remains locked: marker plan hash mismatch")
    if sha256_file(plan_path) != expected_plan_hash:
        raise RuntimeError("offline_test remains locked: frozen plan file changed")
    entry = manifest["offline_splits"]["offline_test"]
    indices = np.load(manifest_path.parent / entry["index_file"], allow_pickle=False)
    ids = ordered_ids(manifest, indices)
    assert_id_hash(ids, entry["sorted_id_sha256"], "offline_test")
    cache_root = Path(manifest["representation_cache"])
    brain, _ = gather_cache_rows(
        cache_root, manifest["cache_row_table"], indices, "brain_roi_features.npy"
    )
    clip, _ = gather_cache_rows(
        cache_root, manifest["cache_row_table"], indices, "clip_layer_features.npy"
    )
    return indices, brain, clip, ids


def paired_subsampling(
    real_plan: np.ndarray,
    baseline_plan: np.ndarray,
    per_stimulus_costs: np.ndarray,
    brain_rdms: np.ndarray,
    clip_rdms: np.ndarray,
    calibration: LossCalibration,
    count: int,
    fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    subset_size = int(round(fraction * len(per_stimulus_costs)))
    rows: List[Dict[str, Any]] = []
    for replicate in range(count):
        positions = np.sort(
            rng.choice(len(per_stimulus_costs), size=subset_size, replace=False)
        )
        feature = per_stimulus_costs[positions].mean(axis=0, dtype=np.float64)
        brain_geometry = relation_matrix_from_rdms(brain_rdms, positions)
        clip_geometry = relation_matrix_from_rdms(clip_rdms, positions)
        real = validation_metrics(real_plan, feature, brain_geometry, clip_geometry, calibration)
        baseline = validation_metrics(
            baseline_plan, feature, brain_geometry, clip_geometry, calibration
        )
        row: Dict[str, Any] = {"replicate": replicate, "subset_size": subset_size}
        for metric in (
            "validation_feature_normalized",
            "validation_gw_normalized",
            "validation_composite",
        ):
            short = metric.removeprefix("validation_")
            row[f"fgw_{short}"] = real[metric]
            row[f"baseline_{short}"] = baseline[metric]
            row[f"fgw_minus_baseline_{short}"] = real[metric] - baseline[metric]
        rows.append(row)
        if (replicate + 1) % 20 == 0:
            print(f"offline-test no-replacement subsamples {replicate + 1}/{count}", flush=True)
    result = {
        "count": count,
        "fraction": fraction,
        "subset_size": subset_size,
        "seed": seed,
        "plans_refit_on_subsamples": False,
        "paired_differences": {
            metric: summary([row[f"fgw_minus_baseline_{metric}"] for row in rows])
            for metric in ("feature_normalized", "gw_normalized", "composite")
        },
    }
    return rows, result


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    manifest_path = Path(args.stimulus_manifest).expanduser().resolve()
    locked_path = Path(args.locked_configuration).expanduser().resolve()
    validation_summary_path = Path(args.validation_summary).expanduser().resolve()
    feature_dir = Path(args.feature_cost_dir).expanduser().resolve()
    geometry_dir = Path(args.geometry_dir).expanduser().resolve()
    validation_dir = Path(args.validation_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot output: {output_dir}")
    output_dir.mkdir(parents=True)

    try:
        manifest, locked, validation_summary, feature_config, geometry_config = verify_locked_inputs(
            manifest_path,
            locked_path,
            validation_summary_path,
            feature_dir,
            geometry_dir,
            validation_dir,
        )
    except Exception as error:
        write_json(
            output_dir / "offline_test_blocked.json",
            {
                "OFFLINE_TEST_STATUS": "BLOCKED_BY_FROZEN_ASSET_MISMATCH",
                "error": str(error),
            },
        )
        raise

    # The following protocol is persisted before any offline-test index or
    # feature array is opened. It is part of the freeze boundary.
    solver = locked["solver_settings"]
    protocol = {
        "protocol_version": "prompt5b_locked_one_shot_replication_v1",
        "subject": manifest["subject"],
        "offline_test_accessed_when_written": False,
        "locked_structural": EXPECTED_LOCK,
        "locked_baseline": {
            key: validation_summary["strongest_nonstructural_baseline"][key]
            for key in (
                "slug",
                "method",
                "structure_weight",
                "coverage_weight",
                "best_discovery_initialization",
            )
        },
        "feature_protocol": {
            "final_plan_cost": "5-fold OOF over combined discovery+validation",
            "test_probe": "same ridge refit once on all combined discovery+validation",
            "fold_count": feature_config["cross_fitting"]["folds"],
            "fold_seed": feature_config["cross_fitting"]["random_seed"],
            "ridge_alpha": feature_config["ridge_regularization"]["alpha"],
            "std_epsilon": feature_config["preprocessing"]["std_epsilon"],
            "cost": feature_config["cost_definition"],
        },
        "geometry_protocol": {
            **geometry_config["geometry_definition"],
            "brain_representation": geometry_config["primary_brain_representation"],
            "anchor_count": geometry_config["anchor_size"],
            "random_seed": geometry_config["random_seed"],
            "random_stream": "NumPy SeedSequence(seed).spawn(5)[0]",
            "variance_standardized_geometry_used": False,
        },
        "loss_calibration_rule": (
            "reuse the subject-specific Prompt-5A discovery calibration exactly, "
            "matching the frozen subj01 final-refit rule; no final-train or test recalibration"
        ),
        "fixed_loss_calibration": locked["fixed_loss_calibration"],
        "solver": solver,
        "null_protocol": {
            "brain_geometry_unique_permutations": 200,
            "clip_geometry_unique_permutations": 200,
            "independent_roi_stimulus_permutations": 200,
            "feature_pairing_permutations": 20,
            "null_fit_initializations": [["uniform", 0], ["feature_informed", 0], ["random", 1001]],
            "seed": 42001,
            "fit_split": "offline_discovery + offline_validation only",
            "test_evaluation": "original unpermuted offline_test matrices",
        },
        "test_subsampling": {
            "count": 200,
            "fraction": 0.8,
            "without_replacement": True,
            "seed": 41001,
            "plans_refit": False,
        },
        "decision_rule": {
            "EXPLORATORY_PASS": (
                "composite difference < 0; paired GW-difference CI upper < 0; "
                "all three geometry-null one-sided p < 0.05; no test fitting; pretest hash valid"
            ),
            "FAIL": "mean GW difference >= 0 or any required geometry-null p >= 0.05",
            "INCONCLUSIVE": "remaining mixed criterion outcomes",
            "feature_pairing_null_affects_status": False,
        },
    }
    write_json(output_dir / "offline_test_protocol_pretest.json", protocol)

    discovery_indices, discovery_brain, discovery_clip, discovery_ids = load_non_test_split(
        manifest, manifest_path, "offline_discovery"
    )
    validation_indices, validation_brain, validation_clip, validation_ids = load_non_test_split(
        manifest, manifest_path, "offline_validation"
    )
    assert_id_hash(
        discovery_ids,
        manifest["offline_splits"]["offline_discovery"]["sorted_id_sha256"],
        "offline_discovery",
    )
    assert_id_hash(
        validation_ids,
        manifest["offline_splits"]["offline_validation"]["sorted_id_sha256"],
        "offline_validation",
    )
    fit_ids = discovery_ids + validation_ids
    expected_fit_count = int(
        manifest["offline_splits"]["offline_discovery"]["unique_stimulus_count"]
    ) + int(manifest["offline_splits"]["offline_validation"]["unique_stimulus_count"])
    if len(fit_ids) != expected_fit_count or len(set(fit_ids)) != expected_fit_count:
        raise RuntimeError(
            f"Final fitting set is not {expected_fit_count} unique stimuli"
        )
    protected_validation = set(
        manifest["downstream_id_sets"]["downstream_validation_evaluation"]["ids"]
    )
    protected_test = set(
        manifest["downstream_id_sets"]["protected_final_downstream_test"]["ids"]
    )
    if set(fit_ids) & protected_validation or set(fit_ids) & protected_test:
        raise RuntimeError("Final fitting IDs overlap protected downstream IDs")
    # Pretest overlap verification relies on the frozen manifest declarations;
    # the actual offline-test IDs are recomputed only after the marker.
    overlaps = manifest["overlap_checks"]["offline_unique_stimulus_overlaps"]
    if overlaps["discovery_test"] != 0 or overlaps["validation_test"] != 0:
        raise RuntimeError("Frozen manifest reports final-fit/offline-test overlap")

    final_brain = np.concatenate([discovery_brain, validation_brain], axis=0)
    final_clip = np.concatenate([discovery_clip, validation_clip], axis=0)
    if final_brain.shape != (expected_fit_count, 8, 1024) or final_clip.shape != (
        expected_fit_count,
        6,
        1024,
    ):
        raise RuntimeError(f"Unexpected final feature shapes {final_brain.shape}/{final_clip.shape}")
    folds = int(feature_config["cross_fitting"]["folds"])
    fold_seed = int(feature_config["cross_fitting"]["random_seed"])
    alpha = float(feature_config["ridge_regularization"]["alpha"])
    std_epsilon = float(feature_config["preprocessing"]["std_epsilon"])
    assignments, fold_ids = build_unique_id_folds(fit_ids, folds, fold_seed)
    print("Building locked final-train OOF feature compatibility", flush=True)
    feature_final, feature_final_per_stimulus = crossfit_cost(
        final_brain,
        final_clip,
        assignments,
        alpha,
        std_epsilon,
        args.probe_device,
    )
    np.save(output_dir / "final_train_feature_cost_oof.npy", feature_final)
    np.save(output_dir / "final_train_feature_cost_oof_per_stimulus.npy", feature_final_per_stimulus)

    print("Fitting frozen full-data probes for later test prediction", flush=True)
    flat_targets = np.ascontiguousarray(final_clip.reshape(len(final_clip), -1))
    probe_models = [
        fit_full_probe(final_brain[:, roi], flat_targets, alpha, std_epsilon, args.probe_device)
        for roi in range(8)
    ]
    probe_hash = hash_probe_models(probe_models)

    anchor_count = int(geometry_config["anchor_size"])
    geometry_seed = int(geometry_config["random_seed"])
    final_anchor_positions = fixed_anchor_positions(len(final_ids := fit_ids), anchor_count, geometry_seed)
    write_json(
        output_dir / "final_train_geometry_anchor_ids.json",
        {
            "combined_fit_positions": final_anchor_positions,
            "stable_stimulus_ids": [final_ids[index] for index in final_anchor_positions],
            "count": anchor_count,
            "ordered_id_sha256": sha256_ids(
                [final_ids[index] for index in final_anchor_positions]
            ),
        },
    )
    print("Building locked final-train primary geometries", flush=True)
    brain_geometry_final, clip_geometry_final, brain_final_rdms, _ = compute_geometry(
        final_brain, final_clip, final_anchor_positions, args.probe_device
    )
    np.save(output_dir / "final_train_brain_geometry.npy", brain_geometry_final)
    np.save(output_dir / "final_train_clip_geometry.npy", clip_geometry_final)

    fixed = locked["fixed_loss_calibration"]
    calibration = LossCalibration(fixed["s_feature"], fixed["s_gw"], fixed["eps_scale"])
    final_uniform_reference = LossCalibration(
        *(
            value
            for value in (
                float(np.sum(feature_final) / (8.0 * 6.0)),
                float(
                    sum(
                        (brain_geometry_final[r, rp] - clip_geometry_final[l, lp]) ** 2
                        for r in range(8)
                        for rp in range(8)
                        for l in range(6)
                        for lp in range(6)
                    )
                    / (8.0 * 8.0 * 6.0 * 6.0)
                ),
                float(fixed["eps_scale"]),
            )
        )
    )
    selected_kind, selected_seed = locked["best_discovery_initialization"]
    print("Fitting exactly one locked final srFGW plan", flush=True)
    try:
        final_fit = fit_one_plan(
            feature_final,
            brain_geometry_final,
            clip_geometry_final,
            calibration,
            0.5,
            0.0,
            str(selected_kind),
            int(selected_seed),
            solver,
            args.solver_device,
        )
    except Exception as error:
        write_json(
            output_dir / "numerical_refit_failure.json",
            {
                "OFFLINE_TEST_STATUS": "NUMERICAL_REFIT_FAILURE",
                "selected_initialization": [selected_kind, selected_seed],
                "error": str(error),
                "offline_test_accessed": False,
            },
        )
        raise
    final_plan = final_fit["plan"]

    baseline_spec = validation_summary["strongest_nonstructural_baseline"]
    baseline_kind, baseline_seed = baseline_spec["best_discovery_initialization"]
    print("Fitting the pre-selected non-structural baseline", flush=True)
    baseline_fit = fit_one_plan(
        feature_final,
        brain_geometry_final,
        clip_geometry_final,
        calibration,
        float(baseline_spec["structure_weight"]),
        float(baseline_spec["coverage_weight"]),
        str(baseline_kind),
        int(baseline_seed),
        solver,
        args.solver_device,
    )
    baseline_plan = baseline_fit["plan"]

    roi_names = list(geometry_config["roi_names"])
    layer_depths = list(geometry_config["clip_layers"])
    layer_labels = [f"L{layer}" for layer in layer_depths]
    final_plan_path = output_dir / "final_transport_plan_pretest.npy"
    baseline_plan_path = output_dir / "baseline_transport_plan_pretest.npy"
    np.save(final_plan_path, final_plan)
    np.save(baseline_plan_path, baseline_plan)
    save_plan_csv(output_dir / "final_transport_plan_pretest.csv", final_plan, roi_names, layer_labels)
    save_plan_csv(output_dir / "baseline_transport_plan_pretest.csv", baseline_plan, roi_names, layer_labels)
    final_diagnostics = coupling_diagnostics(final_plan, roi_names, layer_labels, layer_depths)
    baseline_diagnostics = coupling_diagnostics(
        baseline_plan, roi_names, layer_labels, layer_depths
    )
    write_json(
        output_dir / "final_refit_diagnostics.json",
        {
            "convergence": {
                "converged": final_fit["converged"],
                "iterations": final_fit["iterations"],
                "stopping_reason": final_fit["stopping_reason"],
                "source_marginal_feasibility_error": final_fit[
                    "source_marginal_feasibility_error"
                ],
                "training_components": final_fit["components"],
            },
            "coupling": final_diagnostics,
            "interpretation": "descriptive transport only; no biological hierarchy claim",
        },
    )
    write_json(
        output_dir / "baseline_refit_diagnostics.json",
        {
            "convergence": {
                "converged": baseline_fit["converged"],
                "iterations": baseline_fit["iterations"],
                "stopping_reason": baseline_fit["stopping_reason"],
                "source_marginal_feasibility_error": baseline_fit[
                    "source_marginal_feasibility_error"
                ],
                "training_components": baseline_fit["components"],
            },
            "coupling": baseline_diagnostics,
        },
    )

    # All required null plans are also trained and frozen before test unlock.
    null_protocol = protocol["null_protocol"]
    null_specs = null_protocol["null_fit_initializations"]
    null_rng = np.random.default_rng(int(null_protocol["seed"]))
    brain_permutations = unique_permutations(200, 8, null_rng)
    clip_permutations = unique_permutations(200, 6, null_rng)
    brain_null_plans = []
    clip_null_plans = []
    for index, permutation in enumerate(brain_permutations):
        fitted = fit_null(
            feature_final,
            brain_geometry_final[np.ix_(permutation, permutation)],
            clip_geometry_final,
            calibration,
            null_specs,
            solver,
            args.solver_device,
        )
        brain_null_plans.append(fitted["plan"])
        if (index + 1) % 25 == 0:
            print(f"pretest brain-identity null {index + 1}/200", flush=True)
    for index, permutation in enumerate(clip_permutations):
        fitted = fit_null(
            feature_final,
            brain_geometry_final,
            clip_geometry_final[np.ix_(permutation, permutation)],
            calibration,
            null_specs,
            solver,
            args.solver_device,
        )
        clip_null_plans.append(fitted["plan"])
        if (index + 1) % 25 == 0:
            print(f"pretest CLIP-identity null {index + 1}/200", flush=True)
    independent_plans = []
    independent_geometries = []
    for index in range(200):
        rng = np.random.default_rng(int(null_protocol["seed"]) + 20000 + index)
        geometry_null = independently_permuted_brain_geometry(brain_final_rdms, rng)
        fitted = fit_null(
            feature_final,
            geometry_null,
            clip_geometry_final,
            calibration,
            null_specs,
            solver,
            args.solver_device,
        )
        independent_geometries.append(geometry_null)
        independent_plans.append(fitted["plan"])
        if (index + 1) % 20 == 0:
            print(f"pretest independent-ROI null {index + 1}/200", flush=True)
    feature_null_costs = []
    feature_null_plans = []
    for index in range(20):
        rng = np.random.default_rng(int(null_protocol["seed"]) + 10000 + index)
        cost_null = feature_pairing_null_cost(
            final_brain,
            final_clip,
            assignments,
            alpha,
            std_epsilon,
            rng,
            args.probe_device,
        )
        fitted = fit_null(
            cost_null,
            brain_geometry_final,
            clip_geometry_final,
            calibration,
            null_specs,
            solver,
            args.solver_device,
        )
        feature_null_costs.append(cost_null)
        feature_null_plans.append(fitted["plan"])
        print(f"pretest feature-pairing null {index + 1}/20", flush=True)

    brain_null_array = np.stack(brain_null_plans)
    clip_null_array = np.stack(clip_null_plans)
    independent_plan_array = np.stack(independent_plans)
    feature_null_plan_array = np.stack(feature_null_plans)
    np.save(output_dir / "pretest_brain_identity_null_plans.npy", brain_null_array)
    np.save(output_dir / "pretest_clip_identity_null_plans.npy", clip_null_array)
    np.save(output_dir / "pretest_independent_roi_null_plans.npy", independent_plan_array)
    np.save(output_dir / "pretest_independent_roi_null_geometries.npy", np.stack(independent_geometries))
    np.save(output_dir / "pretest_feature_pairing_null_costs.npy", np.stack(feature_null_costs))
    np.save(output_dir / "pretest_feature_pairing_null_plans.npy", feature_null_plan_array)
    np.save(output_dir / "pretest_brain_identity_permutations.npy", np.stack(brain_permutations))
    np.save(output_dir / "pretest_clip_identity_permutations.npy", np.stack(clip_permutations))

    final_refit_config = {
        **protocol,
        "offline_test_accessed_when_written": False,
        "final_fit_count": len(fit_ids),
        "final_fit_id_sha256": sha256_ids(sorted(fit_ids)),
        "discovery_id_sha256": manifest["offline_splits"]["offline_discovery"]["sorted_id_sha256"],
        "validation_id_sha256": manifest["offline_splits"]["offline_validation"]["sorted_id_sha256"],
        "offline_test_declared_id_sha256_pretest": manifest["offline_splits"]["offline_test"]["sorted_id_sha256"],
        "offline_test_actual_id_hash_recomputed": False,
        "protected_overlap": 0,
        "OOF_fold_id_hashes": [sha256_ids(values) for values in fold_ids],
        "full_probe_parameter_sha256": probe_hash,
        "source_hashes": locked["matrix_hashes"],
        "protocol_config_hashes": {
            "feature_cost_config_sha256": sha256_file(
                feature_dir / "feature_cost_config.json"
            ),
            "geometry_config_sha256": sha256_file(geometry_dir / "geometry_config.json"),
        },
        "final_train_matrix_hashes": {
            "feature": sha256_file(output_dir / "final_train_feature_cost_oof.npy"),
            "brain_geometry": sha256_file(output_dir / "final_train_brain_geometry.npy"),
            "clip_geometry": sha256_file(output_dir / "final_train_clip_geometry.npy"),
        },
        "final_plan_fit_count": 1,
        "subject": manifest["subject"],
        "s_feature_final": calibration.feature_scale,
        "s_gw_final": calibration.gw_scale,
        "final_scale_provenance": protocol["loss_calibration_rule"],
        "uniform_reference_losses_recomputed_on_final_train_for_audit_not_used": {
            "s_feature": final_uniform_reference.feature_scale,
            "s_gw": final_uniform_reference.gw_scale,
        },
        "final_plan_initialization": [selected_kind, selected_seed],
        "baseline_plan_initialization": [baseline_kind, baseline_seed],
        "offline_test_used_for_fitting": False,
        "stage2_run": False,
        "caption_metrics_used": False,
    }
    final_config_path = output_dir / "final_refit_config.json"
    write_json(final_config_path, final_refit_config)
    final_plan_hash = sha256_file(final_plan_path)
    baseline_plan_hash = sha256_file(baseline_plan_path)
    config_hash = sha256_file(final_config_path)
    null_hashes = {
        name: sha256_file(output_dir / name)
        for name in (
            "pretest_brain_identity_null_plans.npy",
            "pretest_clip_identity_null_plans.npy",
            "pretest_independent_roi_null_plans.npy",
            "pretest_feature_pairing_null_plans.npy",
        )
    }
    write_json(
        output_dir / "final_plan_hash.json",
        {
            "subject": manifest["subject"],
            "FINAL_PLAN_HASH": final_plan_hash,
            "hash_algorithm": "SHA256 over exact .npy file bytes",
            "baseline_plan_sha256": baseline_plan_hash,
            "final_refit_config_sha256": config_hash,
            "pretest_null_plan_hashes": null_hashes,
        },
    )
    marker_path = output_dir / "final_plan_frozen.marker"
    durable_marker(
        marker_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "FINAL_PLAN_HASH": final_plan_hash,
            "final_plan_sha256": final_plan_hash,
            "baseline_plan_sha256": baseline_plan_hash,
            "config_sha256": config_hash,
            "training_stimulus_id_sha256": sha256_ids(sorted(fit_ids)),
            "selected_initialization": [selected_kind, selected_seed],
            "stimulus_manifest_sha256": sha256_file(manifest_path),
            "offline_test_accessed_before_marker": False,
            "transport_plan_must_never_change_after_this_marker": True,
        },
    )
    if sha256_file(final_plan_path) != final_plan_hash:
        raise RuntimeError("Frozen final plan hash changed before test unlock")
    print(f"FINAL_PLAN_FROZEN = {final_plan_hash}", flush=True)

    # ------------------------- irreversible test boundary -------------------------
    test_indices, test_brain, test_clip, test_ids = unlock_offline_test(
        marker_path, final_plan_path, final_plan_hash, manifest, manifest_path
    )
    expected_test_count = int(
        manifest["offline_splits"]["offline_test"]["unique_stimulus_count"]
    )
    if len(test_ids) != expected_test_count or len(set(test_ids)) != expected_test_count:
        raise RuntimeError(
            f"offline_test is not {expected_test_count} unique stimuli"
        )
    if set(test_ids) & set(fit_ids):
        raise RuntimeError("offline_test overlaps final fitting IDs")
    if set(test_ids) & protected_validation or set(test_ids) & protected_test:
        raise RuntimeError("offline_test overlaps protected downstream IDs")
    print("Offline test unlocked after durable final-plan freeze", flush=True)

    feature_test, feature_test_per_stimulus = test_feature_cost(
        probe_models, test_brain, test_clip
    )
    np.save(output_dir / "offline_test_feature_cost.npy", feature_test)
    np.save(output_dir / "offline_test_feature_cost_per_stimulus.npy", feature_test_per_stimulus)
    test_anchor_positions = fixed_anchor_positions(len(test_ids), anchor_count, geometry_seed)
    write_json(
        output_dir / "offline_test_geometry_anchor_ids.json",
        {
            "offline_test_positions": test_anchor_positions,
            "stable_stimulus_ids": [test_ids[index] for index in test_anchor_positions],
            "count": anchor_count,
            "ordered_id_sha256": sha256_ids(
                [test_ids[index] for index in test_anchor_positions]
            ),
        },
    )
    brain_test_rdms = cosine_distance_matrices(
        test_brain, np.arange(len(test_brain)), args.probe_device
    )
    clip_test_rdms = cosine_distance_matrices(
        test_clip, np.arange(len(test_clip)), args.probe_device
    )
    brain_test_geometry = relation_matrix_from_rdms(brain_test_rdms, test_anchor_positions)
    clip_test_geometry = relation_matrix_from_rdms(clip_test_rdms, test_anchor_positions)
    np.save(output_dir / "offline_test_brain_geometry.npy", brain_test_geometry)
    np.save(output_dir / "offline_test_clip_geometry.npy", clip_test_geometry)

    real_metrics = validation_metrics(
        final_plan, feature_test, brain_test_geometry, clip_test_geometry, calibration
    )
    baseline_metrics = validation_metrics(
        baseline_plan, feature_test, brain_test_geometry, clip_test_geometry, calibration
    )
    real_metrics = {key.replace("validation_", "offline_test_"): value for key, value in real_metrics.items()}
    baseline_metrics = {
        key.replace("validation_", "offline_test_"): value
        for key, value in baseline_metrics.items()
    }
    difference = {
        name: real_metrics[f"offline_test_{name}"] - baseline_metrics[f"offline_test_{name}"]
        for name in (
            "feature_raw",
            "feature_normalized",
            "gw_raw",
            "gw_normalized",
            "composite",
        )
    }
    offline_test_metrics = {
        **real_metrics,
        "coupling_diagnostics": final_diagnostics,
        "frozen_plan_sha256_verified": sha256_file(final_plan_path) == final_plan_hash,
    }
    write_json(output_dir / "offline_test_metrics.json", offline_test_metrics)
    write_json(
        output_dir / "baseline_offline_test_metrics.json",
        {
            "preselected_baseline": baseline_spec,
            **baseline_metrics,
            "coupling_diagnostics": baseline_diagnostics,
            "fgw_minus_baseline": difference,
        },
    )

    subsample_rows, subsample_result = paired_subsampling(
        final_plan,
        baseline_plan,
        feature_test_per_stimulus,
        brain_test_rdms,
        clip_test_rdms,
        calibration,
        count=200,
        fraction=0.8,
        seed=41001,
    )
    write_dict_rows(output_dir / "offline_test_subsampling.csv", subsample_rows)

    def evaluate_null_plans(plans: np.ndarray) -> List[Dict[str, float]]:
        return [
            validation_metrics(
                plan, feature_test, brain_test_geometry, clip_test_geometry, calibration
            )
            for plan in plans
        ]

    brain_null_metrics = evaluate_null_plans(brain_null_array)
    clip_null_metrics = evaluate_null_plans(clip_null_array)
    independent_null_metrics = evaluate_null_plans(independent_plan_array)
    feature_null_metrics = evaluate_null_plans(feature_null_plan_array)
    real_gw = offline_test_metrics["offline_test_gw_normalized"]

    def structural_null_payload(items: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
        values = [item["validation_gw_normalized"] for item in items]
        composites = [item["validation_composite"] for item in items]
        return {
            "count": len(items),
            "offline_test_gw_normalized": summary(values),
            "offline_test_composite": summary(composites),
            "one_sided_p_real_gw_lower": one_sided_lower_p(real_gw, values),
        }

    feature_null_feature = [
        item["validation_feature_normalized"] for item in feature_null_metrics
    ]
    null_summary = {
        "all_plans_fitted_and_hashed_before_offline_test_access": True,
        "evaluation_matrices": "original unpermuted offline_test matrices",
        "brain_geometry_identity_mismatch": structural_null_payload(brain_null_metrics),
        "clip_geometry_identity_mismatch": structural_null_payload(clip_null_metrics),
        "independent_roi_stimulus_geometry": structural_null_payload(
            independent_null_metrics
        ),
        "feature_pairing": {
            "count": len(feature_null_metrics),
            "offline_test_feature_normalized": summary(feature_null_feature),
            "offline_test_gw_normalized": summary(
                [item["validation_gw_normalized"] for item in feature_null_metrics]
            ),
            "offline_test_composite": summary(
                [item["validation_composite"] for item in feature_null_metrics]
            ),
            "one_sided_p_real_feature_lower": one_sided_lower_p(
                offline_test_metrics["offline_test_feature_normalized"],
                feature_null_feature,
            ),
            "affects_locked_configuration": False,
            "affects_pass_status": False,
        },
    }
    write_json(output_dir / "offline_test_null_summary.json", null_summary)

    gw_difference = subsample_result["paired_differences"]["gw_normalized"]
    geometry_p = {
        "brain_identity": null_summary["brain_geometry_identity_mismatch"][
            "one_sided_p_real_gw_lower"
        ],
        "clip_identity": null_summary["clip_geometry_identity_mismatch"][
            "one_sided_p_real_gw_lower"
        ],
        "independent_roi_stimulus": null_summary[
            "independent_roi_stimulus_geometry"
        ]["one_sided_p_real_gw_lower"],
    }
    criteria = {
        "composite_lower_than_preselected_baseline": difference["composite"] < 0,
        "paired_structural_ci_entirely_below_zero": gw_difference["ci_97_5"] < 0,
        "all_geometry_null_p_below_0_05": all(value < 0.05 for value in geometry_p.values()),
        "offline_test_used_for_fitting_or_selection": False,
        "final_plan_hash_created_before_test_access": True,
    }
    if not criteria["all_geometry_null_p_below_0_05"] or gw_difference["mean"] >= 0:
        status = "FAIL"
    elif all(
        criteria[key]
        for key in (
            "composite_lower_than_preselected_baseline",
            "paired_structural_ci_entirely_below_zero",
            "all_geometry_null_p_below_0_05",
        )
    ):
        status = "EXPLORATORY_PASS"
    else:
        status = "INCONCLUSIVE"

    final_summary = {
        "OFFLINE_TEST_STATUS": status,
        "subject": manifest["subject"],
        "scope": f"locked independent-subject replication / {manifest['subject']}",
        "FINAL_PLAN_HASH": final_plan_hash,
        "plan_frozen_before_test_access": True,
        "final_fit_count": expected_fit_count,
        "offline_test_count": len(test_ids),
        "actual_offline_test_id_sha256": sha256_ids(sorted(test_ids)),
        "offline_test_metrics": offline_test_metrics,
        "preselected_baseline_metrics": baseline_metrics,
        "fgw_minus_baseline": difference,
        "paired_test_subsampling": subsample_result,
        "geometry_null_p_values": geometry_p,
        "feature_pairing_null": null_summary["feature_pairing"],
        "decision_criteria": criteria,
        "interpretation": (
            "The learned correspondence is primarily supported by representational "
            "geometry rather than independent evidence from direct ROI-to-layer "
            "feature compatibility."
        ),
        "data_usage": {
            "offline_test_used_for_fitting": False,
            "offline_test_used_for_hyperparameter_or_initialization_selection": False,
            "stage2_run": False,
            "caption_metrics_used": False,
        },
        "provenance": {
            "manifest_sha256": sha256_file(manifest_path),
            "locked_configuration_sha256": sha256_file(locked_path),
            "validation_summary_sha256": sha256_file(validation_summary_path),
            "final_refit_config_sha256": config_hash,
            "final_plan_sha256_after_test": sha256_file(final_plan_path),
            "baseline_plan_sha256_after_test": sha256_file(baseline_plan_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "pretest_marker": json.loads(marker_path.read_text()),
        },
        "elapsed_seconds": time.time() - started,
    }
    if final_summary["provenance"]["final_plan_sha256_after_test"] != final_plan_hash:
        raise RuntimeError("Final transport plan changed after offline-test access")
    write_json(output_dir / "offline_test_summary.json", final_summary)
    if list(output_dir.rglob("best_transport_plan.npy")):
        raise RuntimeError("Forbidden best_transport_plan.npy was created")
    print(f"OFFLINE_TEST_STATUS = {status}", flush=True)
    print(f"Saved one-shot offline test to {output_dir}", flush=True)
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
