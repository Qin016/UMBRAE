#!/usr/bin/env python
"""Prompt-4A validation, identifiability, stability, and null analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Loading ``models.fgw_correspondence`` through the package executes
# ``models/__init__.py``, which eagerly imports the online CLIP stack.  This
# validation program is deliberately offline and must not depend on
# torchvision/CLIP/UMBRAE, so load the standalone numerical module directly.
_FGW_PATH = Path(__file__).resolve().parents[1] / "models" / "fgw_correspondence.py"
_FGW_SPEC = importlib.util.spec_from_file_location("offline_fgw_correspondence", _FGW_PATH)
if _FGW_SPEC is None or _FGW_SPEC.loader is None:
    raise ImportError(f"Cannot load offline FGW module from {_FGW_PATH}")
_FGW = importlib.util.module_from_spec(_FGW_SPEC)
sys.modules[_FGW_SPEC.name] = _FGW
_FGW_SPEC.loader.exec_module(_FGW)
LossCalibration = _FGW.LossCalibration
calibration_from_uniform_plan = _FGW.calibration_from_uniform_plan
coupling_diagnostics = _FGW.coupling_diagnostics
gw_loss = _FGW.gw_loss
initialization_plan = _FGW.initialization_plan
jensen_shannon_divergence = _FGW.jensen_shannon_divergence
solve_balanced_outer_ot = _FGW.solve_balanced_outer_ot
solve_projected_simplex = _FGW.solve_projected_simplex
from scripts.analyze_representation_geometry import (
    cosine_distance_matrices,
    relation_matrix_from_rdms,
    save_heatmap,
    spearman_value,
    upper_triangle,
)
from scripts.analyze_representation_geometry_v2 import (
    gather_cache_rows,
    relation_from_standardized_ranks,
    standardized_ranks,
)
from scripts.build_fgw_feature_cost_v2 import (
    build_unique_id_folds,
    cosine_costs,
    fit_ridge_predict_multioutput,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen discovery FGW candidates")
    parser.add_argument("--stimulus-manifest", required=True)
    parser.add_argument("--feature-cost-dir", required=True)
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--discovery-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preregistered-grid", required=True)
    parser.add_argument("--probe-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--solver-device", default="cpu")
    return parser.parse_args()


def clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(clean_json(payload), indent=2, allow_nan=False) + "\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_matrix_csv(
    path: Path, matrix: np.ndarray, row_labels: Sequence[str], column_labels: Sequence[str]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", *column_labels])
        for label, row in zip(row_labels, matrix):
            writer.writerow([label, *[f"{float(value):.12g}" for value in row]])


def write_dict_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config_slug(method: str, beta: float, coverage: Optional[float]) -> str:
    def token(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    if method == "source_constrained_feature_routing":
        return method
    if method == "balanced_outer_ot":
        return method
    if method == "source_constrained_feature_coverage":
        return f"{method}_lambda_{token(float(coverage))}"
    if method == "sr_fgw":
        return f"{method}_beta_{token(beta)}"
    return f"{method}_beta_{token(beta)}_lambda_{token(float(coverage))}"


def expand_preregistered_grid(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    result = []
    for method, configs in grid["candidate_grid"].items():
        for config in configs:
            beta = float(config["structure_weight"])
            coverage = config.get("coverage_weight")
            coverage_value = None if coverage is None else float(coverage)
            result.append(
                {
                    "method": method,
                    "structure_weight": beta,
                    "coverage_weight": coverage_value,
                    "hard_uniform_target": bool(config.get("hard_uniform_target", False)),
                    "slug": config_slug(method, beta, coverage_value),
                    "structural": method in {"sr_fgw", "sr_fgw_coverage"},
                }
            )
    expected = int(grid.get("required_candidate_count", 21))
    if len(result) != expected:
        raise ValueError(
            f"Expected {expected} preregistered configs, got {len(result)}"
        )
    return result


def load_split_features(
    manifest: Mapping[str, Any], manifest_path: Path, split: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    entry = manifest["offline_splits"][split]
    indices = np.load(manifest_path.parent / entry["index_file"], allow_pickle=False)
    table = manifest["cache_row_table"]
    by_global = {int(row["global_cache_row"]): row for row in table}
    ids = [by_global[int(index)]["stable_stimulus_id"] for index in indices]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{split} does not contain unique stimulus IDs")
    cache_root = Path(manifest["representation_cache"])
    brain, _ = gather_cache_rows(cache_root, table, indices, "brain_roi_features.npy")
    clip, _ = gather_cache_rows(cache_root, table, indices, "clip_layer_features.npy")
    return indices, brain, clip, ids


def fit_discovery_probe_evaluate_validation(
    discovery_brain: np.ndarray,
    discovery_clip: np.ndarray,
    validation_brain: np.ndarray,
    validation_clip: np.ndarray,
    alpha: float,
    std_epsilon: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n_validation = len(validation_brain)
    costs = np.empty((n_validation, 8, 6), dtype=np.float32)
    y_train = np.ascontiguousarray(discovery_clip.reshape(len(discovery_clip), -1))
    for roi in range(8):
        prediction = fit_ridge_predict_multioutput(
            discovery_brain[:, roi],
            y_train,
            validation_brain[:, roi],
            alpha,
            std_epsilon,
            device,
        ).reshape(n_validation, 6, 1024)
        costs[:, roi, :] = cosine_costs(prediction, validation_clip)
    return costs.mean(axis=0, dtype=np.float64), costs


def validation_metrics(
    plan: np.ndarray,
    feature_cost_validation: np.ndarray,
    brain_geometry_validation: np.ndarray,
    clip_geometry_validation: np.ndarray,
    calibration: LossCalibration,
) -> Dict[str, float]:
    tensor_plan = torch.as_tensor(plan, dtype=torch.float64)
    feature_raw = float(np.sum(plan * feature_cost_validation))
    gw_raw_value = float(
        gw_loss(
            tensor_plan,
            torch.as_tensor(brain_geometry_validation, dtype=torch.float64),
            torch.as_tensor(clip_geometry_validation, dtype=torch.float64),
        )
    )
    feature_norm = feature_raw / (calibration.feature_scale + calibration.eps_scale)
    gw_norm = gw_raw_value / (calibration.gw_scale + calibration.eps_scale)
    return {
        "validation_feature_raw": feature_raw,
        "validation_feature_normalized": feature_norm,
        "validation_gw_raw": gw_raw_value,
        "validation_gw_normalized": gw_norm,
        "validation_composite": 0.5 * feature_norm + 0.5 * gw_norm,
    }


def fit_plan_initializations(
    feature_cost: np.ndarray,
    brain_geometry: np.ndarray,
    clip_geometry: np.ndarray,
    calibration: LossCalibration,
    beta: float,
    coverage: float,
    specs: Sequence[Sequence[Any]],
    solver: Mapping[str, Any],
    validation_inputs: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for kind_raw, seed_raw in specs:
        kind = str(kind_raw)
        seed = int(seed_raw)
        fitted = solve_projected_simplex(
            feature_cost,
            brain_geometry,
            clip_geometry,
            calibration,
            beta,
            coverage,
            initialization_plan(kind, feature_cost, seed),
            entropy_weight=float(solver["entropy_weight"]),
            learning_rate=float(solver["learning_rate"]),
            max_iterations=int(solver["max_iterations"]),
            tolerance=float(solver["tolerance"]),
            patience=int(solver["patience"]),
            eps_cov=float(solver["eps_coverage"]),
            device=device,
        )
        item = {
            "initialization_kind": kind,
            "initialization_seed": seed,
            "plan": fitted["plan"],
            "discovery_components": fitted["components"],
            "discovery_objective": fitted["components"]["total_objective"],
            "converged": fitted["converged"],
            "iterations": fitted["iterations"],
            "stopping_reason": fitted["stopping_reason"],
            "source_marginal_feasibility_error": fitted[
                "source_marginal_feasibility_error"
            ],
        }
        if validation_inputs is not None:
            item.update(
                validation_metrics(
                    fitted["plan"],
                    validation_inputs[0],
                    validation_inputs[1],
                    validation_inputs[2],
                    calibration,
                )
            )
        results.append(item)
    return results


def best_discovery_result(results: Sequence[Mapping[str, Any]]) -> Tuple[int, Mapping[str, Any]]:
    converged = [
        (index, item) for index, item in enumerate(results) if bool(item["converged"])
    ]
    if not converged:
        raise RuntimeError("No converged discovery initialization")
    return min(converged, key=lambda pair: float(pair[1]["discovery_objective"]))


def summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("count", "mean", "std", "median", "ci_2_5", "ci_97_5")}
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "ci_2_5": float(np.percentile(array, 2.5)),
        "ci_97_5": float(np.percentile(array, 97.5)),
    }


def plan_set_identifiability(plans: np.ndarray) -> Dict[str, Any]:
    plans = np.asarray(plans, dtype=np.float64)
    if len(plans) < 2:
        return {
            "plan_count": len(plans),
            "pair_count": 0,
            "status": "SINGLETON_NO_PAIRWISE_IDENTIFIABILITY_ESTIMATE",
        }
    frobenius = []
    row_correlations = []
    row_js = []
    target_cosines = []
    target_l1 = []
    preferred_agreement = []
    per_roi_correlations: List[List[float]] = [[] for _ in range(plans.shape[1])]
    for left in range(len(plans)):
        for right in range(left + 1, len(plans)):
            a = plans[left] * plans.shape[1]
            b = plans[right] * plans.shape[1]
            frobenius.append(float(np.linalg.norm(plans[left] - plans[right])))
            pair_row_corr = []
            pair_row_js = []
            for roi in range(a.shape[0]):
                if np.std(a[roi]) > 0 and np.std(b[roi]) > 0:
                    correlation = float(np.corrcoef(a[roi], b[roi])[0, 1])
                    pair_row_corr.append(correlation)
                    per_roi_correlations[roi].append(correlation)
                pair_row_js.append(jensen_shannon_divergence(a[roi], b[roi]))
            if pair_row_corr:
                row_correlations.append(float(np.mean(pair_row_corr)))
            row_js.append(float(np.mean(pair_row_js)))
            target_a = plans[left].sum(axis=0)
            target_b = plans[right].sum(axis=0)
            denominator = np.linalg.norm(target_a) * np.linalg.norm(target_b)
            target_cosines.append(float(np.dot(target_a, target_b) / denominator))
            target_l1.append(float(np.sum(np.abs(target_a - target_b))))
            preferred_agreement.append(
                float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))
            )
    return {
        "plan_count": len(plans),
        "pair_count": len(frobenius),
        "pairwise_frobenius_distance": summary(frobenius),
        "mean_row_correlation": summary(row_correlations),
        "mean_row_js_divergence": summary(row_js),
        "target_marginal_cosine_similarity": summary(target_cosines),
        "target_marginal_l1_distance": summary(target_l1),
        "preferred_layer_agreement_fraction": summary(preferred_agreement),
        "per_roi_row_correlation": [summary(values) for values in per_roi_correlations],
    }


def analyze_local_minima(
    results: Sequence[Mapping[str, Any]], relative_threshold: float, top_k: int
) -> Dict[str, Any]:
    converged = [item for item in results if bool(item["converged"])]
    converged.sort(key=lambda item: float(item["discovery_objective"]))
    if not converged:
        return {"status": "NO_CONVERGED_INITIALIZATION"}
    best = float(converged[0]["discovery_objective"])
    cutoff = best + relative_threshold * abs(best)
    near = [item for item in converged if float(item["discovery_objective"]) <= cutoff]
    top = converged[: min(top_k, len(converged))]
    return {
        "status": "AVAILABLE",
        "best_converged_discovery_objective": best,
        "near_optimal_relative_threshold": relative_threshold,
        "near_optimal_cutoff": cutoff,
        "converged_count": len(converged),
        "near_optimal_count": len(near),
        "near_optimal_initializations": [
            [item["initialization_kind"], item["initialization_seed"]] for item in near
        ],
        "near_optimal": plan_set_identifiability(
            np.stack([item["plan"] for item in near], axis=0)
        ),
        "top_k": len(top),
        "top_k_initializations": [
            [item["initialization_kind"], item["initialization_seed"]] for item in top
        ],
        "top_k_analysis": plan_set_identifiability(
            np.stack([item["plan"] for item in top], axis=0)
        ),
    }


def unique_permutations(count: int, size: int, rng: np.random.Generator) -> List[np.ndarray]:
    identity = tuple(range(size))
    seen = set()
    result = []
    while len(result) < count:
        permutation = tuple(int(value) for value in rng.permutation(size))
        if permutation == identity or permutation in seen:
            continue
        seen.add(permutation)
        result.append(np.asarray(permutation, dtype=np.int64))
    return result


def fit_null_plan(
    feature_cost: np.ndarray,
    brain_geometry: np.ndarray,
    clip_geometry: np.ndarray,
    calibration: LossCalibration,
    selected: Mapping[str, Any],
    null_specs: Sequence[Sequence[Any]],
    solver: Mapping[str, Any],
    validation_inputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    device: str,
) -> Dict[str, Any]:
    results = fit_plan_initializations(
        feature_cost,
        brain_geometry,
        clip_geometry,
        calibration,
        float(selected["structure_weight"]),
        float(selected["coverage_weight"]),
        null_specs,
        solver,
        validation_inputs,
        device,
    )
    try:
        index, winner = best_discovery_result(results)
    except RuntimeError:
        index = int(np.argmin([item["discovery_objective"] for item in results]))
        winner = results[index]
    return {"winner_index": index, **winner}


def one_sided_lower_p(real: float, null_values: Sequence[float]) -> float:
    values = np.asarray(null_values)
    return float((1 + np.sum(values <= real)) / (1 + len(values)))


def feature_pairing_null_cost(
    brain: np.ndarray,
    clip: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    std_epsilon: float,
    rng: np.random.Generator,
    device: str,
) -> np.ndarray:
    n = len(brain)
    cost_sum = np.zeros((8, 6), dtype=np.float64)
    count = np.zeros((8, 6), dtype=np.int64)
    for fold in range(int(assignments.max()) + 1):
        training = np.flatnonzero(assignments != fold)
        heldout = np.flatnonzero(assignments == fold)
        permuted_train = rng.permutation(training)
        permuted_heldout = rng.permutation(heldout)
        y_train = np.ascontiguousarray(clip[permuted_train].reshape(len(training), -1))
        target_heldout = clip[permuted_heldout]
        for roi in range(8):
            prediction = fit_ridge_predict_multioutput(
                brain[training, roi],
                y_train,
                brain[heldout, roi],
                alpha,
                std_epsilon,
                device,
            ).reshape(len(heldout), 6, 1024)
            costs = cosine_costs(prediction, target_heldout)
            cost_sum[roi] += costs.sum(axis=0, dtype=np.float64)
            count[roi] += len(heldout)
    if not np.all(count == n):
        raise RuntimeError("Feature-pairing null did not cover every OOF stimulus")
    return cost_sum / count


def independently_permuted_brain_geometry(
    brain_rdms: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    node_count, stimulus_count, _ = brain_rdms.shape
    pair_left, pair_right = np.triu_indices(stimulus_count, k=1)
    vectors = np.empty((len(pair_left), node_count), dtype=np.float32)
    for node in range(node_count):
        order = rng.permutation(stimulus_count)
        vectors[:, node] = brain_rdms[node, order[pair_left], order[pair_right]]
    return relation_from_standardized_ranks(standardized_ranks(vectors))


def save_heatmap_rectangular(
    path: Path,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    color_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(column_labels)), labels=column_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(row_labels)), labels=row_labels)
    axis.set_title(title)
    for row in range(len(row_labels)):
        for column in range(len(column_labels)):
            axis.text(column, row, f"{matrix[row,column]:.3f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label=color_label)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    manifest_path = Path(args.stimulus_manifest).expanduser().resolve()
    feature_dir = Path(args.feature_cost_dir).expanduser().resolve()
    geometry_dir = Path(args.geometry_dir).expanduser().resolve()
    discovery_dir = Path(args.discovery_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    prereg_path = Path(args.preregistered_grid).expanduser().resolve()
    if output_dir != prereg_path.parent:
        raise ValueError("Preregistered grid must already live in output directory")
    existing = {path.name for path in output_dir.iterdir()}
    if existing != {prereg_path.name}:
        raise FileExistsError(f"Output must contain only preregistration before run: {existing}")
    prereg_hash_before = sha256_file(prereg_path)
    prereg = json.loads(prereg_path.read_text())
    if not prereg["written_before_offline_validation_feature_access_for_prompt4a"]:
        raise ValueError("Grid is not marked preregistered")
    configs = expand_preregistered_grid(prereg)

    manifest = json.loads(manifest_path.read_text())
    feature_config = json.loads((feature_dir / "feature_cost_config.json").read_text())
    geometry_config = json.loads((geometry_dir / "geometry_config.json").read_text())
    if feature_config["source_split"] != "offline_discovery":
        raise ValueError("Discovery feature cost provenance failed")
    if geometry_config["manifest_offline_split"] != "offline_discovery":
        raise ValueError("Discovery geometry provenance failed")
    discovery_indices, discovery_brain, discovery_clip, discovery_ids = load_split_features(
        manifest, manifest_path, "offline_discovery"
    )
    # This is the only held-out feature split opened in Prompt 4A.
    validation_indices, validation_brain, validation_clip, validation_ids = load_split_features(
        manifest, manifest_path, "offline_validation"
    )
    if set(discovery_ids) & set(validation_ids):
        raise ValueError("Discovery and validation stimulus IDs overlap")

    roi_names = list(geometry_config["roi_names"])
    layer_depths = list(geometry_config["clip_layers"])
    layer_labels = [f"L{layer}" for layer in layer_depths]
    feature_discovery = np.load(feature_dir / "feature_cost_oof.npy", allow_pickle=False)
    brain_geometry_discovery = np.load(geometry_dir / "brain_geometry.npy", allow_pickle=False)
    clip_geometry_discovery = np.load(geometry_dir / "clip_geometry.npy", allow_pickle=False)
    fixed = prereg["fixed_loss_scales"]
    calibration = LossCalibration(fixed["s_feature"], fixed["s_gw"], fixed["eps_scale"])
    recalculated = calibration_from_uniform_plan(
        feature_discovery, brain_geometry_discovery, clip_geometry_discovery, fixed["eps_scale"]
    )
    if abs(recalculated.feature_scale - calibration.feature_scale) > 1e-12 or abs(
        recalculated.gw_scale - calibration.gw_scale
    ) > 1e-12:
        raise ValueError("Preregistered loss scales differ from frozen discovery calibration")

    ridge = feature_config["ridge_regularization"]
    preprocessing = feature_config["preprocessing"]
    print("Fitting frozen discovery probes and evaluating offline_validation", flush=True)
    feature_validation, validation_costs_per_stimulus = fit_discovery_probe_evaluate_validation(
        discovery_brain,
        discovery_clip,
        validation_brain,
        validation_clip,
        float(ridge["alpha"]),
        float(preprocessing["std_epsilon"]),
        args.probe_device,
    )
    np.save(output_dir / "feature_cost_validation.npy", feature_validation)
    np.save(
        output_dir / "feature_cost_validation_per_stimulus.npy",
        validation_costs_per_stimulus,
    )
    write_matrix_csv(
        output_dir / "feature_cost_validation.csv",
        feature_validation,
        roi_names,
        layer_labels,
    )
    validation_feature_rows = []
    for roi, roi_name in enumerate(roi_names):
        for layer, layer_label in enumerate(layer_labels):
            values = validation_costs_per_stimulus[:, roi, layer]
            validation_feature_rows.append(
                {
                    "roi": roi_name,
                    "clip_layer": layer_label,
                    "validation_count": len(values),
                    "mean_cosine_prediction_quality": float(1.0 - values.mean()),
                    "mean_cost": float(values.mean()),
                    "std_cost_ddof1": float(values.std(ddof=1)),
                }
            )
    write_dict_rows(output_dir / "feature_cost_validation_metrics.csv", validation_feature_rows)

    print("Computing evaluation-only validation geometries", flush=True)
    validation_brain_rdms = cosine_distance_matrices(
        validation_brain, np.arange(len(validation_brain)), args.probe_device
    )
    validation_clip_rdms = cosine_distance_matrices(
        validation_clip, np.arange(len(validation_clip)), args.probe_device
    )
    validation_geometry_config = prereg["validation_geometry"]
    validation_anchor_count = int(validation_geometry_config["anchor_count"])
    if validation_anchor_count > len(validation_brain):
        raise ValueError("Validation geometry anchor count exceeds validation sample count")
    validation_anchor_rng = np.random.default_rng(
        int(validation_geometry_config["seed"])
    )
    validation_all = np.sort(
        validation_anchor_rng.choice(
            len(validation_brain), size=validation_anchor_count, replace=False
        )
    )
    write_json(
        output_dir / "validation_geometry_anchor_ids.json",
        {
            "offline_validation_positions": validation_all,
            "stable_stimulus_ids": [validation_ids[index] for index in validation_all],
            "count": validation_anchor_count,
            "selection": validation_geometry_config["anchor_selection"],
            "seed": validation_geometry_config["seed"],
        },
    )
    brain_geometry_validation = relation_matrix_from_rdms(
        validation_brain_rdms, validation_all
    )
    clip_geometry_validation = relation_matrix_from_rdms(
        validation_clip_rdms, validation_all
    )
    np.save(output_dir / "brain_geometry_validation.npy", brain_geometry_validation)
    np.save(output_dir / "clip_geometry_validation.npy", clip_geometry_validation)
    write_matrix_csv(
        output_dir / "brain_geometry_validation.csv",
        brain_geometry_validation,
        roi_names,
        roi_names,
    )
    write_matrix_csv(
        output_dir / "clip_geometry_validation.csv",
        clip_geometry_validation,
        layer_labels,
        layer_labels,
    )
    save_heatmap(
        output_dir / "brain_geometry_validation_heatmap.png",
        brain_geometry_validation,
        roi_names,
        "Validation ROI representational geometry relation matrix\n(not anatomical distance)",
    )
    save_heatmap(
        output_dir / "clip_geometry_validation_heatmap.png",
        clip_geometry_validation,
        layer_labels,
        "Validation CLIP representational geometry relation matrix",
    )
    save_heatmap_rectangular(
        output_dir / "feature_cost_validation_heatmap.png",
        feature_validation,
        roi_names,
        layer_labels,
        "Validation ROI-to-CLIP prediction cost",
        "1 - cosine",
    )

    validation_inputs = (
        feature_validation,
        brain_geometry_validation,
        clip_geometry_validation,
    )
    solver = prereg["solver"]
    init_specs = prereg["structural_initializations"]["specifications"]
    config_root = output_dir / "configs"
    config_root.mkdir()
    config_summaries = []
    local_minimum: Dict[str, Any] = {}
    best_plans: Dict[str, np.ndarray] = {}
    for config_index, config in enumerate(configs):
        print(f"validation grid {config_index + 1}/{len(configs)}: {config['slug']}", flush=True)
        config_dir = config_root / config["slug"]
        config_dir.mkdir()
        if config["hard_uniform_target"]:
            balanced = solve_balanced_outer_ot(feature_discovery)
            plan = balanced["plan"]
            metrics = validation_metrics(
                plan,
                feature_validation,
                brain_geometry_validation,
                clip_geometry_validation,
                calibration,
            )
            results = [
                {
                    "initialization_kind": "linear_program",
                    "initialization_seed": None,
                    "plan": plan,
                    "discovery_objective": balanced["raw_feature_objective"]
                    / (calibration.feature_scale + calibration.eps_scale),
                    "converged": True,
                    "iterations": balanced["iterations"],
                    "stopping_reason": "linear_program_optimum",
                    **metrics,
                }
            ]
            best_index = 0
        else:
            results = fit_plan_initializations(
                feature_discovery,
                brain_geometry_discovery,
                clip_geometry_discovery,
                calibration,
                config["structure_weight"],
                float(config["coverage_weight"]),
                init_specs,
                solver,
                validation_inputs,
                args.solver_device,
            )
            try:
                best_index, _ = best_discovery_result(results)
            except RuntimeError:
                best_index = -1
        plans = np.stack([item["plan"] for item in results], axis=0)
        np.save(config_dir / "initialization_plans.npy", plans)
        metric_rows = [
            {key: value for key, value in item.items() if key not in {"plan", "discovery_components"}}
            for item in results
        ]
        write_dict_rows(config_dir / "initialization_metrics.csv", metric_rows)
        if best_index >= 0:
            best = results[best_index]
            best_plan = best["plan"]
            best_plans[config["slug"]] = best_plan
            np.save(config_dir / "best_discovery_initialization_plan.npy", best_plan)
            diagnostics = coupling_diagnostics(
                best_plan, roi_names, layer_labels, layer_depths
            )
            write_json(config_dir / "best_discovery_diagnostics.json", diagnostics)
            local = analyze_local_minima(
                results,
                float(prereg["near_optimal_relative_threshold"]),
                int(prereg["top_k_identifiability"]),
            )
            local_minimum[config["slug"]] = local
            config_summary = {
                **config,
                "converged_initialization_count": sum(item["converged"] for item in results),
                "best_discovery_initialization_index": best_index,
                "best_discovery_initialization": [
                    best["initialization_kind"],
                    best["initialization_seed"],
                ],
                "best_discovery_objective": best["discovery_objective"],
                **{
                    key: best[key]
                    for key in (
                        "validation_feature_raw",
                        "validation_feature_normalized",
                        "validation_gw_raw",
                        "validation_gw_normalized",
                        "validation_composite",
                    )
                },
                "target_entropy": diagnostics["target_utilization"]["target_entropy"],
                "effective_target_layer_count": diagnostics["target_utilization"][
                    "effective_target_layer_count"
                ],
                "mean_row_entropy": diagnostics["mean_row_entropy"],
                "mean_pairwise_roi_js": diagnostics[
                    "mean_pairwise_jensen_shannon_divergence"
                ],
                "local_minimum_identifiability": local,
            }
        else:
            config_summary = {
                **config,
                "converged_initialization_count": 0,
                "numerical_failure": "NO_CONVERGED_INITIALIZATION",
            }
        config_summaries.append(config_summary)

    write_json(
        output_dir / "local_minimum_identifiability.json",
        {
            "analysis_name": "local_minimum_identifiability",
            "near_optimal_relative_threshold": prereg["near_optimal_relative_threshold"],
            "top_k": prereg["top_k_identifiability"],
            "configs": local_minimum,
            "unstable_plans_averaged_into_consensus": False,
        },
    )
    write_json(output_dir / "validation_candidate_summary.json", {"configs": config_summaries})
    write_dict_rows(
        output_dir / "validation_candidate_summary.csv",
        [
            {
                key: value
                for key, value in item.items()
                if not isinstance(value, (dict, list))
            }
            for item in config_summaries
        ],
    )

    structural = [
        item
        for item in config_summaries
        if item.get("structural") and item.get("converged_initialization_count", 0) > 0
    ]
    if not structural:
        raise RuntimeError("No converged structural validation candidate")
    selected = min(structural, key=lambda item: item["validation_composite"])
    selected_plan = best_plans[selected["slug"]]
    nonstructural = [
        item
        for item in config_summaries
        if not item.get("structural") and item.get("converged_initialization_count", 0) > 0
    ]
    frozen_baseline_slug = prereg.get("frozen_primary_nonstructural_slug")
    if frozen_baseline_slug is None:
        strongest_baseline = min(
            nonstructural, key=lambda item: item["validation_composite"]
        )
    else:
        matching_baselines = [
            item for item in nonstructural if item["slug"] == frozen_baseline_slug
        ]
        if len(matching_baselines) != 1:
            raise ValueError(
                "Frozen primary non-structural baseline is missing or duplicated: "
                f"{frozen_baseline_slug}"
            )
        strongest_baseline = matching_baselines[0]
    baseline_plan = best_plans[strongest_baseline["slug"]]
    balanced_summary = next(item for item in config_summaries if item["method"] == "balanced_outer_ot")
    balanced_plan = best_plans[balanced_summary["slug"]]
    save_heatmap_rectangular(
        output_dir / "selected_structural_plan_heatmap.png",
        selected_plan,
        roi_names,
        layer_labels,
        "Locked structural configuration\n(plan and initialization fit on discovery only)",
        "transport mass",
    )

    # Validation performance stability: recompute evaluation matrices, never refit plans.
    subsampling = prereg["validation_subsampling"]
    subsample_rng = np.random.default_rng(int(subsampling["seed"]))
    subset_size = int(round(float(subsampling["fraction"]) * len(validation_ids)))
    subsample_rows = []
    plan_items = [(item["slug"], best_plans[item["slug"]]) for item in config_summaries if item["slug"] in best_plans]
    for replicate in range(int(subsampling["count"])):
        indices = np.sort(
            subsample_rng.choice(len(validation_ids), size=subset_size, replace=False)
        )
        feature_subset = validation_costs_per_stimulus[indices].mean(axis=0, dtype=np.float64)
        brain_subset = relation_matrix_from_rdms(validation_brain_rdms, indices)
        clip_subset = relation_matrix_from_rdms(validation_clip_rdms, indices)
        for slug, plan in plan_items:
            metrics = validation_metrics(plan, feature_subset, brain_subset, clip_subset, calibration)
            subsample_rows.append({"replicate": replicate, "config": slug, **metrics})
        if (replicate + 1) % 20 == 0:
            print(f"validation no-replacement subsamples {replicate + 1}/{subsampling['count']}", flush=True)
    write_dict_rows(output_dir / "validation_subsampling_metrics.csv", subsample_rows)
    grouped_subsamples: Dict[str, List[Mapping[str, Any]]] = {}
    for slug, _ in plan_items:
        grouped_subsamples[slug] = [row for row in subsample_rows if row["config"] == slug]
    subsample_summary = {
        slug: {
            metric: summary([row[metric] for row in rows])
            for metric in (
                "validation_feature_normalized",
                "validation_gw_normalized",
                "validation_composite",
            )
        }
        for slug, rows in grouped_subsamples.items()
    }
    differences = {}
    for comparison_name, other_slug in (
        ("selected_minus_strongest_nonstructural", strongest_baseline["slug"]),
        ("selected_minus_balanced_outer_ot", balanced_summary["slug"]),
    ):
        selected_rows = grouped_subsamples[selected["slug"]]
        other_rows = grouped_subsamples[other_slug]
        differences[comparison_name] = {
            "other_config": other_slug,
            **{
                metric: summary(
                    [left[metric] - right[metric] for left, right in zip(selected_rows, other_rows)]
                )
                for metric in (
                    "validation_feature_normalized",
                    "validation_gw_normalized",
                    "validation_composite",
                )
            },
        }
    write_json(
        output_dir / "validation_subsampling_summary.json",
        {
            "sampling": "without replacement from offline_validation; plans frozen",
            "count": subsampling["count"],
            "fraction": subsampling["fraction"],
            "subset_size": subset_size,
            "per_config": subsample_summary,
            "paired_differences": differences,
        },
    )

    # Preregistered fixed structural comparisons. These are descriptive checks;
    # they do not alter the locked configuration or initialization selection.
    comparison_rows = []
    comparison_payload = []
    summaries_by_slug = {item["slug"]: item for item in config_summaries}
    for structural_item in structural:
        for baseline_kind, baseline_slug in (
            ("frozen_primary_nonstructural", strongest_baseline["slug"]),
            ("balanced_outer_ot", balanced_summary["slug"]),
        ):
            baseline_item = summaries_by_slug[baseline_slug]
            if baseline_slug not in grouped_subsamples:
                unavailable = {
                    "structural_config": structural_item["slug"],
                    "baseline_kind": baseline_kind,
                    "baseline_config": baseline_slug,
                    "structure_weight": structural_item["structure_weight"],
                    "coverage_weight": structural_item["coverage_weight"],
                    "comparison_status": "BASELINE_NOT_CONVERGED",
                    "baseline_numerical_failure": baseline_item.get(
                        "numerical_failure", "NO_FROZEN_PLAN"
                    ),
                }
                comparison_rows.append(unavailable)
                comparison_payload.append(unavailable)
                continue
            structural_rows = grouped_subsamples[structural_item["slug"]]
            baseline_rows = grouped_subsamples[baseline_slug]
            paired = {
                metric: summary(
                    [
                        left[metric] - right[metric]
                        for left, right in zip(structural_rows, baseline_rows)
                    ]
                )
                for metric in (
                    "validation_feature_normalized",
                    "validation_gw_normalized",
                    "validation_composite",
                )
            }
            record = {
                "structural_config": structural_item["slug"],
                "baseline_kind": baseline_kind,
                "baseline_config": baseline_slug,
                "structure_weight": structural_item["structure_weight"],
                "coverage_weight": structural_item["coverage_weight"],
                "comparison_status": "AVAILABLE",
                "heldout_feature_difference": structural_item[
                    "validation_feature_normalized"
                ]
                - baseline_item["validation_feature_normalized"],
                "heldout_gw_difference": structural_item["validation_gw_normalized"]
                - baseline_item["validation_gw_normalized"],
                "heldout_composite_difference": structural_item[
                    "validation_composite"
                ]
                - baseline_item["validation_composite"],
                "structural_target_entropy": structural_item["target_entropy"],
                "baseline_target_entropy": baseline_item["target_entropy"],
                "structural_mean_row_entropy": structural_item["mean_row_entropy"],
                "baseline_mean_row_entropy": baseline_item["mean_row_entropy"],
                "structural_mean_pairwise_roi_js": structural_item[
                    "mean_pairwise_roi_js"
                ],
                "baseline_mean_pairwise_roi_js": baseline_item[
                    "mean_pairwise_roi_js"
                ],
                "structural_near_optimal_count": structural_item[
                    "local_minimum_identifiability"
                ].get("near_optimal_count"),
                "baseline_near_optimal_count": baseline_item[
                    "local_minimum_identifiability"
                ].get("near_optimal_count"),
                "subsample_feature_difference_ci_2_5": paired[
                    "validation_feature_normalized"
                ]["ci_2_5"],
                "subsample_feature_difference_ci_97_5": paired[
                    "validation_feature_normalized"
                ]["ci_97_5"],
                "subsample_gw_difference_ci_2_5": paired[
                    "validation_gw_normalized"
                ]["ci_2_5"],
                "subsample_gw_difference_ci_97_5": paired[
                    "validation_gw_normalized"
                ]["ci_97_5"],
                "subsample_composite_difference_ci_2_5": paired[
                    "validation_composite"
                ]["ci_2_5"],
                "subsample_composite_difference_ci_97_5": paired[
                    "validation_composite"
                ]["ci_97_5"],
            }
            comparison_rows.append(record)
            comparison_payload.append({**record, "paired_subsampling": paired})
    write_dict_rows(output_dir / "primary_structure_comparisons.csv", comparison_rows)
    write_json(
        output_dir / "primary_structure_comparisons.json",
        {
            "selection_affected": False,
            "comparisons": comparison_payload,
        },
    )

    # Required geometry identity nulls, fit on discovery and evaluate on original validation.
    null_config = prereg["null_experiments"]
    null_specs = null_config["null_fit_initializations"]
    null_rng = np.random.default_rng(int(null_config["seed"]))
    brain_permutations = unique_permutations(
        int(null_config["brain_geometry_unique_permutations"]), 8, null_rng
    )
    clip_permutations = unique_permutations(
        int(null_config["clip_geometry_unique_permutations"]), 6, null_rng
    )
    brain_null_rows = []
    brain_null_plans = []
    for index, permutation in enumerate(brain_permutations):
        geometry_null = brain_geometry_discovery[np.ix_(permutation, permutation)]
        fitted = fit_null_plan(
            feature_discovery,
            geometry_null,
            clip_geometry_discovery,
            calibration,
            selected,
            null_specs,
            solver,
            validation_inputs,
            args.solver_device,
        )
        brain_null_plans.append(fitted["plan"])
        brain_null_rows.append(
            {
                "null_index": index,
                "permutation": " ".join(map(str, permutation.tolist())),
                **{key: value for key, value in fitted.items() if key not in {"plan", "discovery_components"}},
            }
        )
        if (index + 1) % 25 == 0:
            print(f"brain identity-mismatch null {index + 1}/{len(brain_permutations)}", flush=True)
    np.save(output_dir / "brain_geometry_identity_null_plans.npy", np.stack(brain_null_plans))
    write_dict_rows(output_dir / "brain_geometry_identity_null.csv", brain_null_rows)

    clip_null_rows = []
    clip_null_plans = []
    for index, permutation in enumerate(clip_permutations):
        geometry_null = clip_geometry_discovery[np.ix_(permutation, permutation)]
        fitted = fit_null_plan(
            feature_discovery,
            brain_geometry_discovery,
            geometry_null,
            calibration,
            selected,
            null_specs,
            solver,
            validation_inputs,
            args.solver_device,
        )
        clip_null_plans.append(fitted["plan"])
        clip_null_rows.append(
            {
                "null_index": index,
                "permutation": " ".join(map(str, permutation.tolist())),
                **{key: value for key, value in fitted.items() if key not in {"plan", "discovery_components"}},
            }
        )
        if (index + 1) % 25 == 0:
            print(f"CLIP identity-mismatch null {index + 1}/{len(clip_permutations)}", flush=True)
    np.save(output_dir / "clip_geometry_identity_null_plans.npy", np.stack(clip_null_plans))
    write_dict_rows(output_dir / "clip_geometry_identity_null.csv", clip_null_rows)

    real_validation = validation_metrics(
        selected_plan,
        feature_validation,
        brain_geometry_validation,
        clip_geometry_validation,
        calibration,
    )
    geometry_null_summary = {
        "evaluation_matrices": "original unpermuted offline_validation matrices",
        "selected_real": real_validation,
        "brain_identity_mismatch": {
            "unique_permutation_count": len(brain_permutations),
            "validation_gw_normalized": summary(
                [row["validation_gw_normalized"] for row in brain_null_rows]
            ),
            "validation_composite": summary(
                [row["validation_composite"] for row in brain_null_rows]
            ),
            "one_sided_p_real_gw_lower": one_sided_lower_p(
                real_validation["validation_gw_normalized"],
                [row["validation_gw_normalized"] for row in brain_null_rows],
            ),
        },
        "clip_identity_mismatch": {
            "unique_permutation_count": len(clip_permutations),
            "validation_gw_normalized": summary(
                [row["validation_gw_normalized"] for row in clip_null_rows]
            ),
            "validation_composite": summary(
                [row["validation_composite"] for row in clip_null_rows]
            ),
            "one_sided_p_real_gw_lower": one_sided_lower_p(
                real_validation["validation_gw_normalized"],
                [row["validation_gw_normalized"] for row in clip_null_rows],
            ),
        },
    }
    write_json(output_dir / "geometry_identity_null_summary.json", geometry_null_summary)

    # Feature-pairing null: same folds/protocol, independently permuted pairs.
    assignments, _ = build_unique_id_folds(
        discovery_ids,
        int(feature_config["cross_fitting"]["folds"]),
        int(feature_config["cross_fitting"]["random_seed"]),
    )
    feature_null_costs = []
    feature_null_plans = []
    feature_null_rows = []
    for index in range(int(null_config["feature_pairing_permutations"])):
        rng = np.random.default_rng(int(null_config["seed"]) + 10000 + index)
        cost_null = feature_pairing_null_cost(
            discovery_brain,
            discovery_clip,
            assignments,
            float(ridge["alpha"]),
            float(preprocessing["std_epsilon"]),
            rng,
            args.probe_device,
        )
        fitted = fit_null_plan(
            cost_null,
            brain_geometry_discovery,
            clip_geometry_discovery,
            calibration,
            selected,
            null_specs,
            solver,
            validation_inputs,
            args.solver_device,
        )
        feature_null_costs.append(cost_null)
        feature_null_plans.append(fitted["plan"])
        feature_null_rows.append(
            {
                "null_index": index,
                "seed": int(null_config["seed"]) + 10000 + index,
                **{key: value for key, value in fitted.items() if key not in {"plan", "discovery_components"}},
            }
        )
        print(f"feature-pairing null {index + 1}/{null_config['feature_pairing_permutations']}", flush=True)
    np.save(output_dir / "feature_pairing_null_costs.npy", np.stack(feature_null_costs))
    np.save(output_dir / "feature_pairing_null_plans.npy", np.stack(feature_null_plans))
    write_dict_rows(output_dir / "feature_pairing_null_metrics.csv", feature_null_rows)
    write_json(
        output_dir / "feature_pairing_null_summary.json",
        {
            "count": len(feature_null_rows),
            "validation_feature_normalized": summary(
                [row["validation_feature_normalized"] for row in feature_null_rows]
            ),
            "validation_gw_normalized": summary(
                [row["validation_gw_normalized"] for row in feature_null_rows]
            ),
            "validation_composite": summary(
                [row["validation_composite"] for row in feature_null_rows]
            ),
            "one_sided_p_real_feature_lower": one_sided_lower_p(
                real_validation["validation_feature_normalized"],
                [row["validation_feature_normalized"] for row in feature_null_rows],
            ),
            "evaluation_matrices": "original unpermuted offline_validation matrices",
        },
    )

    # Independent ROI stimulus-identity geometry null on frozen discovery anchor IDs.
    anchor_payload = json.loads((geometry_dir / "anchor_ids.json").read_text())
    discovery_global_to_position = {
        int(global_row): position for position, global_row in enumerate(discovery_indices)
    }
    anchor_positions = np.asarray(
        [discovery_global_to_position[int(row)] for row in anchor_payload["global_cache_rows"]],
        dtype=np.int64,
    )
    discovery_anchor_rdms = cosine_distance_matrices(
        discovery_brain, anchor_positions, args.probe_device
    )
    independent_rows = []
    independent_plans = []
    independent_geometries = []
    for index in range(int(null_config["independent_roi_stimulus_permutations"])):
        rng = np.random.default_rng(int(null_config["seed"]) + 20000 + index)
        geometry_null = independently_permuted_brain_geometry(discovery_anchor_rdms, rng)
        fitted = fit_null_plan(
            feature_discovery,
            geometry_null,
            clip_geometry_discovery,
            calibration,
            selected,
            null_specs,
            solver,
            validation_inputs,
            args.solver_device,
        )
        independent_geometries.append(geometry_null)
        independent_plans.append(fitted["plan"])
        independent_rows.append(
            {
                "null_index": index,
                "seed": int(null_config["seed"]) + 20000 + index,
                **{key: value for key, value in fitted.items() if key not in {"plan", "discovery_components"}},
            }
        )
        if (index + 1) % 20 == 0:
            print(f"independent-ROI stimulus null {index + 1}/{null_config['independent_roi_stimulus_permutations']}", flush=True)
    np.save(output_dir / "independent_roi_geometry_null_matrices.npy", np.stack(independent_geometries))
    np.save(output_dir / "independent_roi_geometry_null_plans.npy", np.stack(independent_plans))
    write_dict_rows(output_dir / "independent_roi_geometry_null_metrics.csv", independent_rows)
    write_json(
        output_dir / "independent_roi_geometry_null_summary.json",
        {
            "count": len(independent_rows),
            "validation_gw_normalized": summary(
                [row["validation_gw_normalized"] for row in independent_rows]
            ),
            "validation_composite": summary(
                [row["validation_composite"] for row in independent_rows]
            ),
            "one_sided_p_real_gw_lower": one_sided_lower_p(
                real_validation["validation_gw_normalized"],
                [row["validation_gw_normalized"] for row in independent_rows],
            ),
            "per_roi_permutation_is_independent": True,
            "evaluation_matrices": "original unpermuted offline_validation matrices",
        },
    )

    # Matched-marginal, exactly row-entropy-matched null via unique ROI-row permutations.
    row_permutations = unique_permutations(
        int(null_config["matched_marginal_unique_row_permutations"]), 8, null_rng
    )
    matched_plans = np.stack([selected_plan[permutation] for permutation in row_permutations])
    matched_rows = []
    selected_diag = coupling_diagnostics(selected_plan, roi_names, layer_labels, layer_depths)
    for index, (permutation, plan) in enumerate(zip(row_permutations, matched_plans)):
        metrics = validation_metrics(
            plan,
            feature_validation,
            brain_geometry_validation,
            clip_geometry_validation,
            calibration,
        )
        diag = coupling_diagnostics(plan, roi_names, layer_labels, layer_depths)
        row_corr = []
        for roi in range(8):
            if np.std(plan[roi]) > 0 and np.std(selected_plan[roi]) > 0:
                row_corr.append(float(np.corrcoef(plan[roi], selected_plan[roi])[0, 1]))
        matched_rows.append(
            {
                "null_index": index,
                "row_permutation": " ".join(map(str, permutation.tolist())),
                **metrics,
                "mean_pairwise_roi_js": diag["mean_pairwise_jensen_shannon_divergence"],
                "preferred_layer_diversity": len(np.unique(np.argmax(plan, axis=1))),
                "row_correlation_to_real_mean": float(np.mean(row_corr)) if row_corr else float("nan"),
                "preferred_layer_agreement_to_real": float(
                    np.mean(np.argmax(plan, axis=1) == np.argmax(selected_plan, axis=1))
                ),
            }
        )
    np.save(output_dir / "matched_marginal_random_plans.npy", matched_plans)
    write_dict_rows(output_dir / "matched_marginal_random_plan_metrics.csv", matched_rows)
    matched_summary = {
            "construction": (
                "unique random ROI-row permutations of real T; preserves source and target "
                "marginals exactly and preserves the row-entropy multiset exactly"
            ),
            "count": len(matched_rows),
            "real_mean_pairwise_roi_js": selected_diag[
                "mean_pairwise_jensen_shannon_divergence"
            ],
            "null_mean_pairwise_roi_js": summary(
                [row["mean_pairwise_roi_js"] for row in matched_rows]
            ),
            "real_preferred_layer_diversity": len(np.unique(np.argmax(selected_plan, axis=1))),
            "null_preferred_layer_diversity": summary(
                [row["preferred_layer_diversity"] for row in matched_rows]
            ),
            "row_correlation_to_real": summary(
                [row["row_correlation_to_real_mean"] for row in matched_rows]
            ),
            "preferred_layer_agreement_to_real": summary(
                [row["preferred_layer_agreement_to_real"] for row in matched_rows]
            ),
            "validation_gw_normalized": summary(
                [row["validation_gw_normalized"] for row in matched_rows]
            ),
            "one_sided_p_real_gw_lower": one_sided_lower_p(
                real_validation["validation_gw_normalized"],
                [row["validation_gw_normalized"] for row in matched_rows],
            ),
            "interpretation": "JS/diversity alone cannot distinguish real row identity from this null",
        }
    write_json(
        output_dir / "matched_marginal_random_plan_null.json",
        matched_summary,
    )

    # One post-selection registered variance-standardized geometry sensitivity.
    discovery_mean = discovery_brain.mean(axis=0, dtype=np.float64)
    discovery_std = discovery_brain.std(axis=0, dtype=np.float64)
    discovery_std = np.where(discovery_std > 1e-6, discovery_std, 1.0)
    discovery_standardized = (
        (discovery_brain.astype(np.float64) - discovery_mean) / discovery_std
    ).astype(np.float32)
    validation_standardized = (
        (validation_brain.astype(np.float64) - discovery_mean) / discovery_std
    ).astype(np.float32)
    standardized_discovery_rdms = cosine_distance_matrices(
        discovery_standardized, anchor_positions, args.probe_device
    )
    standardized_validation_rdms = cosine_distance_matrices(
        validation_standardized,
        np.arange(len(validation_standardized)),
        args.probe_device,
    )
    standardized_brain_discovery = relation_matrix_from_rdms(
        standardized_discovery_rdms, np.arange(len(anchor_positions))
    )
    standardized_brain_validation = relation_matrix_from_rdms(
        standardized_validation_rdms, np.arange(len(validation_standardized))
    )
    sensitivity_results = fit_plan_initializations(
        feature_discovery,
        standardized_brain_discovery,
        clip_geometry_discovery,
        calibration,
        float(selected["structure_weight"]),
        float(selected["coverage_weight"]),
        init_specs,
        solver,
        (
            feature_validation,
            standardized_brain_validation,
            clip_geometry_validation,
        ),
        args.solver_device,
    )
    sensitivity_index, sensitivity_best = best_discovery_result(sensitivity_results)
    sensitivity_plan = sensitivity_best["plan"]
    normalized_primary = selected_plan * 8.0
    normalized_sensitivity = sensitivity_plan * 8.0
    row_correlations = []
    row_js_values = []
    for roi in range(8):
        if np.std(normalized_primary[roi]) > 0 and np.std(normalized_sensitivity[roi]) > 0:
            row_correlations.append(
                float(np.corrcoef(normalized_primary[roi], normalized_sensitivity[roi])[0, 1])
            )
        row_js_values.append(
            jensen_shannon_divergence(
                normalized_primary[roi], normalized_sensitivity[roi]
            )
        )
    frobenius_distance = float(np.linalg.norm(selected_plan - sensitivity_plan))
    denominator = np.linalg.norm(selected_plan) * np.linalg.norm(sensitivity_plan)
    variance_sensitivity = {
        "analysis_name": "variance_standardized_geometry_sensitivity",
        "used_for_selection": False,
        "standardization_fit_split": "offline_discovery only",
        "validation_standardization_uses_discovery_statistics": True,
        "selected_primary_config": selected["slug"],
        "sensitivity_best_discovery_initialization_index": sensitivity_index,
        "sensitivity_best_discovery_initialization": [
            sensitivity_best["initialization_kind"],
            sensitivity_best["initialization_seed"],
        ],
        "row_correlation": summary(row_correlations),
        "row_js_divergence": summary(row_js_values),
        "frobenius_distance": frobenius_distance,
        "frobenius_cosine_similarity": float(
            np.sum(selected_plan * sensitivity_plan) / denominator
        ),
        "preferred_layer_agreement": float(
            np.mean(np.argmax(selected_plan, axis=1) == np.argmax(sensitivity_plan, axis=1))
        ),
        "primary_validation_metrics": real_validation,
        "sensitivity_validation_metrics_on_standardized_geometry": {
            key: sensitivity_best[key]
            for key in (
                "validation_feature_normalized",
                "validation_gw_normalized",
                "validation_composite",
            )
        },
    }
    np.save(output_dir / "brain_geometry_discovery_variance_standardized.npy", standardized_brain_discovery)
    np.save(output_dir / "brain_geometry_validation_variance_standardized.npy", standardized_brain_validation)
    np.save(output_dir / "variance_standardized_sensitivity_plan.npy", sensitivity_plan)
    write_json(output_dir / "variance_standardized_geometry_sensitivity.json", variance_sensitivity)

    # Fixed comparisons and decision.
    selected_local = local_minimum[selected["slug"]]
    identifiability_reference = (
        selected_local["near_optimal"]
        if selected_local.get("near_optimal_count", 0) >= 2
        else selected_local["top_k_analysis"]
    )
    thresholds = prereg["decision_thresholds"]
    pref_median = identifiability_reference.get("preferred_layer_agreement_fraction", {}).get(
        "median", float("nan")
    )
    row_corr_median = identifiability_reference.get("mean_row_correlation", {}).get(
        "median", float("nan")
    )
    identifiable = (
        np.isfinite(pref_median)
        and np.isfinite(row_corr_median)
        and pref_median >= thresholds["minimum_median_preferred_layer_agreement"]
        and row_corr_median >= thresholds["minimum_median_row_correlation"]
    )
    selected_subsample = subsample_summary[selected["slug"]]["validation_composite"]
    selected_cv = selected_subsample["std"] / abs(selected_subsample["mean"])
    performance_reproducible = selected_cv <= thresholds[
        "maximum_selected_composite_coefficient_of_variation"
    ]
    baseline_difference = differences["selected_minus_strongest_nonstructural"]
    structural_benefit = (
        baseline_difference["validation_gw_normalized"]["ci_97_5"] < 0
        and baseline_difference["validation_composite"]["ci_97_5"] < 0
    )
    null_p_values = {
        "brain_identity": geometry_null_summary["brain_identity_mismatch"][
            "one_sided_p_real_gw_lower"
        ],
        "clip_identity": geometry_null_summary["clip_identity_mismatch"][
            "one_sided_p_real_gw_lower"
        ],
        "independent_roi_stimulus": one_sided_lower_p(
            real_validation["validation_gw_normalized"],
            [row["validation_gw_normalized"] for row in independent_rows],
        ),
    }
    null_separation = all(
        value <= thresholds["null_separation_one_sided_p_max"]
        for value in null_p_values.values()
    )
    matched_plan_specialization = (
        matched_summary["one_sided_p_real_gw_lower"]
        <= thresholds.get(
            "matched_plan_specialization_requires_real_validation_gw_p_max",
            thresholds["null_separation_one_sided_p_max"],
        )
    )
    matched_plan_reproducibility = True
    if thresholds.get(
        "matched_plan_reproducibility_requires_identifiability_preferred_agreement_above_null_ci_97_5",
        False,
    ):
        matched_plan_reproducibility = matched_plan_reproducibility and (
            pref_median
            > matched_summary["preferred_layer_agreement_to_real"]["ci_97_5"]
        )
    if thresholds.get(
        "matched_plan_reproducibility_requires_identifiability_row_correlation_above_null_ci_97_5",
        False,
    ):
        matched_plan_reproducibility = matched_plan_reproducibility and (
            row_corr_median
            > matched_summary["row_correlation_to_real"]["ci_97_5"]
        )
    matched_plan_pass = matched_plan_specialization and matched_plan_reproducibility
    if (
        performance_reproducible
        and null_separation
        and structural_benefit
        and identifiable
        and matched_plan_pass
    ):
        validation_status = "READY_FOR_OFFLINE_TEST"
    elif not structural_benefit:
        validation_status = "NO_GO"
    else:
        validation_status = "INCONCLUSIVE"

    provenance = {
        "stimulus_manifest_sha256": sha256_file(manifest_path),
        "preregistered_grid_sha256": prereg_hash_before,
        "feature_cost_discovery_sha256": sha256_file(feature_dir / "feature_cost_oof.npy"),
        "brain_geometry_discovery_sha256": sha256_file(geometry_dir / "brain_geometry.npy"),
        "clip_geometry_discovery_sha256": sha256_file(geometry_dir / "clip_geometry.npy"),
        "feature_cost_validation_sha256": sha256_file(output_dir / "feature_cost_validation.npy"),
        "brain_geometry_validation_sha256": sha256_file(output_dir / "brain_geometry_validation.npy"),
        "clip_geometry_validation_sha256": sha256_file(output_dir / "clip_geometry_validation.npy"),
        "validation_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    locked = {
        "lock_scope": "configuration only; not a final transport plan",
        "method": selected["method"],
        "structure_weight": selected["structure_weight"],
        "coverage_weight": selected["coverage_weight"],
        "entropy_weight": solver["entropy_weight"],
        "solver_settings": solver,
        "initialization_specs": init_specs,
        "best_discovery_initialization": selected["best_discovery_initialization"],
        "fixed_loss_calibration": prereg["fixed_loss_scales"],
        "selection_criterion": prereg["selection"]["criterion"],
        "selection_split": (
            "offline_discovery"
            if prereg.get("locked_cross_subject_replication", False)
            else "offline_validation"
        ),
        "initialization_selection_split": "offline_discovery",
        "matrix_hashes": provenance,
        "VALIDATION_STATUS": validation_status,
        "exploratory_single_subject": not bool(
            prereg.get("locked_cross_subject_replication", False)
        ),
        "authorizes_offline_test": validation_status == "READY_FOR_OFFLINE_TEST",
        "best_transport_plan_created": False,
    }
    write_json(output_dir / "fgw_locked_configuration.json", locked)
    final_summary = {
        "VALIDATION_STATUS": validation_status,
        "result_scope": prereg.get("result_scope", "EXPLORATORY subj01"),
        "selected_structural_config": selected,
        "strongest_nonstructural_baseline": strongest_baseline,
        "balanced_outer_ot": balanced_summary,
        "selected_validation_metrics": real_validation,
        "selected_subsample_composite_cv": selected_cv,
        "selected_minus_baseline_subsampling": baseline_difference,
        "decision_checks": {
            "validation_performance_reproducible": performance_reproducible,
            "real_geometry_separates_from_all_required_geometry_nulls": null_separation,
            "structural_benefit_over_strongest_nonstructural": structural_benefit,
            "local_minimum_identifiable": identifiable,
            "matched_plan_specialization": matched_plan_specialization,
            "matched_plan_reproducibility": matched_plan_reproducibility,
            "matched_plan_pass": matched_plan_pass,
            "roi_row_reproducibility_reference": identifiability_reference,
            "offline_test_accessed": False,
        },
        "geometry_null_p_values": null_p_values,
        "feature_pairing_null_p_real_feature_lower": one_sided_lower_p(
            real_validation["validation_feature_normalized"],
            [row["validation_feature_normalized"] for row in feature_null_rows],
        ),
        "matched_marginal_null_p_real_gw_lower": matched_summary[
            "one_sided_p_real_gw_lower"
        ],
        "variance_sensitivity": variance_sensitivity,
        "provenance": provenance,
        "data_usage": {
            "offline_discovery_used_for_fit": True,
            "offline_validation_used_for_evaluation_and_config_selection": not bool(
                prereg.get("locked_cross_subject_replication", False)
            ),
            "offline_validation_used_for_evaluation_only": bool(
                prereg.get("locked_cross_subject_replication", False)
            ),
            "offline_validation_used_to_select_initialization_within_config": False,
            "offline_test_accessed": False,
            "protected_300_validation_accessed": False,
            "protected_982_test_accessed": False,
            "stage2_run": False,
        },
        "best_transport_plan_created": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "validation_summary.json", final_summary)
    write_json(
        output_dir / "full_provenance.json",
        {
            **provenance,
            "preregistered_grid_unchanged_during_run": sha256_file(prereg_path)
            == prereg_hash_before,
            "roi_order": roi_names,
            "clip_layer_order": layer_depths,
            "discovery_count": len(discovery_ids),
            "validation_count": len(validation_ids),
            "offline_test_count_loaded": 0,
            "fixed_geometry_definition": geometry_config["geometry_definition"],
            "fixed_probe_protocol": feature_config,
            "feature_variance_registered_confound": {"rho": 0.7760, "p": 0.0236},
        },
    )
    if list(output_dir.rglob("best_transport_plan.npy")):
        raise RuntimeError("Forbidden best_transport_plan.npy was created")
    print(f"Saved Prompt-4A validation to {output_dir}", flush=True)
    print(f"VALIDATION_STATUS = {validation_status}", flush=True)
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
