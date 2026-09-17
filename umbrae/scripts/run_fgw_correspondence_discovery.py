#!/usr/bin/env python
"""Run the fixed Prompt-3 discovery-only srFGW candidate suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.fgw_correspondence import (
    LossCalibration,
    calibration_from_uniform_plan,
    coupling_diagnostics,
    objective_components,
    solve_balanced_outer_ot,
    solve_multiple_initializations,
    validate_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run discovery-only srFGW candidates")
    parser.add_argument("--stimulus-manifest", required=True)
    parser.add_argument("--feature-cost-dir", required=True)
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=2500)
    parser.add_argument("--tolerance", type=float, default=1e-11)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--eps-scale", type=float, default=1e-12)
    parser.add_argument("--eps-coverage", type=float, default=1e-12)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def write_plan_csv(
    path: Path, plan: np.ndarray, roi_names: Sequence[str], layer_labels: Sequence[str]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["roi", *layer_labels])
        for roi_name, row in zip(roi_names, plan):
            writer.writerow([roi_name, *[f"{float(value):.12g}" for value in row]])


def write_trace(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Optimization trace is empty")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_plan(
    plan: np.ndarray,
    feature_cost: np.ndarray,
    brain_geometry: np.ndarray,
    clip_geometry: np.ndarray,
    calibration: LossCalibration,
    structure_weight: float,
    coverage_weight: float,
    entropy_weight: float,
    eps_cov: float,
) -> Dict[str, float]:
    components = objective_components(
        torch.as_tensor(plan, dtype=torch.float64),
        torch.as_tensor(feature_cost, dtype=torch.float64),
        torch.as_tensor(brain_geometry, dtype=torch.float64),
        torch.as_tensor(clip_geometry, dtype=torch.float64),
        calibration,
        structure_weight,
        coverage_weight,
        entropy_weight,
        eps_cov,
    )
    return {key: float(value.detach().cpu()) for key, value in components.items()}


def initialization_variability(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    plans = [np.asarray(item["plan"]) for item in results]
    objectives = np.asarray([item["components"]["total_objective"] for item in results])
    distances = [
        float(np.linalg.norm(plans[left] - plans[right]))
        for left in range(len(plans))
        for right in range(left + 1, len(plans))
    ]
    return {
        "initialization_count": len(results),
        "objective_min": float(objectives.min()),
        "objective_max": float(objectives.max()),
        "objective_range": float(objectives.max() - objectives.min()),
        "pairwise_plan_frobenius_min": float(min(distances)) if distances else 0.0,
        "pairwise_plan_frobenius_mean": float(np.mean(distances)) if distances else 0.0,
        "pairwise_plan_frobenius_max": float(max(distances)) if distances else 0.0,
        "converged_count": sum(bool(item["converged"]) for item in results),
    }


def load_roi_variance(path: Path, roi_names: Sequence[str]) -> List[float]:
    with path.open() as handle:
        rows = {row["roi"]: row for row in csv.DictReader(handle)}
    return [float(rows[name]["mean_feature_variance"]) for name in roi_names]


def save_candidate(
    candidate_dir: Path,
    method: str,
    structure_weight: float,
    coverage_weight: float,
    feature_cost: np.ndarray,
    brain_geometry: np.ndarray,
    clip_geometry: np.ndarray,
    calibration: LossCalibration,
    roi_names: Sequence[str],
    layer_labels: Sequence[str],
    layer_depths: Sequence[int],
    roi_variance: Sequence[float],
    provenance: Mapping[str, Any],
    args: argparse.Namespace,
    balanced: bool = False,
) -> Dict[str, Any]:
    candidate_dir.mkdir(parents=True, exist_ok=False)
    if balanced:
        balanced_result = solve_balanced_outer_ot(feature_cost)
        plan = balanced_result["plan"]
        components = evaluate_plan(
            plan,
            feature_cost,
            brain_geometry,
            clip_geometry,
            calibration,
            0.0,
            0.0,
            args.entropy_weight,
            args.eps_coverage,
        )
        trace_rows = [
            {
                "initialization_kind": "linear_program",
                "initialization_seed": "",
                "iteration": balanced_result["iterations"],
                **components,
                "gradient_norm": "",
                "step_norm": "",
                "source_marginal_feasibility_error": balanced_result[
                    "source_marginal_feasibility_error"
                ],
                "target_marginal_feasibility_error": balanced_result[
                    "target_marginal_feasibility_error"
                ],
                "objective_improvement": "",
            }
        ]
        initialization_summary = {
            "selection": "not applicable; exact balanced linear program",
            "solver": balanced_result["solver"],
            "converged": True,
            "iterations": balanced_result["iterations"],
        }
        np.savez_compressed(candidate_dir / "all_initializations.npz", linear_program=plan)
    else:
        specs = [
            ("uniform", 0),
            ("random", 11),
            ("random", 23),
            ("random", 37),
            ("feature_informed", 0),
        ]
        solved = solve_multiple_initializations(
            feature_cost,
            brain_geometry,
            clip_geometry,
            calibration,
            structure_weight,
            coverage_weight,
            specs,
            entropy_weight=args.entropy_weight,
            learning_rate=args.learning_rate,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            patience=args.patience,
            eps_cov=args.eps_coverage,
            device=args.device,
        )
        winner = solved["winner"]
        plan = winner["plan"]
        components = winner["components"]
        trace_rows = []
        plans_to_save: Dict[str, np.ndarray] = {}
        result_summaries = []
        for result in solved["all_results"]:
            identifier = f"{result['initialization_kind']}_seed{result['initialization_seed']}"
            plans_to_save[identifier] = result["plan"]
            for trace_row in result["trace"]:
                trace_rows.append(
                    {
                        "initialization_kind": result["initialization_kind"],
                        "initialization_seed": result["initialization_seed"],
                        **trace_row,
                    }
                )
            result_summaries.append(
                {
                    "initialization_kind": result["initialization_kind"],
                    "initialization_seed": result["initialization_seed"],
                    "components": result["components"],
                    "converged": result["converged"],
                    "iterations": result["iterations"],
                    "stopping_reason": result["stopping_reason"],
                    "source_marginal_feasibility_error": result[
                        "source_marginal_feasibility_error"
                    ],
                }
            )
        np.savez_compressed(candidate_dir / "all_initializations.npz", **plans_to_save)
        initialization_summary = {
            "winner_index": solved["winner_index"],
            "winner_kind": winner["initialization_kind"],
            "winner_seed": winner["initialization_seed"],
            "selection_metric": solved["selection_metric"],
            "all_results": result_summaries,
            "variability": initialization_variability(solved["all_results"]),
        }

    diagnostics = coupling_diagnostics(
        plan, roi_names, layer_labels, layer_depths, roi_feature_variance=roi_variance
    )
    diagnostics["optimization"] = {
        **components,
        "method": method,
        "structure_weight": structure_weight,
        "coverage_weight": coverage_weight,
        "entropy_weight": args.entropy_weight,
        "initialization_variability": initialization_summary.get("variability"),
    }
    target = plan.sum(axis=0)
    np.save(candidate_dir / "transport_plan.npy", plan)
    write_plan_csv(candidate_dir / "transport_plan.csv", plan, roi_names, layer_labels)
    write_json(
        candidate_dir / "target_marginal.json",
        {
            "target_marginal": {
                label: float(value) for label, value in zip(layer_labels, target)
            },
            "sum": float(target.sum()),
            "hard_uniform_target_constraint": balanced,
            "coverage_is_soft_and_does_not_guarantee_nonzero_mass": not balanced,
        },
    )
    write_json(candidate_dir / "coupling_diagnostics.json", diagnostics)
    write_trace(candidate_dir / "optimization_trace.csv", trace_rows)
    solver_config = {
        "method": method,
        "candidate_stage": "offline_discovery_only",
        "scientific_method_selected": False,
        "source_marginal": [1.0 / len(roi_names)] * len(roi_names),
        "target_marginal": "uniform hard constraint" if balanced else "free",
        "structure_weight": structure_weight,
        "coverage_weight": coverage_weight,
        "entropy_weight": args.entropy_weight,
        "entropy_role": "disabled; not used as a scientific anti-collapse mechanism",
        "eps_scale": calibration.eps_scale,
        "eps_coverage": args.eps_coverage,
        "loss_scales": {
            "s_feature": calibration.feature_scale,
            "s_gw": calibration.gw_scale,
        },
        "objective": (
            "(1-beta)*feature_raw/(s_feature+eps_scale) + "
            "beta*gw_raw/(s_gw+eps_scale) + lambda_cov*KL(target||uniform) + "
            "epsilon_entropy*sum(T log(T+eps))"
        ),
        "solver": (
            "scipy exact linear programming with hard marginals"
            if balanced
            else "projected gradient with Euclidean row-simplex projection and backtracking"
        ),
        "allows_exact_zero_entries": True,
        "coverage_guarantees_nonzero_target_mass": False,
        "learning_rate": None if balanced else args.learning_rate,
        "max_iterations": None if balanced else args.max_iterations,
        "tolerance": None if balanced else args.tolerance,
        "patience": None if balanced else args.patience,
        "initializations": initialization_summary,
        "roi_order": list(roi_names),
        "clip_layer_order": list(layer_depths),
        "matrix_and_provenance_hashes": dict(provenance),
        "offline_validation_loaded": False,
        "offline_test_loaded": False,
        "protected_validation_loaded": False,
        "protected_test_loaded": False,
    }
    write_json(candidate_dir / "solver_config.json", solver_config)
    return {
        "directory": candidate_dir.name,
        "method": method,
        "structure_weight": structure_weight,
        "coverage_weight": coverage_weight,
        "components": components,
        "target_marginal": target,
        "diagnostics": diagnostics,
        "initializations": initialization_summary,
    }


def run(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.stimulus_manifest).expanduser().resolve()
    feature_dir = Path(args.feature_cost_dir).expanduser().resolve()
    geometry_dir = Path(args.geometry_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite discovery output: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.time()

    manifest = json.loads(manifest_path.read_text())
    feature_config = json.loads((feature_dir / "feature_cost_config.json").read_text())
    geometry_config = json.loads((geometry_dir / "geometry_config.json").read_text())
    if feature_config["source_split"] != "offline_discovery":
        raise ValueError("Feature cost is not discovery-only")
    if geometry_config["manifest_offline_split"] != "offline_discovery":
        raise ValueError("Geometry is not discovery-only")
    if feature_config["offline_validation_loaded"] or feature_config["offline_test_loaded"]:
        raise ValueError("Feature-cost provenance indicates held-out use")
    if geometry_config["offline_validation_features_loaded"] or geometry_config[
        "offline_test_features_loaded"
    ]:
        raise ValueError("Geometry provenance indicates held-out use")
    if manifest["offline_splits"]["offline_discovery"]["cache_row_count"] != 5135:
        raise ValueError("Unexpected offline_discovery count")

    feature_path = feature_dir / "feature_cost_oof.npy"
    brain_path = geometry_dir / "brain_geometry.npy"
    clip_path = geometry_dir / "clip_geometry.npy"
    feature_cost = np.load(feature_path, allow_pickle=False)
    brain_geometry = np.load(brain_path, allow_pickle=False)
    clip_geometry = np.load(clip_path, allow_pickle=False)
    validate_problem(feature_cost, brain_geometry, clip_geometry)
    roi_names = list(geometry_config["roi_names"])
    layer_depths = list(geometry_config["clip_layers"])
    layer_labels = [f"L{layer}" for layer in layer_depths]
    if roi_names != feature_config["roi_order"]:
        raise ValueError("ROI order differs between M and frozen geometry")
    if layer_depths != feature_config["clip_layer_order"]:
        raise ValueError("CLIP layer order differs between M and frozen geometry")

    calibration = calibration_from_uniform_plan(
        feature_cost, brain_geometry, clip_geometry, args.eps_scale
    )
    calibration_payload = {
        "reference_plan": "T_ref[r,l] = (1/8)*(1/6)",
        "reference_plan_shape": [8, 6],
        "reference_plan_total_mass": 1.0,
        "eps_scale": args.eps_scale,
        "s_feature": calibration.feature_scale,
        "s_gw": calibration.gw_scale,
        "feature_cost_scale": {
            "min": float(feature_cost.min()),
            "max": float(feature_cost.max()),
            "mean": float(feature_cost.mean()),
            "std": float(feature_cost.std()),
        },
        "brain_geometry_scale": {
            "min": float(brain_geometry.min()),
            "max": float(brain_geometry.max()),
            "mean": float(brain_geometry.mean()),
            "std": float(brain_geometry.std()),
        },
        "clip_geometry_scale": {
            "min": float(clip_geometry.min()),
            "max": float(clip_geometry.max()),
            "mean": float(clip_geometry.mean()),
            "std": float(clip_geometry.std()),
        },
        "reuse_rule": "fixed canonical discovery scales; never recalibrate per run/null/subsample",
    }
    write_json(output_dir / "loss_scale_calibration.json", calibration_payload)
    calibration_hash = sha256_file(output_dir / "loss_scale_calibration.json")
    provenance = {
        "feature_cost_oof_sha256": sha256_file(feature_path),
        "brain_geometry_sha256": sha256_file(brain_path),
        "clip_geometry_sha256": sha256_file(clip_path),
        "feature_cost_config_sha256": sha256_file(feature_dir / "feature_cost_config.json"),
        "geometry_config_sha256": sha256_file(geometry_dir / "geometry_config.json"),
        "stimulus_manifest_sha256": sha256_file(manifest_path),
        "loss_scale_calibration_sha256": calibration_hash,
    }
    roi_variance = load_roi_variance(geometry_dir / "roi_diagnostics.csv", roi_names)

    candidates: List[Tuple[str, str, float, float, bool]] = [
        ("source_constrained_feature_routing", "source_constrained_feature_routing", 0.0, 0.0, False),
        (
            "source_constrained_feature_coverage_lambda_0p01",
            "source_constrained_feature_coverage",
            0.0,
            0.01,
            False,
        ),
        (
            "source_constrained_feature_coverage_lambda_0p1",
            "source_constrained_feature_coverage",
            0.0,
            0.1,
            False,
        ),
        ("balanced_outer_ot", "balanced_outer_ot", 0.0, 0.0, True),
        ("sr_gw", "sr_gw", 1.0, 0.0, False),
    ]
    for beta in (0.25, 0.5, 0.75):
        beta_name = str(beta).replace(".", "p")
        candidates.append((f"sr_fgw_beta_{beta_name}", "sr_fgw", beta, 0.0, False))
        for coverage in (0.01, 0.1):
            coverage_name = str(coverage).replace(".", "p")
            candidates.append(
                (
                    f"sr_fgw_coverage_beta_{beta_name}_lambda_{coverage_name}",
                    "sr_fgw_coverage",
                    beta,
                    coverage,
                    False,
                )
            )

    summaries = []
    for directory, method, beta, coverage, balanced in candidates:
        print(
            f"running {directory}: method={method} beta={beta} lambda_cov={coverage}",
            flush=True,
        )
        summary = save_candidate(
            output_dir / directory,
            method,
            beta,
            coverage,
            feature_cost,
            brain_geometry,
            clip_geometry,
            calibration,
            roi_names,
            layer_labels,
            layer_depths,
            roi_variance,
            provenance,
            args,
            balanced=balanced,
        )
        summaries.append(summary)

    write_json(
        output_dir / "candidate_index.json",
        {
            "stage": "offline_discovery_only",
            "scientific_best_method_declared": False,
            "best_transport_plan_created": False,
            "candidate_count": len(summaries),
            "candidates": [
                {
                    "directory": item["directory"],
                    "method": item["method"],
                    "structure_weight": item["structure_weight"],
                    "coverage_weight": item["coverage_weight"],
                    "total_objective": item["components"]["total_objective"],
                    "maximum_target_mass": item["diagnostics"]["target_utilization"][
                        "maximum_target_mass"
                    ],
                    "effective_target_layer_count": item["diagnostics"][
                        "target_utilization"
                    ]["effective_target_layer_count"],
                }
                for item in summaries
            ],
            "offline_validation_loaded": False,
            "offline_test_loaded": False,
            "elapsed_seconds": time.time() - started,
        },
    )
    write_json(
        output_dir / "pot_reference_validation.json",
        {
            "POT_AVAILABLE": False,
            "status": "NOT_RUN_DEPENDENCY_UNAVAILABLE",
            "reason": "Python package 'ot' is not installed in the brainx environment",
            "custom_coverage_solver_claimed_as_standard_pot": False,
            "fallback": "synthetic unit tests in tests/test_fgw_correspondence.py",
        },
    )
    write_json(
        output_dir / "mot_asset_audit.json",
        {
            "MOT_STYLE_BASELINE_STATUS": "NOT_CURRENTLY_REPRODUCIBLE",
            "reason": (
                "The current FGW cache contains fixed-dimensional ROI tokens, not "
                "per-voxel stimulus response functions. CLIP assets contain only "
                "mean-patch 1024-dimensional layer embeddings, not a declared inner "
                "unit/patch transport representation. Exact stimulus IDs are aligned, "
                "but both faithful inner domains are not preserved in the cache."
            ),
            "raw_nsd_voxels_exist_outside_cache": True,
            "proposed_future_plan_only": (
                "Build a new stimulus-aligned per-voxel ROI response cache and an "
                "explicit CLIP unit/patch response cache, define inner-domain costs "
                "and regularization, then validate inner and outer transports separately."
            ),
            "implemented_in_prompt3": False,
            "balanced_outer_ot_labeled_as_mot": False,
        },
    )
    print(f"Saved {len(summaries)} discovery candidates to {output_dir}", flush=True)
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
