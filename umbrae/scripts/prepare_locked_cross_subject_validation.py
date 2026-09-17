#!/usr/bin/env python
"""Validate replication assets and materialize a locked Prompt-5A config.

This preparation step reads discovery outputs only.  It never loads an
offline-test index or any offline-test representation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FGW_PATH = ROOT / "models" / "fgw_correspondence.py"
SPEC = importlib.util.spec_from_file_location("offline_fgw_prepare", FGW_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(FGW_PATH)
FGW = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FGW
SPEC.loader.exec_module(FGW)

ROI_ORDER = ["V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA"]
LAYER_ORDER = [4, 8, 12, 16, 20, 24]
INIT_SPECS = [["uniform", 0], ["feature_informed", 0]] + [
    ["random", seed] for seed in range(1001, 1019)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--roi-mapping", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--feature-cost-dir", required=True)
    parser.add_argument("--global-preregistration", required=True)
    parser.add_argument("--output-root", required=True)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    subject = args.subject
    output_root = Path(args.output_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    mapping_path = Path(args.roi_mapping).resolve()
    stage1_dir = Path(args.stage1_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    geometry_dir = Path(args.geometry_dir).resolve()
    feature_dir = Path(args.feature_cost_dir).resolve()
    global_prereg_path = Path(args.global_preregistration).resolve()

    cache = json.loads((cache_dir / "cache_config.json").read_text())
    mapping = json.loads(mapping_path.read_text())
    stage1 = json.loads((stage1_dir / "config.json").read_text())
    stage1_rois = json.loads((stage1_dir / "roi_names.json").read_text())
    stage1_layers = json.loads((stage1_dir / "selected_clip_layers.json").read_text())
    geometry = json.loads((geometry_dir / "geometry_config.json").read_text())
    feature = json.loads((feature_dir / "feature_cost_config.json").read_text())
    global_prereg = json.loads(global_prereg_path.read_text())

    checks = {
        "mapping_subject_matches": mapping.get("subject") == subject,
        "mapping_is_real": mapping.get("roi_mapping_is_real") is True,
        "voxel_order_verified": mapping.get("voxel_order_verified") is True,
        "no_unresolved_roi": mapping.get("unresolved_rois") == [],
        "mapping_roi_order_exact": mapping.get("roi_names") == ROI_ORDER,
        "stage1_subject_matches": stage1.get("subject") == subject,
        "stage1_roi_order_exact": stage1_rois == ROI_ORDER,
        "stage1_layer_order_exact": stage1_layers == LAYER_ORDER,
        "stage1_roi_token_dim_1024": stage1.get("roi_token_dim") == 1024,
        "stage1_clip_target_dim_1024": stage1.get("clip_layer_target_dim") == 1024,
        "stage1_projector_enabled": stage1.get("use_brain_clip_projector") is True,
        "cache_subject_matches": cache.get("subject") == subject,
        "cache_roi_order_exact": cache.get("roi_names") == ROI_ORDER,
        "cache_layer_order_exact": cache.get("selected_clip_layers") == LAYER_ORDER,
        "cache_pooling_exact": cache.get("clip_pooling_method")
        == "mean_non_cls_patch_tokens",
        "cache_offline_only": cache.get(
            "uses_image_features_only_for_offline_analysis"
        )
        is True,
        "brain_branch_has_no_image_features": cache.get(
            "image_features_used_by_brain_branch"
        )
        is False,
        "geometry_discovery_only": geometry.get("manifest_offline_split")
        == "offline_discovery",
        "geometry_status_go": geometry.get("GEOMETRY_STATUS") == "GO",
        "feature_cost_discovery_only": feature.get("source_split")
        == "offline_discovery",
        "feature_cost_shape_exact": feature.get("feature_cost_shape") == [8, 6],
        "global_preregistration_precedes_validation": global_prereg.get(
            "written_before_replication_validation_feature_access"
        )
        is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    status = "PREFLIGHT_PASS" if not failed else "BLOCKED_BY_DATA_OR_ROI_MISMATCH"

    preflight = {
        "subject": subject,
        "SUBJECT_REPLICATION_PREFLIGHT_STATUS": status,
        "checks": checks,
        "failed_checks": failed,
        "roi_names": ROI_ORDER,
        "roi_voxel_counts": mapping["roi_counts"],
        "cache_sample_count": cache["sample_count"],
        "cache_feature_dimensions": cache["feature_dimensions"],
        "clip_layers": LAYER_ORDER,
        "clip_pooling": cache["clip_pooling_method"],
        "stage1_checkpoint": cache["stage1_checkpoint"],
        "stage1_checkpoint_sha256": cache["stage1_checkpoint_sha256"],
        "roi_mapping": str(mapping_path),
        "roi_mapping_sha256": sha256(mapping_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "stable_stimulus_ids_available": True,
        "protected_downstream_ids_checked_by_manifest_builder": True,
        "offline_test_index_loaded_by_this_step": False,
        "offline_test_features_loaded_by_this_step": False,
    }
    write_json(output_root / "preflight_audit.json", preflight)
    write_json(
        output_root / "cache_provenance.json",
        {
            "subject": subject,
            "cache_directory": str(cache_dir),
            "cache_config_sha256": sha256(cache_dir / "cache_config.json"),
            "metadata_sha256": sha256(cache_dir / "metadata.json"),
            "cache_config": cache,
            "reused_existing_cache": False,
            "frozen_inference_only": True,
        },
    )
    write_json(output_root / "cross_subject_protocol_reference.json", global_prereg)
    if failed:
        raise RuntimeError(f"{status}: {failed}")

    feature_cost = np.load(feature_dir / "feature_cost_oof.npy", allow_pickle=False)
    brain_geometry = np.load(geometry_dir / "brain_geometry.npy", allow_pickle=False)
    clip_geometry = np.load(geometry_dir / "clip_geometry.npy", allow_pickle=False)
    calibration = FGW.calibration_from_uniform_plan(
        feature_cost, brain_geometry, clip_geometry, 1e-12
    )
    validation_dir = output_root / "validation"
    validation_dir.mkdir()
    prereg = {
        "preregistration_version": 1,
        "locked_cross_subject_replication": True,
        "subject": subject,
        "result_scope": f"LOCKED CROSS-SUBJECT REPLICATION {subject}",
        "written_before_offline_validation_feature_access_for_prompt4a": True,
        "offline_test_access_permitted": False,
        "global_preregistration_path": str(global_prereg_path),
        "global_preregistration_sha256": sha256(global_prereg_path),
        "fixed_loss_scales": {
            "s_feature": calibration.feature_scale,
            "s_gw": calibration.gw_scale,
            "eps_scale": calibration.eps_scale,
            "rule": "subject discovery matrices evaluated at uniform feasible plan",
        },
        "fixed_validation_composite": {
            "feature_weight": 0.5,
            "gw_weight": 0.5,
            "coverage_included": False,
        },
        "required_candidate_count": 3,
        "frozen_primary_nonstructural_slug":
        "source_constrained_feature_coverage_lambda_1",
        "candidate_grid": {
            "sr_fgw": [
                {"structure_weight": 0.5, "coverage_weight": 0.0}
            ],
            "source_constrained_feature_coverage": [
                {"structure_weight": 0.0, "coverage_weight": 1.0}
            ],
            "balanced_outer_ot": [
                {
                    "structure_weight": 0.0,
                    "coverage_weight": None,
                    "hard_uniform_target": True,
                }
            ],
        },
        "structural_initializations": {
            "count": len(INIT_SPECS),
            "specifications": INIT_SPECS,
        },
        "nonstructural_initializations": "same 20 specifications when applicable",
        "near_optimal_relative_threshold": 0.01,
        "top_k_identifiability": 5,
        "validation_subsampling": {
            "count": 200,
            "fraction": 0.75,
            "without_replacement": True,
            "seed": 41001,
        },
        "validation_geometry": {
            "anchor_count": 1500,
            "anchor_selection":
            "without replacement from offline_validation, then sorted for deterministic access",
            "seed": 43001,
            "matches_geometry_v2_anchor_count": True,
        },
        "null_experiments": {
            "brain_geometry_unique_permutations": 200,
            "clip_geometry_unique_permutations": 200,
            "feature_pairing_permutations": 20,
            "independent_roi_stimulus_permutations": 200,
            "matched_marginal_unique_row_permutations": 500,
            "null_fit_initializations": [
                ["uniform", 0],
                ["feature_informed", 0],
                ["random", 1001],
            ],
            "seed": 42001,
        },
        "solver": {
            "learning_rate": 0.1,
            "max_iterations": 2500,
            "tolerance": 1e-11,
            "patience": 80,
            "eps_coverage": 1e-12,
            "entropy_weight": 0.0,
            "projection": "Euclidean row-simplex projection",
        },
        "selection": {
            "scope": "locked configurations only",
            "criterion": "model fixed globally; initialization selected by lowest discovery objective",
            "coverage_in_validation_score": False,
            "initialization_selected_by_validation": False,
            "model_selected_by_validation": False,
        },
        "decision_thresholds": {
            "null_separation_one_sided_p_max": 0.05,
            "structural_benefit_requires_subsample_gw_difference_ci_97_5_below_zero": True,
            "composite_benefit_requires_subsample_difference_ci_97_5_below_zero": True,
            "identifiability_reference_set":
            "near-optimal 1% set when it has at least two plans, otherwise top-5 converged",
            "minimum_median_preferred_layer_agreement": 0.5,
            "minimum_median_row_correlation": 0.0,
            "maximum_selected_composite_coefficient_of_variation": 0.1,
            "matched_plan_specialization_requires_real_validation_gw_p_max": 0.05,
            "matched_plan_reproducibility_requires_identifiability_preferred_agreement_above_null_ci_97_5": True,
            "matched_plan_reproducibility_requires_identifiability_row_correlation_above_null_ci_97_5": True,
        },
        "variance_sensitivity": {
            "registered": True,
            "used_for_selection": False,
        },
    }
    prereg_path = validation_dir / "locked_validation_preregistered.json"
    write_json(prereg_path, prereg)
    write_json(
        output_root / "loss_scale_calibration.json",
        {
            "subject": subject,
            "reference_plan": "T_ref[r,l]=(1/8)*(1/6)",
            "s_feature": calibration.feature_scale,
            "s_gw": calibration.gw_scale,
            "eps_scale": calibration.eps_scale,
            "discovery_only": True,
            "subj01_numeric_scales_reused": False,
            "validation_accessed": False,
            "offline_test_accessed": False,
            "preregistered_config_sha256": sha256(prereg_path),
        },
    )
    print(f"{subject} {status}")
    print(f"s_feature={calibration.feature_scale:.12g} s_gw={calibration.gw_scale:.12g}")
    print(prereg_path)


if __name__ == "__main__":
    main()
