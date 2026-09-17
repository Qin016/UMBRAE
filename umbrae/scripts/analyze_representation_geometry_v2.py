#!/usr/bin/env python
"""Corrected stimulus-subset stability analysis for cached ROI/CLIP features."""

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
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_representation_geometry import (
    cosine_distance_matrices,
    feature_diagnostics,
    load_voxel_counts,
    relation_matrix_from_rdms,
    save_heatmap,
    spearman_value,
    upper_triangle,
    write_json,
    write_matrix_csv,
)


PRIMARY_BRAIN_ALIAS = "roi_tokens_before_projector"
PRIMARY_BRAIN_FILE = "brain_roi_features.npy"
CLIP_FILE = "clip_layer_features.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected NeuroRoute representational-geometry stability study"
    )
    parser.add_argument("--stimulus-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-size", type=int, default=1500)
    parser.add_argument("--subsample-size", type=int, default=600)
    parser.add_argument("--subsample-count", type=int, default=100)
    parser.add_argument("--null-stimulus-size", type=int, default=500)
    parser.add_argument("--null-count", type=int, default=500)
    parser.add_argument("--single-roi-null-count", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def sha256_ids(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(path.read_text())
    required = {"offline_discovery", "offline_validation", "offline_test"}
    if set(manifest.get("offline_splits", {})) != required:
        raise ValueError("Manifest does not contain the three corrected offline splits")
    checks = manifest.get("overlap_checks", {})
    if any(checks.get("offline_unique_stimulus_overlaps", {}).values()):
        raise ValueError("Manifest offline splits overlap")
    if not checks.get("eligible_ids_covered_exactly_once", False):
        raise ValueError("Manifest does not cover eligible IDs exactly once")
    return manifest


def gather_cache_rows(
    cache_root: Path,
    row_table: Sequence[Mapping[str, Any]],
    global_indices: np.ndarray,
    feature_file: str,
) -> Tuple[np.ndarray, List[str]]:
    table = {int(row["global_cache_row"]): row for row in row_table}
    selected = [table[int(index)] for index in global_indices]
    arrays: Dict[str, np.ndarray] = {}
    source_splits: List[str] = []
    shape_tail = None
    dtype = None
    for row in selected:
        split = str(row["cache_split"])
        if split not in arrays:
            arrays[split] = np.load(cache_root / split / feature_file, mmap_mode="r")
            source_splits.append(split)
            shape_tail = arrays[split].shape[1:] if shape_tail is None else shape_tail
            dtype = arrays[split].dtype if dtype is None else dtype
            if arrays[split].shape[1:] != shape_tail or arrays[split].dtype != dtype:
                raise ValueError("Cache feature shape/dtype differs across source groups")
    if shape_tail is None or dtype is None:
        raise ValueError("No feature rows selected")
    output = np.empty((len(selected), *shape_tail), dtype=dtype)
    for output_row, row in enumerate(selected):
        output[output_row] = arrays[str(row["cache_split"])][int(row["cache_split_row"])]
    if not np.isfinite(output).all():
        raise ValueError(f"Non-finite values in gathered {feature_file}")
    return output, source_splits


def frobenius_metrics(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    distance = float(np.linalg.norm(left - right, ord="fro"))
    denominator = float(np.linalg.norm(left, ord="fro") + np.linalg.norm(right, ord="fro"))
    normalized_distance = distance / denominator if denominator > 0 else float("nan")
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    cosine_denominator = float(np.linalg.norm(flat_left) * np.linalg.norm(flat_right))
    cosine_similarity = (
        float(np.dot(flat_left, flat_right) / cosine_denominator)
        if cosine_denominator > 0
        else float("nan")
    )
    return {
        "frobenius_distance": distance,
        "normalized_frobenius_distance": normalized_distance,
        "frobenius_cosine_similarity": cosine_similarity,
        "one_minus_normalized_frobenius_distance": 1.0 - normalized_distance,
    }


def distribution_summary(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {key: float("nan") for key in ("mean", "std_ddof1", "median", "ci_2_5", "ci_97_5")}
    return {
        "mean": float(np.mean(finite)),
        "std_ddof1": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "median": float(np.median(finite)),
        "ci_2_5": float(np.percentile(finite, 2.5)),
        "ci_97_5": float(np.percentile(finite, 97.5)),
    }


def stimulus_subset_stability(
    rdms: np.ndarray,
    subset_size: int,
    count: int,
    rng: np.random.Generator,
    label: str,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    anchor_count = rdms.shape[1]
    if subset_size < 4 or 2 * subset_size > anchor_count:
        raise ValueError("subsample_size must be >=4 and at most anchor_size/2")
    matrices_a = np.empty((count, rdms.shape[0], rdms.shape[0]), dtype=np.float32)
    matrices_b = np.empty_like(matrices_a)
    metrics = np.empty((count, 5), dtype=np.float64)
    progress_every = max(1, count // 10)
    for replicate in range(count):
        order = rng.permutation(anchor_count)
        indices_a = np.sort(order[:subset_size])
        indices_b = np.sort(order[subset_size : 2 * subset_size])
        matrix_a = relation_matrix_from_rdms(rdms, indices_a)
        matrix_b = relation_matrix_from_rdms(rdms, indices_b)
        matrices_a[replicate] = matrix_a
        matrices_b[replicate] = matrix_b
        frobenius = frobenius_metrics(matrix_a, matrix_b)
        metrics[replicate] = [
            spearman_value(upper_triangle(matrix_a), upper_triangle(matrix_b)),
            frobenius["frobenius_distance"],
            frobenius["normalized_frobenius_distance"],
            frobenius["frobenius_cosine_similarity"],
            frobenius["one_minus_normalized_frobenius_distance"],
        ]
        if (replicate + 1) % progress_every == 0 or replicate + 1 == count:
            print(f"{label} no-replacement subset pairs {replicate + 1}/{count}", flush=True)
    summary = {
        "sampling": (
            "each replicate draws two disjoint unique-stimulus subsets without "
            "replacement from the fixed offline_discovery anchor pool"
        ),
        "replicate_count": count,
        "subset_size_each": subset_size,
        "upper_triangle_spearman": distribution_summary(metrics[:, 0]),
        "frobenius_distance": distribution_summary(metrics[:, 1]),
        "normalized_frobenius_distance": distribution_summary(metrics[:, 2]),
        "frobenius_cosine_similarity": distribution_summary(metrics[:, 3]),
        "one_minus_normalized_frobenius_distance": distribution_summary(metrics[:, 4]),
        "replicate_metrics": [
            {
                "replicate": index,
                "upper_triangle_spearman": row[0],
                "frobenius_distance": row[1],
                "normalized_frobenius_distance": row[2],
                "frobenius_cosine_similarity": row[3],
                "one_minus_normalized_frobenius_distance": row[4],
            }
            for index, row in enumerate(metrics)
        ],
    }
    return summary, matrices_a, matrices_b, metrics


def standardized_ranks(vectors: np.ndarray) -> np.ndarray:
    ranks = np.asarray(rankdata(vectors, axis=0, method="average"), dtype=np.float64)
    ranks -= ranks.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(ranks * ranks, axis=0, keepdims=True))
    if np.any(norms <= 0):
        raise ValueError("A null RDM vector is constant")
    return ranks / norms


def relation_from_standardized_ranks(ranks: np.ndarray) -> np.ndarray:
    correlation = np.clip(ranks.T @ ranks, -1.0, 1.0)
    relation = 1.0 - correlation
    relation = (relation + relation.T) * 0.5
    np.fill_diagonal(relation, 0.0)
    return relation


def shared_organization_strength(relation: np.ndarray) -> float:
    return float(np.mean(1.0 - upper_triangle(relation)))


def independently_permuted_roi_null(
    brain_rdms: np.ndarray,
    null_positions: np.ndarray,
    count: int,
    single_roi_count: int,
    rng: np.random.Generator,
    roi_names: Sequence[str],
) -> Tuple[Dict[str, Any], np.ndarray]:
    local_rdms = brain_rdms[:, null_positions][:, :, null_positions]
    node_count, stimulus_count, _ = local_rdms.shape
    pair_left, pair_right = np.triu_indices(stimulus_count, k=1)
    baseline_vectors = local_rdms[:, pair_left, pair_right].T
    baseline_ranks = standardized_ranks(baseline_vectors)
    real_relation = relation_from_standardized_ranks(baseline_ranks)
    observed = shared_organization_strength(real_relation)

    null_matrices = np.empty((count, node_count, node_count), dtype=np.float32)
    null_strength = np.empty(count, dtype=np.float64)
    progress_every = max(1, count // 10)
    for replicate in range(count):
        vectors = np.empty_like(baseline_vectors)
        for node in range(node_count):
            order = rng.permutation(stimulus_count)
            vectors[:, node] = local_rdms[node, order[pair_left], order[pair_right]]
        relation = relation_from_standardized_ranks(standardized_ranks(vectors))
        null_matrices[replicate] = relation
        null_strength[replicate] = shared_organization_strength(relation)
        if (replicate + 1) % progress_every == 0 or replicate + 1 == count:
            print(f"independent-ROI null {replicate + 1}/{count}", flush=True)

    single_roi: Dict[str, Any] = {}
    for node, roi_name in enumerate(roi_names):
        observed_row = float(np.mean(1.0 - real_relation[node, np.arange(node_count) != node]))
        values = np.empty(single_roi_count, dtype=np.float64)
        for replicate in range(single_roi_count):
            order = rng.permutation(stimulus_count)
            vector = local_rdms[node, order[pair_left], order[pair_right]][:, None]
            permuted_rank = standardized_ranks(vector)[:, 0]
            other = np.arange(node_count) != node
            values[replicate] = float(np.mean(permuted_rank @ baseline_ranks[:, other]))
        summary = distribution_summary(values)
        summary.update(
            {
                "observed_mean_rdm_spearman_to_other_rois": observed_row,
                "one_sided_p": float((1 + np.sum(values >= observed_row)) / (1 + len(values))),
                "null_count": single_roi_count,
            }
        )
        single_roi[roi_name] = summary

    null_summary = distribution_summary(null_strength)
    payload = {
        "primary_null": (
            "independently permute stimulus identities for every ROI; each ROI "
            "uses a separate permutation in every replicate"
        ),
        "stimulus_count": stimulus_count,
        "null_count": count,
        "test_statistic": "mean off-diagonal RDM Spearman correlation across ROI pairs",
        "observed_shared_organization_strength": observed,
        "null_shared_organization_strength": null_summary,
        "one_sided_p": float((1 + np.sum(null_strength >= observed)) / (1 + len(null_strength))),
        "observed_above_null_ci_97_5": observed > null_summary["ci_97_5"],
        "real_relation_on_null_subset": real_relation,
        "null_relation_mean": np.mean(null_matrices, axis=0),
        "null_relation_variance_ddof1": np.var(null_matrices, axis=0, ddof=1),
        "single_roi_permutation_diagnostic": single_roi,
    }
    return payload, null_matrices


def roi_stability_rows(
    roi_names: Sequence[str],
    voxel_counts: Mapping[str, int],
    diagnostics: Sequence[Mapping[str, float]],
    matrices_a: np.ndarray,
    matrices_b: np.ndarray,
    single_roi_null: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    rows: List[Dict[str, Any]] = []
    medians = np.empty(len(roi_names), dtype=np.float64)
    for node, roi_name in enumerate(roi_names):
        mask = np.arange(len(roi_names)) != node
        correlations = np.asarray(
            [
                spearman_value(left[node, mask], right[node, mask])
                for left, right in zip(matrices_a, matrices_b)
            ],
            dtype=np.float64,
        )
        summary = distribution_summary(correlations)
        medians[node] = summary["median"]
        null = single_roi_null[roi_name]
        rows.append(
            {
                "roi": roi_name,
                "voxel_count": voxel_counts.get(roi_name),
                **diagnostics[node],
                "stimulus_subset_profile_spearman_mean": summary["mean"],
                "stimulus_subset_profile_spearman_median": summary["median"],
                "stimulus_subset_profile_spearman_ci_2_5": summary["ci_2_5"],
                "stimulus_subset_profile_spearman_ci_97_5": summary["ci_97_5"],
                "single_roi_real_mean_rdm_spearman": null[
                    "observed_mean_rdm_spearman_to_other_rois"
                ],
                "single_roi_permutation_null_mean": null["mean"],
                "single_roi_permutation_p": null["one_sided_p"],
                "repeat_measurement_reliability": "NOT_AVAILABLE",
            }
        )
    return rows, medians


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation_with_p(left: Sequence[float], right: Sequence[float]) -> Dict[str, float]:
    result = spearmanr(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    return {"spearman": float(result.statistic), "two_sided_p": float(result.pvalue)}


def run(args: argparse.Namespace) -> Path:
    if args.subsample_count < 2 or args.null_count < 2 or args.single_roi_null_count < 2:
        raise ValueError("replicate counts must be at least 2")
    manifest_path = Path(args.stimulus_manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite corrected geometry output: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.time()

    manifest_dir = manifest_path.parent
    discovery_index_path = manifest_dir / manifest["offline_splits"]["offline_discovery"]["index_file"]
    discovery_indices = np.load(discovery_index_path, allow_pickle=False)
    if len(discovery_indices) != manifest["offline_splits"]["offline_discovery"]["cache_row_count"]:
        raise ValueError("offline_discovery index count differs from manifest")
    row_table = manifest["cache_row_table"]
    table_by_global = {int(row["global_cache_row"]): row for row in row_table}
    discovery_ids = [table_by_global[int(index)]["stable_stimulus_id"] for index in discovery_indices]
    if len(discovery_ids) != len(set(discovery_ids)):
        raise ValueError("offline_discovery contains duplicate stimulus IDs")
    cache_root = Path(manifest["representation_cache"])
    brain, brain_source_splits = gather_cache_rows(
        cache_root, row_table, discovery_indices, PRIMARY_BRAIN_FILE
    )
    clip, clip_source_splits = gather_cache_rows(
        cache_root, row_table, discovery_indices, CLIP_FILE
    )
    if brain.shape[0] != clip.shape[0] or brain.shape[1] != 8 or clip.shape[1] != 6:
        raise ValueError(f"Unexpected gathered shapes: {brain.shape}, {clip.shape}")

    root_cache_config = json.loads((cache_root / "cache_config.json").read_text())
    roi_names = list(root_cache_config["roi_names"])
    layer_labels = [f"L{layer}" for layer in root_cache_config["selected_clip_layers"]]
    streams = np.random.SeedSequence(args.random_seed).spawn(5)
    stream_names = (
        "anchor_selection",
        "brain_subsets",
        "clip_subsets",
        "null_subset_selection",
        "null_permutations",
    )
    anchor_rng, brain_subset_rng, clip_subset_rng, null_subset_rng, null_rng = [
        np.random.default_rng(stream) for stream in streams
    ]
    if args.anchor_size > len(discovery_indices):
        raise ValueError("anchor_size exceeds offline_discovery")
    anchor_positions = np.sort(
        anchor_rng.choice(len(discovery_indices), size=args.anchor_size, replace=False)
    )
    anchor_ids = [discovery_ids[index] for index in anchor_positions]
    write_json(
        output_dir / "anchor_ids.json",
        {
            "offline_discovery_positions": anchor_positions,
            "global_cache_rows": discovery_indices[anchor_positions],
            "stable_stimulus_ids": anchor_ids,
            "ordered_id_sha256": sha256_ids(anchor_ids),
        },
    )

    print("Computing corrected raw-ROI-token and CLIP anchor RDMs", flush=True)
    brain_rdms = cosine_distance_matrices(brain, anchor_positions, args.device)
    clip_rdms = cosine_distance_matrices(clip, anchor_positions, args.device)
    all_anchor = np.arange(args.anchor_size, dtype=np.int64)
    brain_geometry = relation_matrix_from_rdms(brain_rdms, all_anchor)
    clip_geometry = relation_matrix_from_rdms(clip_rdms, all_anchor)

    brain_stability, brain_a, brain_b, brain_metrics = stimulus_subset_stability(
        brain_rdms, args.subsample_size, args.subsample_count, brain_subset_rng, "brain"
    )
    clip_stability, clip_a, clip_b, clip_metrics = stimulus_subset_stability(
        clip_rdms, args.subsample_size, args.subsample_count, clip_subset_rng, "CLIP"
    )
    stability_payload = {
        "status": "STIMULUS_SUBSET_STABILITY",
        "not_neural_measurement_reliability": True,
        "source_split": "offline_discovery",
        "anchor_size": args.anchor_size,
        "brain": brain_stability,
        "clip": clip_stability,
        "full_replicate_matrices": "stimulus_subset_geometries.npz",
    }
    write_json(output_dir / "stimulus_subset_stability.json", stability_payload)
    np.savez_compressed(
        output_dir / "stimulus_subset_geometries.npz",
        brain_a=brain_a,
        brain_b=brain_b,
        clip_a=clip_a,
        clip_b=clip_b,
        brain_metrics=brain_metrics,
        clip_metrics=clip_metrics,
    )

    null_positions = np.sort(
        null_subset_rng.choice(args.anchor_size, size=args.null_stimulus_size, replace=False)
    )
    null_payload, null_matrices = independently_permuted_roi_null(
        brain_rdms,
        null_positions,
        args.null_count,
        args.single_roi_null_count,
        null_rng,
        roi_names,
    )
    null_payload["random_seed"] = args.random_seed
    null_payload["random_stream"] = "SeedSequence child 4"
    null_payload["null_anchor_positions"] = null_positions
    write_json(output_dir / "geometry_null_summary.json", null_payload)
    np.savez_compressed(output_dir / "geometry_null_matrices.npz", brain=null_matrices)

    repeat_payload = {
        "REPEAT_RELIABILITY_STATUS": "NOT_AVAILABLE",
        "computed": False,
        "reason": (
            "The existing feature cache stores one ROI-token row after averaging all "
            "valid fMRI repeats for each stimulus. It does not contain independent "
            "repeat-resolved ROI tokens or session/run partitions, so same-stimulus "
            "measurement sets A and B cannot be constructed from this cache."
        ),
        "disjoint_stimulus_subsets_used_as_repeat_proxy": False,
        "crossnobis_computed": False,
        "interpretation_limit": (
            "Stimulus-subset stability does not establish neural measurement reliability."
        ),
    }
    write_json(output_dir / "repeat_based_measurement_reliability.json", repeat_payload)

    voxel_counts, voxel_error = load_voxel_counts(root_cache_config)
    diagnostics = feature_diagnostics(brain)
    roi_rows, roi_medians = roi_stability_rows(
        roi_names,
        voxel_counts,
        diagnostics,
        brain_a,
        brain_b,
        null_payload["single_roi_permutation_diagnostic"],
    )
    write_rows_csv(output_dir / "roi_diagnostics.csv", roi_rows)
    voxel_vector = [voxel_counts[name] for name in roi_names]
    variance_vector = [item["mean_feature_variance"] for item in diagnostics]
    confound_audit = {
        "voxel_count_vs_roi_subset_stability": correlation_with_p(voxel_vector, roi_medians),
        "feature_variance_vs_roi_subset_stability": correlation_with_p(variance_vector, roi_medians),
        "snr_proxy_available": False,
        "repeat_reliability_available": False,
        "regressed_out": False,
        "caution": "n=8 ROIs; association tests are descriptive and low-powered",
    }

    np.save(output_dir / "brain_geometry.npy", brain_geometry)
    np.save(output_dir / "clip_geometry.npy", clip_geometry)
    write_matrix_csv(output_dir / "brain_geometry.csv", brain_geometry, roi_names)
    write_matrix_csv(output_dir / "clip_geometry.csv", clip_geometry, layer_labels)
    brain_heatmap = save_heatmap(
        output_dir / "brain_geometry_heatmap.png",
        brain_geometry,
        roi_names,
        "ROI representational geometry relation matrix (raw ROI tokens)\n(not anatomical distance)",
    )
    clip_heatmap = save_heatmap(
        output_dir / "clip_geometry_heatmap.png",
        clip_geometry,
        layer_labels,
        "CLIP layer representational geometry relation matrix",
    )

    brain_spearman = brain_stability["upper_triangle_spearman"]
    stability_pass = brain_spearman["median"] >= 0.5 and brain_spearman["ci_2_5"] > 0
    null_pass = null_payload["one_sided_p"] <= 0.05 and null_payload["observed_above_null_ci_97_5"]
    pathological_count = int(np.sum(roi_medians <= 0))
    pathology_pass = pathological_count <= 1 and all(
        item["near_zero_variance_fraction"] < 0.5 for item in diagnostics
    )
    if stability_pass and null_pass and pathology_pass:
        geometry_status = "GO"
    elif stability_pass or null_pass:
        geometry_status = "PROCEED_WITH_CAUTION"
    else:
        geometry_status = "NO_GO"

    config = {
        "analysis_version": "geometry_v2_corrected",
        "GEOMETRY_STATUS": geometry_status,
        "gate": {
            "stimulus_subset_stability_pass": stability_pass,
            "independent_roi_permutation_null_pass": null_pass,
            "pathological_roi_check_pass": pathology_pass,
            "roi_profile_median_at_or_below_zero_count": pathological_count,
            "repeat_reliability_status": "NOT_AVAILABLE",
        },
        "subject": manifest["subject"],
        "stimulus_manifest": str(manifest_path),
        "manifest_offline_split": "offline_discovery",
        "offline_discovery_count": len(discovery_indices),
        "offline_validation_features_loaded": False,
        "offline_test_features_loaded": False,
        "feature_cache_source_splits_read": sorted(set(brain_source_splits + clip_source_splits)),
        "primary_brain_representation": PRIMARY_BRAIN_ALIAS,
        "primary_brain_storage_file": PRIMARY_BRAIN_FILE,
        "primary_brain_is_raw_voxel_vector": False,
        "primary_brain_feature_shape": list(brain.shape),
        "clip_feature_shape": list(clip.shape),
        "roi_names": roi_names,
        "clip_layers": root_cache_config["selected_clip_layers"],
        "clip_pooling_method": root_cache_config["clip_pooling_method"],
        "geometry_definition": {
            "stimulus_distance": "1 - cosine similarity",
            "node_relation": "1 - Spearman correlation of RDM strict upper triangles",
            "brain_semantics": "ROI representational geometry relation matrix",
            "anatomical_distance": False,
            "hierarchy_claimed": False,
        },
        "anchor_size": args.anchor_size,
        "anchor_id_sha256": sha256_ids(anchor_ids),
        "subsample_size_each": args.subsample_size,
        "subsample_count": args.subsample_count,
        "subsampling_with_replacement": False,
        "null_stimulus_size": args.null_stimulus_size,
        "null_count": args.null_count,
        "single_roi_null_count": args.single_roi_null_count,
        "random_seed": args.random_seed,
        "random_stream_spawn_keys": {
            name: list(stream.spawn_key) for name, stream in zip(stream_names, streams)
        },
        "random_streams": (
            "NumPy SeedSequence(seed) independent children: anchor, brain subsets, "
            "CLIP subsets, null subset, null permutations"
        ),
        "device": args.device,
        "full_stimulus_rdm_saved": False,
        "legacy_prompt2_output_modified": False,
        "legacy_bootstrap_with_replacement_primary": False,
        "repeat_based_measurement_reliability": repeat_payload,
        "roi_confound_audit": confound_audit,
        "roi_mapping_error": voxel_error,
        "heatmaps_saved": brain_heatmap and clip_heatmap,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "geometry_config.json", config)
    print(f"Saved corrected geometry analysis to {output_dir}", flush=True)
    print(f"GEOMETRY_STATUS = {geometry_status}", flush=True)
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
