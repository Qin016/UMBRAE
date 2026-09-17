#!/usr/bin/env python
"""Estimate offline ROI/CLIP representational-geometry reliability.

The analysis consumes only a representation-cache split.  It does not import or
load UMBRAE, Shikra, CLIP, or Stage-2 components, and it never writes a full
stimulus-by-stimulus RDM to disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


GEOMETRY_TYPE = "cosine_rsa_spearman"
SUPPORTED_BRAIN_FEATURES = (
    "projected_roi_features",
    "brain_roi_features",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure reliability of cached ROI and CLIP representational "
            "geometry before fitting FGW"
        )
    )
    parser.add_argument("--representation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-split", default="geometry_fit")
    parser.add_argument(
        "--geometry-type", choices=[GEOMETRY_TYPE], default=GEOMETRY_TYPE
    )
    parser.add_argument(
        "--brain-feature",
        choices=SUPPORTED_BRAIN_FEATURES,
        default="projected_roi_features",
    )
    parser.add_argument("--bootstrap-count", type=int, default=100)
    parser.add_argument("--stimulus-subset-size", type=int, default=1500)
    parser.add_argument("--null-count", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_cache_split(cache_arg: str, split_name: str) -> Path:
    candidate = Path(cache_arg).expanduser().resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Representation cache does not exist: {candidate}")
    direct_config = candidate / "cache_config.json"
    if direct_config.is_file():
        direct = json.loads(direct_config.read_text())
        if direct.get("analysis_split") == split_name:
            return candidate
    split_dir = candidate / split_name
    if (split_dir / "cache_config.json").is_file():
        return split_dir
    raise FileNotFoundError(
        f"Could not resolve cache split {split_name!r} below {candidate}"
    )


def load_cache(
    cache_dir: Path, brain_feature_name: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    config = json.loads((cache_dir / "cache_config.json").read_text())
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    if not config.get("offline_analysis_only", False):
        raise ValueError("Cache is not marked offline_analysis_only")
    if not config.get("uses_image_features_only_for_offline_analysis", False):
        raise ValueError("Cache lacks the offline image-feature safety marker")
    brain_path = cache_dir / f"{brain_feature_name}.npy"
    clip_path = cache_dir / "clip_layer_features.npy"
    brain = np.load(brain_path, mmap_mode="r")
    clip = np.load(clip_path, mmap_mode="r")
    n = int(config["sample_count"])
    expected_brain = (n, len(config["roi_names"]), 1024)
    expected_clip = (n, len(config["selected_clip_layers"]), 1024)
    expected_dims = config.get("feature_dimensions", {})
    if brain.shape != tuple(expected_dims.get(brain_feature_name, expected_brain)):
        raise ValueError(
            f"Unexpected {brain_feature_name} shape: {brain.shape}; "
            f"cache says {expected_dims.get(brain_feature_name)}"
        )
    if clip.shape != tuple(expected_dims.get("clip_layer_features", expected_clip)):
        raise ValueError(
            f"Unexpected clip_layer_features shape: {clip.shape}; "
            f"cache says {expected_dims.get('clip_layer_features')}"
        )
    if brain.shape[0] != clip.shape[0] or brain.shape[0] != len(metadata["samples"]):
        raise ValueError("Brain, CLIP, and metadata sample counts do not match")
    if brain.shape[1] != 8 or clip.shape[1] != 6:
        raise ValueError(
            f"This study requires 8 ROIs and 6 CLIP layers, got {brain.shape} / {clip.shape}"
        )
    return brain, clip, config, metadata


def select_anchor_indices(
    sample_count: int, subset_size: int, rng: np.random.Generator
) -> np.ndarray:
    if subset_size < 4:
        raise ValueError("stimulus_subset_size must be at least 4")
    if subset_size > sample_count:
        raise ValueError(
            f"stimulus_subset_size={subset_size} exceeds cache N={sample_count}"
        )
    # Sorting keeps tensor access and the persisted anchor ordering deterministic.
    return np.sort(rng.choice(sample_count, size=subset_size, replace=False))


def cosine_distance_matrices(
    features: np.ndarray,
    anchor_indices: np.ndarray,
    device: str,
) -> np.ndarray:
    """Return temporary [nodes, anchors, anchors] cosine RDMs in RAM."""
    selected = np.asarray(features[anchor_indices], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Selected features contain NaN or Inf")
    node_count = selected.shape[1]
    anchor_count = selected.shape[0]
    output = np.empty((node_count, anchor_count, anchor_count), dtype=np.float32)
    for node in range(node_count):
        tensor = torch.from_numpy(np.array(selected[:, node, :], copy=True)).to(device)
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=-1, eps=1e-12)
        similarity = tensor @ tensor.T
        distance = (1.0 - similarity).clamp_(0.0, 2.0)
        distance.fill_diagonal_(0.0)
        output[node] = distance.cpu().numpy()
        del tensor, similarity, distance
    return output


def _rank_correlation_matrix(vectors: np.ndarray) -> np.ndarray:
    """Spearman correlation between columns, including average-rank ties."""
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError(f"Expected [observations,nodes], got {vectors.shape}")
    ranks = np.asarray(rankdata(vectors, axis=0, method="average"), dtype=np.float64)
    ranks -= ranks.mean(axis=0, keepdims=True)
    scales = np.sqrt(np.sum(ranks * ranks, axis=0, keepdims=True))
    if np.any(scales <= 0):
        bad = np.flatnonzero(scales.reshape(-1) <= 0).tolist()
        raise ValueError(f"Constant RDM vector for node(s): {bad}")
    ranks /= scales
    correlations = np.clip(ranks.T @ ranks, -1.0, 1.0)
    np.fill_diagonal(correlations, 1.0)
    return correlations


def relation_matrix_from_rdms(
    rdms: np.ndarray, stimulus_indices: np.ndarray
) -> np.ndarray:
    """Compute 1-Spearman between upper-triangle RDM vectors."""
    stimulus_indices = np.asarray(stimulus_indices, dtype=np.int64)
    pair_left, pair_right = np.triu_indices(len(stimulus_indices), k=1)
    left = stimulus_indices[pair_left]
    right = stimulus_indices[pair_right]
    vectors = np.asarray(rdms[:, left, right].T, dtype=np.float32)
    relation = 1.0 - _rank_correlation_matrix(vectors)
    relation = (relation + relation.T) * 0.5
    np.fill_diagonal(relation, 0.0)
    return relation


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[np.triu_indices(matrix.shape[0], k=1)])


def spearman_value(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(np.asarray(left), np.asarray(right))
    return float(result.statistic)


def permutation_null(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    observed = spearman_value(upper_triangle(matrix_a), upper_triangle(matrix_b))
    values = np.empty(count, dtype=np.float64)
    for index in range(count):
        order = rng.permutation(matrix_b.shape[0])
        permuted = matrix_b[np.ix_(order, order)]
        values[index] = spearman_value(
            upper_triangle(matrix_a), upper_triangle(permuted)
        )
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("All node-label permutation null values are non-finite")
    null_mean = float(np.mean(finite))
    null_std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return {
        "observed_spearman": observed,
        "null_method": "permute node labels of split B, preserving matrix structure",
        "null_count": int(count),
        "null_mean": null_mean,
        "null_std": null_std,
        "null_ci_2_5": float(np.percentile(finite, 2.5)),
        "null_ci_97_5": float(np.percentile(finite, 97.5)),
        "one_sided_p": float((1 + np.sum(finite >= observed)) / (1 + len(finite))),
        "z_vs_null": (
            float((observed - null_mean) / null_std) if null_std > 0 else None
        ),
        "null_values": finite,
    }


def bootstrap_relations(
    rdms: np.ndarray,
    reference: np.ndarray,
    count: int,
    rng: np.random.Generator,
    progress_label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n = rdms.shape[1]
    matrices = np.empty((count, rdms.shape[0], rdms.shape[0]), dtype=np.float32)
    correlations = np.empty(count, dtype=np.float64)
    reference_upper = upper_triangle(reference)
    progress_every = max(1, count // 10)
    for index in range(count):
        sampled = rng.integers(0, n, size=n, dtype=np.int64)
        matrix = relation_matrix_from_rdms(rdms, sampled)
        matrices[index] = matrix
        correlations[index] = spearman_value(reference_upper, upper_triangle(matrix))
        if (index + 1) % progress_every == 0 or index + 1 == count:
            print(f"{progress_label} bootstrap {index + 1}/{count}", flush=True)
    return matrices, correlations


def matrix_summary(
    matrices: np.ndarray, correlations: np.ndarray
) -> Dict[str, Any]:
    return {
        "mean": np.mean(matrices, axis=0),
        "variance_ddof1": (
            np.var(matrices, axis=0, ddof=1)
            if len(matrices) > 1
            else np.zeros_like(matrices[0])
        ),
        "ci_2_5": np.percentile(matrices, 2.5, axis=0),
        "ci_97_5": np.percentile(matrices, 97.5, axis=0),
        "spearman_to_full_geometry": {
            "mean": float(np.nanmean(correlations)),
            "std_ddof1": (
                float(np.nanstd(correlations, ddof=1)) if len(correlations) > 1 else 0.0
            ),
            "median": float(np.nanmedian(correlations)),
            "ci_2_5": float(np.nanpercentile(correlations, 2.5)),
            "ci_97_5": float(np.nanpercentile(correlations, 97.5)),
        },
    }


def load_voxel_counts(config: Mapping[str, Any]) -> Tuple[Dict[str, int], str | None]:
    roi_path_raw = config.get("roi_indices_path")
    if not roi_path_raw:
        return {}, "cache_config has no roi_indices_path"
    roi_path = Path(str(roi_path_raw))
    if not roi_path.is_file():
        return {}, f"ROI mapping not found: {roi_path}"
    payload = json.loads(roi_path.read_text())
    counts: Dict[str, int] = {}
    for name in config["roi_names"]:
        record = payload.get("rois", {}).get(name, {})
        if "num_voxels" in record:
            counts[name] = int(record["num_voxels"])
        elif "indices" in record:
            counts[name] = len(record["indices"])
    return counts, None


def feature_diagnostics(features: np.ndarray) -> List[Dict[str, float]]:
    diagnostics: List[Dict[str, float]] = []
    for node in range(features.shape[1]):
        values = np.asarray(features[:, node, :], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite feature values for node {node}")
        feature_variances = np.var(values, axis=0, ddof=0)
        norms = np.linalg.norm(values, axis=1)
        diagnostics.append(
            {
                "mean_feature_variance": float(np.mean(feature_variances)),
                "median_feature_variance": float(np.median(feature_variances)),
                "near_zero_variance_fraction": float(
                    np.mean(feature_variances < 1e-12)
                ),
                "representation_norm_mean": float(np.mean(norms)),
                "representation_norm_std": float(np.std(norms, ddof=1)),
                "representation_norm_median": float(np.median(norms)),
                "representation_norm_min": float(np.min(norms)),
                "representation_norm_max": float(np.max(norms)),
            }
        )
    return diagnostics


def roi_rows(
    roi_names: Sequence[str],
    voxel_counts: Mapping[str, int],
    diagnostics: Sequence[Mapping[str, float]],
    split_a: np.ndarray,
    split_b: np.ndarray,
    reference: np.ndarray,
    bootstraps: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, name in enumerate(roi_names):
        mask = np.arange(len(roi_names)) != index
        profile_split = spearman_value(split_a[index, mask], split_b[index, mask])
        profile_bootstrap = np.asarray(
            [
                spearman_value(reference[index, mask], matrix[index, mask])
                for matrix in bootstraps
            ],
            dtype=np.float64,
        )
        relation_std = np.std(bootstraps[:, index, :][:, mask], axis=0, ddof=1)
        row: Dict[str, Any] = {
            "roi": name,
            "voxel_count": voxel_counts.get(name),
            **diagnostics[index],
            "split_half_profile_spearman": profile_split,
            "bootstrap_profile_spearman_mean": float(np.nanmean(profile_bootstrap)),
            "bootstrap_profile_spearman_ci_2_5": float(
                np.nanpercentile(profile_bootstrap, 2.5)
            ),
            "bootstrap_profile_spearman_ci_97_5": float(
                np.nanpercentile(profile_bootstrap, 97.5)
            ),
            "mean_bootstrap_relation_std": float(np.mean(relation_std)),
        }
        rows.append(row)
    return rows


def write_matrix_csv(path: Path, matrix: np.ndarray, labels: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(value):.9g}" for value in row]])


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty diagnostic table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(
    path: Path, matrix: np.ndarray, labels: Sequence[str], title: str
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=2.0)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title(title)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] > 1.0 else "black",
            )
    figure.colorbar(image, ax=axis, label="1 - Spearman correlation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def preliminary_assessment(
    brain_split: Mapping[str, Any], brain_bootstrap: Mapping[str, Any]
) -> Dict[str, Any]:
    observed = float(brain_split["observed_spearman"])
    p_value = float(brain_split["one_sided_p"])
    null_high = float(brain_split["null_ci_97_5"])
    boot = brain_bootstrap["spearman_to_full_geometry"]
    boot_low = float(boot["ci_2_5"])
    passes_null = observed > null_high and p_value <= 0.05
    reproducible = boot_low > 0.0 and float(boot["median"]) >= 0.5
    status = "GO" if passes_null and reproducible else "CAUTION"
    return {
        "status": status,
        "rule": (
            "GO requires split-half Spearman above the 97.5th percentile of "
            "the node-label permutation null with one-sided p<=0.05, plus "
            "bootstrap median Spearman-to-full >=0.5 and its 95% CI lower bound >0"
        ),
        "passes_split_half_null": passes_null,
        "passes_bootstrap_reproducibility": reproducible,
        "note": (
            "This is a preliminary representation-reliability gate, not evidence "
            "of anatomical distance or cortical hierarchy."
        ),
    }


def run(args: argparse.Namespace) -> Path:
    if args.bootstrap_count < 2:
        raise ValueError("bootstrap_count must be at least 2")
    if args.null_count < 1:
        raise ValueError("null_count must be positive")
    if args.geometry_type != GEOMETRY_TYPE:
        raise ValueError(f"Unsupported geometry type: {args.geometry_type}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing geometry output: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    started = time.time()

    cache_dir = resolve_cache_split(args.representation_cache, args.cache_split)
    brain, clip, cache_config, metadata = load_cache(cache_dir, args.brain_feature)
    if cache_config.get("analysis_split") != "geometry_fit":
        raise ValueError(
            "Reliability discovery must use analysis_split='geometry_fit'; "
            f"got {cache_config.get('analysis_split')!r}"
        )
    roi_names = list(cache_config["roi_names"])
    layer_labels = [f"L{layer}" for layer in cache_config["selected_clip_layers"]]

    seed_sequence = np.random.SeedSequence(args.random_seed)
    anchor_rng, split_rng, brain_boot_rng, clip_boot_rng, brain_null_rng, clip_null_rng = [
        np.random.default_rng(child) for child in seed_sequence.spawn(6)
    ]
    anchors = select_anchor_indices(len(brain), args.stimulus_subset_size, anchor_rng)
    anchor_samples = [metadata["samples"][int(index)] for index in anchors]
    anchor_ids = [sample["stable_stimulus_id"] for sample in anchor_samples]
    if len(anchor_ids) != len(set(anchor_ids)):
        raise ValueError("Anchor subset contains duplicate stable stimulus IDs")
    write_json(
        output_dir / "anchor_ids.json",
        {
            "cache_row_indices": anchors,
            "stable_stimulus_ids": anchor_ids,
            "stable_stimulus_id_sha256": sha256_lines(anchor_ids),
        },
    )

    print("Computing temporary anchor cosine RDMs (not saved to disk)", flush=True)
    brain_rdms = cosine_distance_matrices(brain, anchors, args.device)
    clip_rdms = cosine_distance_matrices(clip, anchors, args.device)
    local_all = np.arange(len(anchors), dtype=np.int64)
    brain_geometry = relation_matrix_from_rdms(brain_rdms, local_all)
    clip_geometry = relation_matrix_from_rdms(clip_rdms, local_all)

    split_order = split_rng.permutation(len(anchors))
    midpoint = len(anchors) // 2
    split_a_indices = np.sort(split_order[:midpoint])
    split_b_indices = np.sort(split_order[midpoint:])
    if min(len(split_a_indices), len(split_b_indices)) < 2:
        raise ValueError("Split-half subsets are too small")
    brain_a = relation_matrix_from_rdms(brain_rdms, split_a_indices)
    brain_b = relation_matrix_from_rdms(brain_rdms, split_b_indices)
    clip_a = relation_matrix_from_rdms(clip_rdms, split_a_indices)
    clip_b = relation_matrix_from_rdms(clip_rdms, split_b_indices)
    brain_split = permutation_null(brain_a, brain_b, args.null_count, brain_null_rng)
    clip_split = permutation_null(clip_a, clip_b, args.null_count, clip_null_rng)

    print("Starting stimulus bootstrap", flush=True)
    brain_boot, brain_boot_corr = bootstrap_relations(
        brain_rdms,
        brain_geometry,
        args.bootstrap_count,
        brain_boot_rng,
        "brain",
    )
    clip_boot, clip_boot_corr = bootstrap_relations(
        clip_rdms,
        clip_geometry,
        args.bootstrap_count,
        clip_boot_rng,
        "CLIP",
    )
    brain_boot_summary = matrix_summary(brain_boot, brain_boot_corr)
    clip_boot_summary = matrix_summary(clip_boot, clip_boot_corr)

    voxel_counts, voxel_count_error = load_voxel_counts(cache_config)
    roi_diagnostics = feature_diagnostics(brain)
    clip_diagnostics = feature_diagnostics(clip)
    roi_table = roi_rows(
        roi_names,
        voxel_counts,
        roi_diagnostics,
        brain_a,
        brain_b,
        brain_geometry,
        brain_boot,
    )

    np.save(output_dir / "brain_geometry.npy", brain_geometry)
    np.save(output_dir / "clip_geometry.npy", clip_geometry)
    write_matrix_csv(output_dir / "brain_geometry.csv", brain_geometry, roi_names)
    write_matrix_csv(output_dir / "clip_geometry.csv", clip_geometry, layer_labels)
    write_rows_csv(output_dir / "roi_reliability_summary.csv", roi_table)
    np.savez_compressed(
        output_dir / "bootstrap_geometries.npz",
        brain=brain_boot,
        clip=clip_boot,
        brain_spearman_to_full=brain_boot_corr,
        clip_spearman_to_full=clip_boot_corr,
    )

    split_payload = {
        "comparison": (
            "Spearman correlation between upper triangles of independently "
            "estimated relation matrices"
        ),
        "split_sizes": [len(split_a_indices), len(split_b_indices)],
        "split_a_anchor_positions": split_a_indices,
        "split_b_anchor_positions": split_b_indices,
        "split_a_stable_id_sha256": sha256_lines(
            anchor_ids[index] for index in split_a_indices
        ),
        "split_b_stable_id_sha256": sha256_lines(
            anchor_ids[index] for index in split_b_indices
        ),
        "brain": brain_split,
        "clip": clip_split,
        "brain_split_a_geometry": brain_a,
        "brain_split_b_geometry": brain_b,
        "clip_split_a_geometry": clip_a,
        "clip_split_b_geometry": clip_b,
    }
    write_json(output_dir / "split_half_reliability.json", split_payload)
    bootstrap_payload = {
        "method": (
            "stimulus bootstrap with replacement; each replicate contains the "
            "same number of draws as the anchor subset"
        ),
        "count": args.bootstrap_count,
        "brain": brain_boot_summary,
        "clip": clip_boot_summary,
        "full_replicate_matrices_file": "bootstrap_geometries.npz",
    }
    write_json(output_dir / "bootstrap_geometry_summary.json", bootstrap_payload)

    brain_heatmap = save_heatmap(
        output_dir / "brain_geometry_heatmap.png",
        brain_geometry,
        roi_names,
        "ROI representational geometry relation matrix\n(not anatomical distance)",
    )
    clip_heatmap = save_heatmap(
        output_dir / "clip_geometry_heatmap.png",
        clip_geometry,
        layer_labels,
        "CLIP layer representational geometry relation matrix",
    )
    assessment = preliminary_assessment(brain_split, brain_boot_summary)
    repeat_available = bool(metadata.get("repeat_metadata_available", False))
    trial_available = bool(metadata.get("trial_metadata_available", False))
    geometry_config = {
        "analysis_name": "subj01_representational_geometry_reliability_s1",
        "created_unix_time": time.time(),
        "elapsed_seconds": time.time() - started,
        "subject": cache_config["subject"],
        "source_cache_directory": str(cache_dir),
        "source_analysis_split": cache_config["analysis_split"],
        "cache_ordered_sample_id_sha256": metadata["ordered_sample_id_sha256"],
        "brain_feature": args.brain_feature,
        "brain_feature_shape": list(brain.shape),
        "clip_feature_shape": list(clip.shape),
        "roi_names": roi_names,
        "clip_layers": cache_config["selected_clip_layers"],
        "clip_layer_labels": layer_labels,
        "clip_pooling_method": cache_config["clip_pooling_method"],
        "geometry_type": args.geometry_type,
        "stimulus_distance": "1 - cosine similarity",
        "node_relation": "1 - Spearman correlation of upper-triangle RDM vectors",
        "brain_matrix_semantics": "ROI representational geometry relation matrix",
        "brain_matrix_is_anatomical_distance": False,
        "hierarchy_claimed": False,
        "anchor_subset_size": len(anchors),
        "anchor_stable_id_sha256": sha256_lines(anchor_ids),
        "bootstrap_count": args.bootstrap_count,
        "null_count": args.null_count,
        "random_seed": args.random_seed,
        "device": args.device,
        "full_stimulus_rdm_saved": False,
        "memory_strategy": (
            "fixed anchor subset; temporary float32 node RDMs in RAM; "
            "upper-triangle vectors for relation estimates"
        ),
        "repeat_metadata_available": repeat_available,
        "trial_metadata_available": trial_available,
        "repeat_reduction_in_cache": cache_config.get("repeat_reduction"),
        "repeat_aware_geometry": {
            "computed": False,
            "crossnobis_computed": False,
            "reason": (
                "The cache contains one ROI representation after averaging repeats "
                "per stimulus and only one scalar trial_id per sample; it has no "
                "repeat-resolved representations or independent session/run partitions."
            ),
        },
        "voxel_counts": voxel_counts,
        "voxel_count_error": voxel_count_error,
        "clip_feature_diagnostics": {
            label: values for label, values in zip(layer_labels, clip_diagnostics)
        },
        "heatmaps_saved": brain_heatmap and clip_heatmap,
        "leakage_control": {
            "only_geometry_fit_loaded": True,
            "feature_cost_fit_loaded": False,
            "heldout_eval_loaded": False,
            "umbrae_loaded": False,
            "shikra_loaded": False,
            "stage2_modified": False,
        },
        "preliminary_go_no_go": assessment,
    }
    write_json(output_dir / "geometry_config.json", geometry_config)
    print(f"Saved geometry reliability analysis to {output_dir}", flush=True)
    print(f"Preliminary assessment: {assessment['status']}", flush=True)
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
