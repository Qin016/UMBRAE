"""Stimulus-independent pRF fine-unit construction for P11-B0.

The utilities in this module contain no learnable parameters, losses, optimizer,
or training entry points.  They operate only on anatomical/pRF metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .p11_structural_mapping import gaussian_patch_affinity


RETINO_ROIS = ("V1", "V2", "V3", "hV4")
HEMISPHERES = ("left", "right")
HIGH_LEVEL_ROIS = ("FFA", "EBA", "PPA", "OPA")


def clip_patch_visual_angle_map(extent_degrees: float = 8.4, grid_size: int = 16):
    """Return row-major patch centers for a square, fixation-centered stimulus."""
    half = extent_degrees / 2.0
    step = extent_degrees / grid_size
    result = []
    for row in range(grid_size):
        for column in range(grid_size):
            x_norm = 2.0 * (column + 0.5) / grid_size - 1.0
            y_norm = 1.0 - 2.0 * (row + 0.5) / grid_size
            result.append(
                {
                    "patch_index": row * grid_size + column,
                    "row": row,
                    "column": column,
                    "normalized_x": x_norm,
                    "normalized_y": y_norm,
                    "visual_angle_x": -half + (column + 0.5) * step,
                    "visual_angle_y": half - (row + 0.5) * step,
                    "aperture_overlap_fraction": 1.0,
                }
            )
    return result


def largest_remainder_allocation(counts: Mapping[tuple[str, str], int], k: int):
    """Proportionally allocate K with one unit per non-empty group."""
    keys = [(r, h) for r in RETINO_ROIS for h in HEMISPHERES]
    if any(counts.get(key, 0) <= 0 for key in keys) or k < len(keys):
        raise ValueError("all eight ROI x hemisphere groups must be non-empty")
    remaining = k - len(keys)
    total = float(sum(counts[key] for key in keys))
    quotas = {key: remaining * counts[key] / total for key in keys}
    allocation = {key: 1 + int(np.floor(quotas[key])) for key in keys}
    left = k - sum(allocation.values())
    order = sorted(keys, key=lambda key: (-(quotas[key] % 1.0), keys.index(key)))
    for key in order[:left]:
        allocation[key] += 1
    if sum(allocation.values()) != k:
        raise AssertionError("largest-remainder allocation does not sum to K")
    return allocation


def _group_features(table: Mapping[str, np.ndarray], indices: np.ndarray, feature_set: str):
    x = np.asarray(table["prf_x_deg"])[indices]
    y = np.asarray(table["prf_y_deg"])[indices]
    if feature_set == "xy":
        features = np.column_stack([x, y])
    elif feature_set == "xy_logsigma":
        sigma = np.asarray(table["prf_sigma_gaussian_deg"])[indices]
        if np.any(sigma <= 0):
            raise ValueError("log-sigma feature requires positive sigma")
        features = np.column_stack([x, y, np.log(sigma)])
    else:
        raise ValueError(f"unknown feature set: {feature_set}")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    return (features - mean) / scale, mean, scale


def _canonical_clusters(indices: np.ndarray, labels: np.ndarray, table):
    clusters = []
    x_all = np.asarray(table["prf_x_deg"])
    y_all = np.asarray(table["prf_y_deg"])
    for label in np.unique(labels):
        members = np.sort(indices[labels == label])
        x = float(x_all[members].mean())
        y = float(y_all[members].mean())
        eccentricity = float(np.hypot(x, y))
        angle = float(np.degrees(np.arctan2(y, x)) % 360.0)
        clusters.append((eccentricity, angle, members))
    clusters.sort(key=lambda item: (item[0], item[1]))
    return [members for _, _, members in clusters]


def build_candidate(
    table: Mapping[str, np.ndarray],
    patch_centers_xy: np.ndarray,
    quality_policy: str,
    feature_set: str,
    k_retino: int,
    seed: int = 42,
    n_init: int = 10,
):
    """Construct one canonical ROI x hemisphere conditioned mapping."""
    quality_mask = {
        "r2_gt0": np.asarray(table["quality_gt_0"], dtype=bool),
        "r2_ge10p1": np.asarray(table["quality_ge_10p1"], dtype=bool),
    }[quality_policy]
    roi = np.asarray(table["parent_retino_roi"])
    hemi = np.asarray(table["hemisphere"])
    valid = (
        np.asarray(table["valid_full_prf"], dtype=bool)
        & quality_mask
        & np.isin(roi, RETINO_ROIS)
    )
    groups = {
        (r, h): np.flatnonzero(valid & (roi == r) & (hemi == h))
        for r in RETINO_ROIS for h in HEMISPHERES
    }
    allocation = largest_remainder_allocation({key: len(v) for key, v in groups.items()}, k_retino)
    sigma_all = np.asarray(table["prf_sigma_gaussian_deg"], dtype=np.float64)
    x_all = np.asarray(table["prf_x_deg"], dtype=np.float64)
    y_all = np.asarray(table["prf_y_deg"], dtype=np.float64)
    r2_all = np.asarray(table["fit_quality_R2_percent"], dtype=np.float64)
    ecc_all = np.asarray(table["continuous_eccentricity_deg"], dtype=np.float64)
    voxel_affinity = gaussian_patch_affinity(
        np.column_stack([x_all[np.flatnonzero(valid)], y_all[np.flatnonzero(valid)]]),
        sigma_all[np.flatnonzero(valid)], patch_centers_xy,
    )
    affinity_lookup = {int(v): voxel_affinity[i] for i, v in enumerate(np.flatnonzero(valid))}
    units = []
    normalization = {}
    stability_values = []
    group_labels = {}
    for r in RETINO_ROIS:
        for h in HEMISPHERES:
            key = (r, h)
            indices = groups[key]
            features, mean, scale = _group_features(table, indices, feature_set)
            normalization[f"{r}_{h}"] = {"mean": mean.tolist(), "scale": scale.tolist()}
            model = KMeans(n_clusters=allocation[key], random_state=seed, n_init=n_init).fit(features)
            group_labels[key] = (indices, model.labels_)
            # Seed stability is intentionally measured with one initialization
            # per audit seed; the canonical seed-42 construction uses n_init=10.
            for audit_seed in range(5):
                other = KMeans(n_clusters=allocation[key], random_state=audit_seed, n_init=1).fit(features)
                stability_values.append(adjusted_rand_score(model.labels_, other.labels_))
            for local_index, members in enumerate(_canonical_clusters(indices, model.labels_, table)):
                x = x_all[members]; y = y_all[members]; sig = sigma_all[members]; r2 = r2_all[members]
                weights = np.maximum(r2, 0.0)
                stacked = np.stack([affinity_lookup[int(v)] for v in members])
                uniform = stacked.mean(axis=0)
                quality = np.average(stacked, axis=0, weights=weights)
                uniform /= uniform.sum(); quality /= quality.sum()
                mx = float(x.mean()); my = float(y.mean())
                centered_sq = (x - mx) ** 2 + (y - my) ** 2
                entropy = float(-(quality * np.log(np.maximum(quality, 1e-30))).sum())
                top_order = np.argsort(quality)[::-1]
                affinity_x = float(quality @ patch_centers_xy[:, 0])
                affinity_y = float(quality @ patch_centers_xy[:, 1])
                affinity_var = float(
                    quality @ (
                        (patch_centers_xy[:, 0] - affinity_x) ** 2
                        + (patch_centers_xy[:, 1] - affinity_y) ** 2
                    )
                )
                units.append({
                    "unit_id": f"{r}_{h}_u{local_index:03d}", "unit_type": "retinotopic",
                    "parent_roi": r, "hemisphere": h, "voxel_indices": members.tolist(),
                    "num_voxels": int(len(members)), "mean_x": mx, "mean_y": my,
                    "median_x": float(np.median(x)), "median_y": float(np.median(y)),
                    "prf_center_x": mx, "prf_center_y": my,
                    "mean_eccentricity": float(ecc_all[members].mean()),
                    "eccentricity": float(np.hypot(mx, my)),
                    "polar_angle": float(np.degrees(np.arctan2(my, mx)) % 360.0),
                    "mean_sigma": float(sig.mean()), "median_sigma": float(np.median(sig)),
                    "gaussian_sigma": float(sig.mean()),
                    "mean_r2": float(r2.mean()), "median_r2": float(np.median(r2)),
                    "within_unit_spatial_variance": float(centered_sq.mean()),
                    "rms_radius": float(np.sqrt(centered_sq.mean())),
                    "within_unit_sigma_variance": float(sig.var()),
                    "max_quality_weight_contribution": float(weights.max() / weights.sum()),
                    "dominant_patch": int(top_order[0]),
                    "top5_patches": top_order[:5].tolist(),
                    "top10_cumulative_mass": float(quality[top_order[:10]].sum()),
                    "affinity_entropy": entropy,
                    "effective_patch_count": float(np.exp(entropy)),
                    "affinity_center_of_mass_x": affinity_x,
                    "affinity_center_of_mass_y": affinity_y,
                    "affinity_spatial_variance": affinity_var,
                    "patch_affinity": quality.astype(np.float32),
                    "patch_affinity_uniform": uniform.astype(np.float32),
                })
    for token_index, unit in enumerate(units):
        unit["token_index"] = token_index
    return {
        "units": units, "valid_voxels": np.flatnonzero(valid), "allocation": allocation,
        "normalization": normalization, "cluster_stability": float(np.mean(stability_values)),
        "n_init": n_init, "seed": seed,
    }


def affinity_metrics(units: Sequence[Mapping[str, Any]]):
    w = np.stack([np.asarray(unit["patch_affinity"], dtype=np.float64) for unit in units])
    entropy = -(w * np.log(np.maximum(w, 1e-30))).sum(axis=1)
    normalized = w / np.linalg.norm(w, axis=1, keepdims=True)
    cosine = normalized @ normalized.T
    off = cosine[~np.eye(len(w), dtype=bool)]
    singular = np.linalg.svd(w, compute_uv=False)
    probabilities = singular / singular.sum()
    effective_rank = float(np.exp(-(probabilities * np.log(np.maximum(probabilities, 1e-30))).sum()))
    top10_mass = np.sort(w, axis=1)[:, -10:].sum(axis=1)
    covered = np.unique(np.argpartition(w, -5, axis=1)[:, -5:]).size / 256.0
    return {
        "mean_affinity_entropy": float(entropy.mean()),
        "mean_effective_patch_count": float(np.exp(entropy).mean()),
        "mean_top10_cumulative_mass": float(top10_mass.mean()),
        "mean_offdiag_affinity_cos": float(off.mean()),
        "median_offdiag_affinity_cos": float(np.median(off)),
        "p90_offdiag_affinity_cos": float(np.quantile(off, 0.9)),
        "max_offdiag_affinity_cos": float(off.max()),
        "affinity_effective_rank": effective_rank,
        "singular_values": singular.tolist(),
        "patch_coverage_score": float(covered),
    }


def randomize_membership(units: Sequence[Mapping[str, Any]], seed: int = 42):
    """Shuffle memberships only within ROI x hemisphere, preserving teachers."""
    rng = np.random.default_rng(seed)
    result = [dict(unit) for unit in units]
    for r in RETINO_ROIS:
        for h in HEMISPHERES:
            positions = [i for i, u in enumerate(units) if u["parent_roi"] == r and u["hemisphere"] == h]
            sizes = [len(units[i]["voxel_indices"]) for i in positions]
            pool = np.concatenate([np.asarray(units[i]["voxel_indices"], dtype=np.int64) for i in positions])
            shuffled = rng.permutation(pool)
            offset = 0
            for position, size in zip(positions, sizes):
                result[position]["voxel_indices"] = np.sort(shuffled[offset:offset + size]).tolist()
                result[position]["num_voxels"] = size
                offset += size
    return result


class FinePRFUnitMapping:
    """Read-only mapping interface with no torch or learnable state."""

    def __init__(self, mapping: str | Path | Mapping[str, Any]):
        if isinstance(mapping, (str, Path)):
            mapping = json.loads(Path(mapping).read_text())
        self.mapping = dict(mapping)
        self.units = list(self.mapping["units"])
        self.by_id = {unit["unit_id"]: unit for unit in self.units}

    def get_unit_indices(self, unit_id): return tuple(self.by_id[unit_id]["voxel_indices"])
    def get_parent_roi(self, unit_id): return self.by_id[unit_id]["parent_roi"]
    def get_hemisphere(self, unit_id): return self.by_id[unit_id].get("hemisphere")
    def get_patch_affinity(self, unit_id): return self.by_id[unit_id].get("patch_affinity")
    def get_unit_metadata(self, unit_id): return dict(self.by_id[unit_id])
