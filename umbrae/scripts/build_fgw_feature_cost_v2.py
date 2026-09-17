#!/usr/bin/env python
"""Build a cross-fitted ROI-to-CLIP compatibility cost on offline_discovery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_representation_geometry_v2 import gather_cache_rows, sha256_ids
from scripts.analyze_representation_geometry import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-fit the FGW feature cost matrix")
    parser.add_argument("--stimulus-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--std-epsilon", type=float, default=1e-6)
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


def build_unique_id_folds(
    stimulus_ids: Sequence[str], folds: int, seed: int
) -> Tuple[np.ndarray, List[List[str]]]:
    if folds < 2 or folds > len(stimulus_ids):
        raise ValueError("folds must be between 2 and the number of stimuli")
    if len(stimulus_ids) != len(set(stimulus_ids)):
        raise ValueError("offline_discovery contains duplicate stimulus IDs")
    sorted_ids = sorted(stimulus_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sorted_ids))
    fold_id_lists = [[] for _ in range(folds)]
    for position, shuffled_index in enumerate(order):
        fold_id_lists[position % folds].append(sorted_ids[int(shuffled_index)])
    fold_id_lists = [sorted(values) for values in fold_id_lists]
    id_to_fold = {
        stimulus_id: fold
        for fold, values in enumerate(fold_id_lists)
        for stimulus_id in values
    }
    assignments = np.asarray([id_to_fold[value] for value in stimulus_ids], dtype=np.int64)
    return assignments, fold_id_lists


def fit_ridge_predict_multioutput(
    train_x: np.ndarray,
    train_y: np.ndarray,
    heldout_x: np.ndarray,
    alpha: float,
    std_epsilon: float,
    device: str,
) -> np.ndarray:
    """Fit train-only standardized ridge and predict all CLIP targets at once."""
    if alpha < 0:
        raise ValueError("ridge alpha must be non-negative")
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    x_heldout = torch.as_tensor(heldout_x, dtype=torch.float32, device=device)
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, unbiased=False, keepdim=True)
    x_std = torch.where(x_std > std_epsilon, x_std, torch.ones_like(x_std))
    y_mean = y.mean(dim=0, keepdim=True)
    x = (x - x_mean) / x_std
    x_heldout = (x_heldout - x_mean) / x_std
    y = y - y_mean
    gram = x.T @ x
    gram.diagonal().add_(float(alpha))
    right = x.T @ y
    try:
        factor = torch.linalg.cholesky(gram)
        weights = torch.cholesky_solve(right, factor)
    except RuntimeError:
        weights = torch.linalg.solve(gram, right)
    prediction = x_heldout @ weights + y_mean
    return prediction.detach().cpu().numpy().astype(np.float32, copy=False)


def cosine_costs(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sum(prediction * target, axis=-1)
    denominator = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1)
    similarity = numerator / np.maximum(denominator, 1e-12)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def write_fold_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.stimulus_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["overlap_checks"]["offline_unique_stimulus_overlaps"] != {
        "discovery_validation": 0,
        "discovery_test": 0,
        "validation_test": 0,
    }:
        raise ValueError("Corrected offline split overlap check failed")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite feature-cost output: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.time()

    manifest_dir = manifest_path.parent
    discovery = manifest["offline_splits"]["offline_discovery"]
    discovery_indices = np.load(manifest_dir / discovery["index_file"], allow_pickle=False)
    table = manifest["cache_row_table"]
    table_by_global = {int(row["global_cache_row"]): row for row in table}
    stimulus_ids = [table_by_global[int(index)]["stable_stimulus_id"] for index in discovery_indices]
    cache_root = Path(manifest["representation_cache"])
    brain, brain_splits = gather_cache_rows(
        cache_root, table, discovery_indices, "brain_roi_features.npy"
    )
    clip, clip_splits = gather_cache_rows(
        cache_root, table, discovery_indices, "clip_layer_features.npy"
    )
    if brain.shape != (len(stimulus_ids), 8, 1024) or clip.shape != (
        len(stimulus_ids),
        6,
        1024,
    ):
        raise ValueError(f"Unexpected feature shapes: brain={brain.shape}, clip={clip.shape}")
    cache_config = json.loads((cache_root / "cache_config.json").read_text())
    roi_names = list(cache_config["roi_names"])
    layer_labels = [f"L{layer}" for layer in cache_config["selected_clip_layers"]]
    assignments, fold_id_lists = build_unique_id_folds(stimulus_ids, args.folds, args.random_seed)

    n, roi_count, _ = brain.shape
    layer_count = clip.shape[1]
    cost_sum = np.zeros((roi_count, layer_count), dtype=np.float64)
    cost_square_sum = np.zeros_like(cost_sum)
    cost_count = np.zeros_like(cost_sum, dtype=np.int64)
    fold_rows: List[Dict[str, Any]] = []
    for fold in range(args.folds):
        heldout = np.flatnonzero(assignments == fold)
        training = np.flatnonzero(assignments != fold)
        y_train = np.ascontiguousarray(clip[training].reshape(len(training), -1))
        y_heldout = np.asarray(clip[heldout], dtype=np.float32)
        for roi in range(roi_count):
            prediction_flat = fit_ridge_predict_multioutput(
                np.asarray(brain[training, roi], dtype=np.float32),
                y_train,
                np.asarray(brain[heldout, roi], dtype=np.float32),
                args.ridge_alpha,
                args.std_epsilon,
                args.device,
            )
            prediction = prediction_flat.reshape(len(heldout), layer_count, -1)
            costs = cosine_costs(prediction, y_heldout)
            cost_sum[roi] += costs.sum(axis=0, dtype=np.float64)
            cost_square_sum[roi] += np.square(costs, dtype=np.float64).sum(axis=0)
            cost_count[roi] += len(heldout)
            for layer in range(layer_count):
                fold_rows.append(
                    {
                        "fold": fold,
                        "train_count": len(training),
                        "heldout_count": len(heldout),
                        "roi": roi_names[roi],
                        "clip_layer": layer_labels[layer],
                        "mean_oof_cosine_quality": float(1.0 - np.mean(costs[:, layer])),
                        "mean_oof_cost": float(np.mean(costs[:, layer])),
                        "std_oof_cost_ddof1": float(np.std(costs[:, layer], ddof=1)),
                    }
                )
        print(f"completed OOF fold {fold + 1}/{args.folds}", flush=True)

    if not np.all(cost_count == n):
        raise RuntimeError("Every ROI/layer must receive exactly one OOF prediction per stimulus")
    feature_cost = cost_sum / cost_count
    variance = cost_square_sum / cost_count - np.square(feature_cost)
    cost_std = np.sqrt(np.maximum(variance, 0.0))
    if not np.isfinite(feature_cost).all() or not np.isfinite(cost_std).all():
        raise ValueError("Non-finite OOF feature cost")
    np.save(output_dir / "feature_cost_oof.npy", feature_cost)
    with (output_dir / "feature_cost_oof.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["roi", *layer_labels])
        for roi_name, row in zip(roi_names, feature_cost):
            writer.writerow([roi_name, *[f"{float(value):.9g}" for value in row]])
    write_fold_metrics(output_dir / "feature_cost_fold_metrics.csv", fold_rows)

    source_files = {
        "manifest": manifest_path,
        "cache_config": cache_root / "cache_config.json",
    }
    for split in sorted(set(brain_splits + clip_splits)):
        source_files[f"{split}/brain_roi_features"] = cache_root / split / "brain_roi_features.npy"
        source_files[f"{split}/clip_layer_features"] = cache_root / split / "clip_layer_features.npy"
    source_hashes = {name: sha256_file(path) for name, path in source_files.items()}
    combined_digest = hashlib.sha256()
    for name in sorted(source_hashes):
        combined_digest.update(name.encode("utf-8"))
        combined_digest.update(b"=")
        combined_digest.update(source_hashes[name].encode("ascii"))
        combined_digest.update(b"\n")
    layer_mean = feature_cost.mean(axis=0)
    best_layer_index = int(np.argmin(layer_mean))
    sorted_layer_costs = np.sort(layer_mean)
    l24_index = layer_labels.index("L24")
    l24_preference = {
        "layer_mean_costs": {label: float(value) for label, value in zip(layer_labels, layer_mean)},
        "lowest_mean_cost_layer": layer_labels[best_layer_index],
        "l24_is_lowest_mean_cost": best_layer_index == l24_index,
        "l24_margin_to_second_best": (
            float(sorted_layer_costs[1] - sorted_layer_costs[0])
            if best_layer_index == l24_index
            else float(layer_mean[l24_index] - sorted_layer_costs[0])
        ),
        "interpretation": "descriptive compatibility only; not model-depth hierarchy",
    }
    config = {
        "analysis": "cross_fitted_roi_to_clip_feature_compatibility_v2",
        "source_split": "offline_discovery",
        "offline_discovery_count": n,
        "offline_validation_loaded": False,
        "offline_test_loaded": False,
        "protected_validation_loaded": False,
        "protected_test_loaded": False,
        "brain_input": "roi_tokens_before_projector",
        "brain_storage_file": "brain_roi_features.npy",
        "brain_input_is_raw_voxel_vector": False,
        "clip_target": "cached mean_non_cls_patch_tokens per selected layer",
        "probe_type": "multi-output linear ridge with intercept",
        "ridge_regularization": {
            "alpha": args.ridge_alpha,
            "objective": "sum_squared_error + alpha * squared_weight_norm",
            "selection": "pre-specified globally; no validation or per-pair tuning",
        },
        "preprocessing": {
            "brain": "per-dimension center and scale using training-fold mean/std only",
            "clip_target": "center using training-fold mean only; no target scaling",
            "std_epsilon": args.std_epsilon,
            "heldout_statistics_used": False,
        },
        "cross_fitting": {
            "folds": args.folds,
            "assignment_unit": "unique stable stimulus ID",
            "random_seed": args.random_seed,
            "folds_disjoint": True,
            "every_stimulus_predicted_exactly_once": True,
            "fold_stimulus_id_hashes": [sha256_ids(values) for values in fold_id_lists],
            "fold_sizes": [len(values) for values in fold_id_lists],
        },
        "cost_definition": "mean over OOF stimuli of 1 - cosine(predicted CLIP feature, true CLIP feature)",
        "feature_cost_shape": list(feature_cost.shape),
        "roi_order": roi_names,
        "clip_layer_order": cache_config["selected_clip_layers"],
        "clip_layer_labels": layer_labels,
        "oof_cost_mean_per_roi_layer": feature_cost,
        "oof_cost_std_per_roi_layer": cost_std,
        "source_cache_hash": combined_digest.hexdigest(),
        "source_file_hashes": source_hashes,
        "manifest_offline_discovery_id_hash": discovery["sorted_id_sha256"],
        "l24_preference_diagnostic": l24_preference,
        "soft_routing_scores_used_as_feature_cost": False,
        "feature_models_are_probes_only": True,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "feature_cost_config.json", config)
    print(f"Saved OOF feature cost to {output_dir}", flush=True)
    print(json.dumps(l24_preference, indent=2), flush=True)
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
