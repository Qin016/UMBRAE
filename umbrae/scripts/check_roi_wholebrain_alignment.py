#!/usr/bin/env python
"""Sanity-check NSD ROI NIfTI alignment with UMBRAE whole-brain arrays."""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check spatial and nsdgeneral-value alignment between an UMBRAE "
            "wholebrain_3d.npy sample and official NSD ROI NIfTI files."
        )
    )
    parser.add_argument("--wholebrain_path", required=True)
    parser.add_argument("--roi_path", required=True)
    parser.add_argument("--nsdgeneral_npy_path")
    parser.add_argument("--nsdgeneral_nii_path")
    args = parser.parse_args()

    optional_pair = (
        args.nsdgeneral_npy_path is not None,
        args.nsdgeneral_nii_path is not None,
    )
    if any(optional_pair) and not all(optional_pair):
        parser.error(
            "--nsdgeneral_npy_path and --nsdgeneral_nii_path must be "
            "provided together"
        )
    return args


def require_file(path: str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def finite_statistics(arr: np.ndarray) -> Dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    # Accumulate in float64 to avoid float16 reduction overflow.
    finite = finite.astype(np.float64, copy=False)
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def summarize_array(name: str, arr: np.ndarray) -> None:
    stats = finite_statistics(arr)
    print(f"\n{name}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  min: {stats['min']:.8g}")
    print(f"  max: {stats['max']:.8g}")
    print(f"  mean: {stats['mean']:.8g}")
    print(f"  std: {stats['std']:.8g}")
    print(f"  NaN count: {int(np.isnan(arr).sum())}")
    print(f"  nonzero count: {int(np.count_nonzero(arr))}")


def load_wholebrain(path: str) -> np.ndarray:
    resolved = require_file(path, "wholebrain array")
    try:
        wholebrain = np.load(resolved, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read wholebrain array {resolved}: {exc}") from exc
    if wholebrain.ndim not in (3, 4):
        raise ValueError(
            "wholebrain_3d.npy must have shape [X,Y,Z] or [R,X,Y,Z], "
            f"got {wholebrain.shape}"
        )
    summarize_array("Whole-brain fMRI", wholebrain)
    if wholebrain.ndim == 4:
        print(
            f"  repeat dimension detected: R={wholebrain.shape[0]}; "
            f"spatial shape={wholebrain.shape[-3:]}"
        )
    return wholebrain


def load_roi(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    resolved = require_file(path, "ROI NIfTI")
    try:
        image = nib.load(str(resolved))
        roi_data = np.asarray(image.dataobj)
    except Exception as exc:
        raise RuntimeError(f"Failed to read ROI NIfTI {resolved}: {exc}") from exc
    if roi_data.ndim != 3:
        raise ValueError(f"ROI NIfTI must be 3D, got shape {roi_data.shape}")

    labels, counts = np.unique(roi_data, return_counts=True)
    print("\nROI NIfTI")
    print(f"  path: {resolved}")
    print(f"  shape: {roi_data.shape}")
    print(f"  dtype: {roi_data.dtype}")
    print("  affine:")
    for row in image.affine:
        print(f"    {np.array2string(row, precision=6, suppress_small=True)}")
    print(f"  header zooms: {image.header.get_zooms()[:3]}")
    print(f"  all unique labels: {labels.tolist()}")
    negative_background = counts[labels < 0]
    if negative_background.size:
        print(
            "  negative/outside-volume voxels: "
            f"{int(negative_background.sum())}"
        )
    background = counts[labels == 0]
    if background.size:
        print(f"  background label 0 voxels: {int(background[0])}")
    print("  nonzero label voxel counts:")
    nonzero_found = False
    for label, count in zip(labels, counts):
        if label <= 0:
            continue
        nonzero_found = True
        print(f"    label {format_label(label)}: {int(count)}")
    if not nonzero_found:
        print("    [WARNING] no nonzero ROI labels found")
    return roi_data, image


def format_label(label: np.generic) -> str:
    value = float(label)
    return str(int(value)) if value.is_integer() else str(value)


def spatial_shape(wholebrain: np.ndarray) -> Tuple[int, int, int]:
    return tuple(wholebrain.shape[-3:])


def check_shape_alignment(wholebrain: np.ndarray, roi_data: np.ndarray) -> bool:
    wholebrain_spatial_shape = spatial_shape(wholebrain)
    roi_shape = tuple(roi_data.shape)
    print("\nShape alignment")
    print(f"  wholebrain full shape: {wholebrain.shape}")
    print(f"  wholebrain spatial shape: {wholebrain_spatial_shape}")
    print(f"  ROI shape: {roi_shape}")
    if wholebrain_spatial_shape == roi_shape:
        print("[PASS] Shape matched")
        return True
    print(
        "[FAIL] Shape mismatch: "
        f"wholebrain spatial shape={wholebrain_spatial_shape}, ROI shape={roi_shape}"
    )
    return False


def apply_spatial_mask(wholebrain: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if wholebrain.ndim == 3:
        return wholebrain[mask]
    return wholebrain[:, mask]


def normalize_nsdgeneral_array(
    nsdgeneral: np.ndarray, wholebrain: np.ndarray
) -> np.ndarray:
    nsdgeneral = np.asarray(nsdgeneral)
    if wholebrain.ndim == 3:
        return np.squeeze(nsdgeneral)
    if nsdgeneral.ndim == 1 and wholebrain.shape[0] == 1:
        return nsdgeneral[None, :]
    return nsdgeneral


def check_nsdgeneral_consistency(
    wholebrain: np.ndarray,
    nsdgeneral_npy_path: str,
    nsdgeneral_nii_path: str,
) -> bool:
    npy_path = require_file(nsdgeneral_npy_path, "nsdgeneral array")
    nii_path = require_file(nsdgeneral_nii_path, "nsdgeneral NIfTI")
    try:
        nsdgeneral_npy = np.load(npy_path, allow_pickle=False)
        mask_image = nib.load(str(nii_path))
        mask_data = np.asarray(mask_image.dataobj)
    except Exception as exc:
        raise RuntimeError(f"Failed to read nsdgeneral inputs: {exc}") from exc

    print("\nnsdgeneral consistency")
    print(f"  npy path: {npy_path}")
    print(f"  npy shape: {nsdgeneral_npy.shape}, dtype: {nsdgeneral_npy.dtype}")
    print(f"  mask path: {nii_path}")
    print(f"  mask shape: {mask_data.shape}")
    if tuple(mask_data.shape) != spatial_shape(wholebrain):
        print(
            "[FAIL] nsdgeneral mask shape mismatch: "
            f"mask={mask_data.shape}, wholebrain spatial={spatial_shape(wholebrain)}"
        )
        return False

    nsdgeneral_mask = mask_data > 0
    mask_voxel_count = int(nsdgeneral_mask.sum())
    masked = apply_spatial_mask(wholebrain, nsdgeneral_mask)
    expected = normalize_nsdgeneral_array(nsdgeneral_npy, wholebrain)
    print(f"  mask voxel count: {mask_voxel_count}")
    print(f"  extracted masked shape: {masked.shape}")
    print(f"  nsdgeneral comparison shape: {expected.shape}")

    masked_voxel_count = masked.shape[-1]
    expected_voxel_count = expected.shape[-1] if expected.ndim else 0
    if masked_voxel_count != expected_voxel_count:
        print(
            "[FAIL] nsdgeneral voxel length mismatch: "
            f"masked={masked_voxel_count}, npy={expected_voxel_count}"
        )
        return False
    if masked.shape != expected.shape:
        print(
            "[FAIL] nsdgeneral full shape mismatch: "
            f"masked={masked.shape}, npy={expected.shape}"
        )
        return False

    close = bool(np.allclose(masked, expected, equal_nan=True))
    absolute_diff = np.abs(
        masked.astype(np.float64) - expected.astype(np.float64)
    )
    finite_diff = absolute_diff[np.isfinite(absolute_diff)]
    max_abs_diff = float(finite_diff.max()) if finite_diff.size else float("nan")
    mean_abs_diff = float(finite_diff.mean()) if finite_diff.size else float("nan")
    print(f"  np.allclose(equal_nan=True): {close}")
    print(f"  max_abs_diff: {max_abs_diff:.10g}")
    print(f"  mean_abs_diff: {mean_abs_diff:.10g}")
    print(
        "[PASS] nsdgeneral values and mask order matched"
        if close
        else "[FAIL] nsdgeneral values differ"
    )
    return close


def extract_roi_vectors(
    wholebrain: np.ndarray, roi_data: np.ndarray
) -> Dict[str, np.ndarray]:
    print("\nROI extraction test")
    extracted = {}
    labels = np.unique(roi_data)
    # NSD volume ROI files use -1 outside the cortical volume and 0 for
    # unlabeled/background voxels. Only positive integers are ROI labels.
    labels = labels[labels > 0]
    if labels.size == 0:
        print("  [WARNING] ROI image has no nonzero labels")
        return extracted

    for label in labels:
        label_mask = roi_data == label
        voxel_count = int(label_mask.sum())
        label_name = format_label(label)
        if voxel_count == 0:
            print(f"  [WARNING] label {label_name} has zero voxels")
            continue
        roi_vector = apply_spatial_mask(wholebrain, label_mask)
        stats = finite_statistics(roi_vector)
        nan_count = int(np.isnan(roi_vector).sum())
        print(
            f"  label {label_name}: vector shape={roi_vector.shape}, "
            f"mean={stats['mean']:.8g}, std={stats['std']:.8g}, "
            f"NaN count={nan_count}"
        )
        extracted[label_name] = roi_vector
    return extracted


def main() -> None:
    args = parse_args()
    try:
        wholebrain = load_wholebrain(args.wholebrain_path)
        roi_data, _ = load_roi(args.roi_path)
        shape_matches = check_shape_alignment(wholebrain, roi_data)

        nsdgeneral_checked = args.nsdgeneral_npy_path is not None
        nsdgeneral_matches: Optional[bool] = None
        if nsdgeneral_checked:
            nsdgeneral_matches = check_nsdgeneral_consistency(
                wholebrain,
                args.nsdgeneral_npy_path,
                args.nsdgeneral_nii_path,
            )

        if shape_matches:
            extract_roi_vectors(wholebrain, roi_data)
        else:
            print("\nROI extraction skipped because spatial shapes do not match.")

        print("\nFinal conclusion")
        if not shape_matches:
            print(
                "Conclusion: NOT aligned. Check whether wholebrain_3d.npy "
                "is in func1pt8mm space."
            )
        elif nsdgeneral_checked and nsdgeneral_matches:
            print(
                "Conclusion: ROI NIfTI and wholebrain_3d.npy are aligned and "
                "can be used directly for ROI voxel extraction."
            )
        elif nsdgeneral_checked:
            print(
                "Conclusion: Shape matched, but voxel ordering or preprocessing "
                "may differ. Be careful before using ROI masks."
            )
        else:
            print(
                "Conclusion: Spatial shape matched, but nsdgeneral value/order "
                "was not checked. Provide both nsdgeneral paths before treating "
                "ROI voxel extraction as fully verified."
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()
