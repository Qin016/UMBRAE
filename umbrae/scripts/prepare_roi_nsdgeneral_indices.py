#!/usr/bin/env python
"""Convert volume-space ROI indices to positions in UMBRAE nsdgeneral vectors."""

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np


REQUIRED_ROI_FIELDS = {
    "label",
    "name",
    "num_voxels",
    "x",
    "y",
    "z",
    "flat_indices",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert wholebrain C-order ROI flat indices into vector positions "
            "for sampleXXXX.nsdgeneral.npy."
        )
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--roi_name", required=True)
    parser.add_argument("--roi_indices_pkl", required=True)
    parser.add_argument("--nsdgeneral_nii_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--wholebrain_path")
    parser.add_argument("--nsdgeneral_npy_path")
    parser.add_argument("--save_pkl", action="store_true")
    parser.add_argument("--save_npz", action="store_true")
    args = parser.parse_args()
    validation_pair = (
        args.wholebrain_path is not None,
        args.nsdgeneral_npy_path is not None,
    )
    if any(validation_pair) and not all(validation_pair):
        parser.error(
            "--wholebrain_path and --nsdgeneral_npy_path must be provided together"
        )
    return args


def require_file(path: str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def load_roi_indices_pkl(path: str) -> Tuple[Dict[int, Dict[str, object]], Path]:
    resolved = require_file(path, "ROI indices PKL")
    try:
        with resolved.open("rb") as file:
            payload = pickle.load(file)
    except Exception as exc:
        raise RuntimeError(f"Failed to read ROI indices PKL {resolved}: {exc}") from exc

    # Accept both the current direct dictionary and the earlier wrapped format.
    if isinstance(payload, Mapping) and "roi_indices" in payload:
        payload = payload["roi_indices"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("ROI indices PKL must contain a non-empty dictionary")

    roi_indices: Dict[int, Dict[str, object]] = {}
    for raw_label, raw_entry in payload.items():
        label = int(raw_label)
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"label {label} entry must be a dictionary")
        missing_fields = REQUIRED_ROI_FIELDS - set(raw_entry)
        if missing_fields:
            raise ValueError(
                f"label {label} is missing required fields: {sorted(missing_fields)}"
            )
        entry = dict(raw_entry)
        entry["label"] = int(entry["label"])
        entry["name"] = str(entry["name"])
        for field in ("x", "y", "z", "flat_indices"):
            values = np.asarray(entry[field], dtype=np.int64)
            if values.ndim != 1:
                raise ValueError(f"label {label} field {field} must be 1D")
            entry[field] = values
        lengths = {
            len(entry[field]) for field in ("x", "y", "z", "flat_indices")
        }
        if len(lengths) != 1:
            raise ValueError(f"label {label} coordinate/index lengths differ")
        if int(entry["num_voxels"]) != len(entry["flat_indices"]):
            raise ValueError(
                f"label {label} num_voxels does not match flat_indices length"
            )
        if len(entry["flat_indices"]) == 0:
            raise ValueError(f"label {label} ROI is empty")
        roi_indices[label] = entry
    return dict(sorted(roi_indices.items())), resolved


def load_nsdgeneral_mask(
    path: str,
) -> Tuple[np.ndarray, nib.Nifti1Image, Path]:
    resolved = require_file(path, "nsdgeneral NIfTI")
    try:
        image = nib.load(str(resolved))
        data = np.asarray(image.dataobj)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read nsdgeneral NIfTI {resolved}: {exc}"
        ) from exc
    if data.ndim != 3:
        raise ValueError(f"nsdgeneral NIfTI must be 3D, got shape {data.shape}")
    mask = data > 0
    if not mask.any():
        raise ValueError("nsdgeneral mask contains zero selected voxels")
    return mask, image, resolved


def infer_and_validate_roi_shape(
    roi_indices: Mapping[int, Mapping[str, object]],
    nsdgeneral_shape: Sequence[int],
) -> Tuple[int, int, int]:
    shape = tuple(int(value) for value in nsdgeneral_shape)
    volume_size = int(np.prod(shape))
    for label, entry in roi_indices.items():
        x, y, z = entry["x"], entry["y"], entry["z"]
        if (
            np.any(x < 0)
            or np.any(y < 0)
            or np.any(z < 0)
            or np.any(x >= shape[0])
            or np.any(y >= shape[1])
            or np.any(z >= shape[2])
        ):
            raise ValueError(
                f"label {label} coordinates exceed nsdgeneral volume shape {shape}"
            )
        expected_flat = np.ravel_multi_index((x, y, z), shape, order="C")
        if not np.array_equal(expected_flat, entry["flat_indices"]):
            raise ValueError(
                f"label {label} flat_indices do not match x/y/z in C-order "
                f"for shape {shape}"
            )
        if np.any(entry["flat_indices"] < 0) or np.any(
            entry["flat_indices"] >= volume_size
        ):
            raise ValueError(f"label {label} flat_indices exceed volume bounds")
    return shape


def build_flat_to_nsdgeneral_position(
    nsdgeneral_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    nsdgeneral_flat_indices = np.flatnonzero(
        nsdgeneral_mask.ravel(order="C")
    ).astype(np.int64)
    flat_to_nsd_pos = np.full(
        nsdgeneral_mask.size, -1, dtype=np.int64
    )
    flat_to_nsd_pos[nsdgeneral_flat_indices] = np.arange(
        len(nsdgeneral_flat_indices), dtype=np.int64
    )
    return flat_to_nsd_pos, nsdgeneral_flat_indices


def convert_roi_indices_to_nsdgeneral(
    roi_indices: Mapping[int, Mapping[str, object]],
    flat_to_nsd_pos: np.ndarray,
) -> Dict[int, Dict[str, object]]:
    converted: Dict[int, Dict[str, object]] = {}
    for label, entry in roi_indices.items():
        roi_flat = np.asarray(entry["flat_indices"], dtype=np.int64)
        mapped_positions = flat_to_nsd_pos[roi_flat]
        keep_mask = mapped_positions >= 0
        nsdgeneral_indices = mapped_positions[keep_mask].astype(
            np.int64, copy=False
        )
        kept_flat = roi_flat[keep_mask].astype(np.int64, copy=False)
        missing_flat = roi_flat[~keep_mask].astype(np.int64, copy=False)
        wholebrain_count = len(roi_flat)
        kept_count = len(nsdgeneral_indices)
        coverage = float(kept_count / wholebrain_count)

        converted[label] = {
            "label": int(label),
            "name": str(entry["name"]),
            "num_voxels_wholebrain": int(wholebrain_count),
            "num_voxels_in_nsdgeneral": int(kept_count),
            "coverage_ratio": coverage,
            "nsdgeneral_indices": nsdgeneral_indices,
            "wholebrain_flat_indices": kept_flat,
            "missing_wholebrain_flat_indices": missing_flat,
            "missing_count": int(len(missing_flat)),
            "kept_x": np.asarray(entry["x"], dtype=np.int64)[keep_mask],
            "kept_y": np.asarray(entry["y"], dtype=np.int64)[keep_mask],
            "kept_z": np.asarray(entry["z"], dtype=np.int64)[keep_mask],
        }
    return converted


def save_pkl(
    output_path: Path,
    roi_nsdgeneral_indices: Mapping[int, Mapping[str, object]],
) -> None:
    with output_path.open("wb") as file:
        pickle.dump(
            dict(roi_nsdgeneral_indices),
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved PKL: {output_path}")


def save_npz(
    output_path: Path,
    roi_nsdgeneral_indices: Mapping[int, Mapping[str, object]],
) -> None:
    arrays: Dict[str, np.ndarray] = {
        "labels": np.asarray(list(roi_nsdgeneral_indices), dtype=np.int64),
        "label_names": np.asarray(
            [entry["name"] for entry in roi_nsdgeneral_indices.values()],
            dtype=str,
        ),
    }
    for label, entry in roi_nsdgeneral_indices.items():
        prefix = f"label_{label}"
        for field in (
            "nsdgeneral_indices",
            "wholebrain_flat_indices",
            "missing_wholebrain_flat_indices",
            "kept_x",
            "kept_y",
            "kept_z",
        ):
            arrays[f"{prefix}_{field}"] = np.asarray(
                entry[field], dtype=np.int64
            )
    np.savez_compressed(output_path, **arrays)
    print(f"Saved NPZ: {output_path}")


def save_metadata_json(
    output_path: Path, metadata: Mapping[str, object]
) -> None:
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved metadata: {output_path}")


def load_validation_arrays(
    wholebrain_path: str,
    nsdgeneral_npy_path: str,
    roi_shape: Sequence[int],
    nsdgeneral_voxel_count: int,
) -> Tuple[np.ndarray, np.ndarray, Path, Path]:
    wholebrain_resolved = require_file(wholebrain_path, "wholebrain array")
    nsdgeneral_resolved = require_file(
        nsdgeneral_npy_path, "nsdgeneral array"
    )
    try:
        wholebrain = np.load(wholebrain_resolved, allow_pickle=False)
        nsdgeneral = np.load(nsdgeneral_resolved, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read validation arrays: {exc}") from exc

    if wholebrain.ndim not in (3, 4):
        raise ValueError(
            f"wholebrain must be [X,Y,Z] or [R,X,Y,Z], got {wholebrain.shape}"
        )
    if tuple(wholebrain.shape[-3:]) != tuple(roi_shape):
        raise ValueError(
            f"wholebrain spatial shape {wholebrain.shape[-3:]} != {tuple(roi_shape)}"
        )
    if nsdgeneral.ndim not in (1, 2):
        raise ValueError(
            f"nsdgeneral must be [V] or [R,V], got {nsdgeneral.shape}"
        )
    if nsdgeneral.shape[-1] != nsdgeneral_voxel_count:
        raise ValueError(
            f"nsdgeneral V={nsdgeneral.shape[-1]} != mask voxel count "
            f"{nsdgeneral_voxel_count}"
        )
    expected_repeat_shape = (
        wholebrain.shape[0] if wholebrain.ndim == 4 else None
    )
    actual_repeat_shape = nsdgeneral.shape[0] if nsdgeneral.ndim == 2 else None
    if expected_repeat_shape != actual_repeat_shape:
        raise ValueError(
            "wholebrain and nsdgeneral repeat dimensions are incompatible: "
            f"{wholebrain.shape} vs {nsdgeneral.shape}"
        )
    return (
        wholebrain,
        nsdgeneral,
        wholebrain_resolved,
        nsdgeneral_resolved,
    )


def finite_difference_statistics(
    left: np.ndarray, right: np.ndarray
) -> Tuple[float, float]:
    difference = np.abs(
        left.astype(np.float64) - right.astype(np.float64)
    )
    finite = difference[np.isfinite(difference)]
    if not finite.size:
        return float("nan"), float("nan")
    return float(finite.max()), float(finite.mean())


def validate_against_samples(
    roi_nsdgeneral_indices: Mapping[int, Mapping[str, object]],
    wholebrain_path: str,
    nsdgeneral_npy_path: str,
    roi_shape: Sequence[int],
    nsdgeneral_voxel_count: int,
) -> Dict[str, object]:
    (
        wholebrain,
        nsdgeneral,
        wholebrain_resolved,
        nsdgeneral_resolved,
    ) = load_validation_arrays(
        wholebrain_path,
        nsdgeneral_npy_path,
        roi_shape,
        nsdgeneral_voxel_count,
    )

    per_label = {}
    print("\nStrong validation")
    print(f"  wholebrain: {wholebrain_resolved}, shape={wholebrain.shape}")
    print(f"  nsdgeneral: {nsdgeneral_resolved}, shape={nsdgeneral.shape}")
    for label, entry in roi_nsdgeneral_indices.items():
        x, y, z = entry["kept_x"], entry["kept_y"], entry["kept_z"]
        nsd_indices = entry["nsdgeneral_indices"]
        roi_from_wholebrain = (
            wholebrain[x, y, z]
            if wholebrain.ndim == 3
            else wholebrain[:, x, y, z]
        )
        roi_from_nsdgeneral = (
            nsdgeneral[nsd_indices]
            if nsdgeneral.ndim == 1
            else nsdgeneral[:, nsd_indices]
        )
        if roi_from_wholebrain.shape != roi_from_nsdgeneral.shape:
            raise ValueError(
                f"label {label} extracted shape mismatch: "
                f"{roi_from_wholebrain.shape} != {roi_from_nsdgeneral.shape}"
            )

        strict_close = bool(
            np.allclose(
                roi_from_wholebrain,
                roi_from_nsdgeneral,
                rtol=1e-5,
                atol=1e-8,
                equal_nan=True,
            )
        )
        relaxed_close = bool(
            np.allclose(
                roi_from_wholebrain,
                roi_from_nsdgeneral,
                rtol=1e-3,
                atol=2e-3,
                equal_nan=True,
            )
        )
        max_diff, mean_diff = finite_difference_statistics(
            roi_from_wholebrain, roi_from_nsdgeneral
        )
        result = {
            "name": entry["name"],
            "shape": list(roi_from_wholebrain.shape),
            "strict_allclose": strict_close,
            "relaxed_allclose": relaxed_close,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "nan_count_wholebrain": int(np.isnan(roi_from_wholebrain).sum()),
            "nan_count_nsdgeneral": int(np.isnan(roi_from_nsdgeneral).sum()),
        }
        per_label[str(label)] = result
        print(
            f"  label={label}, name={entry['name']}, "
            f"shape={tuple(roi_from_wholebrain.shape)}"
        )
        print(f"    strict allclose: {'PASS' if strict_close else 'FAIL'}")
        print(f"    relaxed allclose: {'PASS' if relaxed_close else 'FAIL'}")
        print(f"    max_abs_diff: {max_diff:.10g}")
        print(f"    mean_abs_diff: {mean_diff:.10g}")
        print(
            f"    NaNs: wholebrain={result['nan_count_wholebrain']}, "
            f"nsdgeneral={result['nan_count_nsdgeneral']}"
        )

    return {
        "wholebrain_path": str(wholebrain_resolved),
        "wholebrain_shape": list(wholebrain.shape),
        "nsdgeneral_npy_path": str(nsdgeneral_resolved),
        "nsdgeneral_shape": list(nsdgeneral.shape),
        "all_relaxed_allclose": all(
            result["relaxed_allclose"] for result in per_label.values()
        ),
        "per_label": per_label,
    }


def summarize_validation(
    roi_nsdgeneral_indices: Mapping[int, Mapping[str, object]],
    validation: Optional[Mapping[str, object]],
) -> Tuple[bool, bool]:
    any_zero_coverage = False
    any_partial_coverage = False
    print("\nROI conversion summary")
    for label, entry in roi_nsdgeneral_indices.items():
        coverage = float(entry["coverage_ratio"])
        print(f"label={label}, name={entry['name']}")
        print(f"  wholebrain voxels: {entry['num_voxels_wholebrain']}")
        print(f"  nsdgeneral voxels: {entry['num_voxels_in_nsdgeneral']}")
        print(f"  coverage ratio: {coverage:.4f}")
        if validation is not None:
            result = validation["per_label"][str(label)]
            print(
                "  validation relaxed allclose: "
                f"{'PASS' if result['relaxed_allclose'] else 'FAIL'}"
            )
        if coverage == 0:
            any_zero_coverage = True
            print(
                f"[ERROR] label={label}, name={entry['name']}, has zero "
                "voxels inside nsdgeneral and cannot be used with "
                "nsdgeneral.npy input."
            )
        elif coverage < 1:
            any_partial_coverage = True
            print(
                f"[WARN] label={label}, name={entry['name']}, coverage "
                f"ratio={coverage:.4f}, some ROI voxels are outside nsdgeneral."
            )
    return any_zero_coverage, any_partial_coverage


def build_metadata(
    args: argparse.Namespace,
    roi_indices_path: Path,
    nsdgeneral_path: Path,
    nsdgeneral_shape: Sequence[int],
    nsdgeneral_voxel_count: int,
    converted: Mapping[int, Mapping[str, object]],
    validation: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    per_label = {
        str(label): {
            "name": entry["name"],
            "num_voxels_wholebrain": entry["num_voxels_wholebrain"],
            "num_voxels_in_nsdgeneral": entry["num_voxels_in_nsdgeneral"],
            "coverage_ratio": entry["coverage_ratio"],
            "missing_count": entry["missing_count"],
        }
        for label, entry in converted.items()
    }
    coverages = [entry["coverage_ratio"] for entry in converted.values()]
    return {
        "subject": args.subject,
        "roi_name": args.roi_name,
        "roi_indices_pkl": str(roi_indices_path),
        "nsdgeneral_nii_path": str(nsdgeneral_path),
        "nsdgeneral_shape": list(nsdgeneral_shape),
        "nsdgeneral_voxel_count": int(nsdgeneral_voxel_count),
        "labels": list(converted),
        "label_names": [entry["name"] for entry in converted.values()],
        "per_label": per_label,
        "total_wholebrain_roi_voxels": int(
            sum(entry["num_voxels_wholebrain"] for entry in converted.values())
        ),
        "total_nsdgeneral_roi_voxels": int(
            sum(
                entry["num_voxels_in_nsdgeneral"]
                for entry in converted.values()
            )
        ),
        "min_coverage_ratio": float(min(coverages)),
        "created_time": datetime.now(timezone.utc).isoformat(),
        "flat_index_order": "numpy_C_order",
        "validation": validation,
    }


def main() -> None:
    args = parse_args()
    try:
        roi_indices, roi_indices_path = load_roi_indices_pkl(
            args.roi_indices_pkl
        )
        (
            nsdgeneral_mask,
            _,
            nsdgeneral_path,
        ) = load_nsdgeneral_mask(args.nsdgeneral_nii_path)
        roi_shape = infer_and_validate_roi_shape(
            roi_indices, nsdgeneral_mask.shape
        )
        (
            flat_to_nsd_pos,
            nsdgeneral_flat_indices,
        ) = build_flat_to_nsdgeneral_position(nsdgeneral_mask)
        converted = convert_roi_indices_to_nsdgeneral(
            roi_indices, flat_to_nsd_pos
        )

        print(f"Subject: {args.subject}")
        print(f"ROI name: {args.roi_name}")
        print(f"nsdgeneral shape: {nsdgeneral_mask.shape}")
        print(f"nsdgeneral voxels: {len(nsdgeneral_flat_indices)}")

        validation = None
        if args.wholebrain_path is not None:
            validation = validate_against_samples(
                converted,
                args.wholebrain_path,
                args.nsdgeneral_npy_path,
                roi_shape,
                len(nsdgeneral_flat_indices),
            )
        any_zero, any_partial = summarize_validation(converted, validation)

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{args.subject}_{args.roi_name}_nsdgeneral"
        if args.save_pkl:
            save_pkl(output_dir / f"{stem}_indices.pkl", converted)
        if args.save_npz:
            save_npz(output_dir / f"{stem}_indices.npz", converted)

        metadata = build_metadata(
            args,
            roi_indices_path,
            nsdgeneral_path,
            nsdgeneral_mask.shape,
            len(nsdgeneral_flat_indices),
            converted,
            validation,
        )
        save_metadata_json(output_dir / f"{stem}_meta.json", metadata)

        validation_failed = (
            validation is not None and not validation["all_relaxed_allclose"]
        )
        print("\nFinal conclusion")
        if any_zero or validation_failed:
            print("Conclusion: FAILED or partially failed.")
            print(
                "Do not use failed ROI indices until voxel mapping/order is fixed."
            )
            raise SystemExit(1)
        if any_partial:
            print(
                "Conclusion: ROI nsdgeneral indices generated with partial coverage."
            )
            print("Review warnings before using these ROIs for training.")
        else:
            print(
                "Conclusion: ROI nsdgeneral indices generated successfully."
            )
            print(
                "They can be used to extract ROI-wise fMRI features directly "
                "from sampleXXXX.nsdgeneral.npy."
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        print("Conclusion: FAILED or partially failed.")
        print(
            "Do not use failed ROI indices until voxel mapping/order is fixed."
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
