#!/usr/bin/env python
"""Build ROI indices aligned to a subject's flattened nsdgeneral vector."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.roi_mapping import (
    EXPECTED_NSDGENERAL_VOXELS,
    validate_roi_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build subject-specific ROI indices for UMBRAE nsdgeneral"
    )
    parser.add_argument("--subject", type=int, required=True, choices=range(1, 9))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--nsdgeneral-mask",
        help="subject func1pt8mm/roi/nsdgeneral.nii.gz",
    )
    source.add_argument(
        "--nsdgeneral-order",
        help="NPY containing volume linear indices [V] or voxel coordinates [V,3]",
    )
    parser.add_argument(
        "--reference-volume",
        help="NIfTI defining volume shape/affine; required with coordinate order",
    )
    parser.add_argument(
        "--roi-spec",
        required=True,
        help="JSON mapping output ROI names to atlas paths and integer labels",
    )
    parser.add_argument("--output", required=True, help="output .json or .npy")
    parser.add_argument(
        "--output-format", choices=["json", "npy"], default=None
    )
    parser.add_argument(
        "--flatten-order",
        choices=["C", "F"],
        default="C",
        help="used only when deriving order from nsdgeneral mask",
    )
    parser.add_argument(
        "--expected-voxel-count",
        type=int,
        default=None,
        help="defaults to known UMBRAE count for the selected subject",
    )
    parser.add_argument(
        "--strict-roi-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_nifti(path: str):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "NIfTI input requires nibabel: pip install nibabel"
        ) from exc
    image = nib.load(path)
    data = np.asarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {data.shape}")
    return data, image.affine


def load_voxel_order(args: argparse.Namespace):
    provenance = {}
    if args.nsdgeneral_mask:
        mask, affine = load_nifti(args.nsdgeneral_mask)
        volume_shape = mask.shape
        order = np.flatnonzero(mask.ravel(order=args.flatten_order) > 0)
        provenance = {
            "source": "nsdgeneral_mask",
            "path": str(Path(args.nsdgeneral_mask).resolve()),
            "flatten_order": args.flatten_order,
            "warning": (
                "Mask-derived order is valid only if upstream nsdgeneral.npy "
                "used the same NumPy flatten order."
            ),
        }
        return order.astype(np.int64), volume_shape, affine, provenance

    raw_order = np.load(args.nsdgeneral_order)
    if raw_order.ndim == 1:
        if args.reference_volume is None:
            raise ValueError(
                "--reference-volume is required with linear nsdgeneral order"
            )
        reference, affine = load_nifti(args.reference_volume)
        volume_shape = reference.shape
        order = raw_order.astype(np.int64)
        if np.any(order < 0) or np.any(order >= np.prod(volume_shape)):
            raise ValueError("nsdgeneral linear order exceeds reference volume")
        representation = "linear_indices"
    elif raw_order.ndim == 2 and raw_order.shape[1] == 3:
        if args.reference_volume is None:
            raise ValueError(
                "--reference-volume is required with coordinate nsdgeneral order"
            )
        reference, affine = load_nifti(args.reference_volume)
        volume_shape = reference.shape
        coordinates = raw_order.astype(np.int64)
        if np.any(coordinates < 0) or np.any(
            coordinates >= np.asarray(volume_shape)[None, :]
        ):
            raise ValueError("nsdgeneral coordinates exceed reference volume")
        order = np.ravel_multi_index(
            coordinates.T, volume_shape, order=args.flatten_order
        )
        representation = "voxel_coordinates"
    else:
        raise ValueError(
            "nsdgeneral order must have shape [V] or [V, 3]"
        )

    provenance = {
        "source": "explicit_nsdgeneral_order",
        "path": str(Path(args.nsdgeneral_order).resolve()),
        "representation": representation,
        "flatten_order": args.flatten_order,
    }
    return order.astype(np.int64), volume_shape, affine, provenance


def validate_atlas_geometry(
    atlas_shape, atlas_affine, volume_shape, reference_affine, strict
):
    errors = []
    if tuple(atlas_shape) != tuple(volume_shape):
        errors.append(
            f"atlas shape {atlas_shape} != reference shape {volume_shape}"
        )
    if not np.allclose(atlas_affine, reference_affine, atol=1e-4):
        errors.append("atlas affine does not match reference affine")
    if strict and errors:
        raise ValueError("; ".join(errors))
    return errors


def build_indices(
    voxel_order: np.ndarray,
    volume_shape,
    reference_affine,
    roi_spec: Mapping[str, Mapping[str, object]],
    flatten_order: str,
    strict: bool,
):
    roi_indices: Dict[str, List[int]] = {}
    geometry_warnings = []
    for roi_name, spec in roi_spec.items():
        atlas, atlas_affine = load_nifti(str(spec["atlas"]))
        geometry_warnings.extend(
            f"{roi_name}: {message}"
            for message in validate_atlas_geometry(
                atlas.shape,
                atlas_affine,
                volume_shape,
                reference_affine,
                strict,
            )
        )
        labels = np.asarray(spec["labels"], dtype=np.int64)
        atlas_at_nsd_voxels = atlas.ravel(order=flatten_order)[voxel_order]
        roi_indices[roi_name] = np.flatnonzero(
            np.isin(atlas_at_nsd_voxels, labels)
        ).astype(np.int64).tolist()
    return roi_indices, geometry_warnings


def main() -> None:
    args = parse_args()
    expected_count = (
        args.expected_voxel_count
        or EXPECTED_NSDGENERAL_VOXELS[args.subject]
    )
    voxel_order, volume_shape, affine, provenance = load_voxel_order(args)
    if len(voxel_order) != expected_count:
        raise ValueError(
            f"S{args.subject} nsdgeneral order has {len(voxel_order)} voxels; "
            f"expected {expected_count}"
        )
    if np.unique(voxel_order).size != len(voxel_order):
        raise ValueError("nsdgeneral voxel order contains duplicates")

    roi_spec_path = Path(args.roi_spec)
    roi_spec = json.loads(roi_spec_path.read_text())
    for spec in roi_spec.values():
        atlas_path = Path(spec["atlas"])
        if not atlas_path.is_absolute():
            spec["atlas"] = str((roi_spec_path.parent / atlas_path).resolve())
    roi_indices, geometry_warnings = build_indices(
        voxel_order,
        volume_shape,
        affine,
        roi_spec,
        args.flatten_order,
        args.strict_roi_check,
    )
    summary = validate_roi_indices(
        roi_indices, expected_count, strict=args.strict_roi_check
    )
    summary.update(
        {
            "subject": args.subject,
            "volume_shape": list(volume_shape),
            "geometry_warnings": geometry_warnings,
            "voxel_order_provenance": provenance,
        }
    )
    payload = {
        "subject": args.subject,
        "voxel_count": expected_count,
        "roi_names": list(roi_indices),
        "roi_indices": roi_indices,
        "summary": summary,
        "roi_spec_path": str(roi_spec_path.resolve()),
        "voxel_order_provenance": provenance,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.output_format or output_path.suffix.lstrip(".").lower()
    if output_format == "json":
        output_path.write_text(json.dumps(payload, indent=2))
    elif output_format == "npy":
        np.save(output_path, payload, allow_pickle=True)
    else:
        raise ValueError("output format must be json or npy")

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Subject S{args.subject}: {expected_count} nsdgeneral voxels")
    for roi_name, count in summary["roi_voxel_counts"].items():
        print(f"  {roi_name}: {count}")
    print(
        f"Union: {summary['union_voxel_count']} "
        f"({summary['union_fraction']:.2%} of nsdgeneral)"
    )
    print(f"Saved mapping: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
