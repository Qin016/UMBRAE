"""Load and validate subject-specific ROI indices aligned to fMRI vectors."""

import json
import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


EXPECTED_NSDGENERAL_VOXELS = {
    1: 15724,
    2: 14278,
    3: 15226,
    4: 13153,
    5: 13039,
    6: 17907,
    7: 12682,
    8: 14386,
}

MVP_ROI_NAMES = ("V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA")


def normalize_subject_id(subject: object) -> str:
    """Normalize ``1``, ``"1"`` and ``"subj01"`` to ``"subj01"``."""
    match = re.fullmatch(r"(?:subj)?0*(\d+)", str(subject).strip().lower())
    if match is None:
        raise ValueError(f"Invalid subject ID: {subject!r}")
    return f"subj{int(match.group(1)):02d}"


def validate_roi_indices(
    roi_indices: Mapping[str, Sequence[int]],
    voxel_count: int,
    strict: bool = True,
) -> Dict[str, object]:
    """Validate ROI indices and return a JSON-serializable summary."""
    normalized: Dict[str, np.ndarray] = {}
    errors = []

    for roi_name, values in roi_indices.items():
        indices = np.asarray(values, dtype=np.int64)
        if indices.ndim != 1:
            errors.append(f"{roi_name}: indices must be one-dimensional")
            continue
        if indices.size == 0:
            errors.append(f"{roi_name}: ROI is empty")
            continue
        if np.unique(indices).size != indices.size:
            errors.append(f"{roi_name}: contains duplicate indices")
        if indices.min(initial=0) < 0 or indices.max(initial=-1) >= voxel_count:
            errors.append(
                f"{roi_name}: indices must be within [0, {voxel_count})"
            )
        normalized[roi_name] = indices

    union = (
        np.unique(np.concatenate(list(normalized.values())))
        if normalized
        else np.empty(0, dtype=np.int64)
    )
    total_assignments = sum(len(indices) for indices in normalized.values())
    summary = {
        "voxel_count": int(voxel_count),
        "roi_count": len(normalized),
        "roi_voxel_counts": {
            name: int(len(indices)) for name, indices in normalized.items()
        },
        "union_voxel_count": int(len(union)),
        "union_fraction": float(len(union) / voxel_count) if voxel_count else 0.0,
        "union_reasonable": bool(0 < len(union) <= voxel_count),
        "overlap_assignments": int(total_assignments - len(union)),
        "errors": errors,
    }
    if not summary["union_reasonable"]:
        errors.append("ROI union must contain between 1 and voxel_count voxels")
    if strict and errors:
        raise ValueError("Invalid ROI mapping: " + "; ".join(errors))
    return summary


def load_roi_indices(
    path: str,
    mapping_format: Optional[str] = None,
    expected_voxel_count: Optional[int] = None,
    strict: bool = True,
    expected_subject: Optional[object] = None,
    roi_order: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Sequence[int]], Dict[str, object]]:
    """Load legacy or verified NeuroRoute ROI mappings.

    Legacy mappings store a top-level ``roi_indices`` dictionary. NeuroRoute
    mappings store indices below ``rois[name].indices`` and must explicitly be
    marked as real and voxel-order verified. NeuroRoute mappings always use the
    fixed MVP ROI order unless an explicit ``roi_order`` is supplied.
    """
    path_obj = Path(path)
    fmt = mapping_format or path_obj.suffix.lstrip(".").lower()
    if fmt not in {"json", "npy"}:
        raise ValueError("mapping_format must be 'json' or 'npy'")

    if fmt == "json":
        payload = json.loads(path_obj.read_text())
    else:
        payload = np.load(path_obj, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.shape == ():
            payload = payload.item()

    if not isinstance(payload, Mapping):
        raise ValueError("ROI mapping file must contain a dictionary")
    is_neuroroute = "rois" in payload
    if is_neuroroute:
        if payload.get("roi_mapping_is_real") is not True:
            raise ValueError("NeuroRoute mapping must set roi_mapping_is_real=true")
        if payload.get("voxel_order_verified") is not True:
            raise ValueError("NeuroRoute mapping must set voxel_order_verified=true")
        if expected_subject is not None:
            actual_subject = payload.get("subject")
            if actual_subject is None:
                raise ValueError("NeuroRoute mapping is missing subject")
            if normalize_subject_id(actual_subject) != normalize_subject_id(expected_subject):
                raise ValueError(
                    "ROI mapping subject mismatch: "
                    f"{actual_subject!r} != {expected_subject!r}"
                )
        declared_names = payload.get("roi_names")
        if not isinstance(declared_names, list):
            raise ValueError("NeuroRoute mapping must contain roi_names")
        if len(declared_names) != len(set(declared_names)):
            raise ValueError(f"NeuroRoute roi_names contains duplicates: {declared_names}")
        ordered_names = list(roi_order or MVP_ROI_NAMES)
        if len(ordered_names) != len(set(ordered_names)):
            raise ValueError(f"Requested ROI order contains duplicates: {ordered_names}")
        missing = [name for name in ordered_names if name not in declared_names]
        extra = [name for name in declared_names if name not in ordered_names]
        if missing or extra:
            raise ValueError(
                f"NeuroRoute ROI set mismatch; missing={missing}, extra={extra}"
            )
        rois = payload["rois"]
        if not isinstance(rois, Mapping):
            raise ValueError("NeuroRoute 'rois' must be a dictionary")
        roi_indices = {}
        for name in ordered_names:
            record = rois.get(name)
            if not isinstance(record, Mapping) or "indices" not in record:
                raise ValueError(f"NeuroRoute ROI {name!r} is missing indices")
            roi_indices[name] = record["indices"]
    else:
        roi_indices = payload.get("roi_indices", payload)
        if not isinstance(roi_indices, Mapping):
            raise ValueError("ROI mapping must contain an 'roi_indices' dictionary")
        if roi_order is not None:
            missing = [name for name in roi_order if name not in roi_indices]
            extra = [name for name in roi_indices if name not in roi_order]
            if missing or extra:
                raise ValueError(
                    f"ROI set mismatch; missing={missing}, extra={extra}"
                )
            roi_indices = {name: roi_indices[name] for name in roi_order}

    file_voxel_count = payload.get("voxel_count")
    voxel_count = expected_voxel_count or file_voxel_count
    if voxel_count is None:
        raise ValueError(
            "voxel_count is missing; pass expected_voxel_count or use builder output"
        )
    if (
        strict
        and expected_voxel_count is not None
        and file_voxel_count is not None
        and int(file_voxel_count) != int(expected_voxel_count)
    ):
        raise ValueError(
            f"Mapping voxel_count={file_voxel_count} does not match "
            f"expected_voxel_count={expected_voxel_count}"
        )

    normalized = {
        str(name): np.asarray(indices, dtype=np.int64).tolist()
        for name, indices in roi_indices.items()
    }
    summary = validate_roi_indices(normalized, int(voxel_count), strict=strict)
    return normalized, summary
