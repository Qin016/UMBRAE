#!/usr/bin/env python
"""Prepare and validate volume-space ROI voxel indices from an NSD ROI NIfTI."""

import argparse
import csv
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np


ID_COLUMNS = {"label", "label_id", "id", "index", "value"}
NAME_COLUMNS = {"label_name", "name", "roi", "description"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate volume-space voxel coordinates and flat indices from "
            "an official NSD ROI NIfTI."
        )
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--roi_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_table_path")
    parser.add_argument("--nsdgeneral_nii_path")
    parser.add_argument("--wholebrain_path")
    parser.add_argument("--roi_name", required=True)
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--save_pkl", action="store_true")
    parser.add_argument("--make_qc_png", action="store_true")
    parser.add_argument("--make_3d_qc_png", action="store_true")
    return parser.parse_args()


def require_file(path: str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def load_roi_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image, Path]:
    resolved = require_file(path, "ROI NIfTI")
    try:
        image = nib.load(str(resolved))
        roi_data = np.asarray(image.dataobj)
    except Exception as exc:
        raise RuntimeError(f"Failed to read ROI NIfTI {resolved}: {exc}") from exc
    if roi_data.ndim != 3:
        raise ValueError(f"ROI NIfTI must be 3D, got shape {roi_data.shape}")
    if not np.all(np.isfinite(roi_data)):
        raise ValueError("ROI NIfTI contains NaN or infinite label values")

    labels, counts = np.unique(roi_data, return_counts=True)
    print(f"ROI path: {resolved}")
    print(f"ROI shape: {roi_data.shape}")
    print("ROI affine:")
    for row in image.affine:
        print(f"  {np.array2string(row, precision=6, suppress_small=True)}")
    print(f"Header zooms: {image.header.get_zooms()[:3]}")
    print(f"Unique labels: {labels.tolist()}")
    print("Label voxel counts:")
    for label, count in zip(labels, counts):
        print(f"  label={format_label(label)}, voxels={int(count)}")
    return roi_data, image, resolved


def format_label(label: object) -> str:
    value = float(label)
    return str(int(value)) if value.is_integer() else str(value)


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_label_id(value: str) -> Optional[int]:
    try:
        numeric = float(value.strip())
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer():
        return None
    return int(numeric)


def read_table_rows(path: Path) -> Tuple[List[List[str]], Optional[str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return [], None

    suffix = path.suffix.lower()
    delimiter = "\t" if suffix == ".tsv" else "," if suffix == ".csv" else None
    if delimiter is not None:
        rows = list(csv.reader(lines, delimiter=delimiter))
    else:
        try:
            dialect = csv.Sniffer().sniff("\n".join(lines[:10]), delimiters=",\t;")
            rows = list(csv.reader(lines, dialect))
            delimiter = dialect.delimiter
        except csv.Error:
            rows = [re.split(r"\s+", line, maxsplit=1) for line in lines]
            delimiter = "whitespace"
    return [[cell.strip() for cell in row] for row in rows], delimiter


def load_label_table(path: Optional[str]) -> Tuple[Dict[int, str], List[str]]:
    warnings: List[str] = []
    if path is None:
        return {}, warnings

    resolved = require_file(path, "label table")
    try:
        rows, delimiter = read_table_rows(resolved)
    except Exception as exc:
        warnings.append(f"Could not read label table {resolved}: {exc}")
        return {}, warnings
    if not rows:
        warnings.append(f"Label table is empty: {resolved}")
        return {}, warnings

    header = [normalize_header(cell) for cell in rows[0]]
    id_col = next((i for i, name in enumerate(header) if name in ID_COLUMNS), None)
    name_col = next(
        (i for i, name in enumerate(header) if name in NAME_COLUMNS), None
    )
    has_header = id_col is not None and name_col is not None
    if not has_header:
        id_col, name_col = 0, 1
        data_rows = rows
        first_row_is_two_column_mapping = (
            len(rows[0]) >= 2 and parse_label_id(rows[0][0]) is not None
        )
        if not first_row_is_two_column_mapping:
            warnings.append(
                "Could not identify label-table headers; using first column as "
                "label id and second column as label name."
            )
    else:
        data_rows = rows[1:]

    mapping: Dict[int, str] = {}
    skipped = 0
    for row in data_rows:
        if len(row) <= max(id_col, name_col):
            skipped += 1
            continue
        label_id = parse_label_id(row[id_col])
        label_name = row[name_col].strip()
        if label_id is None or not label_name:
            skipped += 1
            continue
        if label_id in mapping and mapping[label_id] != label_name:
            warnings.append(
                f"Duplicate label {label_id} has conflicting names "
                f"{mapping[label_id]!r} and {label_name!r}; keeping the first."
            )
            continue
        mapping[label_id] = label_name

    if not mapping:
        warnings.append(
            f"No label-id/name pairs could be parsed from {resolved} "
            f"(detected delimiter: {delimiter!r})."
        )
    elif skipped:
        warnings.append(f"Skipped {skipped} unrecognized label-table rows.")
    print(f"Label table: {resolved}")
    print(f"Parsed label names: {len(mapping)}")
    return mapping, warnings


def get_valid_labels(roi_data: np.ndarray) -> Tuple[List[int], List[object]]:
    unique = np.unique(roi_data)
    non_integral = [
        float(label)
        for label in unique
        if label > 0 and not float(label).is_integer()
    ]
    if non_integral:
        raise ValueError(
            f"Positive ROI labels must be integers, found: {non_integral}"
        )
    valid_labels = sorted(int(label) for label in unique if label > 0)
    excluded_labels = [
        int(label) if float(label).is_integer() else float(label)
        for label in unique
        if label <= 0
    ]
    return valid_labels, excluded_labels


def check_label_table_consistency(
    valid_labels: Sequence[int],
    label_id_to_name: Mapping[int, str],
) -> List[str]:
    warnings = []
    valid_set = set(valid_labels)
    for label in valid_labels:
        if label not in label_id_to_name:
            warnings.append(
                f"Label {label} not found in label table, using label_{label}"
            )
    for label in sorted(
        label for label in set(label_id_to_name) - valid_set if label > 0
    ):
        warnings.append(
            f"Label table contains label {label}, but it does not appear "
            "in ROI NIfTI"
        )
    return warnings


def build_roi_indices(
    roi_data: np.ndarray,
    label_id_to_name: Mapping[int, str],
) -> Dict[int, Dict[str, object]]:
    valid_labels, _ = get_valid_labels(roi_data)
    roi_indices: Dict[int, Dict[str, object]] = {}
    for label in valid_labels:
        mask = roi_data == label
        x, y, z = np.where(mask)
        flat_indices = np.ravel_multi_index((x, y, z), roi_data.shape, order="C")
        name = label_id_to_name.get(label, f"label_{label}")
        roi_indices[label] = {
            "label": label,
            "name": name,
            "num_voxels": int(flat_indices.size),
            "x": x.astype(np.int64),
            "y": y.astype(np.int64),
            "z": z.astype(np.int64),
            "flat_indices": flat_indices.astype(np.int64),
        }
    return roi_indices


def check_voxel_counts(
    roi_indices: Mapping[int, Mapping[str, object]]
) -> List[str]:
    warnings = []
    print("\nROI label summary")
    for label, entry in roi_indices.items():
        count = int(entry["num_voxels"])
        print(f"  label={label}, name={entry['name']}, num_voxels={count}")
        if count < 10:
            warnings.append(
                f"label={label}, name={entry['name']}, has only {count} voxels"
            )
    print(
        "Total valid ROI voxels:",
        sum(int(entry["num_voxels"]) for entry in roi_indices.values()),
    )
    return warnings


def check_roi_overlap(
    roi_data: np.ndarray,
    roi_indices: Mapping[int, Mapping[str, object]],
) -> Tuple[Dict[str, object], List[str]]:
    warnings = []
    masks = [roi_data == label for label in roi_indices]
    sum_counts = sum(int(mask.sum()) for mask in masks)
    union_count = int(np.logical_or.reduce(masks).sum()) if masks else 0
    overlaps = sum_counts - union_count
    print("\nROI overlap check")
    print(f"  sum of per-label voxels: {sum_counts}")
    print(f"  union voxels: {union_count}")
    print(f"  overlap assignments: {overlaps}")
    if union_count != sum_counts:
        warnings.append(
            "ROI label masks overlap: union voxel count differs from the "
            "sum of per-label voxel counts"
        )
    return {
        "sum_label_voxels": sum_counts,
        "union_voxels": union_count,
        "overlap_assignments": overlaps,
        "passed": union_count == sum_counts,
    }, warnings


def check_nsdgeneral_overlap(
    roi_data: np.ndarray,
    roi_indices: Mapping[int, Mapping[str, object]],
    path: Optional[str],
) -> Tuple[Optional[Dict[str, object]], List[str]]:
    if path is None:
        return None, []

    resolved = require_file(path, "nsdgeneral NIfTI")
    try:
        image = nib.load(str(resolved))
        nsdgeneral_data = np.asarray(image.dataobj)
    except Exception as exc:
        raise RuntimeError(f"Failed to read nsdgeneral NIfTI {resolved}: {exc}") from exc
    if tuple(nsdgeneral_data.shape) != tuple(roi_data.shape):
        raise ValueError(
            "nsdgeneral shape does not match ROI shape: "
            f"{nsdgeneral_data.shape} != {roi_data.shape}"
        )

    nsdgeneral_mask = nsdgeneral_data > 0
    warnings = []
    per_label = {}
    print("\nnsdgeneral overlap")
    for label, entry in roi_indices.items():
        roi_mask = roi_data == label
        voxel_count = int(roi_mask.sum())
        overlap = int(np.logical_and(roi_mask, nsdgeneral_mask).sum())
        ratio = float(overlap / voxel_count) if voxel_count else 0.0
        print(
            f"  label={label}, name={entry['name']}, voxels={voxel_count}, "
            f"overlap_with_nsdgeneral={overlap}, overlap_ratio={ratio:.4f}"
        )
        if ratio < 0.5:
            warnings.append(
                f"label={label}, name={entry['name']}, has low "
                f"nsdgeneral overlap ratio {ratio:.4f}"
            )
        per_label[str(label)] = {
            "name": entry["name"],
            "voxels": voxel_count,
            "overlap_voxels": overlap,
            "overlap_ratio": ratio,
        }
    return {
        "path": str(resolved),
        "shape": list(nsdgeneral_data.shape),
        "mask_voxels": int(nsdgeneral_mask.sum()),
        "per_label": per_label,
    }, warnings


def finite_stats(array: np.ndarray) -> Dict[str, object]:
    finite = array[np.isfinite(array)].astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "nan_count": int(np.isnan(array).sum()),
    }


def check_wholebrain_extraction(
    roi_shape: Sequence[int],
    roi_indices: Mapping[int, Mapping[str, object]],
    path: Optional[str],
) -> Tuple[Optional[Dict[str, object]], List[str]]:
    if path is None:
        return None, []

    resolved = require_file(path, "wholebrain array")
    try:
        wholebrain = np.load(resolved, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read wholebrain array {resolved}: {exc}") from exc
    if wholebrain.ndim not in (3, 4):
        raise ValueError(
            "wholebrain must have shape [X,Y,Z] or [R,X,Y,Z], "
            f"got {wholebrain.shape}"
        )
    spatial_shape = tuple(wholebrain.shape[-3:])
    if spatial_shape != tuple(roi_shape):
        raise ValueError(
            f"wholebrain spatial shape {spatial_shape} != ROI shape {tuple(roi_shape)}"
        )

    warnings = []
    per_label = {}
    print("\nWholebrain extraction")
    print(f"  path: {resolved}")
    print(f"  full shape: {wholebrain.shape}")
    for label, entry in roi_indices.items():
        x, y, z = entry["x"], entry["y"], entry["z"]
        roi_vector = (
            wholebrain[x, y, z]
            if wholebrain.ndim == 3
            else wholebrain[:, x, y, z]
        )
        expected_voxels = int(entry["num_voxels"])
        if roi_vector.shape[-1] != expected_voxels:
            raise ValueError(
                f"label {label} extraction returned shape {roi_vector.shape}; "
                f"expected final dimension {expected_voxels}"
            )
        stats = finite_stats(roi_vector)
        print(
            f"  label={label}, name={entry['name']}, "
            f"extracted_shape={tuple(roi_vector.shape)}, "
            f"mean={stats['mean']:.8g}, std={stats['std']:.8g}, "
            f"nan_count={stats['nan_count']}"
        )
        per_label[str(label)] = {"name": entry["name"], **stats}
    return {
        "path": str(resolved),
        "full_shape": list(wholebrain.shape),
        "spatial_shape": list(spatial_shape),
        "shape_matches": True,
        "per_label": per_label,
    }, warnings


def save_indices_pkl(
    output_path: Path,
    subject: str,
    roi_name: str,
    roi_path: Path,
    roi_shape: Sequence[int],
    roi_indices: Mapping[int, Mapping[str, object]],
) -> None:
    with output_path.open("wb") as file:
        pickle.dump(dict(roi_indices), file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved PKL: {output_path}")


def save_indices_npz(
    output_path: Path,
    roi_indices: Mapping[int, Mapping[str, object]],
) -> None:
    arrays: Dict[str, np.ndarray] = {
        "labels": np.asarray(list(roi_indices), dtype=np.int64),
        "label_names": np.asarray(
            [entry["name"] for entry in roi_indices.values()], dtype=str
        ),
    }
    for label, entry in roi_indices.items():
        prefix = f"label_{label}"
        arrays[f"{prefix}_x"] = entry["x"]
        arrays[f"{prefix}_y"] = entry["y"]
        arrays[f"{prefix}_z"] = entry["z"]
        arrays[f"{prefix}_flat_indices"] = entry["flat_indices"]
    np.savez_compressed(output_path, **arrays)
    print(f"Saved NPZ: {output_path}")


def json_safe_label_mapping(mapping: Mapping[int, str]) -> Dict[str, str]:
    return {str(label): name for label, name in sorted(mapping.items())}


def save_metadata_json(output_path: Path, metadata: Mapping[str, object]) -> None:
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved metadata: {output_path}")


def make_qc_png(
    output_path: Path,
    roi_data: np.ndarray,
    subject: str,
    roi_name: str,
    valid_label_count: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "QC PNG requires matplotlib: pip install matplotlib"
        ) from exc

    x_mid, y_mid, z_mid = (size // 2 for size in roi_data.shape)
    display = np.where(roi_data > 0, roi_data, np.nan)
    slices = [
        ("Sagittal", display[x_mid, :, :].T),
        ("Coronal", display[:, y_mid, :].T),
        ("Axial", display[:, :, z_mid].T),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (title, image) in zip(axes, slices):
        axis.imshow(image, origin="lower", interpolation="nearest", cmap="tab20")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{subject} | {roi_name} | shape={tuple(roi_data.shape)} | "
        f"valid labels={valid_label_count}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved QC PNG: {output_path}")


def make_3d_qc_png(
    output_path: Path,
    roi_indices: Mapping[int, Mapping[str, object]],
    roi_shape: Sequence[int],
    subject: str,
    roi_name: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError(
            "3D QC PNG requires matplotlib: pip install matplotlib"
        ) from exc

    labels = list(roi_indices)
    colormap = plt.get_cmap("tab20", max(len(labels), 1))
    figure = plt.figure(figsize=(11, 9))
    axis = figure.add_subplot(111, projection="3d")
    legend_handles = []

    for color_index, label in enumerate(labels):
        entry = roi_indices[label]
        color = colormap(color_index)
        axis.scatter(
            entry["x"],
            entry["y"],
            entry["z"],
            s=5,
            alpha=0.7,
            color=color,
            depthshade=False,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=7,
                label=f"{label}: {entry['name']} (n={entry['num_voxels']})",
            )
        )

    axis.set_xlabel("X voxel")
    axis.set_ylabel("Y voxel")
    axis.set_zlabel("Z voxel")
    axis.set_xlim(0, roi_shape[0] - 1)
    axis.set_ylim(0, roi_shape[1] - 1)
    axis.set_zlim(0, roi_shape[2] - 1)
    axis.set_box_aspect(roi_shape)
    axis.view_init(elev=22, azim=-60)
    axis.set_title(
        f"{subject} | {roi_name} | 3D voxel labels\n"
        f"shape={tuple(roi_shape)} | valid labels={len(labels)}"
    )
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved 3D QC PNG: {output_path}")


def print_warnings(warnings: Sequence[str]) -> None:
    for warning in warnings:
        print(f"[WARN] {warning}")


def main() -> None:
    args = parse_args()
    warnings: List[str] = []
    serious_error = False
    try:
        roi_data, roi_image, roi_path = load_roi_nifti(args.roi_path)
        valid_labels, excluded_labels = get_valid_labels(roi_data)
        if not valid_labels:
            raise ValueError("ROI NIfTI contains no valid labels > 0")

        parsed_mapping, table_warnings = load_label_table(args.label_table_path)
        warnings.extend(table_warnings)
        warnings.extend(check_label_table_consistency(valid_labels, parsed_mapping))
        label_id_to_name = {
            label: parsed_mapping.get(label, f"label_{label}")
            for label in valid_labels
        }

        roi_indices = build_roi_indices(roi_data, label_id_to_name)
        warnings.extend(check_voxel_counts(roi_indices))
        overlap_stats, overlap_warnings = check_roi_overlap(roi_data, roi_indices)
        warnings.extend(overlap_warnings)
        if not overlap_stats["passed"]:
            serious_error = True

        nsdgeneral_stats, nsdgeneral_warnings = check_nsdgeneral_overlap(
            roi_data, roi_indices, args.nsdgeneral_nii_path
        )
        warnings.extend(nsdgeneral_warnings)
        wholebrain_stats, wholebrain_warnings = check_wholebrain_extraction(
            roi_data.shape, roi_indices, args.wholebrain_path
        )
        warnings.extend(wholebrain_warnings)

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{args.subject}_{args.roi_name}"
        if args.save_pkl:
            save_indices_pkl(
                output_dir / f"{stem}_indices.pkl",
                args.subject,
                args.roi_name,
                roi_path,
                roi_data.shape,
                roi_indices,
            )
        if args.save_npz:
            save_indices_npz(
                output_dir / f"{stem}_indices.npz",
                roi_indices,
            )

        metadata = {
            "subject": args.subject,
            "roi_name": args.roi_name,
            "roi_path": str(roi_path),
            "roi_shape": list(roi_data.shape),
            "affine": np.asarray(roi_image.affine).tolist(),
            "zooms": [
                float(value) for value in roi_image.header.get_zooms()[:3]
            ],
            "valid_labels": valid_labels,
            "label_id_to_name": json_safe_label_mapping(label_id_to_name),
            "voxel_counts": {
                str(label): int(entry["num_voxels"])
                for label, entry in roi_indices.items()
            },
            "total_valid_roi_voxels": sum(
                int(entry["num_voxels"]) for entry in roi_indices.values()
            ),
            "excluded_labels": excluded_labels,
            "created_time": datetime.now(timezone.utc).isoformat(),
            "flat_index_order": "numpy_C_order",
            "roi_overlap": overlap_stats,
            "nsdgeneral_overlap": nsdgeneral_stats,
            "wholebrain_extraction": wholebrain_stats,
            "warnings": warnings,
        }
        save_metadata_json(output_dir / f"{stem}_meta.json", metadata)
        if args.make_qc_png:
            make_qc_png(
                output_dir / f"{stem}_qc.png",
                roi_data,
                args.subject,
                args.roi_name,
                len(valid_labels),
            )
        if args.make_3d_qc_png:
            make_3d_qc_png(
                output_dir / f"{stem}_3d_qc.png",
                roi_indices,
                roi_data.shape,
                args.subject,
                args.roi_name,
            )

        print_warnings(warnings)
        print("\nFinal conclusion")
        if serious_error:
            print(
                "Conclusion: FAILED. ROI indices should not be used until "
                "the issue is fixed."
            )
            raise SystemExit(1)
        if warnings:
            print(
                "Conclusion: ROI indices generated, but please review warnings "
                "about label names or nsdgeneral overlap."
            )
        else:
            print("Conclusion: ROI voxel indices generated successfully.")
            print(
                "The label-to-voxel mapping is valid for this ROI NIfTI and "
                "can be used for ROI-wise fMRI extraction."
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        print(
            "Conclusion: FAILED. ROI indices should not be used until "
            "the issue is fixed."
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
