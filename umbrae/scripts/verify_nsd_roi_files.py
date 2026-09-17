#!/usr/bin/env python
"""Verify subject-specific NSD ROI files required for ROI tokenization."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import nibabel as nib
import numpy as np


EXPECTED_NSDGENERAL_VOXELS = {
    1: 15724,
    2: 14278,
    5: 13039,
    7: 12682,
}

DEFAULT_ROIS = [
    "nsdgeneral",
    "prf-visualrois",
    "floc-faces",
    "floc-places",
    "floc-bodies",
    "streams",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify NSD ppdata ROI files and nsdgeneral voxel counts"
    )
    parser.add_argument(
        "--nsd-root",
        required=True,
        help=(
            "NSD root containing ppdata/, or a parent containing nsddata/ppdata/"
        ),
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=[1, 2, 5, 7]
    )
    parser.add_argument("--space", default="func1pt8mm")
    parser.add_argument(
        "--required-rois", nargs="+", default=DEFAULT_ROIS
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="optional path for a machine-readable verification report",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="exit non-zero if files are missing or validation fails",
    )
    return parser.parse_args()


def candidate_ppdata_roots(nsd_root: Path) -> Iterable[Path]:
    """Yield common interpretations of an NSD root without recursive scanning."""
    candidates = [
        nsd_root / "ppdata",
        nsd_root / "nsddata" / "ppdata",
        nsd_root / "NSD" / "nsddata" / "ppdata",
    ]
    if nsd_root.name == "ppdata":
        candidates.insert(0, nsd_root)
    if nsd_root.name == "nsddata":
        candidates.insert(0, nsd_root / "ppdata")

    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def find_ppdata_root(nsd_root: Path, subjects: List[int], space: str) -> Path:
    candidates = list(candidate_ppdata_roots(nsd_root))
    for candidate in candidates:
        if any(
            (candidate / f"subj{subject:02d}" / space).is_dir()
            for subject in subjects
        ):
            return candidate
    # Return the conventional path so missing-file output remains actionable.
    return candidates[0]


def nifti_metadata(path: Path) -> Dict[str, object]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"expected 3D NIfTI, got shape {data.shape}")
    unique_labels = np.unique(data)
    return {
        "shape": list(data.shape),
        "affine": np.asarray(image.affine).tolist(),
        "unique_labels": [
            int(value) if float(value).is_integer() else float(value)
            for value in unique_labels
        ],
        "nonzero_voxel_count": int(np.count_nonzero(data)),
    }


def verify_subject(
    ppdata_root: Path,
    subject: int,
    space: str,
    required_rois: List[str],
) -> Dict[str, object]:
    roi_dir = ppdata_root / f"subj{subject:02d}" / space / "roi"
    expected_count = EXPECTED_NSDGENERAL_VOXELS.get(subject)
    subject_report: Dict[str, object] = {
        "subject": subject,
        "roi_dir": str(roi_dir),
        "expected_nsdgeneral_voxel_count": expected_count,
        "files": {},
        "errors": [],
    }

    reference_shape: Optional[List[int]] = None
    reference_affine: Optional[np.ndarray] = None
    ordered_rois = (
        ["nsdgeneral"]
        + [name for name in required_rois if name != "nsdgeneral"]
        if "nsdgeneral" in required_rois
        else list(required_rois)
    )
    for roi_name in ordered_rois:
        path = roi_dir / f"{roi_name}.nii.gz"
        file_report: Dict[str, object] = {
            "path": str(path),
            "found": path.is_file(),
        }
        if path.is_file():
            try:
                metadata = nifti_metadata(path)
                file_report.update(metadata)
                affine = np.asarray(metadata["affine"])
                if roi_name == "nsdgeneral":
                    reference_shape = metadata["shape"]
                    reference_affine = affine
                    actual_count = metadata["nonzero_voxel_count"]
                    file_report["expected_voxel_count"] = expected_count
                    file_report["voxel_count_matches"] = (
                        expected_count is None or actual_count == expected_count
                    )
                    if not file_report["voxel_count_matches"]:
                        subject_report["errors"].append(
                            f"nsdgeneral voxel count {actual_count} != "
                            f"expected {expected_count}"
                        )
                elif reference_shape is not None:
                    file_report["shape_matches_nsdgeneral"] = (
                        metadata["shape"] == reference_shape
                    )
                    file_report["affine_matches_nsdgeneral"] = bool(
                        np.allclose(affine, reference_affine, atol=1e-4)
                    )
                    if not file_report["shape_matches_nsdgeneral"]:
                        subject_report["errors"].append(
                            f"{roi_name} shape does not match nsdgeneral"
                        )
                    if not file_report["affine_matches_nsdgeneral"]:
                        subject_report["errors"].append(
                            f"{roi_name} affine does not match nsdgeneral"
                        )
            except Exception as exc:
                file_report["error"] = str(exc)
                subject_report["errors"].append(f"{roi_name}: {exc}")
        else:
            subject_report["errors"].append(f"missing {path}")
        subject_report["files"][roi_name] = file_report

    subject_report["valid"] = not subject_report["errors"]
    return subject_report


def print_subject_report(report: Dict[str, object]) -> None:
    subject = report["subject"]
    print(f"\nSubject S{subject}: {report['roi_dir']}")
    print(
        "Expected nsdgeneral voxels:",
        report["expected_nsdgeneral_voxel_count"],
    )
    for roi_name, file_report in report["files"].items():
        if not file_report["found"]:
            print(f"  [MISSING] {roi_name}: {file_report['path']}")
            continue
        if "error" in file_report:
            print(f"  [ERROR] {roi_name}: {file_report['error']}")
            continue
        print(f"  [FOUND] {roi_name}: {file_report['path']}")
        print(f"    shape: {tuple(file_report['shape'])}")
        print("    affine:")
        for row in file_report["affine"]:
            print(f"      {row}")
        print(f"    unique labels: {file_report['unique_labels']}")
        if roi_name == "nsdgeneral":
            print(
                "    mask voxels:",
                file_report["nonzero_voxel_count"],
                "match:",
                file_report["voxel_count_matches"],
            )
    print("  status:", "PASS" if report["valid"] else "FAIL")


def main() -> None:
    args = parse_args()
    nsd_root = Path(args.nsd_root).expanduser().resolve()
    ppdata_root = find_ppdata_root(nsd_root, args.subjects, args.space)
    print(f"NSD root: {nsd_root}")
    print(f"Resolved ppdata root: {ppdata_root}")

    reports = [
        verify_subject(
            ppdata_root,
            subject,
            args.space,
            args.required_rois,
        )
        for subject in args.subjects
    ]
    for report in reports:
        print_subject_report(report)

    complete_subjects = [report["subject"] for report in reports if report["valid"]]
    payload = {
        "nsd_root": str(nsd_root),
        "ppdata_root": str(ppdata_root),
        "space": args.space,
        "required_rois": args.required_rois,
        "complete_subjects": complete_subjects,
        "all_valid": len(complete_subjects) == len(reports),
        "subjects": reports,
    }
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved report: {report_path}")

    print(
        f"\nSummary: {len(complete_subjects)}/{len(reports)} subjects passed"
    )
    if args.strict and not payload["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
