#!/usr/bin/env python
"""Batch-prepare real NSD ROI mappings and validate multiple sample pairs."""

import argparse
import io
import json
import pickle
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np


DEFAULT_COLLECTIONS = [
    "prf-visualrois",
    "floc-faces",
    "floc-bodies",
    "floc-places",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare all label-level NeuroRoute ROI mappings for one subject"
    )
    parser.add_argument("--subject", required=True, help="e.g. subj01")
    parser.add_argument("--nsd-root", required=True)
    parser.add_argument("--webdataset-tar", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--collections", nargs="+", default=DEFAULT_COLLECTIONS)
    parser.add_argument("--num-validation-samples", type=int, default=3)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def subject_number(subject: str) -> str:
    digits = "".join(character for character in subject if character.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse subject number from {subject!r}")
    return f"{int(digits):02d}"


def run_command(command: List[str], expected_output=None) -> None:
    print("\n$", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        return
    if expected_output is not None and expected_output.is_file():
        print(
            f"[WARN] Command exited with code {result.returncode}, but "
            f"the expected partial-coverage output exists: {expected_output}"
        )
        return
    raise subprocess.CalledProcessError(result.returncode, command)


def iter_sample_pairs(
    tar_path: Path, limit: int
) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
    with tarfile.open(tar_path) as archive:
        names = archive.getnames()
        prefixes = sorted(
            name[: -len(".wholebrain_3d.npy")]
            for name in names
            if name.endswith(".wholebrain_3d.npy")
        )
        for prefix in prefixes[:limit]:
            wholebrain = np.load(
                io.BytesIO(
                    archive.extractfile(f"{prefix}.wholebrain_3d.npy").read()
                )
            )
            nsdgeneral = np.load(
                io.BytesIO(
                    archive.extractfile(f"{prefix}.nsdgeneral.npy").read()
                )
            )
            yield prefix, wholebrain, nsdgeneral


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.astype(np.float64, copy=False).ravel()
    right_flat = right.astype(np.float64, copy=False).ravel()
    finite = np.isfinite(left_flat) & np.isfinite(right_flat)
    left_flat, right_flat = left_flat[finite], right_flat[finite]
    if left_flat.size < 2 or left_flat.std() == 0 or right_flat.std() == 0:
        return float("nan")
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def validate_collection(
    mapping: Mapping[int, Mapping[str, object]],
    sample_pairs: List[Tuple[str, np.ndarray, np.ndarray]],
) -> Dict[str, object]:
    comparisons = []
    skipped_empty_labels = set()
    for sample_name, wholebrain, nsdgeneral in sample_pairs:
        for label, entry in mapping.items():
            x = np.asarray(entry["kept_x"], dtype=np.int64)
            y = np.asarray(entry["kept_y"], dtype=np.int64)
            z = np.asarray(entry["kept_z"], dtype=np.int64)
            indices = np.asarray(entry["nsdgeneral_indices"], dtype=np.int64)
            if indices.size == 0:
                skipped_empty_labels.add((int(label), str(entry["name"])))
                continue
            from_wholebrain = (
                wholebrain[x, y, z]
                if wholebrain.ndim == 3
                else wholebrain[:, x, y, z]
            )
            from_nsdgeneral = (
                nsdgeneral[indices]
                if nsdgeneral.ndim == 1
                else nsdgeneral[:, indices]
            )
            if from_wholebrain.shape != from_nsdgeneral.shape:
                raise ValueError(
                    f"{sample_name} label {label}: shape mismatch "
                    f"{from_wholebrain.shape} != {from_nsdgeneral.shape}"
                )
            difference = np.abs(
                from_wholebrain.astype(np.float64)
                - from_nsdgeneral.astype(np.float64)
            )
            finite_difference = difference[np.isfinite(difference)]
            comparisons.append(
                {
                    "sample": sample_name,
                    "label": int(label),
                    "name": entry["name"],
                    "num_repeats": (
                        int(from_wholebrain.shape[0])
                        if from_wholebrain.ndim == 2
                        else 1
                    ),
                    "pearson_correlation": pearson_correlation(
                        from_wholebrain, from_nsdgeneral
                    ),
                    "max_abs_error": (
                        float(finite_difference.max())
                        if finite_difference.size
                        else None
                    ),
                    "mean_abs_error": (
                        float(finite_difference.mean())
                        if finite_difference.size
                        else None
                    ),
                    "strict_allclose": bool(
                        np.allclose(
                            from_wholebrain,
                            from_nsdgeneral,
                            rtol=1e-5,
                            atol=1e-8,
                            equal_nan=True,
                        )
                    ),
                    "relaxed_allclose": bool(
                        np.allclose(
                            from_wholebrain,
                            from_nsdgeneral,
                            rtol=1e-3,
                            atol=2e-3,
                            equal_nan=True,
                        )
                    ),
                }
            )

    correlations = [
        item["pearson_correlation"]
        for item in comparisons
        if np.isfinite(item["pearson_correlation"])
    ]
    max_errors = [
        item["max_abs_error"]
        for item in comparisons
        if item["max_abs_error"] is not None
    ]
    mean_errors = [
        item["mean_abs_error"]
        for item in comparisons
        if item["mean_abs_error"] is not None
    ]
    return {
        "num_samples_checked": len(sample_pairs),
        "num_repeats_checked": sum(
            pair[1].shape[0] if pair[1].ndim == 4 else 1
            for pair in sample_pairs
        ),
        "num_label_comparisons": len(comparisons),
        "skipped_empty_labels": [
            {"label": label, "name": name}
            for label, name in sorted(skipped_empty_labels)
        ],
        "mean_pearson_correlation": (
            float(np.mean(correlations)) if correlations else None
        ),
        "max_abs_error": max(max_errors) if max_errors else None,
        "mean_abs_error": (
            float(np.mean(mean_errors)) if mean_errors else None
        ),
        "strict_allclose": all(
            item["strict_allclose"] for item in comparisons
        ),
        "relaxed_allclose": all(
            item["relaxed_allclose"] for item in comparisons
        ),
        "comparisons": comparisons,
    }


def aggregate_validation(
    subject: str,
    per_collection: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    validations = list(per_collection.values())
    correlations = [
        validation["mean_pearson_correlation"]
        for validation in validations
        if validation["mean_pearson_correlation"] is not None
    ]
    return {
        "subject": subject,
        "num_samples_checked": max(
            (validation["num_samples_checked"] for validation in validations),
            default=0,
        ),
        "num_repeats_checked": max(
            (validation["num_repeats_checked"] for validation in validations),
            default=0,
        ),
        "mean_pearson_correlation": (
            float(np.mean(correlations)) if correlations else None
        ),
        "max_abs_error": max(
            (
                validation["max_abs_error"]
                for validation in validations
                if validation["max_abs_error"] is not None
            ),
            default=None,
        ),
        "mean_abs_error": float(
            np.mean(
                [
                    validation["mean_abs_error"]
                    for validation in validations
                    if validation["mean_abs_error"] is not None
                ]
            )
        ),
        "strict_allclose": all(
            validation["strict_allclose"] for validation in validations
        ),
        "relaxed_allclose": all(
            validation["relaxed_allclose"] for validation in validations
        ),
        "voxel_order_verified": bool(
            validations
            and all(
                validation["num_samples_checked"] > 0
                and validation["relaxed_allclose"]
                for validation in validations
            )
        ),
        "per_collection": per_collection,
        "created_time": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    sub = subject_number(args.subject)
    nsd_root = Path(args.nsd_root).expanduser().resolve()
    tar_path = Path(args.webdataset_tar).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent

    sample_pairs = list(
        iter_sample_pairs(tar_path, args.num_validation_samples)
    )
    print(
        f"Loaded {len(sample_pairs)} real sample pairs from {tar_path}"
    )

    per_collection = {}
    for collection in args.collections:
        roi_path = (
            nsd_root
            / f"ppdata/subj{sub}/func1pt8mm/roi/{collection}.nii.gz"
        )
        ctab_path = (
            nsd_root
            / f"freesurfer/subj{sub}/label/{collection}.mgz.ctab"
        )
        nsdgeneral_path = (
            nsd_root
            / f"ppdata/subj{sub}/func1pt8mm/roi/nsdgeneral.nii.gz"
        )
        wholebrain_pkl = (
            output_dir / f"{args.subject}_{collection}_indices.pkl"
        )
        nsdgeneral_pkl = (
            output_dir
            / f"{args.subject}_{collection}_nsdgeneral_indices.pkl"
        )

        run_command(
            [
                sys.executable,
                str(script_dir / "prepare_roi_indices.py"),
                "--subject",
                args.subject,
                "--roi_name",
                collection,
                "--roi_path",
                str(roi_path),
                "--label_table_path",
                str(ctab_path),
                "--output_dir",
                str(output_dir),
                "--nsdgeneral_nii_path",
                str(nsdgeneral_path),
                "--save_pkl",
            ]
        )
        run_command(
            [
                sys.executable,
                str(script_dir / "prepare_roi_nsdgeneral_indices.py"),
                "--subject",
                args.subject,
                "--roi_name",
                collection,
                "--roi_indices_pkl",
                str(wholebrain_pkl),
                "--nsdgeneral_nii_path",
                str(nsdgeneral_path),
                "--output_dir",
                str(output_dir),
                "--save_pkl",
                "--save_npz",
            ],
            expected_output=nsdgeneral_pkl,
        )
        with nsdgeneral_pkl.open("rb") as file:
            mapping = pickle.load(file)
        validation = validate_collection(mapping, sample_pairs)
        per_collection[collection] = validation
        print(
            f"{collection}: samples={validation['num_samples_checked']}, "
            f"repeats={validation['num_repeats_checked']}, "
            f"pearson={validation['mean_pearson_correlation']:.10f}, "
            f"max_abs={validation['max_abs_error']:.10g}, "
            f"relaxed={validation['relaxed_allclose']}"
        )

    aggregate = aggregate_validation(args.subject, per_collection)
    validation_path = output_dir / f"{args.subject}_validation.json"
    validation_path.write_text(json.dumps(aggregate, indent=2))
    print(f"Saved validation: {validation_path}")
    print(
        f"voxel_order_verified={aggregate['voxel_order_verified']}, "
        f"mean_pearson={aggregate['mean_pearson_correlation']:.10f}, "
        f"max_abs_error={aggregate['max_abs_error']:.10g}"
    )
    if args.strict and not aggregate["voxel_order_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
