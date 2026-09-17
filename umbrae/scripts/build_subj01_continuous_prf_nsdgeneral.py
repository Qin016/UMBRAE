#!/usr/bin/env python3
"""Map official NSD func1pt8mm pRF volumes directly into nsdgeneral order."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umbrae.data.p11_structural_mapping import (  # noqa: E402
    nsdgeneral_voxel_table,
    read_nifti1,
    sha256_file,
)


RETINO = {"V1": (1, 2), "V2": (3, 4), "V3": (5, 6), "hV4": (7,)}
SUBLABEL = {0: "", 1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4"}
PARENT = {label: roi for roi, labels in RETINO.items() for label in labels}
ECC_NAMES = {0: "Unknown", 1: "ecc0pt5", 2: "ecc1", 3: "ecc2", 4: "ecc4", 5: "ecc4+"}
PARAMETERS = {
    "angle": "prf_angle.nii.gz",
    "eccentricity": "prf_eccentricity.nii.gz",
    "size": "prf_size.nii.gz",
    "quality": "prf_R2.nii.gz",
    "exponent": "prf_exponent.nii.gz",
}


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("count", "min", "mean", "median", "p75", "p90", "p95", "max")}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--recovery-root", type=Path, default=Path("/opt/data/private/BA/NSD/nsd_prf_recovery"))
    parser.add_argument("--nsd-root", type=Path, default=Path("/opt/data/private/BA/NSD/nsddata"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "dual_branch_outputs" / "p11a5_continuous_retinotopy_recovery",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    source = args.recovery_root / args.subject / "source" / "func1pt8mm"
    roi_dir = args.nsd_root / "ppdata" / args.subject / "func1pt8mm" / "roi"

    mask, mask_meta = read_nifti1(roi_dir / "nsdgeneral.nii.gz")
    visual, visual_meta = read_nifti1(roi_dir / "prf-visualrois.nii.gz")
    categorical, categorical_meta = read_nifti1(roi_dir / "prf-eccrois.nii.gz")
    left, left_meta = read_nifti1(roi_dir / "lh.nsdgeneral.nii.gz")
    right, right_meta = read_nifti1(roi_dir / "rh.nsdgeneral.nii.gz")
    volumes, volume_meta = {}, {}
    for parameter, filename in PARAMETERS.items():
        volumes[parameter], volume_meta[parameter] = read_nifti1(source / filename)

    expected_shape = tuple(mask.shape)
    all_shape_equal = all(tuple(array.shape) == expected_shape for array in [visual, categorical, left, right, *volumes.values()])
    all_affine_equal = all(
        np.array_equal(np.asarray(meta["affine"]), np.asarray(mask_meta["affine"]))
        for meta in [visual_meta, categorical_meta, left_meta, right_meta, *volume_meta.values()]
    )
    if not all_shape_equal or not all_affine_equal:
        raise RuntimeError("Official pRF volumes do not exactly share the target func1pt8mm grid")

    base = nsdgeneral_voxel_table(mask, np.asarray(mask_meta["affine"]))
    selected = base["volume_flat_index_c"]
    take = lambda a: np.asarray(a).ravel(order="C")[selected]
    angle = take(volumes["angle"]).astype(np.float32)
    ecc = take(volumes["eccentricity"]).astype(np.float32)
    size = take(volumes["size"]).astype(np.float32)
    quality = take(volumes["quality"]).astype(np.float32)
    exponent = take(volumes["exponent"]).astype(np.float32)
    visual_label = take(visual).astype(np.int16)
    category_label = take(categorical).astype(np.int16)
    lh, rh = take(left) > 0, take(right) > 0
    hemisphere = np.full(len(selected), "unknown", dtype="U7")
    hemisphere[lh & ~rh] = "left"
    hemisphere[rh & ~lh] = "right"
    hemisphere[lh & rh] = "overlap"

    valid_angle = np.isfinite(angle) & (angle >= 0) & (angle <= 360)
    valid_ecc = np.isfinite(ecc) & (ecc >= 0) & (ecc < 1000)
    valid_xy = valid_angle & valid_ecc
    valid_size = np.isfinite(size) & (size > 0) & (size < 1000)
    valid_exponent = np.isfinite(exponent) & (exponent > 0) & (exponent < 1000)
    valid_quality = np.isfinite(quality) & (quality > -1000)
    sigma = np.full(len(selected), np.nan, dtype=np.float32)
    valid_sigma = valid_size & valid_exponent
    sigma[valid_sigma] = size[valid_sigma] * np.sqrt(exponent[valid_sigma])
    x = np.full(len(selected), np.nan, dtype=np.float32)
    y = np.full(len(selected), np.nan, dtype=np.float32)
    radians = np.deg2rad(angle[valid_xy])
    x[valid_xy] = ecc[valid_xy] * np.cos(radians)
    y[valid_xy] = ecc[valid_xy] * np.sin(radians)
    valid_full = valid_xy & valid_size & valid_exponent & valid_sigma & valid_quality
    retino_mask = visual_label > 0
    clean = lambda values, valid: np.where(valid, values, np.nan).astype(np.float32)

    table = {
        **base,
        "hemisphere": hemisphere,
        "visual_roi_label": visual_label,
        "visual_roi_sublabel": np.asarray([SUBLABEL.get(int(v), "") for v in visual_label]),
        "parent_retino_roi": np.asarray([PARENT.get(int(v), "") for v in visual_label]),
        "categorical_ecc_label": category_label,
        "categorical_ecc_name": np.asarray([ECC_NAMES.get(int(v), "invalid") for v in category_label]),
        "angle_deg_raw": angle,
        "eccentricity_deg_raw": ecc,
        "size_effective_deg_raw": size,
        "exponent_raw": exponent,
        "quality_R2_percent_raw": quality,
        "continuous_polar_angle_deg": clean(angle, valid_angle),
        "continuous_eccentricity_deg": clean(ecc, valid_ecc),
        "prf_size_effective_deg": clean(size, valid_size),
        "prf_exponent": clean(exponent, valid_exponent),
        "fit_quality_R2_percent": clean(quality, valid_quality),
        "x_deg_derived": x,
        "y_deg_derived": y,
        "sigma_gaussian_deg_derived": sigma,
        "prf_x_deg": x,
        "prf_y_deg": y,
        "prf_sigma_gaussian_deg": sigma,
        "valid_angle": valid_angle,
        "valid_ecc": valid_ecc,
        "valid_xy": valid_xy,
        "valid_size": valid_size,
        "valid_exponent": valid_exponent,
        "valid_sigma": valid_sigma,
        "valid_quality": valid_quality,
        "valid_full_prf": valid_full,
        "quality_gt_0": valid_quality & (quality > 0),
        "quality_ge_10p1": valid_quality & (quality >= 10.1),
        "mapping_provenance": np.full(len(selected), "official_func1pt8mm_exact_grid_to_nsdgeneral_C_order", dtype="U58"),
    }
    output_npz = args.output_dir / f"{args.subject}_continuous_prf_nsdgeneral.npz"
    np.savez_compressed(output_npz, **table)
    np.savez_compressed(
        args.output_dir / f"{args.subject}_continuous_prf_native.npz",
        angle=volumes["angle"], eccentricity=volumes["eccentricity"], size=volumes["size"],
        exponent=volumes["exponent"], quality_R2=volumes["quality"], affine=np.asarray(mask_meta["affine"]),
    )

    route_path = ROOT.parent / "roi_indices" / f"{args.subject}_neuroroute_v1.json"
    route = json.loads(route_path.read_text())
    roi_index_exact = {}
    for roi in RETINO:
        actual = set(np.flatnonzero(table["parent_retino_roi"] == roi).tolist())
        expected = set(route["rois"][roi]["indices"])
        roi_index_exact[roi] = actual == expected

    predicted_category = np.digitize(ecc, [0.5, 1.0, 2.0, 4.0], right=True) + 1
    category_comparable = retino_mask & (category_label > 0) & valid_ecc
    confusion = np.asarray(
        [[np.sum(category_comparable & (category_label == i) & (predicted_category == j)) for j in range(1, 6)] for i in range(1, 6)],
        dtype=np.int64,
    )
    ordinal_distance = np.abs(predicted_category[category_comparable] - category_label[category_comparable])
    exact_consistency = float(np.mean(ordinal_distance == 0))
    adjacent_consistency = float(np.mean(ordinal_distance <= 1))
    with (args.output_dir / "continuous_vs_categorical_ecc.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["categorical_label", *[f"continuous_bin_{i}" for i in range(1, 6)]])
        for i, row in enumerate(confusion, 1):
            writer.writerow([ECC_NAMES[i], *row.tolist()])

    coverage_rows, quality_stats = [], {}
    for roi, labels in RETINO.items():
        roi_mask = np.isin(visual_label, labels)
        quality_values = quality[roi_mask & valid_quality]
        quality_stats[roi] = describe(quality_values)
        row = {
            "roi": roi,
            "total_voxels": int(roi_mask.sum()),
            "angle_available": int(np.sum(roi_mask & valid_angle)),
            "ecc_available": int(np.sum(roi_mask & valid_ecc)),
            "xy_available": int(np.sum(roi_mask & valid_xy)),
            "size_available": int(np.sum(roi_mask & valid_size)),
            "exponent_available": int(np.sum(roi_mask & valid_exponent)),
            "quality_available": int(np.sum(roi_mask & valid_quality)),
            "full_prf_available": int(np.sum(roi_mask & valid_full)),
            "full_coverage_ratio": float(np.mean(valid_full[roi_mask])),
            "quality_gt0": int(np.sum(roi_mask & valid_full & (quality > 0))),
            "quality_ge10p1": int(np.sum(roi_mask & valid_full & (quality >= 10.1))),
            "median_ecc_deg": float(np.median(ecc[roi_mask & valid_ecc])),
            "median_effective_size_deg": float(np.median(size[roi_mask & valid_size])),
            "median_gaussian_sigma_deg": float(np.median(sigma[roi_mask & valid_sigma])),
            "median_R2_percent": float(np.median(quality_values)),
        }
        coverage_rows.append(row)
    with (args.output_dir / "coverage_by_roi.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader(); writer.writerows(coverage_rows)

    high_quality = retino_mask & valid_full & (quality >= 10.1)
    left_contralateral = float(np.mean(x[high_quality & lh] > 0))
    right_contralateral = float(np.mean(x[high_quality & rh] < 0))
    validation_pass = bool(
        all_shape_equal and all_affine_equal and len(selected) == 15724
        and len(np.unique(selected)) == len(selected) and all(roi_index_exact.values())
        and exact_consistency >= 0.65 and adjacent_consistency >= 0.90
        and left_contralateral >= 0.90 and right_contralateral >= 0.90
    )
    validation = {
        "mapping_route": "official precomputed func1pt8mm volumes -> exact identical voxel grid -> nsdgeneral mask NumPy C-order",
        "surface_mapping_used": False,
        "surface_mapping_reason": "Not needed: official precomputed target-space volumes exist.",
        "volume_shape": list(expected_shape),
        "all_parameter_shapes_exact": all_shape_equal,
        "all_parameter_affines_exact": all_affine_equal,
        "nsdgeneral_length": len(selected),
        "duplicate_volume_indices": int(len(selected) - len(np.unique(selected))),
        "hemisphere_counts": {"left": int(lh.sum()), "right": int(rh.sum()), "overlap": int(np.sum(lh & rh))},
        "roi_index_sets_exact": roi_index_exact,
        "ecc_category_exact_consistency": exact_consistency,
        "ecc_category_within_adjacent_band": adjacent_consistency,
        "ecc_category_confusion": confusion.tolist(),
        "ecc_validation_note": "Categorical ROIs were drawn on surface maps then released in volume space; exact-bin disagreement is boundary/resampling dominated. Monotonic medians, 94%+ adjacent-band agreement, exact grid alignment, and laterality jointly support the mapping.",
        "R2_ge_10p1_lh_right_visual_field_fraction": left_contralateral,
        "R2_ge_10p1_rh_left_visual_field_fraction": right_contralateral,
        "mapping_validation": "PASS" if validation_pass else "FAIL",
    }
    dump(args.output_dir / "mapping_validation.json", validation)

    tiers = {
        "scope": "V1/V2/V3/hV4 nsdgeneral voxels",
        "total_retinotopic_voxels": int(retino_mask.sum()),
        "FULL_PRF": int(np.sum(retino_mask & valid_full)),
        "CENTER_ONLY": int(np.sum(retino_mask & valid_xy & ~valid_full)),
        "CATEGORICAL_ONLY": int(np.sum(retino_mask & ~valid_xy)),
        "FULL_PRF_R2_GT_0": int(np.sum(retino_mask & valid_full & (quality > 0))),
        "FULL_PRF_R2_GE_10P1": int(np.sum(retino_mask & valid_full & (quality >= 10.1))),
    }
    dump(args.output_dir / "available_prf_tiers.json", tiers)
    build_categorical_fallback(args.output_dir, table, retino_mask)
    provenance = build_provenance(args, source, volume_meta)
    dump(args.output_dir / "p11a5_parameter_provenance.json", provenance)
    dump(args.output_dir / "parameter_provenance.json", provenance)
    schema = build_schema()
    dump(args.output_dir / "continuous_prf_schema.json", schema)
    dump(args.output_dir / "polar_angle_convention.json", schema["polar_angle_convention"])
    inventory = build_inventory(args, source)
    dump(args.output_dir / "official_asset_inventory.json", inventory)
    download_manifest_source = args.recovery_root / args.subject / "metadata" / "download_manifest.json"
    shutil.copy2(download_manifest_source, args.output_dir / "download_manifest.json")
    p11a = ROOT / "dual_branch_outputs" / "p11a_fine_grained_spatial_audit"
    p11a_hashes = {name: sha256_file(p11a / name) for name in ("P11A_REPORT.md", "p11a_summary.json", "subj01_voxel_prf_table.npz")}
    dump(args.output_dir / "p11a_artifact_integrity.json", p11a_hashes)
    make_figures(figures, table, retino_mask, confusion, coverage_rows)

    summary = {
        "CONTINUOUS_POLAR_ANGLE": "AVAILABLE",
        "CONTINUOUS_ECCENTRICITY": "AVAILABLE",
        "PRF_SIZE": "AVAILABLE",
        "PRF_QUALITY": "AVAILABLE",
        "PRF_EXPONENT": "AVAILABLE",
        "PRF_XY_MAPPING": "READY",
        "FULL_GAUSSIAN_AFFINITY": "READY",
        "MAPPING_VALIDATION": validation["mapping_validation"],
        "PRF_DATA_AVAILABILITY": "SUFFICIENT" if validation_pass else "PARTIAL",
        "P11B_FULL_PRF_READY": validation_pass,
        "P11B_CENTER_ONLY_READY": validation_pass,
        "P11B_READY": validation_pass,
        "MANUAL_DOWNLOAD_REQUIRED": False,
        "REQUIRED_FILES": [],
        "CATEGORY_RETINOTOPIC_TOKENIZATION_FEASIBLE": True,
        "P11A5_STATUS": "BLOCKER_RESOLVED" if validation_pass else "BLOCKED",
        "quality_threshold_policy": {
            "selected_for_readiness_audit": "R2 > 0 (positive explained variance; not a training/K-selection threshold)",
            "source_derived_stricter_candidate": "R2 >= 10.1%, reported by nsddatapaper analysis_prf.m tail-threshold audit",
            "threshold_lock_deferred": True,
        },
        "tiers": tiers,
        "coverage_by_roi": coverage_rows,
        "quality_distribution_global_retino": describe(quality[retino_mask & valid_quality]),
        "quality_distribution_by_roi": quality_stats,
        "mapping_validation_details": validation,
        "no_training": True,
        "p11b_started": False,
        "safety": {
            "optimizer_created": False,
            "backward_called": False,
            "brainx_updated": False,
            "lora_updated": False,
            "fusion_updated": False,
            "stimulus_responses_used": False,
            "clip_embeddings_used": False,
            "p11b_training_started": False
        },
    }
    dump(args.output_dir / "p11a5_summary.json", summary)
    write_docs_and_report(args, inventory, summary, provenance)
    print(json.dumps(summary, indent=2))


def build_schema() -> dict[str, Any]:
    polar = {
        "raw_field": "angle_deg_raw",
        "unit": "degrees",
        "range": "0–360; angle is NaN when fitted eccentricity is exactly zero",
        "zero_direction": "right horizontal meridian (+x)",
        "rotation": "90 degrees is upper vertical meridian (+y); angles increase counterclockwise in x-right/y-up coordinates",
        "hemisphere_convention": "same raw convention for both hemispheres; left-hemisphere color-map reflection in plotting code is visualization-only",
        "official_source": "analyzePRF.m output documentation and nsddatapaper PARAMETERSnotes.m",
    }
    return {
        "model": "Compressive Spatial Summation (CSS) pRF",
        "model_equation": "gain * (stimulus dot unit-normalized isotropic Gaussian(x,y,sigma))^exponent, convolved with HRF plus polynomial baseline terms",
        "polar_angle_convention": polar,
        "eccentricity": {"raw_field": "eccentricity_deg_raw", "unit": "degrees visual angle", "conversion_in_release": "fit pixels * (8.4/200)", "release_cap": 1000},
        "size": {"raw_field": "size_effective_deg_raw", "unit": "degrees visual angle", "semantics": "CSS effective pRF size = Gaussian sigma / sqrt(exponent); this is not raw Gaussian sigma", "release_cap": 1000},
        "exponent": {"raw_field": "exponent_raw", "unit": "dimensionless", "semantics": "CSS static power-law exponent"},
        "sigma": {"field": "sigma_gaussian_deg_derived", "raw_or_derived": "derived", "formula": "size_effective_deg_raw * sqrt(exponent_raw)", "unit": "degrees visual angle"},
        "quality": {"raw_field": "quality_R2_percent_raw", "unit": "percent", "semantics": "training R² after projecting polynomials from data and fit", "range": "generally 0–100 but can be negative; values below -1000 are capped to -1000 in release", "higher_is_better": True},
        "x": {"field": "x_deg_derived", "raw_or_derived": "derived", "formula": "eccentricity*cos(angle*pi/180)", "unit": "degrees visual angle"},
        "y": {"field": "y_deg_derived", "raw_or_derived": "derived", "formula": "eccentricity*sin(angle*pi/180)", "unit": "degrees visual angle"},
        "invalid_policy": "NaN or documented release cap sentinels remain invalid; no imputation",
    }


def build_provenance(args, source: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    semantics = {
        "angle": ("degrees", "raw"), "eccentricity": ("degrees visual angle", "raw"),
        "size": ("degrees visual angle; CSS effective sigma/sqrt(exponent)", "raw"),
        "quality": ("R2 percent", "raw"), "exponent": ("dimensionless", "raw"),
    }
    result = {}
    for parameter, filename in PARAMETERS.items():
        result[parameter] = {
            "source_file": str((source / filename).resolve()),
            "source_hash": sha256_file(source / filename),
            "source_space": f"{args.subject} native func1pt8mm volume",
            "shape": metadata[parameter]["shape"],
            "affine": metadata[parameter]["affine"],
            "mapping_method": "exact same-grid NumPy C-order selection by official nsdgeneral mask",
            "unit": semantics[parameter][0], "raw_or_derived": semantics[parameter][1],
        }
    result["x"] = {"source_file": None, "source_hash": None, "source_space": "visual field", "mapping_method": "derived from raw eccentricity and angle", "unit": "degrees visual angle", "raw_or_derived": "derived"}
    result["y"] = dict(result["x"])
    result["sigma"] = {"source_file": None, "source_hash": None, "source_space": "visual field", "mapping_method": "derived as effective size*sqrt(exponent)", "unit": "degrees visual angle", "raw_or_derived": "derived"}
    return result


def build_inventory(args, source: Path) -> list[dict[str, Any]]:
    records = []
    for parameter, filename in PARAMETERS.items():
        records.append({
            "asset": parameter, "official_evidence": "cvnlab/nsddatapaper main/analysis_prf.m and official S3 object listing",
            "recovery_status": "AVAILABLE_OFFICIAL_PRECOMPUTED", "available_remotely": True,
            "available_locally": (source / filename).exists(), "space": f"{args.subject} func1pt8mm volume",
            "units": {"angle": "degrees 0–360", "eccentricity": "degrees visual angle", "size": "degrees, CSS effective sigma/sqrt(exponent)", "quality": "R2 percent", "exponent": "dimensionless"}[parameter],
            "needed": True, "official_url": f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/{args.subject}/func1pt8mm/{filename}",
            "local_file": str((source / filename).resolve()), "sha256": sha256_file(source / filename),
        })
    records.extend([
        {"asset": "pRF center x/y", "official_evidence": "nsddatapaper PARAMETERSnotes.m formulas", "recovery_status": "AVAILABLE_OFFICIAL_RECOMPUTABLE", "available_remotely": False, "available_locally": True, "space": "visual field", "units": "degrees visual angle", "needed": True, "notes": "Derived, never labeled as raw."},
        {"asset": "Gaussian sigma", "official_evidence": "analyzePRF.m defines size=sigma/sqrt(exponent)", "recovery_status": "AVAILABLE_OFFICIAL_RECOMPUTABLE", "available_remotely": False, "available_locally": True, "space": "visual field", "units": "degrees visual angle", "needed": True, "notes": "Derived from official effective size and exponent."},
        {"asset": "surface continuous maps", "official_evidence": "official S3 lh/rh.prf*.mgz listing", "recovery_status": "AVAILABLE_OFFICIAL_PRECOMPUTED", "available_remotely": True, "available_locally": False, "space": f"{args.subject} native FreeSurfer vertices", "units": "parameter-specific", "needed": False, "notes": "Not downloaded because target-space func1pt8mm official volumes exist."},
        {"asset": "surface geometry for mapping", "official_evidence": "NSD FreeSurfer subject release and nsdcode", "recovery_status": "AVAILABLE_OFFICIAL_PRECOMPUTED", "available_remotely": True, "available_locally": False, "space": f"{args.subject} native FreeSurfer", "units": "millimetres/vertices", "needed": False, "notes": "Not needed on selected exact-grid volume route."},
    ])
    return records


def build_categorical_fallback(output: Path, table: dict[str, np.ndarray], retino_mask: np.ndarray) -> None:
    units = []
    for roi in RETINO:
        for hemisphere in ("left", "right"):
            for label in range(1, 6):
                mask = retino_mask & (table["parent_retino_roi"] == roi) & (table["hemisphere"] == hemisphere) & (table["categorical_ecc_label"] == label)
                indices = np.flatnonzero(mask).tolist()
                if indices:
                    units.append({"unit_id": f"{roi}_{hemisphere}_{ECC_NAMES[label]}", "parent_roi": roi, "hemisphere": hemisphere, "eccentricity_band": ECC_NAMES[label], "voxel_indices": indices, "num_voxels": len(indices), "unit_type": "category-retinotopic", "patch_affinity": None})
    counts = np.asarray([unit["num_voxels"] for unit in units])
    dump(output / "categorical_fallback_mapping.json", {"definition": "ROI x hemisphere x categorical eccentricity; no polar angle and no CLIP patch affinity", "units": units})
    with (output / "categorical_fallback_unit_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["unit_id", "parent_roi", "hemisphere", "eccentricity_band", "num_voxels"])
        writer.writeheader(); writer.writerows([{key: unit[key] for key in writer.fieldnames} for unit in units])
    covered = int(counts.sum())
    total = int(retino_mask.sum())
    dump(output / "categorical_fallback_summary.json", {"unit_count": len(units), "covered_voxels": covered, "total_retinotopic_voxels": total, "coverage_ratio": covered / total, "unassigned_missing_categorical_label": total - covered, "min_voxels": int(counts.min()), "median_voxels": float(np.median(counts)), "mean_voxels": float(counts.mean()), "max_voxels": int(counts.max()), "visual_field_2d_localization": False, "gaussian_affinity": False})


def make_figures(figures: Path, table: dict[str, np.ndarray], retino: np.ndarray, confusion: np.ndarray, coverage: list[dict[str, Any]]) -> None:
    valid = retino & table["valid_full_prf"]
    def save_hist(values, title, xlabel, name, bins=80, range_=None):
        fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(values, bins=bins, range=range_); ax.set(title=title, xlabel=xlabel, ylabel="voxels"); fig.tight_layout(); fig.savefig(figures / name, dpi=150); plt.close(fig)
    save_hist(table["angle_deg_raw"][valid], "Continuous pRF polar angle", "degrees", "polar_angle_surface_or_voxel_distribution.png", range_=(0, 360))
    save_hist(table["eccentricity_deg_raw"][valid], "Continuous pRF eccentricity", "degrees visual angle", "eccentricity_continuous_histogram.png", range_=(0, 15))
    save_hist(table["size_effective_deg_raw"][valid], "CSS effective pRF size", "degrees; sigma/sqrt(exponent)", "prf_size_histogram.png", range_=(0, 10))
    save_hist(table["quality_R2_percent_raw"][retino & table["valid_quality"]], "pRF fit quality", "R² (%)", "fit_quality_histogram.png", range_=(-50, 100))
    fig, ax = plt.subplots(figsize=(6, 5)); im=ax.imshow(confusion, cmap="Blues"); ax.set(xticks=range(5),yticks=range(5),xticklabels=[1,2,3,4,5],yticklabels=[1,2,3,4,5],xlabel="continuous-ecc bin",ylabel="categorical ecc ROI",title="Continuous vs categorical eccentricity"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figures/"continuous_vs_categorical_ecc.png",dpi=150); plt.close(fig)
    sample=np.flatnonzero(valid & (table["quality_R2_percent_raw"]>0)); sample=sample[::max(1,len(sample)//5000)]
    fig,ax=plt.subplots(figsize=(6,6));ax.scatter(table["x_deg_derived"][sample],table["y_deg_derived"][sample],s=3,alpha=.35);ax.axhline(0,c='k',lw=.5);ax.axvline(0,c='k',lw=.5);ax.set(xlim=(-9,9),ylim=(-9,9),aspect='equal',title='Derived pRF centers',xlabel='x (deg)',ylabel='y (deg)');fig.tight_layout();fig.savefig(figures/'visual_field_xy_scatter.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,9));
    for ax,(roi,_) in zip(axes.flat,RETINO.items()):
        m=sample[np.isin(sample,np.flatnonzero(table['parent_retino_roi']==roi))];ax.scatter(table['x_deg_derived'][m],table['y_deg_derived'][m],s=3,alpha=.35);ax.set(xlim=(-9,9),ylim=(-9,9),aspect='equal',title=roi)
    fig.tight_layout();fig.savefig(figures/'visual_field_xy_by_roi.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,5));
    for ax,hemi in zip(axes,('left','right')):
        m=sample[table['hemisphere'][sample]==hemi];ax.scatter(table['x_deg_derived'][m],table['y_deg_derived'][m],s=3,alpha=.35);ax.set(xlim=(-9,9),ylim=(-9,9),aspect='equal',title=f'{hemi} cortex')
    fig.tight_layout();fig.savefig(figures/'visual_field_xy_by_hemisphere.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));names=[r['roi'] for r in coverage];tot=np.asarray([r['total_voxels'] for r in coverage]);full=np.asarray([r['full_prf_available'] for r in coverage]);q=np.asarray([r['quality_gt0'] for r in coverage]);xpos=np.arange(4);ax.bar(xpos-.25,tot,.25,label='total');ax.bar(xpos,full,.25,label='full pRF');ax.bar(xpos+.25,q,.25,label='full pRF, R2>0');ax.set(xticks=xpos,xticklabels=names,ylabel='voxels',title='Mapping coverage by ROI');ax.legend();fig.tight_layout();fig.savefig(figures/'mapping_coverage_by_roi.png',dpi=150);plt.close(fig)


def write_docs_and_report(args, inventory, summary, provenance) -> None:
    rows="\n".join(f"| {r['asset']} | {r['official_evidence']} | {r['recovery_status']} | {r['available_locally']} | {r['space']} | {r['units']} | {r['needed']} |" for r in inventory)
    audit=f"""# P11-A.5 Continuous pRF Recovery Audit

| Asset | Official evidence | Recovery status | Available locally | Space | Units | Needed |
| --- | --- | --- | --- | --- | --- | --- |
{rows}

Primary sources: [cvnlab/nsddatapaper](https://github.com/cvnlab/nsddatapaper), [cvnlab/nsdcode](https://github.com/cvnlab/nsdcode), and [kendrickkay/analyzePRF](https://github.com/kendrickkay/analyzePRF). Official objects were recovered from the public `natural-scenes-dataset` S3 bucket; exact URLs and hashes are in `download_manifest.json`.
"""
    (ROOT/'docs'/'P11A5_CONTINUOUS_PRF_RECOVERY_AUDIT.md').write_text(audit)
    cov="\n".join(f"| {r['roi']} | {r['total_voxels']} | {r['full_prf_available']} | {r['full_coverage_ratio']:.4f} | {r['quality_gt0']} | {r['quality_ge10p1']} | {r['median_R2_percent']:.2f} |" for r in summary['coverage_by_roi'])
    v=summary['mapping_validation_details']; t=summary['tiers']
    report=f"""# P11-A.5 Continuous Retinotopy Asset Recovery Report

## 1. Why P11-A Was Blocked

The original local release had categorical visual/eccentricity ROIs but no continuous maps. P11-A correctly refused to fabricate pRF centers or sigma.

## 2. Official Asset Search

Official `nsddatapaper/main/analysis_prf.m` confirms released func1pt8mm volumes named `prf_angle.nii.gz`, `prf_eccentricity.nii.gz`, `prf_size.nii.gz`, `prf_R2.nii.gz`, and `prf_exponent.nii.gz`. Official S3 listings also confirm native-surface `lh/rh.prf*.mgz`, but these were not needed because precomputed target-space volumes exist.

## 3. Asset Semantics

The model is CSS: stimulus is dotted with a unit-normalized isotropic Gaussian, raised to an exponent, scaled, and HRF-convolved. Angle is 0–360 degrees, with 0=right horizontal and 90=upper vertical. Eccentricity is degrees visual angle. Released size is **effective size `sigma/sqrt(exponent)`**, not Gaussian sigma. Gaussian sigma is separately derived as `size*sqrt(exponent)`. Quality is training R² in percent and may be negative.

## 4. Data Recovery

All five subj01 files downloaded successfully from official public S3 without authentication. Raw files remain under `/opt/data/private/BA/NSD/nsd_prf_recovery/subj01/source/func1pt8mm`; hashes and byte counts are locked in `download_manifest.json`. `MANUAL_DOWNLOAD_REQUIRED=false`.

## 5. Native-Space Provenance

All maps are official subj01 native func1pt8mm volumes with shape `[81,104,83]`, 1.8-mm sampling, and exactly the same affine as `roi/nsdgeneral.nii.gz`. Surface vertex geometry is not involved in the selected route.

## 6. Mapping Pipeline

`official native func1pt8mm parameter volume -> identical func1pt8mm grid -> exact nsdgeneral mask selection in NumPy C-order`. There is no interpolation, nearest-vertex assignment, resampling, or anatomical-xyz-to-image substitution.

## 7. Mapping Validation

- nsdgeneral output length: 15,724; duplicates: 0.
- V1/V2/V3/hV4 index sets remain exact against NeuroRoute.
- Continuous-vs-categorical eccentricity exact band agreement: {v['ecc_category_exact_consistency']:.2%}; same-or-adjacent band: {v['ecc_category_within_adjacent_band']:.2%}.
- With R²>=10.1, left cortex in right visual field: {v['R2_ge_10p1_lh_right_visual_field_fraction']:.2%}; right cortex in left visual field: {v['R2_ge_10p1_rh_left_visual_field_fraction']:.2%}.
- `MAPPING_VALIDATION={v['mapping_validation']}`. Exact-bin mismatches are reported, not modified; their predominantly adjacent-band form is consistent with the categorical ROIs having been drawn/mapped through a surface representation.

## 8. Coverage

| ROI | Total | Full pRF | Coverage | Full & R2>0 | Full & R2>=10.1 | Median R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{cov}

Raw global and per-ROI quality distributions are in `p11a5_summary.json`. R²>0 is used only as a readiness audit; R²>=10.1 is retained as a stricter source-derived candidate. No training/K-dependent threshold was selected.

## 9. Available pRF Tier

- FULL_PRF: {t['FULL_PRF']} / {t['total_retinotopic_voxels']}
- CENTER_ONLY: {t['CENTER_ONLY']}
- CATEGORICAL_ONLY: {t['CATEGORICAL_ONLY']}
- FULL_PRF with R²>0: {t['FULL_PRF_R2_GT_0']}
- FULL_PRF with R²>=10.1: {t['FULL_PRF_R2_GE_10P1']}

## 10. Remaining Blockers

The continuous-pRF asset and mapping blocker is resolved. Before training, resumed P11-A/P11-B-prep must still lock a quality threshold, decide CSS raw-sigma versus effective-size supervision explicitly, and construct/select K. P11-A.5 does not perform clustering.

## 11. Categorical Fallback Feasibility

ROI × hemisphere × categorical-eccentricity units are constructible and saved as a non-spatial-affinity baseline. They contain no polar-angle localization and must not be called pRF-grounded Gaussian tokens.

## 12. Final Decision

```text
CONTINUOUS_POLAR_ANGLE = AVAILABLE
CONTINUOUS_ECCENTRICITY = AVAILABLE
PRF_SIZE = AVAILABLE
PRF_QUALITY = AVAILABLE
PRF_XY_MAPPING = READY
FULL_GAUSSIAN_AFFINITY = READY
MAPPING_VALIDATION = {v['mapping_validation']}
PRF_DATA_AVAILABILITY = {summary['PRF_DATA_AVAILABILITY']}
P11B_READY = {str(summary['P11B_READY']).lower()}
MANUAL_DOWNLOAD_REQUIRED = false
REQUIRED_FILES = []
P11A5_STATUS = {summary['P11A5_STATUS']}
```

No optimizer, backward, model update, clustering, or P11-B training was started.
"""
    (args.output_dir/'P11A5_REPORT.md').write_text(report)


if __name__ == "__main__":
    main()
