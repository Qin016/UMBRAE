#!/usr/bin/env python3
"""P11-A read-only pRF asset audit and blocked-safe construction for NSD."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umbrae.data.p11_structural_mapping import (
    clip_patch_index_map,
    nsdgeneral_voxel_table,
    read_nifti1,
    sha256_file,
)


RETINO_ROIS = ("V1", "V2", "V3", "hV4")
HIGH_LEVEL_ROIS = ("FFA", "EBA", "PPA", "OPA")
VISUAL_LABELS = {1: "V1", 2: "V1", 3: "V2", 4: "V2", 5: "V3", 6: "V3", 7: "hV4"}
VISUAL_SUBLABELS = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d", 7: "hV4"}
ECC_LABELS = {0: "Unknown", 1: "ecc0pt5", 2: "ecc1", 3: "ecc2", 4: "ecc4", 5: "ecc4+"}
CANDIDATE_K = (32, 48, 64, 96, 128)
BLOCKER = (
    "The local NSD copy has categorical prf-visualrois/prf-eccrois labels but no "
    "voxel-wise polar angle or x/y center, continuous eccentricity, pRF size/sigma, "
    "or fit-quality data. Gaussian voxel-to-patch affinity and pRF-coordinate "
    "clustering therefore cannot be constructed without fabricating values."
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--nsd-root", type=Path, default=root.parents[1] / "NSD" / "nsddata")
    parser.add_argument("--roi-root", type=Path, default=root.parent / "roi_indices")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "dual_branch_outputs" / "p11a_fine_grained_spatial_audit",
    )
    parser.add_argument("--docs-dir", type=Path, default=root / "docs")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def infer_subject(path: Path) -> str:
    for part in path.parts:
        if part.startswith("subj") and part[4:].isdigit():
            return part
    return "all/fsaverage"


def infer_space(path: Path) -> str:
    text = str(path)
    if "/func1pt8mm/roi/" in text:
        return "native subject func1pt8mm volume; directly aligned to nsdgeneral mask"
    if "/freesurfer/fsaverage/" in text:
        return "fsaverage surface label metadata"
    if "/freesurfer/" in text:
        return "native subject FreeSurfer label metadata"
    return "unknown"


def field_meaning(path: Path) -> tuple[str, bool, str]:
    name = path.name.lower()
    if "prf-visualrois" in name:
        return (
            "categorical visual-area labels: V1v,V1d,V2v,V2d,V3v,V3d,hV4",
            True,
            "Usable for ROI provenance only; not a voxel-wise pRF estimate.",
        )
    if "prf-eccrois" in name:
        return (
            "categorical eccentricity labels: ecc0pt5,ecc1,ecc2,ecc4,ecc4+",
            True,
            "Usable as coarse categorical provenance only; exact bin boundaries and continuous eccentricity are not encoded in the file.",
        )
    return ("atlas/color-table metadata, not pRF parameters", False, "No continuous pRF fields.")


def inventory_assets(nsd_root: Path) -> list[dict[str, Any]]:
    patterns = ("*prf*", "*retino*", "*eccen*", "*angle*", "*sigma*", "*rfsize*", "*vexpl*", "*polar*")
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in nsd_root.rglob(pattern) if path.is_file())
    inventory = []
    for path in sorted(files):
        meaning, usable, notes = field_meaning(path)
        record: dict[str, Any] = {
            "file": str(path.resolve()),
            "subject": infer_subject(path),
            "space": infer_space(path),
            "shape": None,
            "dtype": "text" if path.suffix == ".ctab" else None,
            "unique_values": None,
            "field_meaning": meaning,
            "usable": usable,
            "notes": notes,
            "sha256": sha256_file(path),
        }
        if path.name.endswith(".nii.gz") or path.suffix == ".nii":
            array, metadata = read_nifti1(path)
            unique, counts = np.unique(array, return_counts=True)
            record.update(
                shape=metadata["shape"],
                dtype=metadata["dtype"],
                unique_values=[
                    {"value": float(value), "count": int(count)}
                    for value, count in zip(unique, counts)
                ],
                nifti=metadata,
            )
        inventory.append(record)
    return inventory


def build_voxel_table(nsd_root: Path, subject: str, output: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    roi_dir = nsd_root / "ppdata" / subject / "func1pt8mm" / "roi"
    paths = {
        "nsdgeneral": roi_dir / "nsdgeneral.nii.gz",
        "visual": roi_dir / "prf-visualrois.nii.gz",
        "ecc": roi_dir / "prf-eccrois.nii.gz",
        "lh": roi_dir / "lh.nsdgeneral.nii.gz",
        "rh": roi_dir / "rh.nsdgeneral.nii.gz",
    }
    arrays, metadata = {}, {}
    for name, path in paths.items():
        arrays[name], metadata[name] = read_nifti1(path)
    shape = arrays["nsdgeneral"].shape
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError("nsdgeneral, ROI, and hemisphere volumes are not shape-aligned")
    base = nsdgeneral_voxel_table(arrays["nsdgeneral"], np.asarray(metadata["nsdgeneral"]["affine"]))
    selected = base["volume_flat_index_c"]
    select = lambda array: array.ravel(order="C")[selected]
    visual_label = select(arrays["visual"]).astype(np.int16)
    ecc_label = select(arrays["ecc"]).astype(np.int16)
    left = select(arrays["lh"]) > 0
    right = select(arrays["rh"]) > 0
    hemisphere = np.full(len(selected), "unknown", dtype="U7")
    hemisphere[left & ~right] = "left"
    hemisphere[right & ~left] = "right"
    hemisphere[left & right] = "overlap"
    table = {
        **base,
        "hemisphere": hemisphere,
        "visual_roi_label": visual_label,
        "visual_roi_sublabel": np.asarray([VISUAL_SUBLABELS.get(int(v), "") for v in visual_label]),
        "parent_retino_roi": np.asarray([VISUAL_LABELS.get(int(v), "") for v in visual_label]),
        "ecc_roi_label": ecc_label,
        "ecc_roi_name": np.asarray([ECC_LABELS.get(int(v), "invalid") for v in ecc_label]),
        "x_prf": np.full(len(selected), np.nan, dtype=np.float32),
        "y_prf": np.full(len(selected), np.nan, dtype=np.float32),
        "eccentricity_prf": np.full(len(selected), np.nan, dtype=np.float32),
        "angle_prf": np.full(len(selected), np.nan, dtype=np.float32),
        "sigma_prf": np.full(len(selected), np.nan, dtype=np.float32),
        "quality_prf": np.full(len(selected), np.nan, dtype=np.float32),
        "valid_prf": np.zeros(len(selected), dtype=bool),
        "invalid_reason": np.full(len(selected), "continuous_prf_asset_unavailable", dtype="U40"),
    }
    np.savez_compressed(output, **table)
    audit = {
        "shape": list(shape),
        "voxel_count": len(selected),
        "hemisphere_counts": {name: int(np.sum(hemisphere == name)) for name in np.unique(hemisphere)},
        "source_files": {name: str(path.resolve()) for name, path in paths.items()},
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "flat_index_order": "numpy_C_order over volume, matching existing UMBRAE ROI conversion",
        "all_volumes_shape_aligned": True,
        "continuous_prf_valid_voxels": 0,
    }
    return table, audit


def write_coverage(path: Path, table: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    with path.open("w", newline="") as handle:
        fields = ["roi", "total_roi_voxels", "categorical_ecc_label_available", "continuous_prf_valid", "quality_passed", "continuous_coverage_ratio", "median_ecc", "median_sigma", "ecc_label_distribution"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for roi in RETINO_ROIS:
            mask = table["parent_retino_roi"] == roi
            ecc = table["ecc_roi_label"][mask]
            distribution = {ECC_LABELS.get(int(v), str(int(v))): int(np.sum(ecc == v)) for v in np.unique(ecc) if v > 0}
            row = {
                "roi": roi,
                "total_roi_voxels": int(mask.sum()),
                "categorical_ecc_label_available": int(np.sum(ecc > 0)),
                "continuous_prf_valid": 0,
                "quality_passed": 0,
                "continuous_coverage_ratio": 0.0,
                "median_ecc": "unavailable",
                "median_sigma": "unavailable",
                "ecc_label_distribution": json.dumps(distribution, sort_keys=True),
            }
            writer.writerow(row)
            rows.append(row)
    return rows


def unavailable_figure(path: Path, title: str, detail: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.axis("off")
    axis.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, weight="bold")
    axis.text(0.5, 0.38, detail, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_figures(figures: Path, table: dict[str, np.ndarray]) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    for roi in RETINO_ROIS:
        unavailable_figure(
            figures / f"prf_visual_field_scatter_{roi}.png",
            f"{roi}: visual-field scatter unavailable",
            "No voxel-wise polar angle/x/y pRF center is present locally. No coordinates were imputed.",
        )
    masks = [table["parent_retino_roi"] == roi for roi in RETINO_ROIS]
    labels = [ECC_LABELS[i] for i in range(1, 6)]
    values = [[int(np.sum(table["ecc_roi_label"][mask] == i)) for i in range(1, 6)] for mask in masks]
    fig, axis = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(4)
    for label_index, label in enumerate(labels):
        column = np.asarray([row[label_index] for row in values])
        axis.bar(RETINO_ROIS, column, bottom=bottom, label=label)
        bottom += column
    axis.set_title("Available categorical eccentricity ROI labels (not continuous pRF eccentricity)")
    axis.set_ylabel("nsdgeneral voxels")
    axis.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "eccentricity_histogram.png", dpi=150); plt.close(fig)
    for filename, title in [
        ("prf_size_histogram.png", "pRF size histogram unavailable"),
        ("selected_unit_centers.png", "Selected unit centers unavailable"),
        ("selected_patch_affinity_heatmaps.png", "Patch affinities unavailable"),
        ("patch_coverage_map.png", "Patch coverage unavailable"),
        ("unit_affinity_similarity_matrix.png", "Unit affinity similarity unavailable"),
    ]:
        unavailable_figure(figures / filename, title, BLOCKER)


def write_candidate_outputs(output: Path, subject: str) -> list[dict[str, Any]]:
    rows = []
    for k in CANDIDATE_K:
        manifest = {
            "subject": subject,
            "requested_k_retino": k,
            "nominal_k_total_with_four_high_level_tokens": k + 4,
            "construction_status": "REJECTED_MISSING_CONTINUOUS_PRF",
            "features": {"xy_only": "unavailable", "xy_plus_size": "unavailable"},
            "units": [],
            "rejection_reason": BLOCKER,
            "stimulus_or_clip_features_used_for_assignment": False,
        }
        json_dump(output / f"candidate_mapping_k{k}.json", manifest)
        np.save(output / f"candidate_patch_affinity_k{k}.npy", np.empty((0, 256), dtype=np.float32))
        with (output / f"fine_unit_metadata_k{k}.csv").open("w", newline="") as handle:
            csv.writer(handle).writerow(["unit_id", "parent_roi", "num_voxels", "mean_prf_x", "mean_prf_y", "mean_sigma", "quality_mean", "dominant_patch", "patch_entropy", "status"])
        rows.append(
            {
                "k_retino": k,
                "k_total": k + 4,
                "min_voxels_per_unit": "unavailable",
                "median_voxels_per_unit": "unavailable",
                "mean_voxels_per_unit": "unavailable",
                "max_voxels_per_unit": "unavailable",
                "p05_voxels_per_unit": "unavailable",
                "spatial_compactness": "unavailable",
                "affinity_redundancy": "unavailable",
                "effective_rank": "unavailable",
                "patch_coverage": "unavailable",
                "status": "REJECTED_MISSING_CONTINUOUS_PRF",
            }
        )
    with (output / "candidate_k_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    inventory = inventory_assets(args.nsd_root)
    json_dump(args.output_dir / "p11_prf_asset_inventory.json", inventory)
    table, voxel_audit = build_voxel_table(
        args.nsd_root, args.subject, args.output_dir / f"{args.subject}_voxel_prf_table.npz"
    )
    coverage = write_coverage(args.output_dir / "prf_coverage_by_roi.csv", table)
    schema = {
        "subject": args.subject,
        "space": "native subject func1pt8mm nsdgeneral vector",
        "voxel_index": "zero-based UMBRAE nsdgeneral vector index in NumPy C-order mask extraction",
        "available_fields": {
            "visual_roi_label": "categorical V1v/V1d/V2v/V2d/V3v/V3d/hV4",
            "ecc_roi_label": "categorical ecc0pt5/ecc1/ecc2/ecc4/ecc4+; exact continuous value unavailable",
            "ijk_xyz": "functional-volume voxel index and affine world millimetres",
            "hemisphere": "derived from aligned lh/rh nsdgeneral masks",
        },
        "unavailable_fields": {
            "x_prf": "no local source file",
            "y_prf": "no local source file",
            "angle_prf": "no local prfangle surface/volume file",
            "eccentricity_prf": "only categorical labels exist; no continuous estimate",
            "sigma_prf": "no local pRF size/sigma file; Gaussian sigma semantics cannot be established",
            "quality_prf": "no local R2/variance-explained/fit-quality file",
        },
        "angle_units_zero_direction_rotation": None,
        "eccentricity_units": "categorical label names suggest degrees but no continuous values are encoded; not used numerically",
        "sigma_semantics": None,
        "quality_metric_semantics": None,
        "default_quality_threshold": None,
        "quality_distribution": None,
        "missing_value_policy": "NaN plus valid_prf=false; no fill, nearest-neighbor, or ROI-mean imputation",
        "official_code_evidence": {
            "repository": "https://github.com/cvnlab/nsdcode",
            "file": "examples/examples_nsdmapdata.py",
            "evidence": "official polar-angle mapping example expects freesurfer/subjXX/label/lh.prfangle.mgz",
            "local_expected_file_present": False,
        },
    }
    json_dump(args.output_dir / "p11_prf_parameter_schema.json", schema)
    patches = {
        "model": "openai/clip-vit-large-patch14",
        "teacher_layer": "hidden_states[-2][:,1:,:]",
        "grid": [16, 16],
        "num_patches": 256,
        "ordering": "row-major: patch_index = row*16 + column",
        "repository_evidence": "models/clip_patch_teacher.py: patch_grid.flatten(2).transpose(1,2)",
        "normalized_coordinate_only": True,
        "absolute_visual_angle_mapping_available": False,
        "image_y_direction": "down",
        "visual_field_y_direction": "up (sign-inverted field provided for geometry only)",
        "patches": clip_patch_index_map(),
    }
    json_dump(args.output_dir / "clip_patch_index_map.json", patches)
    candidates = write_candidate_outputs(args.output_dir, args.subject)
    selected = {
        "subject": args.subject,
        "coordinate_system": "unresolved: continuous pRF coordinate system unavailable",
        "selection_status": "BLOCKED",
        "primary_k_retino": None,
        "primary_k_total": None,
        "secondary_k_retino": None,
        "units": [],
        "high_level_roi_policy": "FFA/EBA/PPA/OPA retained conceptually as one semantic unit each; no unit mapping emitted until retinotopic mapping is valid",
        "blockers": [BLOCKER],
    }
    json_dump(args.output_dir / "selected_mapping.json", selected)
    json_dump(args.output_dir / f"p11_fine_unit_mapping_{args.subject}.json", selected)
    random_control = {
        "protocol": "within_roi_spatial_randomization_v1",
        "design_ready": True,
        "executable_mapping_ready": False,
        "reason": "Protocol is specified, but there is no valid real unit assignment to randomize.",
        "preserve": ["total token count", "unit voxel-count distribution", "parent ROI", "ROI voxel coverage", "pRF quality distribution when available"],
        "destroy": ["within-ROI voxel-to-pRF/unit spatial organization"],
        "algorithm": "For each parent ROI independently, permute the pooled eligible voxels and repartition using the real unit-size sequence with a fixed seed.",
        "seed": 42,
        "whole_brain_random_grouping": False,
    }
    json_dump(args.output_dir / "random_control_protocol.json", random_control)
    write_figures(args.output_dir / "figures", table)
    roi_mapping = args.roi_root / f"{args.subject}_neuroroute_v1.json"
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": args.subject,
        "nsd_root": str(args.nsd_root.resolve()),
        "nsdgeneral_voxel_count": voxel_audit["voxel_count"],
        "voxel_order": voxel_audit["flat_index_order"],
        "source_files": voxel_audit["source_files"],
        "source_hashes": voxel_audit["source_hashes"],
        "roi_mapping": str(roi_mapping.resolve()),
        "roi_mapping_sha256": sha256_file(roi_mapping),
        "pRF_quality_threshold": None,
        "clustering_algorithm": None,
        "clustering_seed": 42,
        "candidate_k": list(CANDIDATE_K),
        "selected_k": None,
        "visual_angle_mapping_convention": "unavailable; normalized CLIP image coordinates only",
        "clip_patch_ordering": "row-major from repository flatten(2).transpose(1,2)",
        "clip_teacher": "openai/clip-vit-large-patch14 hidden_states[-2][:,1:,:]",
        "stimulus_responses_used_for_mapping": False,
        "clip_features_used_for_mapping": False,
        "optimizer_created": False,
        "backward_called": False,
        "training_started": False,
    }
    json_dump(args.output_dir / "provenance.json", provenance)
    summary = {
        "PRF_DATA_AVAILABILITY": "INSUFFICIENT",
        "FINE_GRAINED_RETINOTOPIC_TOKENIZATION": "NOT_FEASIBLE",
        "PRIMARY_K_RETINO": None,
        "PRIMARY_K_TOTAL": None,
        "SECONDARY_K": None,
        "RANDOM_CONTROL_READY": False,
        "RANDOM_CONTROL_PROTOCOL_DESIGNED": True,
        "P11B_READY": False,
        "P11A_STATUS": "BLOCKED",
        "blockers": [
            "Missing voxel-wise pRF polar angle/x/y center.",
            "Missing continuous pRF eccentricity and size/sigma with defined units.",
            "Missing voxel-wise pRF R2/variance-explained quality.",
            "Therefore no traceable Gaussian voxel-to-patch affinity, structural candidate clustering, K selection, or executable matched random control can be constructed.",
        ],
        "asset_count": len(inventory),
        "voxel_audit": voxel_audit,
        "coverage_by_roi": coverage,
        "candidate_results": candidates,
        "no_training": True,
    }
    json_dump(args.output_dir / "p11a_summary.json", summary)
    write_reports(args, inventory, coverage, summary)
    print(json.dumps(summary, indent=2))


def write_reports(args: argparse.Namespace, inventory: list[dict[str, Any]], coverage: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    inventory_rows = [
        f"| `{Path(row['file']).name}` | {row['subject']} | {row['space']} | {row['shape'] or 'N/A'} | {row['field_meaning']} | {'limited' if row['usable'] else 'no'} | {row['notes']} |"
        for row in inventory
    ]
    audit_doc = """# P11 pRF Asset Audit

This is a local, read-only inventory. `Usable=limited` means categorical ROI provenance only, not a continuous voxel-wise pRF estimate.

| File | Subject | Space | Shape | Field Meaning | Usable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(inventory_rows) + "\n\n## Missing required continuous assets\n\nNo local file supplies polar angle/x/y, continuous eccentricity, size/sigma, or fit quality. The official NSD mapping-code example refers to `freesurfer/subjXX/label/lh.prfangle.mgz`; that asset is absent locally.\n"
    (args.docs_dir / "P11_PRF_ASSET_AUDIT.md").write_text(audit_doc)
    coverage_rows = "\n".join(
        f"| {r['roi']} | {r['total_roi_voxels']} | {r['categorical_ecc_label_available']} | 0 | 0 | 0.000 | unavailable | unavailable |"
        for r in coverage
    )
    candidate_rows = "\n".join(
        f"| {k} | {k+4} | unavailable | unavailable | unavailable | unavailable | unavailable | REJECTED_MISSING_CONTINUOUS_PRF |"
        for k in CANDIDATE_K
    )
    report = f"""# P11-A Fine-Grained Spatial Brain Representation Audit

## 1. Motivation

P8–P10 showed that marginal correction, token-wise adaptation, and token mixing cannot recover the Exact CLIP Token Oracle's downstream utility. P11-A therefore audited whether a spatially traceable brain representation can be constructed. No training was started.

## 2. Available pRF / Retinotopy Assets

The local NSD copy contains categorical `prf-visualrois` and `prf-eccrois` volumes and hemisphere-specific variants. It does **not** contain continuous voxel-wise pRF angle/x/y, eccentricity, size/sigma, or R²/variance-explained files. Full inventory: `docs/P11_PRF_ASSET_AUDIT.md` and `p11_prf_asset_inventory.json`.

Official code evidence: `cvnlab/nsdcode/examples/examples_nsdmapdata.py` uses `freesurfer/subjXX/label/lh.prfangle.mgz` for polar-angle mapping; the expected file is absent from this local NSD copy.

## 3. nsdgeneral ↔ pRF Mapping

The reliable part is `nsdgeneral vector index -> C-order volume index -> func1pt8mm ijk/world xyz -> aligned categorical visual ROI/eccentricity ROI/hemisphere`. The constructed `{args.subject}_voxel_prf_table.npz` contains all {summary['voxel_audit']['voxel_count']} voxels. Continuous pRF columns are NaN and `valid_prf=false`, explicitly complying with the no-imputation rule.

There is no reliable continuation from those volume voxels to continuous pRF parameters because the source parameter maps are absent.

## 4. pRF Coverage

| ROI | Total | Categorical ecc label | Valid continuous pRF | Quality passed | Coverage | Median ecc | Median sigma |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
{coverage_rows}

Quality distribution and a defensible threshold cannot be reported because no fit-quality metric exists locally. No threshold was invented.

## 5. Visual-Field Sanity

Hemisphere provenance is available from aligned left/right masks, but contralateral visual-field sanity cannot be tested without polar angle/x/y. The eccentricity figure reports categorical label counts only. Scatter, pRF-size, laterality-affinity, and visual-field coverage panels are explicitly marked unavailable rather than populated with fabricated coordinates.

## 6. CLIP Patch Geometry

The repository's `models/clip_patch_teacher.py` applies `patch_grid.flatten(2).transpose(1,2)`, proving row-major 16×16 ordering (`index=row*16+column`) for the 256 patch tokens. `clip_patch_index_map.json` records normalized image centers. `NORMALIZED_COORDINATE_ONLY=true`; absolute visual-angle centers are null because no validated pRF-to-stimulus convention can be joined locally.

## 7. Voxel-to-Patch Affinity

The intended kernel is `exp(-||c_p-mu_v||²/(2 sigma_v²))`, normalized over patches. It was **not evaluated** because neither `mu_v` nor a defined positive Gaussian sigma is available. No categorical eccentricity midpoint, ROI mean, nearest neighbor, or zero fill was substituted.

## 8. Candidate Fine-Grained Units

XY-only and XY+size ROI-conditioned clustering both require missing continuous pRF data. Rejected manifests and empty `[0,256]` affinity sentinels were emitted for K={{32,48,64,96,128}} so downstream code cannot mistake absence for a valid mapping.

## 9. Candidate Comparison

| K_retino | K_total nominal | Min vox/unit | Median vox/unit | Spatial compactness | Affinity redundancy | Patch coverage | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{candidate_rows}

No PRIMARY or SECONDARY K is selected. Selecting one would use only a requested token count, contrary to the protocol.

## 10. Selected Mapping

`selected_mapping.json` is a blocked manifest with zero units, not a usable tokenizer mapping. `PRIMARY_K_RETINO=null`, `PRIMARY_K_TOTAL=null`, and `SECONDARY_K=null`.

## 11. High-Level ROI Handling

FFA/EBA/PPA/OPA remain conceptually one non-retinotopic semantic unit per ROI. They were not assigned fake spatial pRF centers or face/body/place visual teachers. Existing atlas overlaps remain documented in the NeuroRoute mapping and were not silently deduplicated.

## 12. Random Spatial Control

The future protocol is defined as within-parent-ROI permutation followed by repartition using the real unit-size sequence. It preserves token count, ROI provenance, unit sizes, voxel coverage, and (when present) quality distribution while destroying spatial organization. It is not executable until a valid real mapping exists, hence `RANDOM_CONTROL_READY=false`.

## 13. Future Spatial Teacher

Once real affinities exist, the retinotopic teacher is `V_teacher_r = sum_p W[r,p] V_clip[p]`. This differs from P3 by conditioning each unit on a traceable visual-field kernel instead of a global semantic target. No teacher tensor was generated and no training was started.

## 14. Risks

- Continuous pRF assets may need a separate NSD data release and a verified surface-to-func1pt8mm mapping.
- Angle convention, eccentricity units, pRF-size definition, and fit-quality semantics must be sourced from the matching release.
- hV4 coverage may differ from V1–V3 after quality filtering.
- Excessive K could fragment low-coverage voxels; it cannot be evaluated yet.
- High-level ROIs do not have an automatic retinotopic teacher.
- Subject-agnostic code is present, but cross-subject consistency cannot be evaluated before subj01 construction is valid.

## 15. Final Decision

```text
PRF_DATA_AVAILABILITY = INSUFFICIENT
FINE_GRAINED_RETINOTOPIC_TOKENIZATION = NOT_FEASIBLE
PRIMARY_K_RETINO = null
PRIMARY_K_TOTAL = null
SECONDARY_K = null
RANDOM_CONTROL_READY = false
P11B_READY = false
P11A_STATUS = BLOCKED
```

Blocker: {BLOCKER}
"""
    (args.output_dir / "P11A_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
