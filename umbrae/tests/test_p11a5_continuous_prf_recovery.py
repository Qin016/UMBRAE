import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dual_branch_outputs" / "p11a5_continuous_retinotopy_recovery"
SOURCE = Path("/opt/data/private/BA/NSD/nsd_prf_recovery/subj01/source/func1pt8mm")
P11A = ROOT / "dual_branch_outputs" / "p11a_fine_grained_spatial_audit"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table():
    return np.load(OUT / "subj01_continuous_prf_nsdgeneral.npz", allow_pickle=False)


def test_source_metadata_and_official_assets_exist():
    manifest = json.loads((OUT / "download_manifest.json").read_text())
    assert manifest["manual_download_required"] is False
    assert {asset["parameter"] for asset in manifest["assets"]} == {"angle", "eccentricity", "size", "R2", "exponent"}
    assert all(asset["download_success"] for asset in manifest["assets"])


def test_file_hashes_are_recorded_and_exact():
    manifest = json.loads((OUT / "download_manifest.json").read_text())
    for asset in manifest["assets"]:
        path = SOURCE / asset["file"]
        assert asset["sha256"] == sha256(path)
        assert asset["bytes"] == path.stat().st_size == asset["expected_bytes_from_official_listing"]


def test_native_parameter_arrays_align_to_expected_space():
    native = np.load(OUT / "subj01_continuous_prf_native.npz", allow_pickle=False)
    for key in ("angle", "eccentricity", "size", "exponent", "quality_R2"):
        assert native[key].shape == (81, 104, 83)
    assert native["affine"].shape == (4, 4)


def test_nsdgeneral_output_length_and_indices():
    values = table()
    assert values["voxel_index"].shape == (15724,)
    np.testing.assert_array_equal(values["voxel_index"], np.arange(15724))
    assert len(np.unique(values["volume_flat_index_c"])) == 15724
    assert values["volume_flat_index_c"].min() >= 0
    assert values["volume_flat_index_c"].max() < 81 * 104 * 83


def test_known_roi_voxel_index_sets_unchanged():
    values = table()
    route = json.loads((ROOT.parent / "roi_indices/subj01_neuroroute_v1.json").read_text())
    for roi in ("V1", "V2", "V3", "hV4"):
        actual = set(np.flatnonzero(values["parent_retino_roi"] == roi).tolist())
        assert actual == set(route["rois"][roi]["indices"])


def test_invalid_clean_parameters_and_derived_values_remain_nan():
    values = table()
    field_masks = {
        "continuous_polar_angle_deg": "valid_angle",
        "continuous_eccentricity_deg": "valid_ecc",
        "prf_size_effective_deg": "valid_size",
        "prf_exponent": "valid_exponent",
        "fit_quality_R2_percent": "valid_quality",
        "prf_x_deg": "valid_xy",
        "prf_y_deg": "valid_xy",
        "prf_sigma_gaussian_deg": "valid_sigma",
    }
    for field, mask in field_masks.items():
        assert np.isnan(values[field][~values[mask]]).all()
        assert np.isfinite(values[field][values[mask]]).all()


def test_raw_and_derived_parameter_provenance_are_distinct():
    provenance = json.loads((OUT / "p11a5_parameter_provenance.json").read_text())
    assert provenance["angle"]["raw_or_derived"] == "raw"
    assert provenance["eccentricity"]["raw_or_derived"] == "raw"
    assert provenance["x"]["raw_or_derived"] == "derived"
    assert provenance["y"]["raw_or_derived"] == "derived"
    assert provenance["sigma"]["raw_or_derived"] == "derived"
    assert provenance["sigma"]["source_file"] is None


def test_derived_xy_and_sigma_follow_official_formulas():
    values = table()
    mask = values["valid_full_prf"]
    radians = np.deg2rad(values["angle_deg_raw"][mask])
    np.testing.assert_allclose(values["prf_x_deg"][mask], values["eccentricity_deg_raw"][mask] * np.cos(radians), atol=2e-6)
    np.testing.assert_allclose(values["prf_y_deg"][mask], values["eccentricity_deg_raw"][mask] * np.sin(radians), atol=2e-6)
    np.testing.assert_allclose(values["prf_sigma_gaussian_deg"][mask], values["size_effective_deg_raw"][mask] * np.sqrt(values["exponent_raw"][mask]), atol=2e-6)


def test_continuous_ecc_category_consistency_is_reproducible():
    values = table()
    comparable = (values["visual_roi_label"] > 0) & (values["categorical_ecc_label"] > 0) & values["valid_ecc"]
    predicted = np.digitize(values["continuous_eccentricity_deg"], [0.5, 1.0, 2.0, 4.0], right=True) + 1
    distance = np.abs(predicted[comparable] - values["categorical_ecc_label"][comparable])
    validation = json.loads((OUT / "mapping_validation.json").read_text())
    assert np.isclose(np.mean(distance == 0), validation["ecc_category_exact_consistency"])
    assert np.isclose(np.mean(distance <= 1), validation["ecc_category_within_adjacent_band"])


def test_mapping_is_deterministic_and_has_single_provenance_route():
    values = table()
    expected_x = values["eccentricity_deg_raw"] * np.cos(np.deg2rad(values["angle_deg_raw"]))
    np.testing.assert_allclose(values["prf_x_deg"][values["valid_xy"]], expected_x[values["valid_xy"]], atol=2e-6)
    assert set(values["mapping_provenance"].tolist()) == {"official_func1pt8mm_exact_grid_to_nsdgeneral_C_order"}


def test_no_training_modules_or_operations_were_used():
    summary = json.loads((OUT / "p11a5_summary.json").read_text())
    assert summary["no_training"] is True and summary["p11b_started"] is False
    assert all(value is False for value in summary["safety"].values())


def test_p11a_artifacts_remain_intact():
    recorded = json.loads((OUT / "p11a_artifact_integrity.json").read_text())
    for name, digest in recorded.items():
        assert sha256(P11A / name) == digest


def test_mapping_readiness_and_coverage_are_supported():
    summary = json.loads((OUT / "p11a5_summary.json").read_text())
    assert summary["MAPPING_VALIDATION"] == "PASS"
    assert summary["P11A5_STATUS"] == "BLOCKER_RESOLVED"
    assert summary["P11B_READY"] is True
    assert summary["tiers"]["FULL_PRF"] == 4656
    assert summary["tiers"]["total_retinotopic_voxels"] == 4657


def test_categorical_fallback_preserves_scope_without_patch_affinity():
    mapping = json.loads((OUT / "categorical_fallback_mapping.json").read_text())
    assert len(mapping["units"]) == 40
    flattened = [index for unit in mapping["units"] for index in unit["voxel_indices"]]
    assert len(flattened) == len(set(flattened))
    assert all(unit["patch_affinity"] is None for unit in mapping["units"])
