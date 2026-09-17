import json
from pathlib import Path

import numpy as np
import torch

from umbrae.data.p11_structural_mapping import (
    FineGrainedStructuralTokenizer,
    clip_patch_index_map,
    gaussian_patch_affinity,
    within_roi_random_control,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dual_branch_outputs" / "p11a_fine_grained_spatial_audit"


def load_table():
    return np.load(OUTPUT / "subj01_voxel_prf_table.npz", allow_pickle=False)


def test_nsdgeneral_indices_are_valid_and_complete():
    table = load_table()
    indices = table["voxel_index"]
    assert indices.shape == (15724,)
    np.testing.assert_array_equal(indices, np.arange(15724))


def test_no_out_of_range_voxel_or_volume_index():
    table = load_table()
    assert int(table["voxel_index"].min()) == 0
    assert int(table["voxel_index"].max()) == 15723
    assert int(table["volume_flat_index_c"].min()) >= 0
    assert int(table["volume_flat_index_c"].max()) < 81 * 104 * 83
    assert len(np.unique(table["volume_flat_index_c"])) == 15724


def test_missing_prf_is_explicit_and_never_zero_filled():
    table = load_table()
    assert not table["valid_prf"].any()
    for field in ("x_prf", "y_prf", "eccentricity_prf", "angle_prf", "sigma_prf", "quality_prf"):
        assert np.isnan(table[field]).all()
    assert set(table["invalid_reason"].tolist()) == {"continuous_prf_asset_unavailable"}


def test_gaussian_affinity_has_k_by_256_shape_and_normalizes():
    patches = clip_patch_index_map()
    centers = np.asarray([[p["image_x_normalized"], p["visual_field_y_up_normalized"]] for p in patches])
    affinity = gaussian_patch_affinity(
        np.asarray([[-0.5, 0.25], [0.2, -0.4], [0.0, 0.0]]),
        np.asarray([0.2, 0.4, 0.8]),
        centers,
    )
    assert affinity.shape == (3, 256)
    assert np.isfinite(affinity).all()
    np.testing.assert_allclose(affinity.sum(axis=1), 1.0, atol=1e-6)


def test_rejected_candidates_cannot_masquerade_as_valid_affinity():
    for k in (32, 48, 64, 96, 128):
        manifest = json.loads((OUTPUT / f"candidate_mapping_k{k}.json").read_text())
        affinity = np.load(OUTPUT / f"candidate_patch_affinity_k{k}.npy")
        assert manifest["construction_status"] == "REJECTED_MISSING_CONTINUOUS_PRF"
        assert manifest["units"] == []
        assert affinity.shape == (0, 256)


def synthetic_units():
    return [
        {"unit_id": "V1_0", "parent_roi": "V1", "voxel_indices": [0, 1, 2], "num_voxels": 3},
        {"unit_id": "V1_1", "parent_roi": "V1", "voxel_indices": [3, 4], "num_voxels": 2},
        {"unit_id": "V2_0", "parent_roi": "V2", "voxel_indices": [5, 6, 7, 8], "num_voxels": 4},
        {"unit_id": "V2_1", "parent_roi": "V2", "voxel_indices": [9], "num_voxels": 1},
    ]


def test_each_valid_synthetic_retinotopic_voxel_assigned_once():
    flattened = [index for unit in synthetic_units() for index in unit["voxel_indices"]]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))


def test_unit_voxel_counts_match_mapping_and_tokenizer_masks():
    units = synthetic_units()
    assert all(unit["num_voxels"] == len(unit["voxel_indices"]) for unit in units)
    tokenizer = FineGrainedStructuralTokenizer({"units": units})
    raw, mask = tokenizer(torch.arange(20, dtype=torch.float32).reshape(2, 10))
    assert raw.shape == (2, 4, 4)
    assert mask.sum(dim=1).tolist() == [3, 2, 4, 1]


def test_mapping_control_is_deterministic_under_fixed_seed():
    first = within_roi_random_control(synthetic_units(), seed=42)
    second = within_roi_random_control(synthetic_units(), seed=42)
    assert first == second


def test_random_control_preserves_size_roi_and_coverage():
    source = synthetic_units()
    control = within_roi_random_control(source, seed=42)
    for roi in ("V1", "V2"):
        source_roi = [u for u in source if u["parent_roi"] == roi]
        control_roi = [u for u in control if u["parent_roi"] == roi]
        assert [u["num_voxels"] for u in source_roi] == [u["num_voxels"] for u in control_roi]
        assert sorted(i for u in source_roi for i in u["voxel_indices"]) == sorted(i for u in control_roi for i in u["voxel_indices"])


def test_clip_patch_indices_are_exact_row_major_0_to_255():
    patches = clip_patch_index_map()
    assert [patch["patch_index"] for patch in patches] == list(range(256))
    for patch in patches:
        assert patch["patch_index"] == patch["row"] * 16 + patch["column"]
    assert patches[0]["row"] == 0 and patches[0]["column"] == 0
    assert patches[-1]["row"] == 15 and patches[-1]["column"] == 15


def test_selected_mapping_and_summary_are_safely_blocked():
    selected = json.loads((OUTPUT / "selected_mapping.json").read_text())
    summary = json.loads((OUTPUT / "p11a_summary.json").read_text())
    assert selected["selection_status"] == "BLOCKED" and selected["units"] == []
    assert summary["PRF_DATA_AVAILABILITY"] == "INSUFFICIENT"
    assert summary["FINE_GRAINED_RETINOTOPIC_TOKENIZATION"] == "NOT_FEASIBLE"
    assert summary["P11B_READY"] is False
    assert summary["no_training"] is True


def test_provenance_records_no_training_or_semantic_mapping_leakage():
    provenance = json.loads((OUTPUT / "provenance.json").read_text())
    assert provenance["optimizer_created"] is False
    assert provenance["backward_called"] is False
    assert provenance["training_started"] is False
    assert provenance["stimulus_responses_used_for_mapping"] is False
    assert provenance["clip_features_used_for_mapping"] is False
