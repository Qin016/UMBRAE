import json
from pathlib import Path

import numpy as np
import torch

from umbrae.data.prf_fine_unit_mapping import (
    FinePRFUnitMapping,
    clip_patch_visual_angle_map,
    largest_remainder_allocation,
)
from umbrae.data.spatial_teacher_builder import SpatialTeacherBuilder


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dual_branch_outputs/p11b0_prf_grounded_unit_construction"
TABLE = ROOT / "dual_branch_outputs/p11a5_continuous_retinotopy_recovery/subj01_continuous_prf_nsdgeneral.npz"


def mapping(name="selected_mapping_real.json"):
    return json.loads((OUT / name).read_text())


def retino_units(name="selected_mapping_real.json"):
    return [u for u in mapping(name)["units"] if u["unit_type"] == "retinotopic"]


def test_selected_indices_valid_unique_and_exact_quality_pool():
    table = np.load(TABLE, allow_pickle=False)
    units = retino_units()
    indices = [v for unit in units for v in unit["voxel_indices"]]
    assert len(indices) == len(set(indices)) == 3956
    assert min(indices) >= 0 and max(indices) < 15724
    expected = np.flatnonzero(
        table["valid_full_prf"]
        & table["quality_ge_10p1"]
        & np.isin(table["parent_retino_roi"], ["V1", "V2", "V3", "hV4"])
    )
    np.testing.assert_array_equal(np.sort(indices), expected)


def test_no_cross_roi_or_hemisphere_membership():
    table = np.load(TABLE, allow_pickle=False)
    for unit in retino_units():
        idx = unit["voxel_indices"]
        assert set(table["parent_retino_roi"][idx]) == {unit["parent_roi"]}
        assert set(table["hemisphere"][idx]) == {unit["hemisphere"]}


def test_largest_remainder_allocation_sums_and_is_deterministic():
    counts = {(r, h): i + 10 for i, (r, h) in enumerate(
        [(r, h) for r in ("V1", "V2", "V3", "hV4") for h in ("left", "right")]
    )}
    first = largest_remainder_allocation(counts, 64)
    assert first == largest_remainder_allocation(counts, 64)
    assert sum(first.values()) == 64 and min(first.values()) >= 1


def test_real_patch_affinity_shape_normalization_and_finiteness():
    w = np.load(OUT / "selected_patch_affinity_real.npy")
    assert w.shape == (64, 256)
    assert np.isfinite(w).all() and (w >= 0).all()
    np.testing.assert_allclose(w.sum(1), 1, atol=1e-6)


def test_cluster_order_is_spatially_canonical_and_deterministic():
    units = retino_units()
    for roi in ("V1", "V2", "V3", "hV4"):
        for hemi in ("left", "right"):
            group = [u for u in units if u["parent_roi"] == roi and u["hemisphere"] == hemi]
            keys = [(np.hypot(u["mean_x"], u["mean_y"]), u["polar_angle"]) for u in group]
            assert keys == sorted(keys)
    assert mapping() == mapping()


def test_random_control_preserves_all_matching_invariants():
    real = retino_units()
    random = retino_units("selected_mapping_random_seed42.json")
    assert [u["num_voxels"] for u in real] == [u["num_voxels"] for u in random]
    assert [u["parent_roi"] for u in real] == [u["parent_roi"] for u in random]
    assert [u["hemisphere"] for u in real] == [u["hemisphere"] for u in random]
    assert sorted(v for u in real for v in u["voxel_indices"]) == sorted(v for u in random for v in u["voxel_indices"])
    assert all(np.array_equal(a["patch_affinity"], b["patch_affinity"]) for a, b in zip(real, random))
    assert any(a["voxel_indices"] != b["voxel_indices"] for a, b in zip(real, random))


def test_random_voxels_assigned_once_and_remain_in_group():
    table = np.load(TABLE, allow_pickle=False)
    units = retino_units("selected_mapping_random_seed42.json")
    flat = [v for u in units for v in u["voxel_indices"]]
    assert len(flat) == len(set(flat))
    for u in units:
        idx = u["voxel_indices"]
        assert set(table["parent_retino_roi"][idx]) == {u["parent_roi"]}
        assert set(table["hemisphere"][idx]) == {u["hemisphere"]}


def test_high_level_roi_affinity_is_null():
    high = [u for u in mapping()["units"] if u["unit_type"] == "high_level_functional"]
    assert [u["parent_roi"] for u in high] == ["FFA", "EBA", "PPA", "OPA"]
    assert all(u["patch_affinity"] is None for u in high)


def test_clip_patch_map_count_order_and_y_convention():
    patches = clip_patch_visual_angle_map()
    assert len(patches) == 256
    assert [p["patch_index"] for p in patches] == list(range(256))
    assert all(p["patch_index"] == p["row"] * 16 + p["column"] for p in patches)
    assert patches[0]["visual_angle_y"] > 0 and patches[-1]["visual_angle_y"] < 0
    assert patches[0]["visual_angle_x"] < 0 and patches[15]["visual_angle_x"] > 0


def test_laterality_sanity_on_selected_affinity():
    units = retino_units(); w = np.load(OUT / "selected_patch_affinity_real.npy")
    patch_x = np.array([p["visual_angle_x"] for p in clip_patch_visual_angle_map()])
    com_x = w @ patch_x
    left = [i for i, u in enumerate(units) if u["hemisphere"] == "left"]
    right = [i for i, u in enumerate(units) if u["hemisphere"] == "right"]
    assert np.mean(com_x[left]) > 0
    assert np.mean(com_x[right]) < 0


def test_spatial_teacher_shape_modes_and_no_parameters():
    w = torch.from_numpy(np.load(OUT / "selected_patch_affinity_real.npy"))
    builder = SpatialTeacherBuilder(w)
    x = torch.randn(2, 256, 1024)
    assert builder(x).shape == (2, 64, 1024)
    assert builder(x, normalize_patches=True).shape == (2, 64, 1024)
    assert list(builder.parameters()) == []


def test_fine_mapping_interface_has_no_learnable_state():
    interface = FinePRFUnitMapping(OUT / "selected_mapping_real.json")
    uid = retino_units()[0]["unit_id"]
    assert interface.get_unit_indices(uid)
    assert interface.get_parent_roi(uid) == "V1"
    assert interface.get_hemisphere(uid) == "left"
    assert len(interface.get_patch_affinity(uid)) == 256
    assert not hasattr(interface, "parameters")


def test_no_training_or_semantic_leakage_provenance():
    provenance = json.loads((OUT / "provenance.json").read_text())
    assert provenance["optimizer_created"] is False
    assert provenance["backward_called"] is False
    assert provenance["training_started"] is False
    assert provenance["stimulus_responses_used_for_mapping"] is False
    assert provenance["clip_features_used_for_mapping"] is False
    summary = json.loads((OUT / "p11b0_summary.json").read_text())
    assert summary["NO_TRAINING_STARTED"] is True
    assert summary["P11B0_STATUS"] == "COMPLETE"
