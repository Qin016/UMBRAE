import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.roi_mapping import MVP_ROI_NAMES, load_roi_indices


def payload(subject="subj01"):
    return {
        "subject": subject,
        "roi_mapping_is_real": True,
        "voxel_order_verified": True,
        "roi_names": list(reversed(MVP_ROI_NAMES)),
        "rois": {
            name: {"indices": [2 * index, 2 * index + 1]}
            for index, name in enumerate(MVP_ROI_NAMES)
        },
    }


def write(tmp_path, value):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(value))
    return path


def test_neuroroute_mapping_uses_fixed_order(tmp_path):
    indices, summary = load_roi_indices(
        str(write(tmp_path, payload())),
        expected_voxel_count=16,
        expected_subject="1",
    )
    assert list(indices) == list(MVP_ROI_NAMES)
    assert summary["roi_count"] == 8


@pytest.mark.parametrize("field", ["roi_mapping_is_real", "voxel_order_verified"])
def test_neuroroute_requires_verified_flags(tmp_path, field):
    value = payload()
    value[field] = False
    with pytest.raises(ValueError, match=field):
        load_roi_indices(str(write(tmp_path, value)), expected_voxel_count=16)


def test_neuroroute_rejects_subject_mismatch(tmp_path):
    with pytest.raises(ValueError, match="subject mismatch"):
        load_roi_indices(
            str(write(tmp_path, payload("subj02"))),
            expected_voxel_count=16,
            expected_subject="subj01",
        )


def test_neuroroute_rejects_missing_duplicate_and_out_of_bounds(tmp_path):
    missing = payload()
    missing["roi_names"].remove("OPA")
    with pytest.raises(ValueError, match="missing"):
        load_roi_indices(str(write(tmp_path, missing)), expected_voxel_count=16)
    duplicate = payload()
    duplicate["roi_names"][0] = duplicate["roi_names"][1]
    with pytest.raises(ValueError, match="duplicates"):
        load_roi_indices(str(write(tmp_path, duplicate)), expected_voxel_count=16)
    bounds = payload()
    bounds["rois"]["OPA"]["indices"] = [15, 16]
    with pytest.raises(ValueError, match="within"):
        load_roi_indices(str(write(tmp_path, bounds)), expected_voxel_count=16)
