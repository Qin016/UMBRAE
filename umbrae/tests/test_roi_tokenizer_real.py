"""Real-data validation for merged NeuroRoute ROI mappings."""

import io
import json
import os
import sys
import tarfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.roi_tokenizer import ROITokenizer


REQUIRED_ROIS = ["V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA"]
RSC_EXCLUSION_REASON = (
    "official floc-places:RSC has zero overlap with nsdgeneral "
    "across S1/S2/S5/S7"
)


def _paths():
    subject = os.environ.get("NEUROROUTE_TEST_SUBJECT", "subj01")
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    mapping_path = workspace_root / "roi_indices" / f"{subject}_neuroroute_v1.json"
    tar_path = (
        project_root
        / "nsd"
        / "webdataset_avg_split"
        / "test"
        / f"test_{subject}_0.tar"
    )
    return subject, mapping_path, tar_path


def _load_first_nsdgeneral(tar_path: Path) -> np.ndarray:
    with tarfile.open(tar_path) as archive:
        name = next(
            member.name
            for member in archive.getmembers()
            if member.name.endswith(".nsdgeneral.npy")
        )
        return np.load(io.BytesIO(archive.extractfile(name).read()))


def test_real_neuroroute_mapping_and_tokenizer():
    subject, mapping_path, tar_path = _paths()
    if not mapping_path.is_file() or not tar_path.is_file():
        raise unittest.SkipTest(
            f"Real NeuroRoute fixtures are unavailable for {subject}: "
            f"{mapping_path}, {tar_path}"
        )

    mapping = json.loads(mapping_path.read_text())
    assert mapping["roi_mapping_is_real"] is True
    assert mapping["voxel_order_verified"] is True
    assert mapping["roi_names"] == REQUIRED_ROIS
    assert mapping["required_rois"] == REQUIRED_ROIS
    assert mapping["unresolved_rois"] == []
    assert "RSC" not in mapping["rois"]
    assert "RSC" in mapping["optional_rois"]
    assert "RSC" in mapping["excluded_rois"]
    assert (
        mapping["excluded_roi_details"]["RSC"]["reason"]
        == RSC_EXCLUSION_REASON
    )

    roi_indices = {
        name: mapping["rois"][name]["indices"] for name in REQUIRED_ROIS
    }
    assert all(roi_indices[name] for name in REQUIRED_ROIS)

    fmri = _load_first_nsdgeneral(tar_path)
    if fmri.ndim == 1:
        fmri = fmri[None, None, :]
    elif fmri.ndim == 2:
        fmri = fmri[None, :, :]
    else:
        raise AssertionError(f"Unexpected real fMRI shape: {fmri.shape}")

    tokenizer = ROITokenizer(
        roi_names=REQUIRED_ROIS,
        roi_indices=roi_indices,
        token_dim=32,
        tokenizer_type="shared_mlp",
    )
    output = tokenizer(torch.from_numpy(fmri.astype(np.float32)))

    assert tuple(output["roi_tokens"].shape) == (1, 8, 32)
    assert output["roi_names"] == REQUIRED_ROIS


if __name__ == "__main__":
    try:
        test_real_neuroroute_mapping_and_tokenizer()
    except unittest.SkipTest as error:
        print(f"[SKIP] {error}")
    else:
        print("Real NeuroRoute ROITokenizer validation passed")
