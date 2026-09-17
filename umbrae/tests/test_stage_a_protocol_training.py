import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_stage_a_structural import StageACacheDataset, token_diagnostics


def test_stage_a_cache_sample_alignment(tmp_path):
    fmri = np.arange(15, dtype=np.float32).reshape(3, 5)
    patch = np.zeros((3, 256, 4), dtype=np.float16)
    np.save(tmp_path / "fmri.npy", fmri)
    np.save(tmp_path / "patch.npy", patch)
    (tmp_path / "ids.json").write_text(json.dumps(["a", "b", "c"]))
    metadata = {
        "sample_id_index": str(tmp_path / "ids.json"),
        "fmri_path": str(tmp_path / "fmri.npy"),
        "clip_path": str(tmp_path / "patch.npy"),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    dataset = StageACacheDataset(tmp_path)
    assert dataset[1]["sample_id"] == "b"
    assert np.array_equal(dataset[1]["fmri"], fmri[1])
    assert dataset[1]["v_patch"].shape == (256, 4)


def test_roi_token_diagnostics_do_not_include_diagonal():
    tokens = torch.eye(4).unsqueeze(0)
    norm, cosine = token_diagnostics(tokens)
    assert torch.isclose(norm, torch.tensor(1.0))
    assert torch.isclose(cosine, torch.tensor(0.0))
