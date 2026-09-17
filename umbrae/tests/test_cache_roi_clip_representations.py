import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cache_roi_clip_representations import (
    ANALYSIS_SPLIT_NAMES,
    POOLING_METHOD,
    assign_stimulus_splits,
    cache_representations,
)


ROI_NAMES = ["V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA"]
CLIP_LAYERS = [4, 8, 12, 16, 20, 24]


class SyntheticDataset(Dataset):
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return {
            "fmri": torch.tensor([float(index), 1.0, 2.0]),
            "image": torch.full((3, 4, 4), float(index) / self.count),
            "record_index": index,
            "valid_repeats": 3,
        }


class SyntheticROIEncoder(nn.Module):
    def forward(self, fmri):
        base = fmri[:, :1, None]
        roi = torch.arange(8, device=fmri.device, dtype=fmri.dtype)[None, :, None]
        dim = torch.arange(4, device=fmri.device, dtype=fmri.dtype)[None, None, :]
        raw = base + roi + dim
        projected = torch.cat([raw, raw[..., :1] * 0.5], dim=-1)
        return {
            "brain_roi_features": raw,
            "projected_roi_features": projected,
        }


class SyntheticCLIPBank(nn.Module):
    def forward(self, images):
        base = images.mean(dim=(1, 2, 3))[:, None, None]
        layers = torch.arange(6, device=images.device, dtype=images.dtype)[None, :, None]
        dim = torch.arange(5, device=images.device, dtype=images.dtype)[None, None, :]
        return {"pooled_tokens": base + layers + dim}


def test_synthetic_cache_shapes_ids_and_metadata(tmp_path):
    count = 10
    records = [
        {
            "sample_index": index,
            "sample_key": f"sample{index:012d}",
            "stable_stimulus_id": f"coco73k:{1000 + index}",
            "coco73k_id": 1000 + index,
            "trial_id": 2000 + index,
            "number_of_repeats": 3,
            "source_tar": "synthetic.tar",
        }
        for index in range(count)
    ]
    split_ids = assign_stimulus_splits(records, seed=42)
    output_dir = tmp_path / "cache"
    cache_representations(
        records=records,
        roi_encoder=SyntheticROIEncoder(),
        clip_bank=SyntheticCLIPBank(),
        output_dir=output_dir,
        roi_names=ROI_NAMES,
        selected_layers=CLIP_LAYERS,
        roi_dim=4,
        clip_dim=5,
        subject="subj01",
        source_split="synthetic",
        seed=42,
        batch_size=3,
        device="cpu",
        provenance={"stage1_checkpoint": "synthetic_checkpoint.pt"},
        dataset_override=SyntheticDataset(count),
    )

    root_config = json.loads((output_dir / "cache_config.json").read_text())
    root_metadata = json.loads((output_dir / "metadata.json").read_text())
    assert root_config["sample_count"] == count
    assert root_config["roi_names"] == ROI_NAMES
    assert root_config["selected_clip_layers"] == CLIP_LAYERS
    assert root_config["clip_pooling_method"] == POOLING_METHOD
    assert root_config["uses_image_features_only_for_offline_analysis"] is True
    assert root_config["image_features_used_by_brain_branch"] is False
    assert sum(len(ids) for ids in split_ids.values()) == count
    assert set(root_metadata["split_ids"]) == set(ANALYSIS_SPLIT_NAMES)

    observed_ids = []
    for split_name in ANALYSIS_SPLIT_NAMES:
        split_dir = output_dir / split_name
        metadata = json.loads((split_dir / "metadata.json").read_text())
        split_config = json.loads((split_dir / "cache_config.json").read_text())
        split_count = metadata["sample_count"]
        brain = np.load(split_dir / "brain_roi_features.npy")
        projected = np.load(split_dir / "projected_roi_features.npy")
        clip = np.load(split_dir / "clip_layer_features.npy")
        assert brain.shape == (split_count, 8, 4)
        assert projected.shape == (split_count, 8, 5)
        assert clip.shape == (split_count, 6, 5)
        assert np.isfinite(brain).all()
        assert np.isfinite(projected).all()
        assert np.isfinite(clip).all()
        assert split_config["feature_dimensions"]["D_roi"] == 4
        assert split_config["feature_dimensions"]["D_clip"] == 5
        assert metadata["repeat_metadata_available"] is True
        assert metadata["trial_metadata_available"] is True
        split_ids_from_rows = [
            sample["stable_stimulus_id"] for sample in metadata["samples"]
        ]
        assert set(split_ids_from_rows) == set(
            root_metadata["split_ids"][split_name]
        )
        observed_ids.extend(split_ids_from_rows)
        for row, sample in enumerate(metadata["samples"]):
            assert brain[row, 0, 0] == sample["sample_index"]

    assert sorted(observed_ids) == [f"coco73k:{1000 + i}" for i in range(count)]

    try:
        cache_representations(
            records=records,
            roi_encoder=SyntheticROIEncoder(),
            clip_bank=SyntheticCLIPBank(),
            output_dir=output_dir,
            roi_names=ROI_NAMES,
            selected_layers=CLIP_LAYERS,
            roi_dim=4,
            clip_dim=5,
            subject="subj01",
            source_split="synthetic",
            seed=42,
            dataset_override=SyntheticDataset(count),
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing cache directory was not protected")


def test_stimulus_split_is_deterministic_and_grouped():
    records = [
        {
            "sample_index": index,
            "sample_key": f"sample{index:012d}",
            "stable_stimulus_id": f"coco73k:{index // 2}",
        }
        for index in range(12)
    ]
    first = assign_stimulus_splits(records, seed=7)
    first_assignments = [record["analysis_split"] for record in records]
    second = assign_stimulus_splits(records, seed=7)
    assert first == second
    assert first_assignments == [record["analysis_split"] for record in records]
    for left, right in zip(records[::2], records[1::2]):
        assert left["analysis_split"] == right["analysis_split"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_synthetic_cache_shapes_ids_and_metadata(Path(directory))
    test_stimulus_split_is_deterministic_and_grouped()
    print("Synthetic representation-cache tests passed")
