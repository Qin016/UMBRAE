import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dual_branch_cache import build_feature_cache_metadata, validate_cache_stage


def test_cache_metadata_and_stage_c_guard(tmp_path):
    checkpoint = tmp_path / "brainx.pt"
    checkpoint.write_bytes(b"checkpoint")
    metadata = build_feature_cache_metadata(
        sample_id="sample1",
        subject="subj01",
        brainx_checkpoint=str(checkpoint),
        clip_model="openai/clip-vit-large-patch14",
        clip_layer_definition="hidden_states[-2][:,1:,:]",
        repeat_aggregation_policy="mean",
        dtype="float16",
        z_sem_shape=(256, 1024),
        v_patch_shape=(256, 1024),
    )
    assert metadata["h_struct_cached"] is False
    assert metadata["z_sem_valid_stages"] == ["B"]
    assert len(metadata["brainx_checkpoint_sha256"]) == 64
    validate_cache_stage("A")
    validate_cache_stage("B")
    with pytest.raises(ValueError, match="forbidden in Stage C"):
        validate_cache_stage("C")
