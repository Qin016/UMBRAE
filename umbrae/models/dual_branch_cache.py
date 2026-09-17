"""Metadata contract for Stage-B frozen feature caches.

Z_sem caching is valid only while the semantic backbone is fully frozen.
It must be disabled once Stage C introduces trainable LoRA inside BrainX.
H_struct is intentionally never cacheable because the structural branch trains.
"""

import hashlib
from pathlib import Path
from typing import Dict, Sequence


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache_stage(stage: str) -> None:
    normalized = stage.upper()
    if normalized == "C":
        raise ValueError("Z_sem cache is forbidden in Stage C because LoRA changes Z_sem")
    if normalized not in {"A", "B"}:
        raise ValueError(f"Unsupported stage: {stage!r}")


def build_feature_cache_metadata(
    *,
    sample_id: str,
    subject: str,
    brainx_checkpoint: str,
    clip_model: str,
    clip_layer_definition: str,
    repeat_aggregation_policy: str,
    dtype: str,
    z_sem_shape: Sequence[int],
    v_patch_shape: Sequence[int],
) -> Dict[str, object]:
    return {
        "sample_id": str(sample_id),
        "subject": str(subject),
        "brainx_checkpoint_path": str(Path(brainx_checkpoint).expanduser().resolve()),
        "brainx_checkpoint_sha256": sha256_file(brainx_checkpoint),
        "clip_model": str(clip_model),
        "clip_layer_definition": str(clip_layer_definition),
        "repeat_aggregation_policy": str(repeat_aggregation_policy),
        "dtype": str(dtype),
        "z_sem_shape": [int(value) for value in z_sem_shape],
        "v_patch_shape": [int(value) for value in v_patch_shape],
        "h_struct_cached": False,
        "z_sem_valid_stages": ["B"],
        "z_sem_forbidden_stage": "C",
    }
