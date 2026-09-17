#!/usr/bin/env python
"""Cache frozen NeuroRoute ROI and CLIP representations for offline analysis.

This script is intentionally inference-only. It never imports UMBRAE/Shikra,
never constructs an optimizer, and never writes to existing experiment roots.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import math
import os
import re
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.brain_clip_projector import BrainToCLIPProjector
from models.clip_layer_bank import CLIPLayerBank
from models.roi_tokenizer import ROITokenizer


DEFAULT_LAYERS = [4, 8, 12, 16, 20, 24]
ANALYSIS_SPLIT_NAMES = ["geometry_fit", "feature_cost_fit", "heldout_eval"]
ANALYSIS_SPLIT_RATIOS = [0.6, 0.2, 0.2]
POOLING_METHOD = "mean_non_cls_patch_tokens"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache frozen subj01 NeuroRoute ROI and CLIP representations "
            "for offline RSA/FGW analysis"
        )
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--data-tar",
        "--dataset-path",
        dest="data_paths",
        nargs="+",
        required=True,
        help="Tar paths, glob patterns, or directories containing tar shards",
    )
    parser.add_argument("--roi-indices-path", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument(
        "--clip-model-name-or-path",
        help="Defaults to the exact value stored in the Stage-1 checkpoint",
    )
    parser.add_argument(
        "--selected-clip-layers",
        type=int,
        nargs="+",
        help="Defaults to checkpoint layers; must match the checkpoint",
    )
    parser.add_argument("--clip-layer-target-dim", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=ANALYSIS_SPLIT_RATIOS,
        metavar=("GEOMETRY", "FEATURE_COST", "HELDOUT"),
    )
    return parser.parse_args()


def _natural_key(path: Path) -> List[object]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path))
    ]


def expand_data_paths(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in values:
        expanded = Path(raw).expanduser()
        if expanded.is_dir():
            matches = [str(path) for path in expanded.rglob("*.tar")]
        else:
            matches = glob.glob(str(expanded))
            if not matches and expanded.is_file():
                matches = [str(expanded)]
        paths.extend(Path(match).resolve() for match in matches)
    unique = sorted(dict.fromkeys(paths), key=_natural_key)
    if not unique:
        raise FileNotFoundError(f"No tar shards matched: {values}")
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing tar shards: {missing}")
    return unique


def load_verified_roi_mapping(
    path: str, expected_subject: Optional[str] = None
) -> tuple[List[str], Dict[str, List[int]]]:
    mapping_path = Path(path).expanduser().resolve()
    payload = json.loads(mapping_path.read_text())
    if not payload.get("roi_mapping_is_real", False):
        raise ValueError("ROI mapping is not marked as real")
    if not payload.get("voxel_order_verified", False):
        raise ValueError("ROI mapping voxel order is not verified")
    if expected_subject is not None and payload.get("subject") != expected_subject:
        raise ValueError(
            "ROI mapping subject does not match --subject: "
            f"{payload.get('subject')!r} != {expected_subject!r}"
        )
    roi_names = list(payload["roi_names"])
    roi_indices = {
        name: list(payload["rois"][name]["indices"]) for name in roi_names
    }
    empty = [name for name, values in roi_indices.items() if not values]
    if empty:
        raise ValueError(f"Required ROI mappings are empty: {empty}")
    return roi_names, roi_indices


def _member_reference(member: tarfile.TarInfo) -> Dict[str, int]:
    return {"offset": int(member.offset_data), "size": int(member.size)}


def _sample_number(sample_key: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", sample_key)
    return (int(match.group(1)), sample_key) if match else (math.inf, sample_key)


def build_sample_index(
    tar_paths: Sequence[Path], max_samples: Optional[int] = None
) -> List[Dict[str, object]]:
    """Index valid fMRI/image groups without decoding large members."""
    records: List[Dict[str, object]] = []
    for tar_path in tar_paths:
        grouped: Dict[str, Dict[str, tarfile.TarInfo]] = {}
        with tarfile.open(tar_path) as archive:
            for member in archive.getmembers():
                if not member.isfile() or "." not in member.name:
                    continue
                prefix, suffix = member.name.split(".", 1)
                grouped.setdefault(prefix, {})[suffix] = member

            for sample_key, members in grouped.items():
                fmri_member = members.get("nsdgeneral.npy")
                image_suffix = next(
                    (suffix for suffix in ("jpg", "png") if suffix in members),
                    None,
                )
                if fmri_member is None or image_suffix is None:
                    continue
                coco_member = members.get("coco73k.npy")
                repeat_member = members.get("num_uniques.npy")
                trial_member = members.get("trial.npy")
                coco_id = (
                    int(
                        np.asarray(
                            np.load(
                                io.BytesIO(
                                    archive.extractfile(coco_member).read()
                                ),
                                allow_pickle=False,
                            )
                        ).reshape(-1)[0]
                    )
                    if coco_member is not None
                    else None
                )
                num_repeats = (
                    int(
                        np.asarray(
                            np.load(
                                io.BytesIO(
                                    archive.extractfile(repeat_member).read()
                                ),
                                allow_pickle=False,
                            )
                        ).reshape(-1)[0]
                    )
                    if repeat_member is not None
                    else None
                )
                trial_id = (
                    int(
                        np.asarray(
                            np.load(
                                io.BytesIO(
                                    archive.extractfile(trial_member).read()
                                ),
                                allow_pickle=False,
                            )
                        ).reshape(-1)[0]
                    )
                    if trial_member is not None
                    else None
                )
                stable_id = (
                    f"coco73k:{coco_id}"
                    if coco_id is not None
                    else f"sample:{sample_key}"
                )
                records.append(
                    {
                        "sample_key": sample_key,
                        "stable_stimulus_id": stable_id,
                        "coco73k_id": coco_id,
                        "trial_id": trial_id,
                        "number_of_repeats": num_repeats,
                        "source_tar": str(tar_path),
                        "_fmri_member": _member_reference(fmri_member),
                        "_image_member": _member_reference(
                            members[image_suffix]
                        ),
                        "_image_suffix": image_suffix,
                    }
                )
    records.sort(
        key=lambda record: (
            _sample_number(str(record["sample_key"])),
            str(record["source_tar"]),
        )
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        records = records[:max_samples]
    if not records:
        raise ValueError("No nsdgeneral/image sample pairs were found")
    for index, record in enumerate(records):
        record["sample_index"] = index
    return records


def _normalized_split_counts(count: int, ratios: Sequence[float]) -> List[int]:
    values = np.asarray(ratios, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("split_ratios must contain three finite values")
    if np.any(values < 0) or values.sum() <= 0:
        raise ValueError("split_ratios must be non-negative with positive sum")
    values = values / values.sum()
    expected = values * count
    result = np.floor(expected).astype(np.int64)
    remainder = int(count - result.sum())
    fractional_order = np.argsort(-(expected - result), kind="stable")
    for position in fractional_order[:remainder]:
        result[position] += 1
    return result.tolist()


def assign_stimulus_splits(
    records: List[Dict[str, object]],
    seed: int,
    ratios: Sequence[float] = ANALYSIS_SPLIT_RATIOS,
) -> Dict[str, List[str]]:
    """Assign all occurrences of one stable stimulus ID to one fixed split."""
    unique_ids = sorted(
        {str(record["stable_stimulus_id"]) for record in records}
    )
    generator = np.random.default_rng(seed)
    shuffled = [unique_ids[index] for index in generator.permutation(len(unique_ids))]
    counts = _normalized_split_counts(len(shuffled), ratios)
    split_ids: Dict[str, List[str]] = {}
    offset = 0
    for name, count in zip(ANALYSIS_SPLIT_NAMES, counts):
        split_ids[name] = sorted(shuffled[offset : offset + count])
        offset += count
    id_to_split = {
        stimulus_id: name
        for name, ids in split_ids.items()
        for stimulus_id in ids
    }
    local_counts = {name: 0 for name in ANALYSIS_SPLIT_NAMES}
    for record in records:
        name = id_to_split[str(record["stable_stimulus_id"])]
        record["analysis_split"] = name
        record["split_sample_index"] = local_counts[name]
        local_counts[name] += 1
    return split_ids


class OfflineNSDCacheDataset(Dataset):
    """Read mean-valid-repeat fMRI and images in deterministic index order."""

    def __init__(self, records: Sequence[Mapping[str, object]]) -> None:
        self.records = list(records)
        self._handles: Dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _read(self, path: str, member: Mapping[str, object]) -> bytes:
        handle = self._handles.get(path)
        if handle is None or handle.closed:
            handle = Path(path).open("rb")
            self._handles[path] = handle
        handle.seek(int(member["offset"]))
        payload = handle.read(int(member["size"]))
        if len(payload) != int(member["size"]):
            raise IOError(f"Short read from {path}")
        return payload

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        path = str(record["source_tar"])
        fmri = np.load(
            io.BytesIO(self._read(path, record["_fmri_member"])),
            allow_pickle=False,
        )
        if fmri.ndim == 1:
            valid_repeats = 1
            mean_fmri = fmri.astype(np.float32)
        elif fmri.ndim == 2:
            declared = record.get("number_of_repeats")
            valid_repeats = int(declared) if declared is not None else fmri.shape[0]
            if not 1 <= valid_repeats <= fmri.shape[0]:
                raise ValueError(
                    f"Invalid number_of_repeats={valid_repeats} for "
                    f"{record['sample_key']} with fMRI shape {fmri.shape}"
                )
            mean_fmri = fmri[:valid_repeats].astype(np.float32).mean(axis=0)
        else:
            raise ValueError(
                f"Expected nsdgeneral [V] or [R,V], got {fmri.shape}"
            )
        image = Image.open(
            io.BytesIO(self._read(path, record["_image_member"]))
        ).convert("RGB")
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return {
            "fmri": torch.from_numpy(mean_fmri),
            "image": torch.from_numpy(image_array.copy()).permute(2, 0, 1),
            "record_index": index,
            "valid_repeats": valid_repeats,
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()


class FrozenROIRepresentationEncoder(nn.Module):
    """The exact Stage-1 fMRI tokenizer/projector branch, without CLIP/router."""

    def __init__(
        self,
        roi_names: Sequence[str],
        roi_indices: Mapping[str, Sequence[int]],
        checkpoint: Mapping[str, object],
    ) -> None:
        super().__init__()
        config = checkpoint.get("config", {})
        state = checkpoint.get("model", checkpoint)
        dim = int(config.get("roi_token_dim", 1024))
        self.roi_tokenizer = ROITokenizer(
            roi_names=roi_names,
            roi_indices=roi_indices,
            token_dim=dim,
            tokenizer_type="shared_mlp",
            use_roi_embeddings=True,
        )
        use_projector = bool(config.get("use_brain_clip_projector", True))
        self.brain_clip_projector = (
            BrainToCLIPProjector(
                dim,
                int(config.get("clip_layer_target_dim", dim)),
                projector_type=config.get("projector_type", "mlp"),
                hidden_dim=config.get("projector_hidden_dim"),
                dropout=float(config.get("projector_dropout", 0.1)),
            )
            if use_projector
            else nn.Identity()
        )
        tokenizer_state = {
            key[len("roi_tokenizer.") :]: value
            for key, value in state.items()
            if key.startswith("roi_tokenizer.")
        }
        projector_state = {
            key[len("brain_clip_projector.") :]: value
            for key, value in state.items()
            if key.startswith("brain_clip_projector.")
        }
        self.roi_tokenizer.load_state_dict(tokenizer_state, strict=True)
        if use_projector:
            self.brain_clip_projector.load_state_dict(
                projector_state, strict=True
            )
        self.roi_names = list(roi_names)
        self.roi_dim = dim
        self.output_dim = int(config.get("clip_layer_target_dim", dim))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, fmri: Tensor) -> Dict[str, Tensor]:
        raw = self.roi_tokenizer(fmri)["roi_tokens"]
        projected = self.brain_clip_projector(raw)
        return {
            "brain_roi_features": raw,
            "projected_roi_features": projected,
        }


def _torch_load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        return torch.load(
            str(path), map_location="cpu", mmap=True, weights_only=False
        )
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def build_frozen_models(
    checkpoint_path: Path,
    roi_names: Sequence[str],
    roi_indices: Mapping[str, Sequence[int]],
    clip_model_name_or_path: Optional[str],
    selected_layers: Optional[Sequence[int]],
    target_dim: Optional[int],
    expected_subject: Optional[str] = None,
) -> tuple[nn.Module, nn.Module, Dict[str, object]]:
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    config = dict(checkpoint.get("config", {}))
    checkpoint_subject = config.get("subject")
    if (
        expected_subject is not None
        and checkpoint_subject is not None
        and checkpoint_subject != expected_subject
    ):
        raise ValueError(
            "Stage-1 checkpoint subject does not match --subject: "
            f"{checkpoint_subject!r} != {expected_subject!r}"
        )
    checkpoint_layers = list(config.get("selected_clip_layers", DEFAULT_LAYERS))
    layers = list(selected_layers) if selected_layers is not None else checkpoint_layers
    if layers != checkpoint_layers:
        raise ValueError(
            "selected CLIP layers must match the Stage-1 checkpoint exactly: "
            f"requested={layers}, checkpoint={checkpoint_layers}"
        )
    checkpoint_target_dim = int(config.get("clip_layer_target_dim", 1024))
    resolved_target_dim = int(target_dim or checkpoint_target_dim)
    if resolved_target_dim != checkpoint_target_dim:
        raise ValueError(
            "clip-layer-target-dim must match the Stage-1 checkpoint: "
            f"{resolved_target_dim} != {checkpoint_target_dim}"
        )
    model_path = clip_model_name_or_path or config.get(
        "clip_model_name_or_path", "openai/clip-vit-large-patch14"
    )
    roi_encoder = FrozenROIRepresentationEncoder(
        roi_names, roi_indices, checkpoint
    )
    clip_bank = CLIPLayerBank(
        model_name_or_path=model_path,
        selected_layers=layers,
        target_dim=resolved_target_dim,
        freeze_clip=True,
    )
    state = checkpoint.get("model", checkpoint)
    clip_state = {
        key[len("clip_layer_bank.") :]: value
        for key, value in state.items()
        if key.startswith("clip_layer_bank.")
    }
    if not clip_state:
        raise ValueError("Stage-1 checkpoint contains no clip_layer_bank weights")
    clip_bank.load_state_dict(clip_state, strict=True)
    clip_bank.requires_grad_(False)
    clip_bank.eval()
    resolved = {
        "clip_model_name_or_path": str(model_path),
        "selected_clip_layers": layers,
        "clip_layer_target_dim": resolved_target_dim,
        "roi_token_dim": roi_encoder.roi_dim,
        "stage1_config": config,
    }
    return roi_encoder, clip_bank, resolved


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ordered_id_sha256(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["stable_stimulus_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sample_key"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def public_metadata_record(
    record: Mapping[str, object], subject: str, source_split: str
) -> Dict[str, object]:
    return {
        "sample_index": int(record["sample_index"]),
        "split_sample_index": int(record["split_sample_index"]),
        "sample_key": str(record["sample_key"]),
        "stable_stimulus_id": str(record["stable_stimulus_id"]),
        "coco73k_id": record.get("coco73k_id"),
        "trial_id": record.get("trial_id"),
        "number_of_repeats": record.get("number_of_repeats"),
        "subject": subject,
        "dataset_split": source_split,
        "analysis_split": str(record["analysis_split"]),
        "source_tar": str(record["source_tar"]),
    }


def _unique_statistics(records: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    ids = [str(record["stable_stimulus_id"]) for record in records]
    unique = len(set(ids))
    return {
        "sample_count": len(ids),
        "unique_stimulus_id_count": unique,
        "duplicate_stimulus_row_count": len(ids) - unique,
    }


def _open_split_arrays(
    root: Path,
    split_records: Mapping[str, Sequence[Mapping[str, object]]],
    roi_count: int,
    layer_count: int,
    roi_dim: int,
    clip_dim: int,
) -> Dict[str, Dict[str, np.memmap]]:
    result: Dict[str, Dict[str, np.memmap]] = {}
    for name, records in split_records.items():
        split_dir = root / name
        split_dir.mkdir(parents=True, exist_ok=False)
        count = len(records)
        result[name] = {
            "brain_roi_features": np.lib.format.open_memmap(
                split_dir / "brain_roi_features.npy",
                mode="w+",
                dtype=np.float32,
                shape=(count, roi_count, roi_dim),
            ),
            "projected_roi_features": np.lib.format.open_memmap(
                split_dir / "projected_roi_features.npy",
                mode="w+",
                dtype=np.float32,
                shape=(count, roi_count, clip_dim),
            ),
            "clip_layer_features": np.lib.format.open_memmap(
                split_dir / "clip_layer_features.npy",
                mode="w+",
                dtype=np.float32,
                shape=(count, layer_count, clip_dim),
            ),
        }
    return result


def _validate_feature_batch(
    raw: Tensor,
    projected: Tensor,
    clip: Tensor,
    batch_size: int,
    roi_count: int,
    layer_count: int,
    roi_dim: int,
    clip_dim: int,
) -> None:
    expected = {
        "brain_roi_features": (batch_size, roi_count, roi_dim),
        "projected_roi_features": (batch_size, roi_count, clip_dim),
        "clip_layer_features": (batch_size, layer_count, clip_dim),
    }
    actual = {
        "brain_roi_features": tuple(raw.shape),
        "projected_roi_features": tuple(projected.shape),
        "clip_layer_features": tuple(clip.shape),
    }
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(
                f"{name} shape {actual[name]} != expected {expected[name]}"
            )
    for name, value in (
        ("brain_roi_features", raw),
        ("projected_roi_features", projected),
        ("clip_layer_features", clip),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")


def cache_representations(
    records: List[Dict[str, object]],
    roi_encoder: nn.Module,
    clip_bank: nn.Module,
    output_dir: Path,
    roi_names: Sequence[str],
    selected_layers: Sequence[int],
    roi_dim: int,
    clip_dim: int,
    subject: str,
    source_split: str,
    seed: int,
    batch_size: int = 16,
    num_workers: int = 0,
    device: str = "cpu",
    provenance: Optional[Mapping[str, object]] = None,
    dataset_override: Optional[Dataset] = None,
) -> Dict[str, object]:
    """Compute and atomically publish three offline NumPy caches."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing cache directory: {output_dir}"
        )
    temp_dir = output_dir.with_name(
        f".{output_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    temp_dir.mkdir(parents=True, exist_ok=False)
    split_records = {
        name: [
            record for record in records if record["analysis_split"] == name
        ]
        for name in ANALYSIS_SPLIT_NAMES
    }
    if sum(len(value) for value in split_records.values()) != len(records):
        raise ValueError("Analysis split assignment does not cover every record")
    dataset = OfflineNSDCacheDataset(records) if "_fmri_member" in records[0] else None
    try:
        arrays = _open_split_arrays(
            temp_dir,
            split_records,
            len(roi_names),
            len(selected_layers),
            roi_dim,
            clip_dim,
        )
        active_dataset = dataset or dataset_override
        if active_dataset is None:
            raise ValueError("A dataset is required for cache computation")
        loader = DataLoader(
            active_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
        )
        roi_encoder = roi_encoder.to(device).eval()
        clip_bank = clip_bank.to(device).eval()
        processed = 0
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                fmri = batch["fmri"].to(device, non_blocking=True)
                images = batch["image"].to(device, non_blocking=True)
                roi_output = roi_encoder(fmri)
                clip_output = clip_bank(images)
                raw = roi_output["brain_roi_features"]
                projected = roi_output["projected_roi_features"]
                clip = clip_output["pooled_tokens"]
                current_batch = int(fmri.shape[0])
                _validate_feature_batch(
                    raw,
                    projected,
                    clip,
                    current_batch,
                    len(roi_names),
                    len(selected_layers),
                    roi_dim,
                    clip_dim,
                )
                raw_np = raw.detach().cpu().to(torch.float32).numpy()
                projected_np = (
                    projected.detach().cpu().to(torch.float32).numpy()
                )
                clip_np = clip.detach().cpu().to(torch.float32).numpy()
                record_indices = batch["record_index"].cpu().numpy().tolist()
                for row, record_index in enumerate(record_indices):
                    record = records[int(record_index)]
                    split = str(record["analysis_split"])
                    target = int(record["split_sample_index"])
                    arrays[split]["brain_roi_features"][target] = raw_np[row]
                    arrays[split]["projected_roi_features"][target] = projected_np[row]
                    arrays[split]["clip_layer_features"][target] = clip_np[row]
                processed += current_batch
                if batch_index == 0 or (batch_index + 1) % 10 == 0:
                    print(
                        f"cached {processed}/{len(records)} samples",
                        flush=True,
                    )
        if processed != len(records):
            raise RuntimeError(f"Processed N={processed}, expected {len(records)}")
        for split_arrays in arrays.values():
            for array in split_arrays.values():
                array.flush()
        del arrays

        split_summaries = {}
        for name, selected in split_records.items():
            split_dir = temp_dir / name
            expected_shapes = {
                "brain_roi_features": [len(selected), len(roi_names), roi_dim],
                "projected_roi_features": [
                    len(selected),
                    len(roi_names),
                    clip_dim,
                ],
                "clip_layer_features": [
                    len(selected),
                    len(selected_layers),
                    clip_dim,
                ],
            }
            for feature_name, expected_shape in expected_shapes.items():
                array = np.load(
                    split_dir / f"{feature_name}.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                if list(array.shape) != expected_shape:
                    raise ValueError(
                        f"Saved {feature_name} shape {array.shape} != "
                        f"{expected_shape}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"Saved {feature_name} contains NaN/Inf")
            metadata_records = [
                public_metadata_record(record, subject, source_split)
                for record in selected
            ]
            unique_stats = _unique_statistics(selected)
            repeat_counts: Dict[str, int] = {}
            for record in selected:
                key = str(record.get("number_of_repeats"))
                repeat_counts[key] = repeat_counts.get(key, 0) + 1
            metadata = {
                "subject": subject,
                "dataset_split": source_split,
                "analysis_split": name,
                "sample_count": len(selected),
                "unique_stimulus_id_statistics": unique_stats,
                "ordered_sample_id_sha256": ordered_id_sha256(selected),
                "repeat_metadata_available": any(
                    record.get("number_of_repeats") is not None
                    for record in selected
                ),
                "trial_metadata_available": any(
                    record.get("trial_id") is not None for record in selected
                ),
                "number_of_repeats_histogram": repeat_counts,
                "samples": metadata_records,
            }
            (split_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )
            split_config = {
                "offline_analysis_only": True,
                "uses_image_features_only_for_offline_analysis": True,
                "image_features_used_by_brain_branch": False,
                "image_features_exported_to_training_or_inference": False,
                "subject": subject,
                "dataset_split": source_split,
                "analysis_split": name,
                "sample_count": len(selected),
                "roi_names": list(roi_names),
                "selected_clip_layers": list(selected_layers),
                "clip_pooling_method": POOLING_METHOD,
                "feature_dimensions": {
                    "brain_roi_features": expected_shapes[
                        "brain_roi_features"
                    ],
                    "projected_roi_features": expected_shapes[
                        "projected_roi_features"
                    ],
                    "clip_layer_features": expected_shapes[
                        "clip_layer_features"
                    ],
                    "D_roi": roi_dim,
                    "D_clip": clip_dim,
                },
                "storage": {
                    "format": "numpy_npy_float32",
                    "files": [
                        "brain_roi_features.npy",
                        "projected_roi_features.npy",
                        "clip_layer_features.npy",
                    ],
                },
                "deterministic_ordering": {
                    "enabled": True,
                    "source_order": "numeric sample_key then source_tar",
                    "ordered_sample_id_sha256": ordered_id_sha256(selected),
                    "random_seed": seed,
                },
                **dict(provenance or {}),
            }
            (split_dir / "cache_config.json").write_text(
                json.dumps(split_config, indent=2)
            )
            split_summaries[name] = {
                "sample_count": len(selected),
                "unique_stimulus_id_count": unique_stats[
                    "unique_stimulus_id_count"
                ],
                "ordered_sample_id_sha256": ordered_id_sha256(selected),
                "relative_path": name,
            }

        root_metadata = {
            "subject": subject,
            "dataset_split": source_split,
            "sample_count": len(records),
            "unique_stimulus_id_statistics": _unique_statistics(records),
            "split_ids": {
                name: sorted(
                    {
                        str(record["stable_stimulus_id"])
                        for record in selected
                    }
                )
                for name, selected in split_records.items()
            },
            "splits": split_summaries,
        }
        (temp_dir / "metadata.json").write_text(
            json.dumps(root_metadata, indent=2)
        )
        root_config = {
            "offline_analysis_only": True,
            "uses_image_features_only_for_offline_analysis": True,
            "image_features_used_by_brain_branch": False,
            "image_features_exported_to_training_or_inference": False,
            "subject": subject,
            "dataset_split": source_split,
            "sample_count": len(records),
            "roi_names": list(roi_names),
            "selected_clip_layers": list(selected_layers),
            "clip_pooling_method": POOLING_METHOD,
            "feature_dimensions": {
                "D_roi": roi_dim,
                "D_clip": clip_dim,
                "R": len(roi_names),
                "L": len(selected_layers),
            },
            "random_seed": seed,
            "analysis_split_names": ANALYSIS_SPLIT_NAMES,
            "splits": split_summaries,
            "integrity_checks": {
                "no_nan_or_inf": True,
                "matching_sample_axis": True,
                "exact_tensor_dimensions": True,
                "deterministic_ordering": True,
                "unique_stimulus_statistics_recorded": True,
            },
            **dict(provenance or {}),
        }
        (temp_dir / "cache_config.json").write_text(
            json.dumps(root_config, indent=2)
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_dir, output_dir)
        return root_config
    except Exception:
        print(f"Incomplete cache retained for diagnosis: {temp_dir}", flush=True)
        raise
    finally:
        if dataset is not None:
            dataset.close()


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    tar_paths = expand_data_paths(args.data_paths)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing cache directory: {output_dir}"
        )
    mapping_path = Path(args.roi_indices_path).expanduser().resolve()
    checkpoint_path = Path(args.stage1_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    roi_names, roi_indices = load_verified_roi_mapping(
        str(mapping_path), expected_subject=args.subject
    )
    print(f"Indexing {len(tar_paths)} tar shards...", flush=True)
    records = build_sample_index(tar_paths, args.max_samples)
    split_ids = assign_stimulus_splits(
        records, args.random_seed, args.split_ratios
    )
    print(
        "Indexed "
        f"{len(records)} rows / {len({r['stable_stimulus_id'] for r in records})} "
        "unique stimuli; split unique-ID counts="
        + str({name: len(ids) for name, ids in split_ids.items()}),
        flush=True,
    )
    roi_encoder, clip_bank, resolved = build_frozen_models(
        checkpoint_path,
        roi_names,
        roi_indices,
        args.clip_model_name_or_path,
        args.selected_clip_layers,
        args.clip_layer_target_dim,
        expected_subject=args.subject,
    )
    provenance = {
        "stage1_checkpoint": str(checkpoint_path),
        "stage1_checkpoint_sha256": sha256_file(checkpoint_path),
        "roi_indices_path": str(mapping_path),
        "roi_indices_sha256": sha256_file(mapping_path),
        "clip_model_name_or_path": resolved["clip_model_name_or_path"],
        "source_tar_manifest": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in tar_paths
        ],
        "repeat_reduction": "mean_of_first_num_uniques_rows_in_float32",
        "split_protocol": {
            "type": "fixed_stimulus_level",
            "names": ANALYSIS_SPLIT_NAMES,
            "ratios_requested": list(args.split_ratios),
            "assignment": (
                "sorted unique stable IDs, NumPy default_rng(seed) permutation, "
                "largest-remainder counts"
            ),
            "seed": args.random_seed,
        },
    }
    config = cache_representations(
        records=records,
        roi_encoder=roi_encoder,
        clip_bank=clip_bank,
        output_dir=output_dir,
        roi_names=roi_names,
        selected_layers=resolved["selected_clip_layers"],
        roi_dim=resolved["roi_token_dim"],
        clip_dim=resolved["clip_layer_target_dim"],
        subject=args.subject,
        source_split=args.split_name,
        seed=args.random_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        provenance=provenance,
    )
    print(
        f"Saved offline-only representation cache to {output_dir} "
        f"with N={config['sample_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
