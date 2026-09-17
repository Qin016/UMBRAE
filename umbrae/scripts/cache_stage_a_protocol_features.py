#!/usr/bin/env python
"""Build the locked Stage-A fMRI/CLIP cache from protocol manifests."""

import argparse
import io
import json
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.clip_patch_teacher import FixedCLIPPatchTeacher
from models.dual_branch_cache import sha256_file


def parse_args():
    parser = argparse.ArgumentParser(description="Cache Protocol-V1 Stage-A inputs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-version", default="protocol_v1")
    parser.add_argument("--clip-model", default=FixedCLIPPatchTeacher.MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verification-samples", type=int, default=3)
    return parser.parse_args()


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def read_npy(archive, name):
    handle = archive.extractfile(name)
    if handle is None:
        raise FileNotFoundError(name)
    return np.load(io.BytesIO(handle.read()), allow_pickle=False)


def load_row(archive, row):
    key = row["sample_id"]
    fmri = read_npy(archive, f"{key}.nsdgeneral.npy")
    repeats = int(read_npy(archive, f"{key}.num_uniques.npy").reshape(-1)[0])
    if fmri.ndim == 2:
        if repeats < 1 or repeats > fmri.shape[0]:
            raise ValueError(f"{key}: invalid num_uniques={repeats}")
        fmri = fmri[:repeats].astype(np.float32).mean(axis=0)
    elif fmri.ndim != 1:
        raise ValueError(f"{key}: unexpected fMRI shape {fmri.shape}")
    image_handle = archive.extractfile(row["image_id"])
    if image_handle is None:
        raise FileNotFoundError(row["image_id"])
    image = Image.open(io.BytesIO(image_handle.read())).convert("RGB")
    image = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return np.asarray(fmri, dtype=np.float32), image


def infer_patches(teacher, images, device):
    images = torch.stack(images).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")
    ):
        patches = teacher(images)
    return patches.float().cpu().numpy()


def run(args):
    if args.clip_model != FixedCLIPPatchTeacher.MODEL_NAME:
        raise ValueError("Protocol V1 forbids changing the CLIP teacher")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_rows(args.manifest)
    if any(row["split"] != args.split for row in rows):
        raise ValueError("Manifest contains rows from a different split")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest sample IDs are not unique")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fmri_path = output / "fmri_mean_valid_repeats.npy"
    patch_path = output / "clip_patch_fp16.npy"
    ids_path = output / "sample_ids.json"
    teacher = FixedCLIPPatchTeacher(args.clip_model).to(args.device).eval()
    voxel_count = None
    fmri_cache = patch_cache = None
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["shard"]].append((index, row))
    completed = 0
    for shard, entries in grouped.items():
        with tarfile.open(shard) as archive:
            for start in range(0, len(entries), args.batch_size):
                chunk = entries[start : start + args.batch_size]
                fmri_batch, image_batch = [], []
                for _, row in chunk:
                    fmri, image = load_row(archive, row)
                    fmri_batch.append(fmri)
                    image_batch.append(image)
                if voxel_count is None:
                    voxel_count = int(fmri_batch[0].shape[0])
                    fmri_cache = np.lib.format.open_memmap(
                        fmri_path, mode="w+", dtype=np.float32,
                        shape=(len(rows), voxel_count),
                    )
                    patch_cache = np.lib.format.open_memmap(
                        patch_path, mode="w+", dtype=np.float16,
                        shape=(len(rows), teacher.num_patch_tokens, teacher.hidden_dim),
                    )
                indices = [index for index, _ in chunk]
                fmri_cache[indices] = np.stack(fmri_batch)
                patch_cache[indices] = infer_patches(teacher, image_batch, args.device).astype(np.float16)
                completed += len(chunk)
                if completed % 512 < len(chunk) or completed == len(rows):
                    print(json.dumps({"split": args.split, "cached": completed, "total": len(rows)}), flush=True)
    fmri_cache.flush()
    patch_cache.flush()
    ids_path.write_text(json.dumps([row["sample_id"] for row in rows], indent=2))

    checks = []
    chosen = random.Random(args.seed).sample(range(len(rows)), min(args.verification_samples, len(rows)))
    for index in chosen:
        row = rows[index]
        shard_entries = grouped[row["shard"]]
        local_position = next(
            position for position, (global_index, _) in enumerate(shard_entries)
            if global_index == index
        )
        chunk_start = (local_position // args.batch_size) * args.batch_size
        verification_chunk = shard_entries[chunk_start : chunk_start + args.batch_size]
        images = []
        with tarfile.open(row["shard"]) as archive:
            for _, verification_row in verification_chunk:
                _, image = load_row(archive, verification_row)
                images.append(image)
        online = infer_patches(teacher, images, args.device)[local_position - chunk_start]
        cached = np.asarray(patch_cache[index], dtype=np.float32)
        difference = np.abs(online - cached)
        online_flat = online.reshape(-1).astype(np.float64)
        cached_flat = cached.reshape(-1).astype(np.float64)
        checks.append({
            "sample_id": row["sample_id"],
            "comparison_batch_size": len(verification_chunk),
            "cosine_similarity": float(
                np.dot(online_flat, cached_flat)
                / (np.linalg.norm(online_flat) * np.linalg.norm(cached_flat))
            ),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
        })
    metadata = {
        "protocol_version": args.protocol_version,
        "split": args.split,
        "sample_count": len(rows),
        "sample_id_index": str(ids_path),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "repeat_policy": "mean_over_num_uniques_valid_repeats",
        "fmri_path": str(fmri_path),
        "fmri_dtype": "float32",
        "fmri_shape": [len(rows), voxel_count],
        "clip_path": str(patch_path),
        "clip_model": args.clip_model,
        "clip_feature_definition": FixedCLIPPatchTeacher.LAYER_DEFINITION,
        "clip_preprocessing": {
            "input_color": "RGB", "resize": 224, "interpolation": "bicubic",
            "antialias": True, "center_crop": 224,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
        "clip_dtype": "float16",
        "clip_shape": [len(rows), teacher.num_patch_tokens, teacher.hidden_dim],
        "online_equivalence": checks,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    run(parse_args())
