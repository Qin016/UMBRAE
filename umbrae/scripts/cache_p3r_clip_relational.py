#!/usr/bin/env python
"""Cache frozen multi-layer CLIP pools and block-23 global targets for P3-R."""

import argparse
import json
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.clip_layer_bank import CLIPLayerBank
from models.dual_branch_cache import sha256_file
from scripts.cache_stage_a_protocol_features import load_row, read_rows


LAYERS = [4, 8, 12, 16, 20, 24]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--stage-a-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verification-samples", type=int, default=3)
    return parser.parse_args()


def infer(bank, images, device):
    images = torch.stack(images).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")
    ):
        output = bank(images)["pooled_tokens"]
    return output.float().cpu().numpy()


def run(args):
    rows = read_rows(args.manifest)
    stage_a = Path(args.stage_a_cache).expanduser().resolve()
    base_metadata = json.loads((stage_a / "metadata.json").read_text())
    sample_ids = json.loads(Path(base_metadata["sample_id_index"]).read_text())
    if sample_ids != [row["sample_id"] for row in rows]:
        raise ValueError("P3 cache and manifest sample ordering disagree")
    if base_metadata["split"] != args.split:
        raise ValueError("Stage-A cache split mismatch")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    layer_path = output / "clip_layer_pools_fp16.npy"
    global_path = output / "clip_block23_global_fp16.npy"
    layer_cache = np.lib.format.open_memmap(
        layer_path, mode="w+", dtype=np.float16, shape=(len(rows), len(LAYERS), 1024)
    )
    global_cache = np.lib.format.open_memmap(
        global_path, mode="w+", dtype=np.float16, shape=(len(rows), 1024)
    )
    # Reuse the already verified Protocol-V1 block-23 patch cache.
    block23_patches = np.load(base_metadata["clip_path"], mmap_mode="r")
    for start in range(0, len(rows), 256):
        end = min(start + 256, len(rows))
        global_cache[start:end] = np.asarray(block23_patches[start:end], dtype=np.float32).mean(axis=1).astype(np.float16)
    bank = CLIPLayerBank(selected_layers=LAYERS, target_dim=1024, freeze_clip=True).to(args.device)
    bank.requires_grad_(False).eval()
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["shard"]].append((index, row))
    completed = 0
    for shard, entries in grouped.items():
        with tarfile.open(shard) as archive:
            for start in range(0, len(entries), args.batch_size):
                chunk = entries[start : start + args.batch_size]
                images = [load_row(archive, row)[1] for _, row in chunk]
                indices = [index for index, _ in chunk]
                layer_cache[indices] = infer(bank, images, args.device).astype(np.float16)
                completed += len(chunk)
                if completed % 512 < len(chunk) or completed == len(rows):
                    print(json.dumps({"split": args.split, "cached": completed, "total": len(rows)}), flush=True)
    layer_cache.flush()
    global_cache.flush()
    chosen = random.Random(args.seed).sample(range(len(rows)), min(args.verification_samples, len(rows)))
    checks = []
    for index in chosen:
        row = rows[index]
        entries = grouped[row["shard"]]
        position = next(i for i, (global_index, _) in enumerate(entries) if global_index == index)
        start = position // args.batch_size * args.batch_size
        chunk = entries[start : start + args.batch_size]
        with tarfile.open(row["shard"]) as archive:
            images = [load_row(archive, value)[1] for _, value in chunk]
        online = infer(bank, images, args.device)[position - start]
        cached = np.asarray(layer_cache[index], dtype=np.float32)
        left, right = online.reshape(-1).astype(np.float64), cached.reshape(-1).astype(np.float64)
        checks.append({
            "sample_id": row["sample_id"], "comparison_batch_size": len(chunk),
            "cosine_similarity": float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))),
            "mean_abs_error": float(np.abs(online - cached).mean()),
            "max_abs_error": float(np.abs(online - cached).max()),
        })
    metadata = {
        "protocol_version": "protocol_v1", "split": args.split,
        "sample_count": len(rows), "sample_ids": str(Path(base_metadata["sample_id_index"])),
        "manifest": str(Path(args.manifest).resolve()), "manifest_sha256": sha256_file(args.manifest),
        "selected_clip_layers": LAYERS, "layer_numbering": "one_based_transformer_block",
        "clip_model": "openai/clip-vit-large-patch14",
        "layer_pools_path": str(layer_path), "layer_pools_dtype": "float16",
        "layer_pools_shape": [len(rows), len(LAYERS), 1024],
        "global_visual_path": str(global_path), "global_visual_definition": "hidden_states[-2][:,1:,:].mean(patch_axis)",
        "global_visual_shape": [len(rows), 1024],
        "source_stage_a_cache_metadata": str(stage_a / "metadata.json"),
        "online_equivalence": checks,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    run(parse_args())
