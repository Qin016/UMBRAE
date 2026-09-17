#!/usr/bin/env python
"""Cache frozen Protocol-V1 BrainX semantic tokens for Stage B."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dual_branch_cache import sha256_file
from models.umbrae_backbone import FrozenUMBRAEEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-cache", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--brainx-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verification-samples", type=int, default=3)
    return parser.parse_args()


def infer(model, fmri, device):
    values = torch.from_numpy(np.array(fmri, dtype=np.float32, copy=True)).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")
    ):
        output = model(values)
    return output.float().cpu().numpy()


def run(args):
    source = Path(args.stage_a_cache).expanduser().resolve()
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["split"] != args.split or metadata["repeat_policy"] != "mean_over_num_uniques_valid_repeats":
        raise ValueError("Stage-A source cache violates Stage-B protocol")
    fmri = np.load(metadata["fmri_path"], mmap_mode="r")
    sample_ids = json.loads(Path(metadata["sample_id_index"]).read_text())
    output = Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    semantic_path = output / "z_sem_fp16.npy"
    semantic = np.lib.format.open_memmap(
        semantic_path, mode="w+", dtype=np.float16, shape=(len(fmri), 256, 1024)
    )
    model = FrozenUMBRAEEncoder(args.brainx_checkpoint, "subj01").to(args.device).eval()
    for start in range(0, len(fmri), args.batch_size):
        end = min(start + args.batch_size, len(fmri))
        semantic[start:end] = infer(model, fmri[start:end], args.device).astype(np.float16)
        if end % 512 < args.batch_size or end == len(fmri):
            print(json.dumps({"split": args.split, "cached": end, "total": len(fmri)}), flush=True)
    semantic.flush()
    checks = []
    for index in random.Random(args.seed).sample(range(len(fmri)), min(args.verification_samples, len(fmri))):
        start = index // args.batch_size * args.batch_size
        end = min(start + args.batch_size, len(fmri))
        online = infer(model, fmri[start:end], args.device)[index - start]
        cached = np.asarray(semantic[index], dtype=np.float32)
        left, right = online.reshape(-1).astype(np.float64), cached.reshape(-1).astype(np.float64)
        checks.append({
            "sample_id": sample_ids[index], "comparison_batch_size": end - start,
            "cosine_similarity": float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))),
            "mean_abs_error": float(np.abs(online - cached).mean()),
            "max_abs_error": float(np.abs(online - cached).max()),
        })
    result = {
        "protocol_version": "protocol_v1", "split": args.split,
        "sample_count": len(fmri), "sample_ids": metadata["sample_id_index"],
        "manifest_sha256": metadata["manifest_sha256"],
        "repeat_policy": metadata["repeat_policy"],
        "brainx_checkpoint": str(Path(args.brainx_checkpoint).resolve()),
        "brainx_checkpoint_sha256": sha256_file(args.brainx_checkpoint),
        "brain_model_kind": model.encoder_kind,
        "z_sem_path": str(semantic_path), "z_sem_dtype": "float16",
        "z_sem_shape": [len(fmri), 256, 1024],
        "z_sem_valid_stages": ["B"], "z_sem_forbidden_stage": "C",
        "z_cal_cached": False, "h_struct_cached": False,
        "source_stage_a_cache_metadata": str(source / "metadata.json"),
        "online_equivalence": checks,
    }
    (output / "metadata.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    run(parse_args())
