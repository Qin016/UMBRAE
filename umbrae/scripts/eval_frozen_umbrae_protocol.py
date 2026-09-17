#!/usr/bin/env python
"""Frozen BASELINE_UMBRAE_V1 retrieval/RSA evaluation; no training."""

import argparse
import json
import random
import sys
from pathlib import Path

import braceexpand
import numpy as np
import torch
import webdataset as wds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.clip_patch_teacher import FixedCLIPPatchTeacher
from models.dual_branch_cache import sha256_file
from models.protocol_evaluation import (
    mean_pool_and_normalize,
    retrieval_metrics,
    rsa_metrics,
)
from models.protocol_metadata import save_experiment_snapshot
from models.umbrae_backbone import FrozenUMBRAEEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--brainx-checkpoint", required=True)
    parser.add_argument("--roi-mapping", required=True)
    parser.add_argument("--protocol-version", default="protocol_v1")
    parser.add_argument("--clip-model", default=FixedCLIPPatchTeacher.MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_shards(data_root: Path, split: str, subject: str):
    values = sorted((data_root / split).glob(f"{split}_{subject}_*.tar"))
    if not values:
        raise FileNotFoundError(f"No {split} shards found for {subject}")
    return values


def make_loader(shards, batch_size, num_workers):
    urls = [str(path.resolve()) for path in shards]
    dataset = (
        wds.WebDataset(urls, resampled=False, nodesplitter=lambda values: values)
        .decode("torch")
        .rename(
            images="jpg;png",
            voxels="nsdgeneral.npy",
            reps="num_uniques.npy",
            coco="coco73k.npy",
        )
        .to_tuple("__key__", "voxels", "images", "reps", "coco")
        .batched(batch_size, partial=True)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=num_workers, shuffle=False
    )


def valid_repeat_mean(voxels: torch.Tensor, repeats: torch.Tensor) -> torch.Tensor:
    if voxels.ndim == 2:
        return voxels.float()
    if voxels.ndim != 3:
        raise ValueError(f"Expected [B,R,V] fMRI, got {tuple(voxels.shape)}")
    counts = repeats.reshape(-1).to(dtype=torch.long)
    rows = []
    for row, count in zip(voxels, counts):
        count_value = int(count)
        if count_value < 1 or count_value > row.shape[0]:
            raise ValueError(f"Invalid num_uniques={count_value}")
        rows.append(row[:count_value].float().mean(dim=0))
    return torch.stack(rows)


def run(args):
    if args.split == "test" and not args.allow_test:
        raise PermissionError(
            "Test set is sealed. Pass --allow-test only for a preregistered final evaluation."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = split_shards(Path(args.data_root).expanduser(), args.split, args.subject)
    loader = make_loader(shards, args.batch_size, args.num_workers)
    brain = FrozenUMBRAEEncoder(args.brainx_checkpoint, args.subject).to(args.device)
    teacher = FixedCLIPPatchTeacher(args.clip_model).to(args.device)
    brain.eval()
    teacher.eval()
    brain_embeddings, visual_embeddings = [], []
    brain_norms, visual_norms, sample_ids, coco_ids = [], [], [], []
    with torch.inference_mode():
        for keys, voxels, images, repeats, coco in loader:
            fmri = valid_repeat_mean(voxels, repeats).to(args.device)
            images = images.to(args.device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=str(args.device).startswith("cuda"),
            ):
                z_sem = brain(fmri)
                v_patch = teacher(images)
            q_sem, q_sem_normalized = mean_pool_and_normalize(z_sem.float())
            q_visual, q_visual_normalized = mean_pool_and_normalize(v_patch.float())
            brain_embeddings.append(q_sem_normalized.cpu())
            visual_embeddings.append(q_visual_normalized.cpu())
            brain_norms.append(q_sem.norm(dim=-1).cpu())
            visual_norms.append(q_visual.norm(dim=-1).cpu())
            sample_ids.extend(str(key) for key in keys)
            coco_ids.extend(int(value) for value in coco.reshape(-1))
    brain_array = torch.cat(brain_embeddings).numpy()
    visual_array = torch.cat(visual_embeddings).numpy()
    brain_norm_array = torch.cat(brain_norms).numpy()
    visual_norm_array = torch.cat(visual_norms).numpy()
    retrieval = retrieval_metrics(brain_array, visual_array)
    rsa = rsa_metrics(brain_array, visual_array)
    metrics = {
        "baseline_name": "BASELINE_UMBRAE_V1",
        "subject": args.subject,
        "split": args.split,
        "repeat_policy": "mean_over_num_uniques_valid_repeats",
        "pooling": "mean_over_256_tokens_then_l2_normalize",
        "sample_count": len(sample_ids),
        "retrieval": retrieval,
        "rsa": rsa,
        "semantic_embedding_norm_mean": float(brain_norm_array.mean()),
        "semantic_embedding_norm_std": float(brain_norm_array.std()),
        "visual_embedding_norm_mean": float(visual_norm_array.mean()),
        "visual_embedding_norm_std": float(visual_norm_array.std()),
    }
    np.save(output_dir / "q_sem_normalized.npy", brain_array)
    np.save(output_dir / "q_visual_normalized.npy", visual_array)
    (output_dir / "sample_ids.json").write_text(json.dumps(sample_ids, indent=2))
    (output_dir / "coco73k_ids.json").write_text(json.dumps(coco_ids, indent=2))
    snapshot_config = vars(args).copy()
    for key in ("data_root", "brainx_checkpoint", "roi_mapping", "output_dir"):
        snapshot_config[key] = str(Path(snapshot_config[key]).expanduser().resolve())
    save_experiment_snapshot(
        str(output_dir),
        config=snapshot_config,
        protocol_version=args.protocol_version,
        checkpoint_sha256=sha256_file(args.brainx_checkpoint),
        roi_mapping_sha256=sha256_file(args.roi_mapping),
        seed=args.seed,
        metrics=metrics,
        worktree=str(Path(__file__).resolve().parents[2]),
    )
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run(parse_args())
