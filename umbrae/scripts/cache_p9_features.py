#!/usr/bin/env python
"""Cache frozen P9 sources and projected CLIP teachers."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.downstream_brain_encoder import DownstreamBrainEncoder
from models.dual_branch_cache import sha256_file
from scripts.cache_p7_downstream_features import load_projector, resolve_config


SPLIT_SIZES = {"train": 8559, "val": 300}


def split_paths(split):
    root = ROOT / "protocol_outputs/protocol_v1/subj01"
    stage = root / "stage_a_cache" / split
    return {
        "fmri": stage / "fmri_mean_valid_repeats.npy",
        "clip": stage / "clip_patch_fp16.npy",
        "base": root / "stage_b_semantic_cache" / split / "z_sem_fp16.npy",
    }


def metadata_path(output):
    return Path(output) / "metadata.json"


def init(args):
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    created = {}
    for split, count in SPLIT_SIZES.items():
        directory = output / split; directory.mkdir(exist_ok=True)
        for name, shape in (("p6_full_real_fp16.npy", (count, 256, 1024)), ("clip_projected_fp16.npy", (count, 256, 4096))):
            path = directory / name
            if not path.exists():
                value = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape); value.flush(); del value
            created[str(path)] = list(shape)
    print(json.dumps(created, indent=2))


def bounds(args, count):
    low = args.range_lo if args.range_lo is not None else args.worker_index * count // args.num_workers
    high = args.range_hi if args.range_hi is not None else (args.worker_index + 1) * count // args.num_workers
    return low, high


def cache_source(args):
    count = SPLIT_SIZES[args.split]; low, high = bounds(args, count); paths = split_paths(args.split)
    target = np.load(Path(args.output_dir) / args.split / "p6_full_real_fp16.npy", mmap_mode="r+")
    fmri = np.load(paths["fmri"], mmap_mode="r")
    config = resolve_config(args.config); model = DownstreamBrainEncoder("full_real", config).to(args.device).eval()
    for start in range(low, high, args.batch_size):
        end = min(start + args.batch_size, high)
        values = torch.from_numpy(np.array(fmri[start:end], dtype=np.float32, copy=True)).to(args.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16): output = model(values)
        target[start:end] = output.cpu().numpy().astype(np.float16)
    target.flush(); print(json.dumps({"kind":"p6_source","split":args.split,"range":[low,high],"device":args.device}))


def project_teacher(args):
    count = SPLIT_SIZES[args.split]; low, high = bounds(args, count); paths = split_paths(args.split)
    target = np.load(Path(args.output_dir) / args.split / "clip_projected_fp16.npy", mmap_mode="r+")
    visual = np.load(paths["clip"], mmap_mode="r"); projector = load_projector(resolve_config(args.config)["mm_projector"], args.device)
    for start in range(low, high, args.batch_size):
        end = min(start + args.batch_size, high)
        values = torch.from_numpy(np.array(visual[start:end], dtype=np.float32, copy=True)).to(args.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16): output = projector(values)
        target[start:end] = output.cpu().numpy().astype(np.float16)
    target.flush(); print(json.dumps({"kind":"projected_teacher","split":args.split,"range":[low,high],"device":args.device}))


def compare(left, right):
    left, right = left.float(), right.float(); difference = (left-right).abs()
    return {"max_abs_error":float(difference.max()),"mean_abs_error":float(difference.mean()),"cosine":float(F.cosine_similarity(left.flatten(1),right.flatten(1)).mean())}


def verify(args):
    output = Path(args.output_dir).resolve(); config = resolve_config(args.config); result={"splits":{}}
    for split,count in SPLIT_SIZES.items():
        paths=split_paths(split); fmri=np.load(paths['fmri'],mmap_mode='r'); base=np.load(paths['base'],mmap_mode='r'); p6=np.load(output/split/'p6_full_real_fp16.npy',mmap_mode='r'); clip=np.load(paths['clip'],mmap_mode='r'); projected=np.load(output/split/'clip_projected_fp16.npy',mmap_mode='r')
        if base.shape!=(count,256,1024) or p6.shape!=(count,256,1024) or clip.shape!=(count,256,1024) or projected.shape!=(count,256,4096):raise ValueError(f'{split} cache shape mismatch')
        indices=[0,min(17,count-1)]; x=torch.from_numpy(np.array(fmri[indices],dtype=np.float32,copy=True)).to(args.device); projector=load_projector(config['mm_projector'],args.device); model=DownstreamBrainEncoder('full_real',config).to(args.device).eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):online_p6=model(x);online_projected=projector(torch.from_numpy(np.array(clip[indices],dtype=np.float32,copy=True)).to(args.device))
        result['splits'][split]={"p6_cache_vs_online":compare(torch.from_numpy(np.array(p6[indices])).to(args.device),online_p6),"teacher_projected_cache_vs_online":compare(torch.from_numpy(np.array(projected[indices])).to(args.device),online_projected),"base_cache_shape":list(base.shape),"p6_all_finite":bool(np.isfinite(p6).all()),"teacher_all_finite":bool(np.isfinite(projected).all())}
        del model,projector;gc.collect();torch.cuda.empty_cache()
    result.update({"source_base":"existing frozen UMBRAE Stage-B semantic cache","source_p6":"frozen P6 Full Real","mm_projector":config['mm_projector'],"mm_projector_sha256":sha256_file(config['mm_projector']),"test_used":False})
    metadata_path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('init','cache-source','project-teacher','verify'));p.add_argument('--config',default='configs/dual_branch/p7_downstream_protocol_v1.json');p.add_argument('--output-dir',default='protocol_outputs/protocol_v1/subj01/p9_cache');p.add_argument('--split',choices=('train','val'),default='train');p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--worker-index',type=int,default=0);p.add_argument('--num-workers',type=int,default=1);p.add_argument('--range-lo',type=int);p.add_argument('--range-hi',type=int);a=p.parse_args();{'init':init,'cache-source':cache_source,'project-teacher':project_teacher,'verify':verify}[a.action](a)
if __name__=='__main__':main()
