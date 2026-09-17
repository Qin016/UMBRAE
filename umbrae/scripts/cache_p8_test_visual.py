#!/usr/bin/env python
"""Cache the locked CLIP block-23 pooled visual target for P8 test diagnosis."""

import argparse, json, sys, tarfile
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.clip_patch_teacher import FixedCLIPPatchTeacher
from models.dual_branch_cache import sha256_file
from scripts.cache_stage_a_protocol_features import load_row, read_rows

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",default="protocol_outputs/protocol_v1/subj01/test_manifest.jsonl");p.add_argument("--output-dir",default="dual_branch_outputs/p8_diagnosis");p.add_argument("--batch-size",type=int,default=64);p.add_argument("--device",default="cuda");a=p.parse_args()
    rows=read_rows(a.manifest)
    if len(rows)!=982 or any(r["split"]!="test" for r in rows): raise ValueError("locked test manifest mismatch")
    output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True);path=output/"test_clip_block23_global_fp16.npy";cache=np.lib.format.open_memmap(path,mode="w+",dtype=np.float16,shape=(982,1024));teacher=FixedCLIPPatchTeacher().to(a.device).eval();teacher.requires_grad_(False)
    grouped=defaultdict(list)
    for index,row in enumerate(rows): grouped[row["shard"]].append((index,row))
    done=0
    for shard,entries in grouped.items():
        with tarfile.open(shard) as archive:
            for start in range(0,len(entries),a.batch_size):
                chunk=entries[start:start+a.batch_size];images=[load_row(archive,row)[1] for _,row in chunk]
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16,enabled=a.device.startswith("cuda")): patches=teacher(torch.stack(images).to(a.device))
                cache[[i for i,_ in chunk]]=patches.float().mean(1).cpu().numpy().astype(np.float16);done+=len(chunk);print(json.dumps({"cached":done,"total":982}),flush=True)
    cache.flush(); metadata={"protocol_version":"protocol_v1","split":"test","sample_count":982,"manifest":str(Path(a.manifest).resolve()),"manifest_sha256":sha256_file(a.manifest),"sample_ids":[r["sample_id"] for r in rows],"clip_model":FixedCLIPPatchTeacher.MODEL_NAME,"feature_definition":"hidden_states[-2][:,1:,:].mean(patch_axis)","shape":[982,1024],"dtype":"float16","path":str(path.resolve()),"frozen":True}
    (output/"test_visual_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")

if __name__=="__main__":main()
