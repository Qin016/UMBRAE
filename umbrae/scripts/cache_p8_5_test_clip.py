#!/usr/bin/env python
import argparse,json,sys,tarfile
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.clip_patch_teacher import FixedCLIPPatchTeacher
from scripts.cache_stage_a_protocol_features import read_rows,load_row
def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--output-dir',default='dual_branch_outputs/p8_5_distribution_bias');a=p.parse_args();rows=read_rows('protocol_outputs/protocol_v1/subj01/test_manifest.jsonl');out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);path=out/'test_clip_patch_fp16.npy';cache=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(982,256,1024));teacher=FixedCLIPPatchTeacher().to(a.device).eval();groups=defaultdict(list)
 for i,r in enumerate(rows):groups[r['shard']].append((i,r))
 for shard,entries in groups.items():
  with tarfile.open(shard) as ar:
   for start in range(0,len(entries),32):
    chunk=entries[start:start+32];images=torch.stack([load_row(ar,r)[1] for _,r in chunk]).to(a.device)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):x=teacher(images)
    cache[[i for i,_ in chunk]]=x.cpu().numpy().astype(np.float16)
 cache.flush();print(json.dumps({'test_clip_cache':str(path),'shape':[982,256,1024],'fit_usage':False}))
if __name__=='__main__':main()
