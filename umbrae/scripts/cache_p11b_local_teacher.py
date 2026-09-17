#!/usr/bin/env python3
"""Cache locked P11-B local teachers from frozen CLIP patch caches."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.dual_branch_cache import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage-a-root',type=Path,required=True);p.add_argument('--affinity',type=Path,required=True);p.add_argument('--mapping',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=64);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    w=np.load(a.affinity).astype(np.float32)
    if w.shape!=(64,256) or not np.allclose(w.sum(1),1,atol=1e-6): raise ValueError('locked affinity must be [64,256] and normalized')
    wt=torch.from_numpy(w).to(a.device)
    result={"mapping":str(a.mapping.resolve()),"mapping_sha256":sha256_file(str(a.mapping)),"affinity":str(a.affinity.resolve()),"affinity_sha256":sha256_file(str(a.affinity)),"teacher_definition":"einsum(kp,bpd->bkd) over cached frozen CLIP hidden_states[-2][:,1:,:]","dtype":"float16","splits":{},"test_loaded":False}
    with torch.inference_mode():
        for split,expected in [('train',8559),('val',300)]:
            meta=json.loads((a.stage_a_root/split/'metadata.json').read_text());clip_path=Path(meta['clip_path']);clip=np.load(clip_path,mmap_mode='r')
            if clip.shape!=(expected,256,1024): raise ValueError(f'{split} CLIP cache shape mismatch')
            path=a.output_dir/f'{split}_local_teacher_fp16.npy';out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(expected,64,1024))
            for start in range(0,expected,a.batch_size):
                x=torch.from_numpy(np.asarray(clip[start:start+a.batch_size],dtype=np.float32)).to(a.device)
                out[start:start+len(x)]=torch.einsum('kp,bpd->bkd',wt,x).half().cpu().numpy()
            out.flush()
            result['splits'][split]={"sample_count":expected,"source_clip":str(clip_path.resolve()),"source_clip_sha256":sha256_file(str(clip_path)),"manifest_sha256":meta['manifest_sha256'],"sample_id_index":meta['sample_id_index'],"path":str(path.resolve()),"shape":[expected,64,1024],"sha256":sha256_file(str(path))}
    (a.output_dir/'metadata.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__': main()
