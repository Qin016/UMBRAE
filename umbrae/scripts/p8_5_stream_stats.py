#!/usr/bin/env python
"""Compute one P8.5 train-token population moment file on one GPU."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.downstream_brain_encoder import DownstreamBrainEncoder
from models.distribution_alignment import finalize_moments
from scripts.cache_p7_downstream_features import resolve_config

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',choices=['umbrae','clip','lora','full_real','full_random'],required=True);p.add_argument('--device',required=True);p.add_argument('--output-dir',default='dual_branch_outputs/p8_5_distribution_bias');p.add_argument('--batch-size',type=int,default=8);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 paths={'umbrae':'protocol_outputs/protocol_v1/subj01/stage_b_semantic_cache/train/z_sem_fp16.npy','clip':'protocol_outputs/protocol_v1/subj01/stage_a_cache/train/clip_patch_fp16.npy'};fmri=np.load('protocol_outputs/protocol_v1/subj01/stage_a_cache/train/fmri_mean_valid_repeats.npy',mmap_mode='r');cache=np.load(paths[a.source],mmap_mode='r') if a.source in paths else None;model=None if cache is not None else DownstreamBrainEncoder(a.source,resolve_config('configs/dual_branch/p7_downstream_protocol_v1.json')).to(a.device).eval()
 count=0;total=np.zeros(1024,dtype=np.float64);cross=np.zeros((1024,1024),dtype=np.float64)
 for start in range(0,8559,a.batch_size):
  end=min(start+a.batch_size,8559)
  if cache is not None:x=torch.from_numpy(np.array(cache[start:end],dtype=np.float32,copy=True)).to(a.device)
  else:
   batch=torch.from_numpy(np.array(fmri[start:end],dtype=np.float32,copy=True)).to(a.device)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):x=model(batch).float()
  flat=x.reshape(-1,1024);total+=flat.double().sum(0).cpu().numpy();cross+=(flat.T@flat).double().cpu().numpy();count+=len(flat)
  if start%1024<a.batch_size:print(json.dumps({'source':a.source,'samples':end,'device':a.device}),flush=True)
 mean,std,cov=finalize_moments(count,total,cross);np.savez_compressed(out/f'train_stats_{a.source}.npz',count=count,mean=mean,std=std,cov=cov,total=total,cross=cross);(out/f'train_stats_{a.source}.json').write_text(json.dumps({'source':a.source,'train_samples':8559,'token_count':count,'shape':[8559,256,1024],'device':a.device,'accumulation':'FP64 sums with per-batch GPU FP32 cross-product','no_training':True},indent=2));print(json.dumps({'source':a.source,'complete':True,'count':count}))
if __name__=='__main__':main()
