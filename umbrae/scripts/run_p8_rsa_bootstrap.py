#!/usr/bin/env python
"""Stimulus bootstrap of the locked P7 validation RSA representations."""

import argparse,json,sys
from pathlib import Path
import numpy as np,torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.downstream_brain_encoder import DownstreamBrainEncoder,MODES
from models.p8_diagnostics import paired_summary
from models.protocol_evaluation import rsa_metrics
from scripts.cache_p7_downstream_features import resolve_config

PAIRS={"lora_minus_umbrae":("lora","umbrae"),"full_real_minus_umbrae":("full_real","umbrae"),"full_random_minus_umbrae":("full_random","umbrae"),"full_real_minus_full_random":("full_real","full_random")}

def normalized_rdm(values):
    x=values.astype(np.float64);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);return 1-x@x.T

def weighted_ranks(weights,order,self_weight):
    sorted_w=weights[:,order];r=self_weight[:,None]+np.cumsum(sorted_w,axis=1)-.5*(sorted_w-1);result=np.empty_like(r,dtype=np.float64);result[:,order]=r;return result

def weighted_corr(left,right,weights,self_weight):
    total=weights.sum(1)+self_weight;self_rank=.5*(self_weight+1);lm=(np.sum(weights*left,1)+self_weight*self_rank)/total;rm=(np.sum(weights*right,1)+self_weight*self_rank)/total;lc=left-lm[:,None];rc=right-rm[:,None];num=np.sum(weights*lc*rc,1)+self_weight*(self_rank-lm)*(self_rank-rm);ld=np.sum(weights*lc**2,1)+self_weight*(self_rank-lm)**2;rd=np.sum(weights*rc**2,1)+self_weight*(self_rank-rm)**2;return num/np.sqrt(ld*rd)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/dual_branch/p7_downstream_protocol_v1.json");p.add_argument("--output-dir",default="dual_branch_outputs/p8_diagnosis");p.add_argument("--iterations",type=int,default=10000);p.add_argument("--seed",type=int,default=8142);p.add_argument("--device",default="cuda");a=p.parse_args();config=resolve_config(a.config);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    fmri=np.load(config['validation_fmri'],mmap_mode='r');visual=np.asarray(np.load('protocol_outputs/protocol_v1/subj01/p3r_cache/val/clip_block23_global_fp16.npy'),dtype=np.float32);pooled={}
    for mode in MODES:
        model=DownstreamBrainEncoder(mode,config).to(a.device).eval();parts=[]
        for start in range(0,300,32):
            x=torch.from_numpy(np.array(fmri[start:start+32],dtype=np.float32,copy=True)).to(a.device)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=a.device.startswith('cuda')):parts.append(model(x).float().mean(1).cpu().numpy())
        pooled[mode]=np.concatenate(parts);del model;torch.cuda.empty_cache()
    # Reuse the exact persisted P7/Protocol-V1 baseline embedding rather than
    # an AMP re-encoding; this makes the baseline point estimate bit-for-bit
    # compatible with the previously reported RSA.
    pooled["umbrae"]=np.load("protocol_outputs/protocol_v1/subj01/umbrae_s1_baseline_seed42_protocolv1/q_sem_normalized.npy")
    point={m:rsa_metrics(pooled[m],visual) for m in MODES};rdms={m:normalized_rdm(pooled[m]) for m in MODES};rdms['visual']=normalized_rdm(visual);upper=np.triu_indices(300,1);vectors={m:rdms[m][upper] for m in (*MODES,'visual')};orders={m:np.argsort(vectors[m],kind='stable') for m in (*MODES,'visual')};rng=np.random.default_rng(a.seed);values={m:np.empty(a.iterations) for m in MODES}
    for start in range(0,a.iterations,20):
        size=min(20,a.iterations-start);draws=rng.integers(0,300,size=(size,300));counts=np.stack([np.bincount(row,minlength=300) for row in draws]);weights=counts[:,upper[0]]*counts[:,upper[1]];self_weight=np.sum(counts*(counts-1)//2,axis=1);visual_rank=weighted_ranks(weights,orders['visual'],self_weight)
        for mode in MODES: values[mode][start:start+size]=weighted_corr(weighted_ranks(weights,orders[mode],self_weight),visual_rank,weights,self_weight)
    result={"seed":a.seed,"iterations":a.iterations,"split":"validation","stimulus_count":300,"brain_vision_synchronized_sampling":True,"same_indices_all_models":True,"method":"exact multiplicity-weighted stimulus bootstrap Spearman ranks (no RDM expansion)","point_estimates":point,"model_confidence_intervals":{m:paired_summary(values[m]) for m in MODES},"comparisons":{name:paired_summary(values[l]-values[r]) for name,(l,r) in PAIRS.items()}}
    (out/'p8_bootstrap_rsa.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({"rsa_bootstrap":"complete","iterations":a.iterations,"point_estimates":{m:point[m]['spearman_rsa'] for m in MODES}}))

if __name__=='__main__':main()
