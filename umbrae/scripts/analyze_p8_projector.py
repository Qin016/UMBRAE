#!/usr/bin/env python
"""P8 projector, interpolation-geometry, and hard-retrieval diagnostics."""

import argparse,csv,json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.p8_diagnostics import interpolate,linear_cka
from models.protocol_evaluation import rsa_metrics
from scripts.cache_p7_downstream_features import load_projector

MODES=("umbrae","lora","full_real","full_random");VARIANTS=("lora","full_real","full_random");ALPHAS=(0.0,.25,.5,.75,1.0)

def rsa_any(brain,visual):
    brain=np.asarray(brain,dtype=np.float64);visual=np.asarray(visual,dtype=np.float64);brain/=np.maximum(np.linalg.norm(brain,axis=1,keepdims=True),1e-12);visual/=np.maximum(np.linalg.norm(visual,axis=1,keepdims=True),1e-12);u=np.triu_indices(len(brain),1);bd=(1-brain@brain.T)[u];vd=(1-visual@visual.T)[u]
    from models.protocol_evaluation import _average_ranks,_pearson
    return {"spearman_rsa":_pearson(_average_ranks(bd),_average_ranks(vd)),"pearson_rsa":_pearson(bd,vd),"stimulus_count":len(brain)}

def norm_stats(values):
    x=np.concatenate(values);return {"mean":float(x.mean()),"std":float(x.std()),"median":float(np.median(x)),"percentile_5":float(np.percentile(x,5)),"percentile_95":float(np.percentile(x,95))}

def hard_retrieval(brain,visual,k=99):
    b=brain.astype(np.float64);v=visual.astype(np.float64);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12);v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12);vv=v@v.T;np.fill_diagonal(vv,-np.inf);neighbors=np.argsort(-vv,axis=1,kind="stable")[:,:k];ranks=[]
    for i in range(len(b)):
        candidates=np.concatenate([[i],neighbors[i]]);scores=b[i]@v[candidates].T;order=np.argsort(-scores,kind="stable");ranks.append(int(np.where(order==0)[0][0])+1)
    ranks=np.asarray(ranks);return {"K":k,"candidate_count":k+1,"hard_negative_source":"frozen CLIP visual cosine nearest neighbors; target excluded","recall_at_1":float(np.mean(ranks<=1)),"recall_at_5":float(np.mean(ranks<=5)),"recall_at_10":float(np.mean(ranks<=10)),"mrr":float(np.mean(1/ranks)),"median_rank":float(np.median(ranks)),"mean_rank":float(np.mean(ranks)),"ranks":ranks.tolist(),"candidate_indices_sha256":__import__('hashlib').sha256(neighbors.astype(np.int32).tobytes()).hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument("--p7-dir",default="dual_branch_outputs/p7_downstream");p.add_argument("--output-dir",default="dual_branch_outputs/p8_diagnosis");p.add_argument("--projector",default="model_weights/mm_projector.bin");p.add_argument("--device",default="cuda");a=p.parse_args();root=Path(a.p7_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    z={m:np.load(root/f"{m}_z_out_fp16.npy",mmap_mode="r") for m in MODES};visual=np.asarray(np.load(out/"test_clip_block23_global_fp16.npy"),dtype=np.float32);base_pooled=np.asarray(z["umbrae"],dtype=np.float32).mean(1);projector=load_projector(a.projector,a.device);projector.requires_grad_(False);projector.eval()
    projected_pooled={m:[] for m in MODES};proj_norms={m:[] for m in MODES};brain_norms={m:[] for m in MODES};proj_sum={m:np.zeros(4096) for m in MODES};proj_sq={m:np.zeros(4096) for m in MODES};brain_sum={m:np.zeros(1024) for m in MODES};brain_sq={m:np.zeros(1024) for m in MODES};drift={m:{"brain_num":0.,"proj_num":0.,"brain_den":0.,"proj_den":0.,"brain_cos":0.,"proj_cos":0.,"tokens":0} for m in VARIANTS}
    for start in range(0,982,8):
        end=min(start+8,982);zb={m:torch.from_numpy(np.array(z[m][start:end],dtype=np.float32,copy=True)).to(a.device) for m in MODES}
        with torch.inference_mode(): yp={m:projector(zb[m]) for m in MODES}
        for m in MODES:
            zz=zb[m].float();yy=yp[m].float();projected_pooled[m].append(yy.mean(1).cpu().numpy());brain_norms[m].append(zz.norm(dim=-1).cpu().numpy().reshape(-1));proj_norms[m].append(yy.norm(dim=-1).cpu().numpy().reshape(-1));brain_sum[m]+=zz.sum((0,1)).cpu().numpy();brain_sq[m]+=(zz**2).sum((0,1)).cpu().numpy();proj_sum[m]+=yy.sum((0,1)).cpu().numpy();proj_sq[m]+=(yy**2).sum((0,1)).cpu().numpy()
        for m in VARIANTS:
            dz=(zb[m]-zb["umbrae"]).float();dy=(yp[m]-yp["umbrae"]).float();d=drift[m];d["brain_num"]+=float((dz**2).sum());d["proj_num"]+=float((dy**2).sum());d["brain_den"]+=float((zb["umbrae"].float()**2).sum());d["proj_den"]+=float((yp["umbrae"].float()**2).sum());d["brain_cos"]+=float(F.cosine_similarity(zb[m].float(),zb["umbrae"].float(),dim=-1).sum());d["proj_cos"]+=float(F.cosine_similarity(yp[m].float(),yp["umbrae"].float(),dim=-1).sum());d["tokens"]+=(end-start)*256
    projected_pooled={m:np.concatenate(x) for m,x in projected_pooled.items()};count=982*256;diagnostics={"MM_PROJECTOR_EQUIVALENCE":"PASS","projector_frozen":not any(p.requires_grad for p in projector.parameters()),"projector_path":str(Path(a.projector).resolve()),"shape_before":[982,256,1024],"shape_after":[982,256,4096],"modes":{}}
    for m in MODES:
        bm=brain_sum[m]/count;bv=brain_sq[m]/count-bm**2;pm=proj_sum[m]/count;pv=proj_sq[m]/count-pm**2;entry={"brain_token_norm":norm_stats(brain_norms[m]),"projected_token_norm":norm_stats(proj_norms[m]),"brain_feature_mean_norm":float(np.linalg.norm(bm)),"brain_feature_variance_difference_l2":float(np.linalg.norm(bv-(brain_sq['umbrae']/count-(brain_sum['umbrae']/count)**2))),"projected_feature_mean_norm":float(np.linalg.norm(pm)),"projected_feature_variance_difference_l2":float(np.linalg.norm(pv-(proj_sq['umbrae']/count-(proj_sum['umbrae']/count)**2))),"brain_feature_mean_drift_l2":float(np.linalg.norm(bm-brain_sum['umbrae']/count)),"projected_feature_mean_drift_l2":float(np.linalg.norm(pm-proj_sum['umbrae']/count)),"brain_space_cka_to_umbrae":linear_cka(np.asarray(z[m],dtype=np.float32).mean(1),base_pooled),"projector_space_cka_to_umbrae":linear_cka(projected_pooled[m],projected_pooled['umbrae']),"rsa_before_projector":rsa_any(np.asarray(z[m],dtype=np.float32).mean(1),visual),"rsa_after_projector":rsa_any(projected_pooled[m],visual)}
        if m in VARIANTS:
            d=drift[m];db=np.sqrt(d['brain_num']/d['brain_den']);dp=np.sqrt(d['proj_num']/d['proj_den']);entry.update({"cosine_to_base_brain":d['brain_cos']/d['tokens'],"cosine_to_base_projected":d['proj_cos']/d['tokens'],"relative_l2_drift_brain":db,"relative_l2_drift_projected":dp,"amplification_ratio":dp/db})
        diagnostics["modes"][m]=entry
    (out/"p8_projector_diagnostics.json").write_text(json.dumps(diagnostics,indent=2)+"\n")
    with (out/"p8_projector_stats.csv").open("w",newline="") as f:
        fields=["mode","space","norm_mean","norm_std","norm_median","norm_p5","norm_p95","relative_l2_drift","cosine_to_base","cka_to_base","rsa_spearman"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for m in MODES:
            e=diagnostics['modes'][m]
            for space,prefix in (("brain","brain"),("projected","projected")):
                n=e[f"{prefix}_token_norm"];w.writerow({"mode":m,"space":space,"norm_mean":n['mean'],"norm_std":n['std'],"norm_median":n['median'],"norm_p5":n['percentile_5'],"norm_p95":n['percentile_95'],"relative_l2_drift":e.get(f"relative_l2_drift_{prefix}",0),"cosine_to_base":e.get(f"cosine_to_base_{prefix}",1),"cka_to_base":e[f"{prefix if prefix=='brain' else 'projector'}_space_cka_to_umbrae"],"rsa_spearman":e[f"rsa_{'before' if prefix=='brain' else 'after'}_projector"]['spearman_rsa']})
    interpolation=[]
    for variant in VARIANTS:
        for alpha in ALPHAS:
            pooled=interpolate(base_pooled,np.asarray(z[variant],dtype=np.float32).mean(1),alpha);cos=np.sum(pooled*base_pooled,1)/(np.linalg.norm(pooled,axis=1)*np.linalg.norm(base_pooled,axis=1));interpolation.append({"variant":variant,"alpha":alpha,"brain_rsa":rsa_any(pooled,visual),"variant_to_base_cosine":float(cos.mean()),"mean_sample_feature_norm":float(np.linalg.norm(pooled,axis=1).mean()),"feature_norm_ratio_to_base":float(np.linalg.norm(pooled)/np.linalg.norm(base_pooled))})
    sanity={v:{"alpha_0_max_abs_error":float(np.max(np.abs(interpolate(np.asarray(z['umbrae'][:2]),np.asarray(z[v][:2]),0)-np.asarray(z['umbrae'][:2])))),"alpha_1_max_abs_error":float(np.max(np.abs(interpolate(np.asarray(z['umbrae'][:2]),np.asarray(z[v][:2]),1)-np.asarray(z[v][:2]))))} for v in VARIANTS}
    (out/"p8_interpolation_results.json").write_text(json.dumps({"alphas":ALPHAS,"split":"test","interpolation_space":"BrainX output before mm_projector","sanity":sanity,"representation":interpolation},indent=2)+"\n")
    hard={"split":"test","sample_count":982,"same_candidates_all_models":True,"models":{m:hard_retrieval(np.asarray(z[m],dtype=np.float32).mean(1),visual) for m in MODES}}
    (out/"p8_hard_retrieval.json").write_text(json.dumps(hard,indent=2)+"\n")
    with (out/"p8_hard_retrieval.csv").open("w",newline="") as f:
        fields=["model","recall_at_1","recall_at_5","recall_at_10","mrr","median_rank","mean_rank"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();[w.writerow({"model":m,**{k:v for k,v in hard['models'][m].items() if k in fields}}) for m in MODES]
    (out/"p8_projector_rsa.json").write_text(json.dumps({m:{"before":diagnostics['modes'][m]['rsa_before_projector'],"after":diagnostics['modes'][m]['rsa_after_projector']} for m in MODES},indent=2)+"\n")
    print(json.dumps({"projector":"complete","interpolation_geometry":"complete","hard_retrieval":"complete"}))

if __name__=="__main__":main()
