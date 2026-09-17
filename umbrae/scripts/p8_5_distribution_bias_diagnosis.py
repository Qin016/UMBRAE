#!/usr/bin/env python
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.distribution_alignment import fit_transforms,apply_transform
from models.downstream_brain_encoder import DownstreamBrainEncoder
from models.protocol_evaluation import rsa_metrics
from models.p8_diagnostics import linear_cka
from scripts.analyze_p8_projector import hard_retrieval,rsa_any
from scripts.cache_p7_downstream_features import load_projector,resolve_config
from models.dual_branch_cache import sha256_file

SOURCES=('umbrae','lora','full_real','full_random','clip');CORRECTIONS=('mean','affine','wct')
def corr(cov,std):return cov/np.maximum(np.outer(std,std),1e-12)
def dist_metrics(s,c):
 dm=s['mean']-c['mean'];ratio=s['std']/np.maximum(c['std'],1e-8);cv=np.linalg.norm(s['cov']-c['cov'])/np.linalg.norm(c['cov']);rd=np.linalg.norm(corr(s['cov'],s['std'])-corr(c['cov'],c['std']))
 return {'mean_shift_l2':float(np.linalg.norm(dm)),'relative_mean_shift':float(np.linalg.norm(dm)/(np.linalg.norm(c['mean'])+1e-12)),'mean_cosine':float(s['mean']@c['mean']/(np.linalg.norm(s['mean'])*np.linalg.norm(c['mean']))),'mean_absolute_channel_mean_difference':float(np.abs(dm).mean()),'maximum_channel_mean_difference':float(np.abs(dm).max()),'mean_absolute_std_difference':float(np.abs(s['std']-c['std']).mean()),'relative_std_difference':float(np.linalg.norm(s['std']-c['std'])/np.linalg.norm(c['std'])),'std_ratio_mean':float(ratio.mean()),'std_ratio_median':float(np.median(ratio)),'std_ratio_p5':float(np.percentile(ratio,5)),'std_ratio_p95':float(np.percentile(ratio,95)),'std_channel_correlation':float(np.corrcoef(s['std'],c['std'])[0,1]),'normalized_covariance_frobenius_distance':float(cv),'coral_distance':float(np.linalg.norm(s['cov']-c['cov'])**2/(4*1024**2)),'correlation_matrix_frobenius_distance':float(rd)}
def mmd(x,y,device):
 x=torch.from_numpy(x.astype(np.float32)).to(device);y=torch.from_numpy(y.astype(np.float32)).to(device);z=torch.cat([x[:1024],y[:1024]]);d=torch.pdist(z);bw=torch.median(d[d>0])**2;total=[0.,0.,0.]
 for a,b,k in ((x,x,0),(y,y,1),(x,y,2)):
  for i in range(0,len(a),512):
   for j in range(0,len(b),512):total[k]+=float(torch.exp(-torch.cdist(a[i:i+512],b[j:j+512])**2/(2*bw)).sum())
 return {'mmd2_biased':total[0]/len(x)**2+total[1]/len(y)**2-2*total[2]/(len(x)*len(y)),'bandwidth_squared':float(bw),'sample_tokens':len(x),'seed':85242}
def main():
 out=Path('dual_branch_outputs/p8_5_distribution_bias');p7=Path('dual_branch_outputs/p7_downstream');stats={s:{k:v for k,v in np.load(out/f'train_stats_{s}.npz').items()} for s in SOURCES};clip=stats['clip'];trans={s:fit_transforms(stats[s]['mean'],stats[s]['std'],stats[s]['cov'],clip['mean'],clip['std'],clip['cov'],1e-4) for s in SOURCES[:-1]}
 eigen={};rows=[]
 for s in SOURCES:
  vals=np.maximum(np.linalg.eigvalsh(stats[s]['cov'])[::-1],0);p=vals/vals.sum();er=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))));eigen[s]=vals
  row={'source':s,'effective_rank':er,**{f'top_{k}_explained':float(p[:k].sum()) for k in (8,16,32,64,128,256)}};rows.append(row)
 with (out/'pca_spectrum.csv').open('w',newline='') as f:
  fields=['source','component','eigenvalue','normalized_eigenvalue','cumulative_explained'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for s,v in eigen.items():
   p=v/v.sum();cum=np.cumsum(p)
   for i in range(256):w.writerow({'source':s,'component':i+1,'eigenvalue':v[i],'normalized_eigenvalue':p[i],'cumulative_explained':cum[i]})
 metrics={s:dist_metrics(stats[s],clip) for s in SOURCES[:-1]};metrics['pca_summary']=rows
 # Same seeded 32 samples x 256 tokens = 8192 tokens for auxiliary MMD.
 rng=np.random.default_rng(85242);ids=np.sort(rng.choice(8559,32,replace=False));fmri=np.load('protocol_outputs/protocol_v1/subj01/stage_a_cache/train/fmri_mean_valid_repeats.npy',mmap_mode='r');samples={'umbrae':np.asarray(np.load('protocol_outputs/protocol_v1/subj01/stage_b_semantic_cache/train/z_sem_fp16.npy',mmap_mode='r')[ids],dtype=np.float32).reshape(-1,1024),'clip':np.asarray(np.load('protocol_outputs/protocol_v1/subj01/stage_a_cache/train/clip_patch_fp16.npy',mmap_mode='r')[ids],dtype=np.float32).reshape(-1,1024)};cfg=resolve_config('configs/dual_branch/p7_downstream_protocol_v1.json')
 for i,s in enumerate(('lora','full_real','full_random')):
  model=DownstreamBrainEncoder(s,cfg).to(f'cuda:{i%2}').eval();x=torch.from_numpy(np.array(fmri[ids],dtype=np.float32,copy=True)).to(f'cuda:{i%2}')
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):samples[s]=model(x).float().cpu().numpy().reshape(-1,1024)
 for s in SOURCES[:-1]:metrics[s]['mmd']=mmd(samples[s],samples['clip'],'cuda:0')
 (out/'distribution_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
 np.savez_compressed(out/'train_distribution_stats.npz',**{f'{s}_{k}':stats[s][k] for s in SOURCES for k in ('mean','std','cov')},**{f'{s}_eigenvalues':eigen[s] for s in SOURCES})
 np.savez_compressed(out/'analytic_transforms.npz',**{f'{s}_{k}':v for s,t in trans.items() for k,v in t.items() if isinstance(v,np.ndarray)})
 metadata={'fit_split':'train','train_samples':8559,'train_token_count':2191104,'epsilon_cov':1e-4,'PCA_method':'full symmetric eigendecomposition; first 256 reported','MMD_seed':85242,'MMD_sample_tokens':8192,'TEST_STATUS':'POST_HOC_REUSED_TEST','test_derived_fit_statistics':False,'transforms':{s:{k:v for k,v in t.items() if not isinstance(v,np.ndarray)} for s,t in trans.items()},'no_trainable_parameters':True,'no_optimizer':True,'no_backward':True}
 (out/'transformation_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 # Create three Full-Real corrected test caches with train-fitted transforms only.
 raw=np.load(p7/'full_real_z_out_fp16.npy',mmap_mode='r');visual=np.asarray(np.load('dual_branch_outputs/p8_diagnosis/test_clip_block23_global_fp16.npy'),dtype=np.float32);rep={};projector=load_projector('model_weights/mm_projector.bin','cuda:0');base=np.load(p7/'umbrae_z_out_fp16.npy',mmap_mode='r')
 for kind in CORRECTIONS:
  path=out/f'full_real_{kind}_test_fp16.npy';target=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=raw.shape);pooled=[];projected=[];norms=[];cos=[]
  for start in range(0,982,8):
   z=apply_transform(np.asarray(raw[start:start+8],dtype=np.float32),kind,stats['full_real']['mean'],stats['full_real']['std'],clip['mean'],clip['std'],trans['full_real']['wct_matrix']);target[start:start+len(z)]=z.astype(np.float16);pooled.append(z.mean(1));zt=torch.from_numpy(z).to('cuda:0')
   with torch.inference_mode():yp=projector(zt).float();yb=projector(torch.from_numpy(np.asarray(base[start:start+len(z)],dtype=np.float32)).to('cuda:0')).float()
   projected.append(yp.mean(1).cpu().numpy());norms.append(yp.norm(dim=-1).cpu().numpy());cos.append(torch.nn.functional.cosine_similarity(yp,yb,dim=-1).cpu().numpy())
  target.flush();q=np.concatenate(pooled);pq=np.concatenate(projected);rep[kind]={'cache':str(path.resolve()),'rsa':rsa_any(q,visual),'hard_retrieval':hard_retrieval(q,visual),'projector_rsa':rsa_any(pq,visual),'projected_token_norm_mean':float(np.concatenate(norms).mean()),'projected_cosine_to_raw_umbrae':float(np.concatenate(cos).mean()),'shape':[982,256,1024],'all_finite':bool(np.isfinite(target).all())}
 (out/'representation_metrics.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps({'distribution_analysis':'complete','corrected_caches':list(rep)}))
if __name__=='__main__':main()
