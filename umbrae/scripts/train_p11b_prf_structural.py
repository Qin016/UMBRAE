#!/usr/bin/env python3
"""Matched P11-B pRF-local representation training (train/validation only)."""

from __future__ import annotations

import argparse, csv, hashlib, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from losses.prf_local_alignment_loss import PRFLocalAlignmentLoss
from models.dual_branch_cache import sha256_file
from umbrae.models.prf_structural_encoder import PRFStructuralEncoder


ROIS=("V1","V2","V3","hV4")


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def parameter_hash(model):
    d=hashlib.sha256()
    for n,p in model.named_parameters():
        d.update(n.encode());d.update(str(tuple(p.shape)).encode());d.update(p.detach().cpu().numpy().tobytes())
    return d.hexdigest()


def load_units(path):
    payload=json.loads(Path(path).read_text());units=[u for u in payload['units'] if u['unit_type']=='retinotopic']
    if len(units)!=64: raise ValueError('mapping must contain exactly 64 retinotopic units')
    return payload,units


class P11BDataset(Dataset):
    def __init__(self,stage_cache,teacher_meta,split,include_global=False):
        self.stage=json.loads((Path(stage_cache)/'metadata.json').read_text());tm=json.loads(Path(teacher_meta).read_text());self.teacher_meta=tm;entry=tm['splits'][split]
        if self.stage['split']!=split or self.stage['sample_count']!=entry['sample_count'] or self.stage['manifest_sha256']!=entry['manifest_sha256']: raise ValueError('stage/teacher cache mismatch')
        self.ids=json.loads(Path(self.stage['sample_id_index']).read_text());self.fmri=np.load(self.stage['fmri_path'],mmap_mode='r');self.teacher=np.load(entry['path'],mmap_mode='r')
        if len(self.ids)!=len(self.fmri) or self.teacher.shape!=(len(self.ids),64,1024): raise ValueError('cache shape mismatch')
        self.global_visual=None
        if include_global:
            clip=np.load(self.stage['clip_path'],mmap_mode='r');self.global_visual=np.asarray(clip.mean(axis=1),dtype=np.float32)
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        out={'fmri':np.array(self.fmri[i],dtype=np.float32,copy=True),'teacher':np.array(self.teacher[i],dtype=np.float16,copy=True)}
        if self.global_visual is not None: out['global_visual']=self.global_visual[i]
        return out


def ranks_from_similarity(sim):
    order=np.argsort(-sim,axis=-1,kind='stable');target=np.arange(sim.shape[-2])
    shape=[1]*(sim.ndim-2)+[sim.shape[-2],1]
    return np.argmax(order==target.reshape(shape),axis=-1)+1


def rank_summary(ranks):
    return {'r1':float(np.mean(ranks<=1)),'r5':float(np.mean(ranks<=5)),'mrr':float(np.mean(1.0/ranks)),'mean_rank':float(np.mean(ranks))}


def relation_spearman(h,t):
    gh=np.einsum('brd,bsd->brs',h,h);gt=np.einsum('brd,bsd->brs',t,t);upper=np.triu_indices(64,k=1)
    a=rankdata(gh[:,upper[0],upper[1]],axis=1);b=rankdata(gt[:,upper[0],upper[1]],axis=1);a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);return np.divide((a*b).sum(1),den,out=np.zeros(len(a)),where=den>0)


def full_metrics(h,t,g,units):
    h=np.asarray(h,dtype=np.float32);t=np.asarray(t,dtype=np.float32);g=np.asarray(g,dtype=np.float32)
    hn=h/np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-12);tn=t/np.maximum(np.linalg.norm(t,axis=-1,keepdims=True),1e-12);gn=g/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
    local=(hn*tn).sum(-1);mse=((h-t)**2).mean(-1)
    loc_sim=np.einsum('brd,bsd->brs',hn,tn);loc_ranks=ranks_from_similarity(loc_sim)
    n=len(h);sample_ranks=np.empty((n,64),dtype=np.int32)
    for r in range(64): sample_ranks[:,r]=ranks_from_similarity(hn[:,r]@tn[:,r].T)
    rel=relation_spearman(hn,tn);global_cos=np.einsum('brd,bd->br',hn,gn);gap=local-global_cos
    pair_h=np.einsum('brd,bsd->brs',hn,hn);mask=~np.eye(64,dtype=bool);brain_pair=pair_h[:,mask].mean(1)
    pair_t=np.einsum('brd,bsd->brs',tn,tn);teacher_pair=pair_t[:,mask].mean(1)
    per_unit=[]
    for r,u in enumerate(units):
        per_unit.append({'unit_id':u['unit_id'],'ROI':u['parent_roi'],'hemisphere':u['hemisphere'],'mean_x':u['mean_x'],'mean_y':u['mean_y'],'ecc':u['mean_eccentricity'],'sigma':u['mean_sigma'],'mean_R2':u['mean_r2'],'voxels':u['num_voxels'],'local_cos':float(local[:,r].mean()),'local_mse':float(mse[:,r].mean()),'localization_r1':float(np.mean(loc_ranks[:,r]<=1)),'localization_r5':float(np.mean(loc_ranks[:,r]<=5)),'localization_mrr':float(np.mean(1/loc_ranks[:,r])),'sample_r1':float(np.mean(sample_ranks[:,r]<=1)),'sample_r5':float(np.mean(sample_ranks[:,r]<=5)),'sample_mrr':float(np.mean(1/sample_ranks[:,r])),'locality_gap':float(gap[:,r].mean())})
    roi_metrics={}
    for roi in ROIS:
        idx=[i for i,u in enumerate(units) if u['parent_roi']==roi]
        roi_metrics[roi]={'local_cos':float(local[:,idx].mean()),'local_mse':float(mse[:,idx].mean()),'unit_localization_r1':float(np.mean(loc_ranks[:,idx]<=1)),'unit_localization_mrr':float(np.mean(1/loc_ranks[:,idx])),'local_sample_r1':float(np.mean(sample_ranks[:,idx]<=1)),'local_sample_mrr':float(np.mean(1/sample_ranks[:,idx])),'locality_gap':float(gap[:,idx].mean())}
    def quantile_diag(field):
        v=np.array([u[field] for u in units]);cuts=np.quantile(v,[1/3,2/3]);labels=np.where(v<=cuts[0],'low',np.where(v<=cuts[1],'mid','high'));return {label:{'unit_count':int(np.sum(labels==label)),'local_cos':float(local[:,labels==label].mean()),'localization_mrr':float(np.mean(1/loc_ranks[:,labels==label]))} for label in ('low','mid','high')}
    sample_metrics={'local_cos':local.mean(1),'unit_localization_mrr':(1/loc_ranks).mean(1),'local_sample_mrr':(1/sample_ranks).mean(1),'unit_relation_spearman':rel,'locality_gap':gap.mean(1)}
    return {'local_cos':float(local.mean()),'local_cos_loss':float(1-local.mean()),'local_mse':float(mse.mean()),'unit_localization':rank_summary(loc_ranks),'local_sample_retrieval':rank_summary(sample_ranks),'unit_relation_spearman':float(rel.mean()),'locality_gap':float(gap.mean()),'brain_local_to_global_cos':float(global_cos.mean()),'mean_pairwise_brain_unit_cos':float(brain_pair.mean()),'mean_pairwise_teacher_unit_cos':float(teacher_pair.mean()),'per_unit':per_unit,'roi_metrics':roi_metrics,'eccentricity_quantiles':quantile_diag('mean_eccentricity'),'sigma_quantiles':quantile_diag('mean_sigma'),'quality_local_cos_spearman':float(spearmanr([u['mean_r2'] for u in units],[x['local_cos'] for x in per_unit]).statistic),'sample_metrics':{k:v.tolist() for k,v in sample_metrics.items()}}


def evaluate(model,criterion,loader,device,amp,units):
    model.eval();sums={k:0. for k in ('loss','local_cos_loss','local_mse','unit_rel_loss','local_nce')};count=0;hs=[];ts=[];gs=[]
    with torch.inference_mode():
        for b in loader:
            x=b['fmri'].to(device,non_blocking=True);t=b['teacher'].to(device,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp): h=model(x)
            loss=criterion(h,t);n=len(x);count+=n
            for k in sums:sums[k]+=float(loss[k])*n
            hs.append(h.half().cpu());ts.append(t.half().cpu());gs.append(b['global_visual'].float())
    rep=full_metrics(torch.cat(hs).numpy(),torch.cat(ts).numpy(),torch.cat(gs).numpy(),units);rep['losses']={k:v/count for k,v in sums.items()};rep['sample_count']=count;return rep


def score(metrics): return (metrics['unit_localization']['mrr'],metrics['local_sample_retrieval']['mrr'],-metrics['local_cos_loss'])


def save_csv(path,rows):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def parse_args():
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--config');known,_=pre.parse_known_args();defaults=json.loads(Path(known.config).read_text()) if known.config else {}
    p=argparse.ArgumentParser(parents=[pre]);req=not bool(defaults)
    for name in ('protocol','mapping','teacher_metadata','teacher_affinity','train_cache','val_cache','output_dir','experiment_name'):p.add_argument('--'+name.replace('_','-'),required=req)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--num-workers',type=int,default=4);p.add_argument('--max-epochs',type=int,default=30);p.add_argument('--min-epochs',type=int,default=10);p.add_argument('--patience',type=int,default=7);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--gradient-clip-norm',type=float,default=1.0);p.add_argument('--temperature',type=float,default=.07);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0');p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True);p.set_defaults(**defaults);return p.parse_args()


def run(a):
    locked={'batch_size':64,'max_epochs':30,'min_epochs':10,'patience':7,'lr':1e-4,'weight_decay':1e-4,'gradient_clip_norm':1.0,'temperature':.07,'seed':42};bad={k:(getattr(a,k),v) for k,v in locked.items() if getattr(a,k)!=v}
    if bad:raise ValueError(f'locked P11-B config mismatch: {bad}')
    protocol=json.loads(Path(a.protocol).read_text());payload,units=load_units(a.mapping);teacher_meta=json.loads(Path(a.teacher_metadata).read_text());w=np.load(a.teacher_affinity)
    if w.shape!=(64,256) or teacher_meta['affinity_sha256']!=sha256_file(a.teacher_affinity):raise ValueError('teacher affinity lock mismatch')
    if payload['quality_policy']!='r2_ge10p1' or payload['feature_set']!='xy' or payload['k_retino']!=64:raise ValueError('not primary P11-B0 mapping')
    seed_all(a.seed);model=PRFStructuralEncoder(units).to(a.device);init_hash=parameter_hash(model);report=model.parameter_report()
    criterion=PRFLocalAlignmentLoss(temperature=a.temperature);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay);amp=bool(a.amp and str(a.device).startswith('cuda'));scaler=torch.cuda.amp.GradScaler(enabled=amp)
    train=P11BDataset(a.train_cache,a.teacher_metadata,'train');val=P11BDataset(a.val_cache,a.teacher_metadata,'val',include_global=True)
    if len(train)!=8559 or len(val)!=300:raise ValueError('P11-B must use locked train/val only')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pth').exists():raise FileExistsError(f'run already exists: {out}')
    cfg=vars(a).copy();cfg.update({'architecture':'unit-specific Linear(Vr,256) + shared GELU/LayerNorm/Linear(256,1024)','unit_embedding':False,'roi_embedding':False,'hemisphere_embedding':False,'prf_coordinate_input':False,'objective_weights':{'local_cos':1.0,'local_mse':.1,'unit_rel':.1,'local_nce':.1},'parameter_report':report,'initial_parameter_sha256':init_hash,'brainx_used':False,'shikra_used':False,'fusion_used':False,'lora_used':False,'mm_projector_used':False,'test_loaded':False});(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    val_loader=DataLoader(val,batch_size=64,shuffle=False,num_workers=a.num_workers,pin_memory=amp);initial=evaluate(model,criterion,val_loader,a.device,amp,units);(out/'initial_val_metrics.json').write_text(json.dumps(initial,indent=2)+'\n');print(json.dumps({'initial_hash':init_hash,'parameter_report':report,'initial_metrics':{k:initial[k] for k in ('local_cos','unit_localization','local_sample_retrieval','unit_relation_spearman','locality_gap')}}),flush=True)
    best_score=(-float('inf'),)*3;best_epoch=0;bad_epochs=0;records=[];started=time.time()
    for epoch in range(1,a.max_epochs+1):
        generator=torch.Generator().manual_seed(a.seed+epoch);loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=generator,num_workers=a.num_workers,pin_memory=amp)
        model.train();sums={k:0. for k in ('loss','local_cos_loss','local_mse','unit_rel_loss','local_nce','grad_norm')};count=0
        for b in loader:
            x=b['fmri'].to(a.device,non_blocking=True);t=b['teacher'].to(a.device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):h=model(x)
            losses=criterion(h,t);scaler.scale(losses['loss']).backward();scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),a.gradient_clip_norm)
            if not torch.isfinite(grad) or not torch.isfinite(losses['loss']):raise RuntimeError('nonfinite optimization')
            scaler.step(optimizer);scaler.update();n=len(x);count+=n
            for k in sums:sums[k]+=float(grad if k=='grad_norm' else losses[k])*n
        vm=evaluate(model,criterion,val_loader,a.device,amp,units);current=score(vm);improved=current>best_score
        if improved:best_score=current;best_epoch=epoch;bad_epochs=0
        else:bad_epochs+=1
        rec={'epoch':epoch,**{f'train_{k}':v/count for k,v in sums.items()},'val_total_loss':vm['losses']['loss'],'val_local_cos':vm['local_cos'],'val_local_mse':vm['local_mse'],'val_unit_localization_r1':vm['unit_localization']['r1'],'val_unit_localization_mrr':vm['unit_localization']['mrr'],'val_local_sample_r1':vm['local_sample_retrieval']['r1'],'val_local_sample_mrr':vm['local_sample_retrieval']['mrr'],'val_unit_relation_spearman':vm['unit_relation_spearman'],'val_locality_gap':vm['locality_gap'],'selection_score':list(current)};records.append(rec);(out/'metrics_per_epoch.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'selection_metric':'lexicographic max(unit_localization_mrr, local_sample_mrr, -local_cos_loss)','selection_score':current,'best_epoch':best_epoch,'mapping_sha256':sha256_file(a.mapping),'teacher_affinity_sha256':sha256_file(a.teacher_affinity),'initial_parameter_sha256':init_hash};torch.save(state,out/'last.pth')
        if improved:torch.save(state,out/'best.pth')
        print(json.dumps(rec),flush=True)
        if epoch>=a.min_epochs and bad_epochs>=a.patience:print(json.dumps({'early_stop':epoch,'best_epoch':best_epoch}),flush=True);break
    state=torch.load(out/'best.pth',map_location=a.device);model.load_state_dict(state['model']);final=evaluate(model,criterion,val_loader,a.device,amp,units);final.update({'best_epoch':state['epoch'],'selection_metric':state['selection_metric'],'selection_score':list(state['selection_score'])});(out/'final_val_metrics.json').write_text(json.dumps(final,indent=2)+'\n');save_csv(out/'per_unit_metrics.csv',final['per_unit']);save_csv(out/'roi_metrics.csv',[{'ROI':r,**v} for r,v in final['roi_metrics'].items()]);(out/'spatial_diagnostics.json').write_text(json.dumps({k:final[k] for k in ('eccentricity_quantiles','sigma_quantiles','quality_local_cos_spearman','mean_pairwise_brain_unit_cos','mean_pairwise_teacher_unit_cos','brain_local_to_global_cos','locality_gap')},indent=2)+'\n')
    provenance={'protocol':str(Path(a.protocol).resolve()),'protocol_sha256':sha256_file(a.protocol),'mapping':str(Path(a.mapping).resolve()),'mapping_sha256':sha256_file(a.mapping),'teacher_affinity':str(Path(a.teacher_affinity).resolve()),'teacher_affinity_sha256':sha256_file(a.teacher_affinity),'teacher_cache_metadata':str(Path(a.teacher_metadata).resolve()),'teacher_cache_metadata_sha256':sha256_file(a.teacher_metadata),'train_manifest_sha256':train.stage['manifest_sha256'],'val_manifest_sha256':val.stage['manifest_sha256'],'repeat_policy':'mean_over_num_uniques_valid_repeats','seed':a.seed,'batch_order_seed_rule':'seed+epoch','initial_parameter_sha256':init_hash,'gpu':torch.cuda.get_device_name(torch.device(a.device).index or 0),'training_seconds':time.time()-started,'test_loaded':False,'brainx_used':False,'shikra_used':False};(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps({'final':{k:final[k] for k in ('best_epoch','local_cos','unit_localization','local_sample_retrieval','unit_relation_spearman','locality_gap')},'provenance':provenance},indent=2),flush=True)


if __name__=='__main__':run(parse_args())
