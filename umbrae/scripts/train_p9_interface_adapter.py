#!/usr/bin/env python
"""Train only the P9 token-wise residual interface adapter."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from losses.interface_distillation_loss import InterfaceDistillationLoss
from models.dual_branch_cache import sha256_file
from models.pre_projector_interface_adapter import PreProjectorInterfaceAdapter,adapter_parameter_counts
from scripts.analyze_p8_projector import hard_retrieval,rsa_any
from scripts.cache_p7_downstream_features import load_projector,resolve_config


def load_config(path):
    cfg=json.loads(Path(path).read_text());cfg['_config_path']=str(Path(path).resolve())
    for key in ('source_train','source_val','visual_train','visual_val','projected_teacher_train','projected_teacher_val','output_dir','p7_config'):
        cfg[key]=str(Path(cfg[key]).resolve())
    return cfg


def arrays(cfg,split):
    return tuple(np.load(cfg[f'{name}_{split}'],mmap_mode='r') for name in ('source','visual','projected_teacher'))


def forward_loss(adapter,projector,criterion,source,visual,teacher,amp):
    with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
        output=adapter(source,return_delta=True);projected=projector(output['z_hat']);losses=criterion(source,output['z_hat'],visual,projected,teacher)
    return output,projected,losses


def evaluate(adapter,projector,criterion,cfg,device):
    source,visual,teacher=arrays(cfg,'val');sums={k:0.0 for k in ('loss','proj_cos','proj_mse','preproj_cos','preserve','norm')};tokens=samples=0;pooled=[];visual_pooled=[];correction=[];projected_norm=[];teacher_norm=[]
    adapter.eval()
    for start in range(0,len(source),cfg['batch_size']):
        end=min(start+cfg['batch_size'],len(source));z=torch.from_numpy(np.array(source[start:end],dtype=np.float32,copy=True)).to(device);v=torch.from_numpy(np.array(visual[start:end],dtype=np.float32,copy=True)).to(device);h=torch.from_numpy(np.array(teacher[start:end],dtype=np.float32,copy=True)).to(device)
        with torch.inference_mode():out,p,loss=forward_loss(adapter,projector,criterion,z,v,h,cfg['amp'])
        for key in sums:sums[key]+=float(loss[key])*(end-start)
        pooled.append(out['z_hat'].float().mean(1).cpu().numpy());visual_pooled.append(v.float().mean(1).cpu().numpy());correction.extend((out['z_hat'].float()-z.float()).flatten(1).norm(dim=1).div(z.float().flatten(1).norm(dim=1).clamp_min(1e-8)).cpu().tolist());projected_norm.append(p.float().norm(dim=-1).cpu().numpy());teacher_norm.append(h.float().norm(dim=-1).cpu().numpy());samples+=end-start
    pooled=np.concatenate(pooled);visual_pooled=np.concatenate(visual_pooled);projected_norm=np.concatenate(projected_norm);teacher_norm=np.concatenate(teacher_norm);losses={key:value/samples for key,value in sums.items()};retrieval=hard_retrieval(pooled,visual_pooled,k=99);retrieval={key:value for key,value in retrieval.items() if key!='ranks'}
    return {**{f'val_{key}':value for key,value in losses.items()},'projected_paired_token_cosine':1-losses['proj_cos'],'projected_oracle_gap':losses['proj_cos'],'preprojector_paired_cosine':1-losses['preproj_cos'],'adapter_correction_ratio':float(np.mean(correction)),'projected_norm_ratio':float(projected_norm.mean()/teacher_norm.mean()),'gate':float(adapter.gate),'rsa':rsa_any(pooled,visual_pooled),'hard_retrieval':retrieval,'sample_count':samples}


def state(adapter,optimizer,scaler,cfg,epoch,metrics):
    return {'adapter':adapter.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'metrics':metrics,'metadata':dict(cfg['_checkpoint_metadata'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();cfg=load_config(a.config);out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps({k:v for k,v in cfg.items() if k!='_config_path'},indent=2)+'\n')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);source,visual,teacher=arrays(cfg,'train');val=arrays(cfg,'val')
    expected=((8559,256,1024),(8559,256,1024),(8559,256,4096));
    if tuple(x.shape for x in (source,visual,teacher))!=expected or tuple(x.shape[0] for x in val)!=(300,300,300):raise ValueError('P9 cache shape mismatch')
    downstream=resolve_config(cfg['p7_config']);projector=load_projector(downstream['mm_projector'],a.device);assert not any(x.requires_grad for x in projector.parameters())
    adapter=PreProjectorInterfaceAdapter(1024,256,cfg['gate_logit_init']).to(a.device);criterion=InterfaceDistillationLoss(**cfg['loss_weights']).to(a.device);counts=adapter_parameter_counts(adapter);non_gate=[x for name,x in adapter.named_parameters() if name!='gate_logit'];optimizer=torch.optim.AdamW([{'params':non_gate,'lr':cfg['adapter_lr'],'weight_decay':cfg['weight_decay']},{'params':[adapter.gate_logit],'lr':cfg['gate_lr'],'weight_decay':0.0}]);scaler=torch.cuda.amp.GradScaler(enabled=cfg['amp'])
    probe=torch.from_numpy(np.array(source[:2],dtype=np.float32,copy=True)).to(a.device)
    with torch.inference_mode():initial=adapter(probe);difference=(initial-probe).abs();initialization={'max_abs_error':float(difference.max()),'mean_abs_error':float(difference.mean()),'cosine_similarity':float(F.cosine_similarity(initial.flatten(1),probe.flatten(1)).mean()),'gate':float(adapter.gate)}
    if initialization['max_abs_error']!=0:raise RuntimeError('Adapter initialization is not identity')
    cfg['_checkpoint_metadata']={'source_kind':cfg['source_kind'],'source_checkpoint':cfg['source_checkpoint'],'source_checkpoint_sha256':sha256_file(cfg['source_checkpoint']),'mm_projector':downstream['mm_projector'],'mm_projector_sha256':sha256_file(downstream['mm_projector']),'clip_teacher':'openai/clip-vit-large-patch14 hidden_states[-2][:,1:,:]','adapter_architecture':'LayerNorm(1024)->Linear(1024,256)->GELU->Linear(256,1024), gated residual','protocol':'protocol_v1','test_used':False}
    provenance={'experiment_name':cfg['experiment_name'],'source_kind':cfg['source_kind'],'source_checkpoint':cfg['source_checkpoint'],'source_checkpoint_sha256':cfg['_checkpoint_metadata']['source_checkpoint_sha256'],'trainable_policy':'adapter_and_gate_only','parameter_counts':counts,'trainable_over_brainx_percent':100*counts['total_trainable']/cfg['brainx_parameter_count'],'initialization_equivalence':initialization,'mm_projector':downstream['mm_projector'],'mm_projector_sha256':cfg['_checkpoint_metadata']['mm_projector_sha256'],'clip_teacher':cfg['_checkpoint_metadata']['clip_teacher'],'brain_representation_frozen':True,'projector_frozen':True,'shikra_used_in_training':False,'test_used':False,'seed':cfg['seed']};(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    generator=torch.Generator().manual_seed(cfg['seed']);best_gap=float('inf');bad=0;started=time.time();metrics_path=out/'metrics_per_epoch.jsonl'
    if metrics_path.exists():raise RuntimeError('Refusing to append to an existing P9 training run')
    for epoch in range(1,cfg['max_epochs']+1):
        adapter.train();order=torch.randperm(len(source),generator=generator).numpy();sums={k:0.0 for k in ('loss','proj_cos','proj_mse','preproj_cos','preserve','norm')};count=0;grad_sum=0.0
        for start in range(0,len(order),cfg['batch_size']):
            ids=order[start:start+cfg['batch_size']];z=torch.from_numpy(np.array(source[ids],dtype=np.float32,copy=True)).to(a.device);v=torch.from_numpy(np.array(visual[ids],dtype=np.float32,copy=True)).to(a.device);h=torch.from_numpy(np.array(teacher[ids],dtype=np.float32,copy=True)).to(a.device);optimizer.zero_grad(set_to_none=True);_,_,loss=forward_loss(adapter,projector,criterion,z,v,h,cfg['amp']);scaler.scale(loss['loss']).backward();scaler.unscale_(optimizer);grad=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),cfg['gradient_clip']));scaler.step(optimizer);scaler.update();size=len(ids);count+=size;grad_sum+=grad*size
            for key in sums:sums[key]+=float(loss[key])*size
        validation=evaluate(adapter,projector,criterion,cfg,a.device);record={'epoch':epoch,**{f'train_{key}':value/count for key,value in sums.items()},'train_grad_norm':grad_sum/count,**validation,'epoch_wall_seconds':time.time()-started};checkpoint=state(adapter,optimizer,scaler,cfg,epoch,record);torch.save(checkpoint,out/f'epoch_{epoch:03d}.pth');torch.save(checkpoint,out/'last.pth')
        with metrics_path.open('a') as handle:handle.write(json.dumps(record)+'\n')
        print(json.dumps({'experiment':cfg['experiment_name'],'epoch':epoch,'gap':record['projected_oracle_gap'],'gate':record['gate'],'correction':record['adapter_correction_ratio']}),flush=True)
        if record['projected_oracle_gap']<best_gap-1e-7:best_gap=record['projected_oracle_gap'];bad=0;torch.save(checkpoint,out/'best_interface.pth')
        else:bad+=1
        if epoch>=cfg['minimum_epochs'] and bad>=cfg['patience']:break
    (out/'interface_training_summary.json').write_text(json.dumps({'epochs_completed':epoch,'best_interface_gap':best_gap,'wall_seconds':time.time()-started,'downstream_selection_pending':True},indent=2)+'\n')

if __name__=='__main__':main()
