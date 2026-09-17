#!/usr/bin/env python
"""Locked P9 validation caption/grounding and downstream checkpoint selection."""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'BrainHub'))
from eval_bbox_rec import calculate_metric
from models.pre_projector_interface_adapter import PreProjectorInterfaceAdapter
from scripts.run_p7_downstream import core_caption_metrics,full_prompt,generate,load_decoder
from scripts.train_p9_interface_adapter import evaluate as evaluate_interface,load_config
from losses.interface_distillation_loss import InterfaceDistillationLoss
from scripts.cache_p7_downstream_features import load_projector,resolve_config
from utils import extract_boxes,extract_id_bbox_caption


def load_json(path):return json.loads(Path(path).read_text())
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def labels(args,out):
    if args.action=='baseline':return [('source_baseline',None,0)]
    checkpoints=[]
    for path in sorted(out.glob('epoch_*.pth')):
        epoch=int(path.stem.split('_')[-1])
        if epoch%args.downstream_every==0:checkpoints.append((f'epoch_{epoch:03d}',path,epoch))
    last=torch.load(out/'last.pth',map_location='cpu')['epoch']
    last_path=out/f'epoch_{last:03d}.pth'
    if not any(epoch==last for _,_,epoch in checkpoints):checkpoints.append((f'epoch_{last:03d}',last_path,last))
    return checkpoints


def adapter_for(checkpoint,device):
    if checkpoint is None:return None
    adapter=PreProjectorInterfaceAdapter().to(device).eval();adapter.load_state_dict(torch.load(checkpoint,map_location='cpu')['adapter'],strict=True);adapter.requires_grad_(False);return adapter


def projected(cache,indices,adapter,projector,device):
    source=torch.from_numpy(np.stack([np.asarray(cache[index],dtype=np.float32) for index in indices])).to(device)
    with torch.inference_mode():adapted=source if adapter is None else adapter(source);return projector(adapted)


def run_label(label,checkpoint,epoch,cfg,args):
    out=Path(cfg['output_dir']);downstream=resolve_config(cfg['p7_config']);source=np.load(cfg['source_val'],mmap_mode='r');refs=load_json(args.caption_references);annotations=load_json(args.grounding_annotations);tokenizer,decoder,projector=load_decoder(downstream,args.device);adapter=adapter_for(checkpoint,args.device)
    jobs={'caption':[(i,None,full_prompt(downstream['caption_prompt'])) for i in range(300)],'grounding':[(i,e,full_prompt(downstream['grounding_prompt'],e)) for i in range(300) for e in annotations[str(i)]]};all_records={}
    for task,items in jobs.items():
        path=out/f'validation_{task}_{label}.jsonl';existing={}
        if path.exists():
            for line in path.read_text().splitlines():row=json.loads(line);existing[(row['sample_id'],row.get('expression'))]=row
        groups=defaultdict(list)
        for item in items:groups[len(tokenizer(item[2],add_special_tokens=True).input_ids)].append(item)
        with path.open('a') as handle:
            for length in sorted(groups):
                for start in range(0,len(groups[length]),downstream['generation_job_batch_size']):
                    entries=[x for x in groups[length][start:start+downstream['generation_job_batch_size']] if (x[0],x[1]) not in existing]
                    if not entries:continue
                    features=projected(source,[x[0] for x in entries],adapter,projector,args.device)
                    try:responses=generate(tokenizer,decoder,features,[x[2] for x in entries],downstream,args.device);error=None
                    except Exception as exc:responses=['']*len(entries);error=f'{type(exc).__name__}: {exc}'
                    for (sample,expression,_),response in zip(entries,responses):
                        common={'sample_id':sample,'label':label,'epoch':epoch,'response':response,'generation_success':error is None,'output_length':len(tokenizer(response,add_special_tokens=False).input_ids) if response else 0,'error':error}
                        if task=='caption':
                            caption=extract_id_bbox_caption(response)[1] if response else '';row={**common,'caption':caption,'status':'failed' if error else ('empty' if not caption.strip() else 'success')}
                        else:
                            boxes=extract_boxes(response) if response else [];row={**common,'expression':expression,'boxes':boxes,'status':'failed' if error else ('parse_failure' if not boxes else 'success')}
                        handle.write(json.dumps(row)+'\n');handle.flush();existing[(sample,expression)]=row
        all_records[task]=existing;print(json.dumps({'experiment':cfg['experiment_name'],'label':label,'task':task,'completed':len(existing)}),flush=True)
    cap=[all_records['caption'][(i,None)] for i in range(300)];caption=core_caption_metrics([refs[str(i)] for i in range(300)],[x['caption'] for x in cap]);caption.update({'sample_count':300,'success':sum(x['status']=='success' for x in cap),'empty':sum(x['status']=='empty' for x in cap),'failed':sum(x['status']=='failed' for x in cap),'mean_output_length':float(np.mean([x['output_length'] for x in cap])),'median_output_length':float(np.median([x['output_length'] for x in cap]))})
    predictions=[];targets=[]
    for i in range(300):
        for expression,boxes in annotations[str(i)].items():
            parsed=all_records['grounding'][(i,expression)]['boxes'];predictions.append(None if not parsed else parsed[0][0]);targets.append(boxes[0])
    value=calculate_metric(predictions,targets,.5);ground={'query_count':len(targets),'mean_iou':value['iou'],'grounding_accuracy':value['accuracy'],'parse_failures':value['failed'],'target_failures':value['target_failed']};row={'label':label,'epoch':epoch,'checkpoint':None if checkpoint is None else str(Path(checkpoint).resolve()),'caption':caption,'grounding':ground}
    with (out/'validation_downstream.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
    if label=='source_baseline':dump(out/'source_downstream_baseline.json',row)
    del adapter,projector,decoder;torch.cuda.empty_cache();return row


def baseline_interface(cfg,device):
    downstream=resolve_config(cfg['p7_config']);projector=load_projector(downstream['mm_projector'],device);adapter=PreProjectorInterfaceAdapter().to(device).eval().requires_grad_(False);criterion=InterfaceDistillationLoss(**cfg['loss_weights']).to(device);metrics=evaluate_interface(adapter,projector,criterion,cfg,device);metrics['identity_adapter_used_for_measurement']=True;metrics['adapter_max_abs_correction']=0.0;dump(Path(cfg['output_dir'])/'source_interface_baseline.json',metrics);return metrics


def finalize(cfg):
    out=Path(cfg['output_dir']);rows=[json.loads(x) for x in (out/'validation_downstream.jsonl').read_text().splitlines()];unique={row['label']:row for row in rows};baseline=unique['source_baseline'];interface={int(row['epoch']):row for row in map(json.loads,(out/'metrics_per_epoch.jsonl').read_text().splitlines())};candidates=[]
    for label,row in unique.items():
        if label=='source_baseline':continue
        epoch=int(row['epoch']);compatible=row['caption']['CIDEr']>=baseline['caption']['CIDEr'] and row['grounding']['grounding_accuracy']>=baseline['grounding']['grounding_accuracy'];candidates.append({**row,'projected_oracle_gap':interface[epoch]['projected_oracle_gap'],'downstream_compatible':compatible})
    eligible=[row for row in candidates if row['downstream_compatible']]
    if eligible:
        selected=min(eligible,key=lambda row:(row['projected_oracle_gap'],-row['caption']['CIDEr']));status='DOWNSTREAM_COMPATIBLE_CHECKPOINT_FOUND';shutil.copy2(selected['checkpoint'],out/'best.pth')
    else:
        selected=None;status='INTERFACE_DISTILLATION_REDUCES_GAP_BUT_NOT_UTILITY'
    result={'selection_rule':'CIDEr >= source AND grounding accuracy >= source; then lowest projected oracle gap, CIDEr tie-break','source_baseline':baseline,'candidates':candidates,'selected':selected,'selection_status':status,'test_used':False};dump(out/'final_val_metrics.json',result);dump(out/'adapter_diagnostics.json',{'source_interface':load_json(out/'source_interface_baseline.json'),'selected_interface':None if selected is None else interface[selected['epoch']]});print(json.dumps({'experiment':cfg['experiment_name'],'selection_status':status,'selected':None if selected is None else selected['label']}))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('baseline','checkpoints','finalize'));p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--caption-references',default='protocol_outputs/protocol_v1/subj01/p9_validation_targets/caption_references.json');p.add_argument('--grounding-annotations',default='protocol_outputs/protocol_v1/subj01/p9_validation_targets/grounding_annotations.json');p.add_argument('--downstream-every',type=int,default=2);a=p.parse_args();cfg=load_config(a.config);Path(cfg['output_dir']).mkdir(parents=True,exist_ok=True)
    if a.action=='finalize':finalize(cfg);return
    if a.action=='baseline':baseline_interface(cfg,a.device)
    existing=set()
    path=Path(cfg['output_dir'])/'validation_downstream.jsonl'
    if path.exists():existing={json.loads(x)['label'] for x in path.read_text().splitlines()}
    for label,checkpoint,epoch in labels(a,Path(cfg['output_dir'])):
        if label not in existing:run_label(label,checkpoint,epoch,cfg,a)
if __name__=='__main__':main()
