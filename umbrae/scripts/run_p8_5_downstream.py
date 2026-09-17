#!/usr/bin/env python
"""Two-worker frozen Shikra evaluation for P8.5 analytic corrections."""
import argparse,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'BrainHub'))
from eval_bbox_rec import calculate_metric
from scripts.cache_p7_downstream_features import resolve_config
from scripts.run_p7_downstream import core_caption_metrics,full_prompt,generate,load_decoder
from utils import extract_boxes,extract_id_bbox_caption
MODES=('mean','affine','wct')
def projected(entries,caches,projector,device):
 x=np.stack([np.asarray(caches[m][i],dtype=np.float32) for i,m,_,_ in entries]);
 with torch.inference_mode():return projector(torch.from_numpy(x).to(device))
def worker(a):
 cfg=resolve_config(a.config);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);lo=a.range_lo if a.range_lo is not None else a.worker_index*982//a.num_workers;hi=a.range_hi if a.range_hi is not None else (a.worker_index+1)*982//a.num_workers;label=a.worker_label or str(a.worker_index);caches={m:np.load(out/f'full_real_{m}_test_fp16.npy',mmap_mode='r') for m in MODES};tok,dec,proj=load_decoder(cfg,a.device)
 annotations=json.loads(Path(cfg['grounding_annotations']).read_text());jobs={'caption':[(i,None,full_prompt(cfg['caption_prompt'])) for i in range(lo,hi)],'grounding':[(i,e,full_prompt(cfg['grounding_prompt'],e)) for i in range(lo,hi) for e in annotations[str(i)]]}
 for task,items in jobs.items():
  path=out/f'{task}_worker{label}.jsonl';existing={};sources=[path]
  if a.resume_worker is not None:sources.insert(0,out/f'{task}_worker{a.resume_worker}.jsonl')
  for source in sources:
   if source.exists():
    for line in source.read_text().splitlines():r=json.loads(line);existing[(r['sample_id'],r['mode'],r.get('expression'))]=r
  groups=defaultdict(list)
  for x in items:groups[len(tok(x[2],add_special_tokens=True).input_ids)].append(x)
  with path.open('a') as f:
   for length in sorted(groups):
    for start in range(0,len(groups[length]),4):
     entries=[]
     for i,e,p in groups[length][start:start+4]:
      for m in MODES:
       if (i,m,e) not in existing:entries.append((i,m,e,p))
     if not entries:continue
     try:responses=generate(tok,dec,projected(entries,caches,proj,a.device),[x[3] for x in entries],cfg,a.device);error=None
     except Exception as ex:responses=['']*len(entries);error=f'{type(ex).__name__}: {ex}'
     for x,response in zip(entries,responses):
      i,m,e,_=x
      if task=='caption':value=extract_id_bbox_caption(response)[1] if response else '';r={'sample_id':i,'mode':m,'caption':value,'response':response,'status':'failed' if error else ('empty' if not value.strip() else 'success'),'error':error}
      else:boxes=extract_boxes(response) if response else [];r={'sample_id':i,'mode':m,'expression':e,'boxes':boxes,'response':response,'status':'failed' if error else ('parse_failure' if not boxes else 'success'),'error':error}
      f.write(json.dumps(r)+'\n');f.flush();existing[(i,m,e)]=r
  print(json.dumps({'worker':label,'device':a.device,'task':task,'entries':len(existing)}),flush=True)
def evaluate(a):
 cfg=resolve_config(a.config);out=Path(a.output_dir);refs=json.loads(Path(cfg['caption_references']).read_text());ann=json.loads(Path(cfg['grounding_annotations']).read_text());caps=[];grounds=[]
 for path in out.glob('caption_worker*.jsonl'):caps += [json.loads(x) for x in path.read_text().splitlines()]
 for path in out.glob('grounding_worker*.jsonl'):grounds += [json.loads(x) for x in path.read_text().splitlines()]
 cm={};gm={}
 for m in MODES:
  selected={r['sample_id']:r for r in caps if r['mode']==m};cand=[selected[i]['caption'] for i in range(982)];v=core_caption_metrics([refs[str(i)] for i in range(982)],cand);v.update({'sample_count':982,'success':sum(r['status']=='success' for r in selected.values()),'empty':sum(r['status']=='empty' for r in selected.values()),'failed':sum(r['status']=='failed' for r in selected.values())});cm[m]=v
  sel={(r['sample_id'],r['expression']):r for r in grounds if r['mode']==m};pred=[];target=[]
  for i in range(982):
   for e,b in ann[str(i)].items():x=sel[(i,e)]['boxes'];pred.append(None if not x else x[0][0]);target.append(b[0])
  v=calculate_metric(pred,target,.5);gm[m]={'mean_iou':v['iou'],'grounding_accuracy':v['accuracy'],'parse_failures':v['failed'],'target_failures':v['target_failed'],'query_count':2419}
 (out/'caption_metrics.json').write_text(json.dumps(cm,indent=2)+'\n');(out/'grounding_metrics.json').write_text(json.dumps(gm,indent=2)+'\n');print(json.dumps({'caption':cm,'grounding':gm}))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/dual_branch/p7_downstream_protocol_v1.json');p.add_argument('--output-dir',default='dual_branch_outputs/p8_5_distribution_bias');p.add_argument('--device',default='cuda:0');p.add_argument('--worker-index',type=int,default=0);p.add_argument('--num-workers',type=int,default=2);p.add_argument('--range-lo',type=int);p.add_argument('--range-hi',type=int);p.add_argument('--worker-label');p.add_argument('--resume-worker');p.add_argument('--evaluate-only',action='store_true');a=p.parse_args();evaluate(a) if a.evaluate_only else worker(a)
if __name__=='__main__':main()
