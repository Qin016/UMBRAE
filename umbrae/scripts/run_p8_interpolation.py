#!/usr/bin/env python
"""Frozen-decoder downstream evaluation of the preregistered P8 alpha grid."""

import argparse,csv,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'BrainHub'))
from eval_bbox_rec import calculate_metric
from models.p8_diagnostics import interpolate
from scripts.cache_p7_downstream_features import resolve_config
from scripts.run_p7_downstream import core_caption_metrics,full_prompt,generate,load_decoder
from utils import extract_boxes,extract_id_bbox_caption

VARIANTS=('lora','full_real','full_random');ALPHAS=(.25,.5,.75);MODES=tuple(f'{v}_a{int(a*100):03d}' for a in ALPHAS for v in VARIANTS)

def parse_mode(mode):
    variant,tag=mode.rsplit('_a',1);return variant,int(tag)/100

def project(entries,caches,projector,device):
    values=[]
    for sample,mode,_,_ in entries:
        variant,alpha=parse_mode(mode);base=np.asarray(caches['umbrae'][sample],dtype=np.float32);other=np.asarray(caches[variant][sample],dtype=np.float32);values.append(base+alpha*(other-base))
    with torch.inference_mode():return projector(torch.from_numpy(np.stack(values)).to(device))

def run_task(task,jobs,caches,tokenizer,decoder,projector,config,device,output):
    path=output/f'p8_interpolation_{task}_predictions.jsonl';existing={}
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line);existing[(row['sample_id'],row['mode'],row.get('expression'))]=row
    groups=defaultdict(list)
    for sample,expression,prompt in jobs:groups[len(tokenizer(prompt,add_special_tokens=True).input_ids)].append((sample,expression,prompt))
    started=time.time();generated=0
    with path.open('a') as handle:
        for alpha in ALPHAS:
            alpha_modes=[f'{v}_a{int(alpha*100):03d}' for v in VARIANTS]
            for length in sorted(groups):
                group=groups[length]
                for start in range(0,len(group),4):
                    entries=[]
                    for sample,expression,prompt in group[start:start+4]:
                        for mode in alpha_modes:
                            if (sample,mode,expression) not in existing:entries.append((sample,mode,expression,prompt))
                    if not entries:continue
                    projected=project(entries,caches,projector,device);begin=time.time()
                    try:responses=generate(tokenizer,decoder,projected,[x[3] for x in entries],config,device);error=None
                    except Exception as exc:responses=['']*len(entries);error=f'{type(exc).__name__}: {exc}'
                    elapsed=time.time()-begin
                    for entry,response in zip(entries,responses):
                        sample,mode,expression,_=entry
                        if task=='caption':
                            value=extract_id_bbox_caption(response)[1] if response else '';row={'sample_id':sample,'mode':mode,'response':response,'caption':value,'status':'failed' if error else ('empty' if not value.strip() else 'success'),'error':error}
                        else:
                            boxes=extract_boxes(response) if response else [];row={'sample_id':sample,'mode':mode,'expression':expression,'response':response,'boxes':boxes,'status':'failed' if error else ('parse_failure' if not boxes else 'success'),'error':error}
                        row['batch_elapsed_seconds']=elapsed;handle.write(json.dumps(row)+'\n');handle.flush();existing[(sample,mode,expression)]=row;generated+=1
                    if generated and generated%180<len(entries):print(json.dumps({'task':task,'generated':generated,'total':len(jobs)*len(MODES)}),flush=True)
    return list(existing.values()),time.time()-started

def evaluate(caption_records,ground_records,config,output):
    refs=json.loads(Path(config['caption_references']).read_text());annotations=json.loads(Path(config['grounding_annotations']).read_text());caption_rows=[];ground_rows=[]
    for mode in MODES:
        variant,alpha=parse_mode(mode);cap={int(x['sample_id']):x for x in caption_records if x['mode']==mode};candidates=[cap[i]['caption'] for i in range(982)];metrics=core_caption_metrics([refs[str(i)] for i in range(982)],candidates);statuses={s:sum(x['status']==s for x in cap.values()) for s in ('success','empty','failed')};caption_rows.append({'variant':variant,'alpha':alpha,**metrics,'sample_count':982,**statuses})
        selected={(int(x['sample_id']),x['expression']):x for x in ground_records if x['mode']==mode};preds=[];targets=[]
        for sample in range(982):
            for expression,boxes in annotations[str(sample)].items():
                parsed=selected[(sample,expression)]['boxes'];preds.append(None if not parsed else parsed[0][0]);targets.append(boxes[0])
        metric=calculate_metric(preds,targets,threshold=.5);ground_rows.append({'variant':variant,'alpha':alpha,'mean_iou':metric['iou'],'grounding_accuracy':metric['accuracy'],'parse_failures':metric['failed'],'target_failures':metric['target_failed'],'query_count':len(targets)})
    for name,rows in [('caption',caption_rows),('grounding',ground_rows)]:
        with (output/f'p8_interpolation_{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (output/'p8_interpolation_downstream.json').write_text(json.dumps({'caption':caption_rows,'grounding':ground_rows},indent=2)+'\n');return caption_rows,ground_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/dual_branch/p7_downstream_protocol_v1.json');p.add_argument('--output-dir',default='dual_branch_outputs/p8_diagnosis');p.add_argument('--device',default='cuda');a=p.parse_args();config=resolve_config(a.config);output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True);meta=json.loads(Path(config['output_dir'],'feature_cache_metadata.json').read_text());caches={m:np.load(meta['representations'][m],mmap_mode='r') for m in ('umbrae',*VARIANTS)}
    if any(x.shape!=(982,256,1024) for x in caches.values()):raise ValueError('P7 cache shape mismatch')
    for variant in VARIANTS:
        if not np.array_equal(np.asarray(caches['umbrae'][0]),interpolate(caches['umbrae'][0],caches[variant][0],0)):raise ValueError('alpha=0 invariant failed')
        if not np.array_equal(np.asarray(caches[variant][0]),interpolate(caches['umbrae'][0],caches[variant][0],1)):raise ValueError('alpha=1 invariant failed')
    tokenizer,decoder,projector=load_decoder(config,a.device);caption_jobs=[(i,None,full_prompt(config['caption_prompt'])) for i in range(982)];caption_records,caption_time=run_task('caption',caption_jobs,caches,tokenizer,decoder,projector,config,a.device,output);annotations=json.loads(Path(config['grounding_annotations']).read_text());ground_jobs=[(i,e,full_prompt(config['grounding_prompt'],e)) for i in range(982) for e in annotations[str(i)]];ground_records,ground_time=run_task('grounding',ground_jobs,caches,tokenizer,decoder,projector,config,a.device,output);captions,grounding=evaluate(caption_records,ground_records,config,output);(output/'p8_interpolation_compute.json').write_text(json.dumps({'intermediate_mode_count':len(MODES),'caption_wall_seconds':caption_time,'grounding_wall_seconds':ground_time,'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/1024**3,'decoder_equivalence':'PASS','mm_projector_equivalence':'PASS','training':False},indent=2)+'\n');print(json.dumps({'interpolation_downstream':'complete','caption':captions,'grounding':grounding}))

if __name__=='__main__':main()
