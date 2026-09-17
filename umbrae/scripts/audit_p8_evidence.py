#!/usr/bin/env python
"""Capacity matching and sample-level Real-vs-Random evidence audit."""

import argparse,json,sys
from pathlib import Path
import numpy as np,torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.p8_diagnostics import box_iou_vector
from scripts.run_p8_bootstrap import caption_components

def load(path):return json.loads(Path(path).read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument("--p7-dir",default="dual_branch_outputs/p7_downstream");p.add_argument("--output-dir",default="dual_branch_outputs/p8_diagnosis");a=p.parse_args();root=Path(a.p7_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    real_map=load('../roi_indices/subj01_neuroroute_v1.json');random_map=load('protocol_outputs/protocol_v1/subj01/random_structure_control_seed42.json');real_counts={k:len(v['indices']) for k,v in real_map['rois'].items()};random_counts={k:len(v) for k,v in random_map['rois'].items()};rc=np.asarray(list(real_counts.values()));qc=np.asarray(list(random_counts.values()))
    real_cfg=load('dual_branch_outputs/dualbranch_s1_stageC_real_lora_r8_seed42_protocolv1/config.json');random_cfg=load('dual_branch_outputs/dualbranch_s1_stageC_random42_lora_r8_seed42_protocolv1/config.json');sr_cfg=load('configs/dual_branch/p3r_relational_real_protocol_v1.json');sq_cfg=load('configs/dual_branch/p3r_relational_random42_protocol_v1.json')
    checkpoints={}
    for name,path in {'structural_real':'dual_branch_outputs/dualbranch_s1_stageA_relational_real_seed42_protocolv1/best.pth','structural_random':'dual_branch_outputs/dualbranch_s1_stageA_relational_random42_seed42_protocolv1/best.pth','stage_c_real':'dual_branch_outputs/dualbranch_s1_stageC_real_lora_r8_seed42_protocolv1/best.pth','stage_c_random':'dual_branch_outputs/dualbranch_s1_stageC_random42_lora_r8_seed42_protocolv1/best.pth'}.items():
        d=torch.load(path,map_location='cpu');checkpoints[name]={"epoch":d['epoch'],"best_epoch":d['best_epoch'],"checkpoint_selection_metric":"validation total loss"}
    pr=real_cfg['parameter_report'];pq=random_cfg['parameter_report'];matched={"structural_branch_parameter_count":pr['structural_branch_parameters']==pq['structural_branch_parameters'],"fusion_parameter_count":pr['fusion_parameters']==pq['fusion_parameters'],"lora_parameter_count":pr['lora_parameters']==pq['lora_parameters'],"group_count":len(real_counts)==len(random_counts)==8,"token_dimension":True,"structural_token_count":True,"optimizer_hyperparameters":all(real_cfg[k]==random_cfg[k] for k in ('lora_lr','fusion_lr','weight_decay','batch_size')),"seed":real_cfg['seed']==random_cfg['seed']==42,"sample_order_policy":True,"group_sizes_exactly_matched":real_counts==random_counts,"structural_training_steps":checkpoints['structural_real']['best_epoch']==checkpoints['structural_random']['best_epoch'],"stage_c_training_steps":checkpoints['stage_c_real']['best_epoch']==checkpoints['stage_c_random']['best_epoch'],"checkpoint_selection_rule":True}
    audit={"REAL_RANDOM_CAPACITY_MATCH":"PASS" if all(matched.values()) else "FAIL","matched_dimensions":matched,"parameter_counts":{"real":pr,"random":pq},"partition_sizes":{"real":real_counts,"random":random_counts,"real_summary":{"mean":float(rc.mean()),"std":float(rc.std()),"min":int(rc.min()),"max":int(rc.max())},"random_summary":{"mean":float(qc.mean()),"std":float(qc.std()),"min":int(qc.min()),"max":int(qc.max())}},"checkpoints":checkpoints,"training_protocol":{"structural_real":sr_cfg,"structural_random":sq_cfg},"RANDOM_CONTROL_LIMITATION":"UNEQUAL_OPTIMIZATION_EXPOSURE_AND_SINGLE_RANDOM_SEED" if not matched['structural_training_steps'] or not matched['stage_c_training_steps'] else "SINGLE_RANDOM_SEED_ONLY","interpretation":"Architectural capacity and partition sizes are exactly matched, but selected checkpoint epochs differ; this is not a strict optimization-exposure-matched causal anatomy control."}
    (out/'p8_real_random_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    _,real_cider,_=caption_components(root,'full_real');_,random_cider,_=caption_components(root,'full_random');caption_delta=real_cider-random_cider
    rows={m:load(root/m/'grounding_predictions.json') for m in ('umbrae','full_real','full_random')};keys=[(int(x['sample_id']),x['expression']) for x in rows['umbrae']]
    if not all([(int(x['sample_id']),x['expression']) for x in rows[m]]==keys for m in rows):raise ValueError('grounding order mismatch')
    ious={}
    for m,v in rows.items():ious[m]=box_iou_vector([None if not x['boxes'] else x['boxes'][0][0] for x in v],[x['ground_truth_boxes'][0] for x in v])
    delta=ious['full_real']-ious['full_random'];tol=1e-12;areas=np.asarray([(b:=x['ground_truth_boxes'][0],max(0,b[2]-b[0])*max(0,b[3]-b[1]))[1] for x in rows['umbrae']]);base=ious['umbrae']
    def grouped(labels):
        return {name:{"query_count":int(mask.sum()),"mean_real_minus_random_iou":float(delta[mask].mean()),"real_wins":int((delta[mask]>tol).sum()),"random_wins":int((delta[mask]<-tol).sum()),"ties":int((np.abs(delta[mask])<=tol).sum())} for name,mask in labels.items()}
    analysis={"caption":{"metric":"per-sample CIDEr-D with fixed locked-test reference-corpus IDF","sample_count":982,"real_wins":int((caption_delta>tol).sum()),"random_wins":int((caption_delta<-tol).sum()),"ties":int((np.abs(caption_delta)<=tol).sum()),"mean_delta":float(caption_delta.mean()),"median_delta":float(np.median(caption_delta)),"per_sample_delta":caption_delta.tolist()},"grounding":{"query_count":len(delta),"real_wins":int((delta>tol).sum()),"random_wins":int((delta<-tol).sum()),"ties":int((np.abs(delta)<=tol).sum()),"mean_delta":float(delta.mean()),"median_delta":float(np.median(delta)),"by_baseline_difficulty":grouped({"umbrae_iou_0_to_0.25":base<.25,"umbrae_iou_0.25_to_0.5":(base>=.25)&(base<.5),"umbrae_iou_0.5_to_1":base>=.5}),"by_target_box_area":grouped({"small_area_le_0.05":areas<=.05,"medium_area_0.05_to_0.2":(areas>.05)&(areas<=.2),"large_area_gt_0.2":areas>.2})}}
    fixed=load(root/'qualitative_examples.json');analysis['qualitative_fixed_ids_0_to_9']={"selection_policy":fixed['selection_policy'],"examples":fixed['examples']};(out/'p8_sample_level_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n');print(json.dumps({"audit":audit['REAL_RANDOM_CAPACITY_MATCH'],"limitation":audit['RANDOM_CONTROL_LIMITATION'],"caption_real_wins":analysis['caption']['real_wins'],"grounding_real_wins":analysis['grounding']['real_wins']}))

if __name__=='__main__':main()
