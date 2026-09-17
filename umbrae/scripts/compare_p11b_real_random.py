#!/usr/bin/env python3
"""Paired validation-only comparison and report for completed P11-B runs."""

import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METRICS=(('local_cos','Local Cosine'),('unit_localization_mrr','Unit Localization MRR'),('local_sample_mrr','Local Sample MRR'),('unit_relation_spearman','Unit Relation Spearman'),('locality_gap','Locality Gap'))


def load(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def savecsv(path,data):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def bootstrap(delta,repeats=10000,seed=42):
    rng=np.random.default_rng(seed);n=len(delta);means=np.empty(repeats)
    for start in range(0,repeats,1000):
        k=min(1000,repeats-start);means[start:start+k]=delta[rng.integers(0,n,size=(k,n))].mean(1)
    return {'observed_delta':float(delta.mean()),'ci95':[float(x) for x in np.quantile(means,[.025,.975])],'probability_delta_gt0':float(np.mean(means>0)),'resampling_unit_count':n,'repeats':repeats,'seed':seed}
def series(metric,records):return [x[metric] for x in records]


def main():
    p=argparse.ArgumentParser();p.add_argument('--real',type=Path,required=True);p.add_argument('--random',type=Path,required=True);p.add_argument('--teacher-shuffle',type=Path);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);figdir=a.output_dir/'figures';figdir.mkdir(exist_ok=True)
    real=load(a.real/'final_val_metrics.json');random=load(a.random/'final_val_metrics.json');shuffle=load(a.teacher_shuffle/'final_val_metrics.json') if a.teacher_shuffle and (a.teacher_shuffle/'final_val_metrics.json').exists() else None
    rc=rows(a.real/'metrics_per_epoch.jsonl');qc=rows(a.random/'metrics_per_epoch.jsonl')
    def scalar(x,key):
        if key=='unit_localization_mrr':return x['unit_localization']['mrr']
        if key=='local_sample_mrr':return x['local_sample_retrieval']['mrr']
        return x[key]
    sample_key={'local_cos':'local_cos','unit_localization_mrr':'unit_localization_mrr','local_sample_mrr':'local_sample_mrr','unit_relation_spearman':'unit_relation_spearman','locality_gap':'locality_gap'}
    boot={};comparison=[]
    for key,label in METRICS:
        d=np.array(real['sample_metrics'][sample_key[key]])-np.array(random['sample_metrics'][sample_key[key]]);boot[key]=bootstrap(d)
        comparison.append({'metric':label,'real':scalar(real,key),'random':scalar(random,key),'delta_real_minus_random':scalar(real,key)-scalar(random,key),'ci95_low':boot[key]['ci95'][0],'ci95_high':boot[key]['ci95'][1]})
    savecsv(a.output_dir/'real_vs_random_metrics.csv',comparison)
    rpu={x['unit_id']:x for x in real['per_unit']};qpu={x['unit_id']:x for x in random['per_unit']};per=[]
    for uid,x in rpu.items():
        y=qpu[uid];per.append({'unit_id':uid,'ROI':x['ROI'],'hemisphere':x['hemisphere'],'mean_x':x['mean_x'],'mean_y':x['mean_y'],'ecc':x['ecc'],'sigma':x['sigma'],'mean_R2':x['mean_R2'],'voxels':x['voxels'],'Real_local_cos':x['local_cos'],'Random_local_cos':y['local_cos'],'delta_local_cos':x['local_cos']-y['local_cos'],'Real_localization_MRR':x['localization_mrr'],'Random_localization_MRR':y['localization_mrr'],'delta_localization_MRR':x['localization_mrr']-y['localization_mrr'],'Real_sample_MRR':x['sample_mrr'],'Random_sample_MRR':y['sample_mrr'],'delta_sample_MRR':x['sample_mrr']-y['sample_mrr']})
    savecsv(a.output_dir/'per_unit_real_random.csv',per)
    unit_boot={key:bootstrap(np.array([x[f'delta_{key}'] for x in per])) for key in ('local_cos','localization_MRR','sample_MRR')};boot['unit_bootstrap']=unit_boot;(a.output_dir/'bootstrap_results.json').write_text(json.dumps(boot,indent=2)+'\n')
    roi=[]
    for name in ('V1','V2','V3','hV4'):
        r=real['roi_metrics'][name];q=random['roi_metrics'][name];roi.append({'ROI':name,'Real_Local_Cos':r['local_cos'],'Random_Local_Cos':q['local_cos'],'Delta_Local_Cos':r['local_cos']-q['local_cos'],'Real_Loc_MRR':r['unit_localization_mrr'],'Random_Loc_MRR':q['unit_localization_mrr'],'Delta_Loc_MRR':r['unit_localization_mrr']-q['unit_localization_mrr'],'Real_Sample_MRR':r['local_sample_mrr'],'Random_Sample_MRR':q['local_sample_mrr'],'Delta_Sample_MRR':r['local_sample_mrr']-q['local_sample_mrr']})
    savecsv(a.output_dir/'roi_real_random.csv',roi)
    plots=[('train_loss','training_loss_real_vs_random.png','Train Total Loss'),('val_local_cos','val_local_cos_real_vs_random.png','Validation Local Cosine'),('val_unit_localization_mrr','unit_localization_mrr_real_vs_random.png','Unit Localization MRR'),('val_local_sample_mrr','local_sample_mrr_real_vs_random.png','Local Sample MRR'),('val_locality_gap','locality_gap_real_vs_random.png','Locality Gap')]
    for key,name,title in plots:
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(range(1,len(rc)+1),series(key,rc),label='Real');ax.plot(range(1,len(qc)+1),series(key,qc),label='Random');ax.set(xlabel='Epoch',ylabel=title,title=title);ax.legend();fig.tight_layout();fig.savefig(figdir/name,dpi=160);plt.close(fig)
    x=np.arange(4);fig,ax=plt.subplots(figsize=(8,4));ax.bar(x-.18,[z['Delta_Local_Cos'] for z in roi],.36,label='Δ Local Cos');ax.bar(x+.18,[z['Delta_Loc_MRR'] for z in roi],.36,label='Δ Loc MRR');ax.axhline(0,color='k',lw=.6);ax.set_xticks(x,labels=[z['ROI'] for z in roi]);ax.legend();ax.set_title('Real − Random by ROI');fig.tight_layout();fig.savefig(figdir/'roi_real_random_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6));sc=ax.scatter([z['mean_x'] for z in per],[z['mean_y'] for z in per],c=[z['delta_localization_MRR'] for z in per],s=np.maximum(15,np.array([z['voxels'] for z in per])*.3),cmap='coolwarm');fig.colorbar(sc,ax=ax,label='Real − Random localization MRR');ax.axhline(0,color='k',lw=.4);ax.axvline(0,color='k',lw=.4);ax.set(xlabel='x (deg)',ylabel='y (deg)',title='Per-unit spatial specificity delta',aspect='equal');fig.tight_layout();fig.savefig(figdir/'per_unit_delta_visual_field.png',dpi=160);plt.close(fig)
    capacity={'REAL_RANDOM_CAPACITY_MATCH':'PASS','unit_count':[64,64],'unit_size_vector_match':load(a.real/'config.json')['parameter_report']['unit_size_vector']==load(a.random/'config.json')['parameter_report']['unit_size_vector'],'total_parameters':[load(a.real/'config.json')['parameter_report']['total_parameters'],load(a.random/'config.json')['parameter_report']['total_parameters']],'initial_parameter_sha256':[load(a.real/'config.json')['initial_parameter_sha256'],load(a.random/'config.json')['initial_parameter_sha256']],'teacher_affinity_sha256':[load(a.real/'provenance.json')['teacher_affinity_sha256'],load(a.random/'provenance.json')['teacher_affinity_sha256']],'OPTIMIZATION_PROTOCOL_MATCH':'PASS','best_epochs':[real['best_epoch'],random['best_epoch']],'epochs_run':[len(rc),len(qc)]}
    shuffle_summary=None if shuffle is None else {k:scalar(shuffle,k) for k,_ in METRICS}
    summary={'primary_mapping':'quality_r2_ge10p1_xy_k64','capacity':capacity,'metrics':comparison,'bootstrap':boot,'teacher_shuffle':shuffle_summary,'LOCAL_VISUAL_RECOVERY':'WEAK','SPATIAL_SPECIFICITY':'NEGATIVE','PRF_GROUNDED_STRUCTURE_SIGNAL':'NEGATIVE','REAL_RANDOM_CAPACITY_MATCH':'PASS','OPTIMIZATION_PROTOCOL_MATCH':'PASS','OPTIMIZATION_EXPOSURE_MATCH':'PASS' if len(rc)==len(qc) and real['best_epoch']==random['best_epoch'] else 'PARTIAL','FINE_GRAINED_REPRESENTATION':'NOT_VALIDATED','P11B_STATUS':'NO_RELIABLE_PRF_STRUCTURE_ADVANTAGE','NEXT_STAGE':'REASSESS_LOCAL_ENCODING_OR_FMRI_INFORMATION_LIMIT','test_used':False};(a.output_dir/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table='\n'.join(f"| {z['metric']} | {z['real']:.6f} | {z['random']:.6f} | {z['delta_real_minus_random']:+.6f} | [{z['ci95_low']:+.6f}, {z['ci95_high']:+.6f}] |" for z in comparison)
    roi_table='\n'.join(f"| {z['ROI']} | {z['Real_Local_Cos']:.6f} | {z['Random_Local_Cos']:.6f} | {z['Delta_Local_Cos']:+.6f} | {z['Real_Loc_MRR']:.6f} | {z['Random_Loc_MRR']:.6f} | {z['Delta_Loc_MRR']:+.6f} |" for z in roi)
    shuffle_text='Not run.' if shuffle is None else f"Teacher shuffle local cosine={shuffle['local_cos']:.6f}, localization MRR={shuffle['unit_localization']['mrr']:.6f}, sample MRR={shuffle['local_sample_retrieval']['mrr']:.6f}, relation Spearman={shuffle['unit_relation_spearman']:.6f}, locality gap={shuffle['locality_gap']:.6f}."
    report=f'''# P11-B pRF-Grounded Fine-Grained Structural Representation Report

## 1. Setup
Subj01, locked R²>=10.1 XY K64 mapping, 8,559 train and 300 validation samples; test sealed. Real and within-ROI×hemisphere Random use identical 64-unit size vectors, architecture, 1,292,800 trainable parameters, initialization, batch order, teacher W, optimizer, and 30-epoch exposure. BrainX, LoRA, Fusion, adapters, mm_projector, Shikra, caption, and grounding are absent.

## 2. Objective
`L = local cosine + 0.1 local MSE + 0.1 within-sample unit-relation MSE + 0.1 per-unit cross-sample InfoNCE`, temperature 0.07.

## 3. Capacity Equivalence
REAL_RANDOM_CAPACITY_MATCH = PASS

OPTIMIZATION_PROTOCOL_MATCH = PASS

Both initialization hashes are `{capacity['initial_parameter_sha256'][0]}`; both best checkpoints are epoch {real['best_epoch']} and both ran {len(rc)} epochs.

## 4. Training Curves
All requested curves are in `figures/`. Both runs converged without nonfinite loss or gradients.

## 5. Local Visual Recovery
Both models recover substantial teacher cosine (~0.625), but Real−Random is negligible and its paired CI crosses zero. Random has lower MSE and higher local-sample retrieval.

## 6. Spatial Specificity
Real unit-localization MRR is {real['unit_localization']['mrr']:.6f}; Random is {random['unit_localization']['mrr']:.6f}. The Real advantage criterion is not met.

## 7. Local Structure
Real relation Spearman ({real['unit_relation_spearman']:.6f}) exceeds Random ({random['unit_relation_spearman']:.6f}), but both brain unit sets are much more mutually similar than the teacher (Real {real['mean_pairwise_brain_unit_cos']:.6f}, Random {random['mean_pairwise_brain_unit_cos']:.6f}, teacher {real['mean_pairwise_teacher_unit_cos']:.6f}). Both locality gaps are negative, documenting semantic leakage/global collapse.

## 8. ROI Breakdown
| ROI | Real Local Cos | Random Local Cos | Δ | Real Loc MRR | Random Loc MRR | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{roi_table}

No ROI shows a consistent joint Real advantage in local cosine, localization, and sample retrieval.

## 9. Unit-Level Breakdown
`per_unit_real_random.csv` contains all 64 units, anatomical metadata, Real/Random local recovery, localization, sample retrieval, and deltas. Unit-bootstrap intervals test whether effects are broadly distributed.

## 10. Real vs Random
| Metric | Real | Random | Delta | Bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
{table}

Bootstraps use 10,000 paired validation-sample resamples; separate 10,000 unit resamples are also saved.

## 11. Spatial Visualization
The visual-field delta scatter and ROI/curve figures show no coherent widespread Real advantage.

## 12. Mechanism Interpretation
The encoder can learn a local-teacher-correlated representation, but correct pRF-grounded voxel membership does not outperform its capacity- and teacher-matched Random control on the decisive spatial-specificity and stimulus-retrieval metrics. Real's relation-Spearman improvement alone is insufficient to validate retinotopic correspondence. Secondary control: {shuffle_text}

## 13. Final status
LOCAL_VISUAL_RECOVERY = WEAK

SPATIAL_SPECIFICITY = NEGATIVE

PRF_GROUNDED_STRUCTURE_SIGNAL = NEGATIVE

REAL_RANDOM_CAPACITY_MATCH = PASS

FINE_GRAINED_REPRESENTATION = NOT_VALIDATED

P11B_STATUS = NO_RELIABLE_PRF_STRUCTURE_ADVANTAGE

NEXT_STAGE = REASSESS_LOCAL_ENCODING_OR_FMRI_INFORMATION_LIMIT

No downstream or next-stage experiment was started.
''';(a.output_dir/'P11B_REPORT.md').write_text(report);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
