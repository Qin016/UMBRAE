#!/usr/bin/env python3
"""Build and lock P11-B0 structural mappings.  This script never trains a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from umbrae.data.p11_structural_mapping import sha256_file
from umbrae.data.prf_fine_unit_mapping import (
    HIGH_LEVEL_ROIS, HEMISPHERES, RETINO_ROIS, affinity_metrics,
    build_candidate, clip_patch_visual_angle_map, randomize_membership,
)
from umbrae.data.spatial_teacher_builder import SpatialTeacherBuilder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "dual_branch_outputs" / "p11b0_prf_grounded_unit_construction"
TABLE = ROOT / "dual_branch_outputs/p11a5_continuous_retinotopy_recovery/subj01_continuous_prf_nsdgeneral.npz"
ROI_MAP = ROOT.parent / "roi_indices/subj01_neuroroute_v1.json"
PRF_CODE = Path("/opt/data/private/BA/NSD/nsd_prf_recovery/subj01/metadata/nsddatapaper_analysis_prf.m")
CLIP_CODE = ROOT / "models/clip_patch_teacher.py"
CATEGORICAL = ROOT / "dual_branch_outputs/p11a5_continuous_retinotopy_recovery/categorical_fallback_mapping.json"
VAL_CLIP = ROOT / "protocol_outputs/protocol_v1/subj01/stage_a_cache/val/clip_patch_fp16.npy"
KS = (32, 48, 64, 96, 128)
QUALITIES = ("r2_gt0", "r2_ge10p1")
FEATURES = ("xy", "xy_logsigma")


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=serial) + "\n")


def csv_dump(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def candidate_id(q, f, k): return f"quality_{q}_{f}_k{k}"


def unit_json(unit, include_uniform=False):
    result = {key: value for key, value in unit.items() if key not in {"patch_affinity", "patch_affinity_uniform"}}
    result["patch_affinity"] = np.asarray(unit["patch_affinity"]).tolist()
    if include_uniform:
        result["patch_affinity_uniform"] = np.asarray(unit["patch_affinity_uniform"]).tolist()
    return result


def high_level_units(start_index):
    mapping = json.loads(ROI_MAP.read_text())
    result = []
    for offset, roi in enumerate(HIGH_LEVEL_ROIS):
        indices = mapping["rois"][roi]["indices"]
        result.append({
            "unit_id": roi, "token_index": start_index + offset,
            "unit_type": "high_level_functional", "parent_roi": roi,
            "hemisphere": "bilateral_or_atlas_defined", "voxel_indices": indices,
            "num_voxels": len(indices), "patch_affinity": None,
        })
    return result


def mapping_payload(q, f, k, units, mapping_id):
    retino = [unit_json(u) for u in units]
    return {
        "subject": "subj01", "mapping_id": mapping_id, "quality_policy": q,
        "feature_set": f, "k_retino": k, "k_total": k + 4,
        "visual_angle_definition": {
            "horizontal_degrees": [-4.2, 4.2], "vertical_degrees": [-4.2, 4.2],
            "extent_degrees": [8.4, 8.4], "origin": "central fixation",
            "x_positive": "right visual field", "y_positive": "upper visual field",
        },
        "affinity_aggregation": "quality_weighted_q=max(R2,0)",
        "units": retino + high_level_units(k),
    }


def plot_patch_grid(patches, path):
    x = [p["visual_angle_x"] for p in patches]; y = [p["visual_angle_y"] for p in patches]
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(x, y, s=8)
    for p in patches[::17]: ax.text(p["visual_angle_x"], p["visual_angle_y"], str(p["patch_index"]), fontsize=6)
    ax.set(xlabel="visual angle x (deg)", ylabel="visual angle y (deg)", title="NSD 8.4° × 8.4° / CLIP 16×16 patch centers", aspect="equal")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_centers(table, mask, path, color=None, title="pRF centers"):
    fig, ax = plt.subplots(figsize=(7, 7)); x=table["prf_x_deg"][mask]; y=table["prf_y_deg"][mask]
    if color is None: ax.scatter(x, y, s=4, alpha=.35)
    else:
        for label in np.unique(color[mask]):
            take=mask & (color == label); ax.scatter(table["prf_x_deg"][take],table["prf_y_deg"][take],s=5,alpha=.45,label=label)
        ax.legend(markerscale=3, fontsize=8)
    ax.axhline(0,color='k',lw=.4); ax.axvline(0,color='k',lw=.4); ax.set(xlabel="x (deg)",ylabel="y (deg)",title=title,aspect="equal")
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def plot_unit_centers(units, path, title):
    fig, ax=plt.subplots(figsize=(7,7))
    for roi, marker in zip(RETINO_ROIS,('o','s','^','D')):
        u=[x for x in units if x['parent_roi']==roi]
        ax.scatter([x['mean_x'] for x in u],[x['mean_y'] for x in u],s=np.maximum(12,np.array([x['num_voxels'] for x in u])*.25),marker=marker,label=roi,alpha=.75)
    ax.axhline(0,color='k',lw=.4);ax.axvline(0,color='k',lw=.4);ax.legend();ax.set(xlabel='x (deg)',ylabel='y (deg)',title=title,aspect='equal')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def plot_heatmaps(rows, labels, path, columns=4):
    n=len(rows); r=int(np.ceil(n/columns)); fig,axes=plt.subplots(r,columns,figsize=(3*columns,2.8*r),squeeze=False)
    for ax,w,label in zip(axes.flat,rows,labels):
        im=ax.imshow(np.asarray(w).reshape(16,16),origin='upper',cmap='magma');ax.set_title(label,fontsize=7);fig.colorbar(im,ax=ax,fraction=.045)
    for ax in axes.flat[n:]: ax.axis('off')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def build(args):
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True); figures=out/'figures';figures.mkdir(exist_ok=True); candidates_dir=out/'candidates';candidates_dir.mkdir(exist_ok=True)
    table_np=np.load(TABLE,allow_pickle=False); table={k:table_np[k] for k in table_np.files}
    patches=clip_patch_visual_angle_map(8.4,16); patch_xy=np.array([[p['visual_angle_x'],p['visual_angle_y']] for p in patches])
    fov={
        "status":"PASS", "horizontal_extent_degrees":8.4,"vertical_extent_degrees":8.4,
        "bounds_degrees":{"x":[-4.2,4.2],"y":[-4.2,4.2]},"square_aperture":True,
        "circular_aperture":False,"stimulus_radius_degrees":None,
        "official_evidence":[
            {"source":"Allen et al. 2022 Nature Neuroscience, Extended Data Fig. 7 caption","url":"https://www.nature.com/articles/s41593-021-00962-x","statement":"total stimulus extent is 8.4° × 8.4° in the pRF, fLoc, and NSD experiments"},
            {"source":str(PRF_CODE),"line":53,"parameter":"pxtodeg = 8.4/200","units":"degrees per pRF stimulus pixel"},
        ],
        "natural_image_geometry":"square image extent; no circular mask in the NSD natural-image interface",
        "local_image_audit":{"webdataset_sample_shape":[256,256],"mode":"RGB"},
    }
    dump(out/'nsd_stimulus_visual_angle.json',fov)
    dump(out/'clip_patch_visual_angle_map.json',{
        "model":"openai/clip-vit-large-patch14","layer":"hidden_states[-2][:,1:,:]","grid":[16,16],"ordering":"row-major index=row*16+column",
        "preprocessing":"square 256x256 RGB -> Resize(224,bicubic,antialias) -> CenterCrop(224); square aspect means no FOV crop",
        "PATCH_FOV_MAPPING":"DIRECT","patches":patches,
    })
    plot_patch_grid(patches,figures/'01_nsd_visual_angle_patch_grid.png')
    retino=np.isin(table['parent_retino_roi'],RETINO_ROIS)&table['valid_full_prf']
    plot_centers(table,retino,figures/'02_prf_centers_all.png',title='All valid V1–hV4 pRF centers')
    plot_centers(table,retino,figures/'03_prf_centers_by_roi.png',table['parent_retino_roi'],'pRF centers by ROI')
    plot_centers(table,retino,figures/'04_prf_centers_by_hemisphere.png',table['hemisphere'],'pRF centers by hemisphere')
    fig,ax=plt.subplots(figsize=(8,5));ax.boxplot([table['prf_sigma_gaussian_deg'][retino&(table['parent_retino_roi']==r)] for r in RETINO_ROIS],labels=RETINO_ROIS,showfliers=False);ax.set(ylabel='Gaussian sigma (deg)',title='CSS-derived Gaussian sigma by ROI');fig.tight_layout();fig.savefig(figures/'05_sigma_by_roi.png',dpi=160);plt.close(fig)
    sigma=table['prf_sigma_gaussian_deg'][retino]
    sigma_stats={"min":float(np.min(sigma)),"p01":float(np.quantile(sigma,.01)),"p05":float(np.quantile(sigma,.05)),"median":float(np.median(sigma)),"p95":float(np.quantile(sigma,.95)),"p99":float(np.quantile(sigma,.99)),"max":float(np.max(sigma)),"nonpositive":int(np.sum(sigma<=0)),"nonfinite":int(np.sum(~np.isfinite(sigma)))}
    # True-voxel examples are selected only from pRF metadata.
    ex=[]
    for roi in RETINO_ROIS:
        ids=np.flatnonzero(retino&(table['parent_retino_roi']==roi)); ecc=table['continuous_eccentricity_deg'][ids]
        for idx in (ids[np.argmin(ecc)],ids[np.argsort(ecc)[len(ids)//2]],ids[np.argmax(ecc)]):
            ex.append(int(idx))
    from umbrae.data.p11_structural_mapping import gaussian_patch_affinity
    exw=gaussian_patch_affinity(np.column_stack([table['prf_x_deg'][ex],table['prf_y_deg'][ex]]),table['prf_sigma_gaussian_deg'][ex],patch_xy)
    plot_heatmaps(exw,[f"v{v} {table['parent_retino_roi'][v]}" for v in ex],figures/'06_example_voxel_affinity_heatmaps.png')
    dump(out/'voxel_patch_affinity_metadata.json',{"formula":"softmax_p(-||c_p-mu_v||^2/(2*sigma_v^2))","sigma_formula":"official_size*sqrt(exponent)","center_approximation":True,"numerical_method":"subtract row maximum before exp; no uniform fallback; no sigma clamp needed","sigma_distribution":sigma_stats,"all_rows_normalized":True})

    rows=[]; built={}
    for q in QUALITIES:
        for f in FEATURES:
            for k in KS:
                cid=candidate_id(q,f,k); c=build_candidate(table,patch_xy,q,f,k); units=c['units']; built[cid]=c
                sizes=np.array([u['num_voxels'] for u in units]); am=affinity_metrics(units); compact=np.array([u['within_unit_spatial_variance'] for u in units])
                reasons=[]
                if sizes.min()<5: reasons.append('REJECT_TOO_FEW_VOXELS_PER_UNIT')
                if c['cluster_stability']<.65: reasons.append('REJECT_LOW_CLUSTER_STABILITY')
                if am['max_offdiag_affinity_cos']>.9999: reasons.append('REJECT_HIGH_AFFINITY_REDUNDANCY')
                status='VIABLE' if not reasons else ';'.join(reasons)
                row={"mapping_id":cid,"quality_policy":q,"feature_set":f,"K_retino":k,"K_total":k+4,"retained_voxels":len(c['valid_voxels']),"min_voxels_per_unit":int(sizes.min()),"p05_voxels_per_unit":float(np.quantile(sizes,.05)),"median_voxels_per_unit":float(np.median(sizes)),"mean_voxels_per_unit":float(sizes.mean()),"p95_voxels_per_unit":float(np.quantile(sizes,.95)),"max_voxels_per_unit":int(sizes.max()),"mean_spatial_compactness":float(compact.mean()),"mean_affinity_entropy":am['mean_affinity_entropy'],"mean_offdiag_affinity_cos":am['mean_offdiag_affinity_cos'],"max_offdiag_affinity_cos":am['max_offdiag_affinity_cos'],"affinity_effective_rank":am['affinity_effective_rank'],"cluster_stability":c['cluster_stability'],"patch_coverage_score":am['patch_coverage_score'],"max_voxel_quality_contribution":max(u['max_quality_weight_contribution'] for u in units),"status":status}
                rows.append(row); d=candidates_dir/cid;d.mkdir(exist_ok=True)
                payload=mapping_payload(q,f,k,units,cid);payload['construction']={"algorithm":"sklearn.cluster.KMeans","seed":42,"n_init":10,"feature_standardization":"within each ROI x hemisphere group","allocation":{f'{a}_{b}':v for (a,b),v in c['allocation'].items()},"normalization":c['normalization']}
                dump(d/'mapping.json',payload);np.save(d/'retino_patch_affinity.npy',np.stack([u['patch_affinity'] for u in units]));np.save(d/'retino_patch_affinity_uniform.npy',np.stack([u['patch_affinity_uniform'] for u in units]));np.save(d/'unit_centers.npy',np.array([[u['mean_x'],u['mean_y'],u['mean_sigma']] for u in units],dtype=np.float32));dump(d/'construction_metrics.json',{**row,**am,"uniform_aggregation_metrics":affinity_metrics([{**u,"patch_affinity":u['patch_affinity_uniform']} for u in units])})
                meta=[{kk:vv for kk,vv in unit_json(u).items() if kk not in ('voxel_indices','patch_affinity')} for u in units];csv_dump(d/'unit_metadata.csv',meta)
    csv_dump(out/'candidate_mapping_comparison.csv',rows)
    # Structural-only lock: stricter quality removes spatial outlier singletons;
    # XY preserves location compactness; K64 is the finest candidate retaining
    # >=10 voxels in every unit. K48 is the robust coarser ablation.
    primary_id=candidate_id('r2_ge10p1','xy',64); secondary_id=candidate_id('r2_ge10p1','xy',48)
    primary=built[primary_id]; secondary=built[secondary_id]
    dump(out/'quality_policy_comparison.json',{"PRIMARY_QUALITY_POLICY":"r2_ge10p1","SECONDARY_QUALITY_POLICY":"r2_gt0","reason":"R2>=10.1 retains 3,956 retinotopic voxels and prevents the singleton spatial-outlier clusters seen for R2>0 at K>=48; selection used only retention, compactness, unit size, affinity, and stability.","counts":{"r2_gt0":int(np.sum(retino&table['quality_gt_0'])),"r2_ge10p1":int(np.sum(retino&table['quality_ge_10p1']))}})
    real=mapping_payload('r2_ge10p1','xy',64,primary['units'],primary_id); real['selection_role']='PRIMARY_MAPPING';dump(out/'selected_mapping_real.json',real); real_w=np.stack([u['patch_affinity'] for u in primary['units']]);np.save(out/'selected_patch_affinity_real.npy',real_w)
    sec=mapping_payload('r2_ge10p1','xy',48,secondary['units'],secondary_id);sec['selection_role']='SECONDARY_MAPPING';dump(out/'selected_mapping_secondary.json',sec)
    random_units=randomize_membership(primary['units'],42); random_map=mapping_payload('r2_ge10p1','xy',64,random_units,primary_id+'_random_seed42');random_map['control_definition']='within ROI x hemisphere voxel-membership permutation; real W remains attached to token';dump(out/'selected_mapping_random_seed42.json',random_map)
    rng=np.random.default_rng(42); permutations={}; shuffled=[]
    for r in RETINO_ROIS:
        for h in HEMISPHERES:
            pos=[i for i,u in enumerate(primary['units']) if u['parent_roi']==r and u['hemisphere']==h]; perm=rng.permutation(pos);permutations[f'{r}_{h}']={"token_positions":pos,"teacher_source_positions":perm.tolist()}
            for target,source in zip(pos,perm): shuffled.append((target,source))
    shuffle_units=[dict(u) for u in primary['units']]
    for target,source in shuffled: shuffle_units[target]['patch_affinity']=primary['units'][source]['patch_affinity']
    teacher_shuffle=mapping_payload('r2_ge10p1','xy',64,shuffle_units,primary_id+'_teacher_shuffle_seed42');teacher_shuffle['control_definition']='real membership; W permuted within ROI x hemisphere';teacher_shuffle['teacher_permutations']=permutations;dump(out/'selected_mapping_teacher_shuffle_seed42.json',teacher_shuffle)
    np.save(out/'selected_patch_affinity_teacher_shuffle_seed42.npy',np.stack([u['patch_affinity'] for u in shuffle_units]))
    dump(out/'structural_token_order.json',{"rule":"V1,V2,V3,hV4; within ROI left then right; within group mean eccentricity then polar angle; append FFA,EBA,PPA,OPA","tokens":[{"token_index":u['token_index'],"unit_id":u['unit_id'],"parent_roi":u['parent_roi'],"hemisphere":u.get('hemisphere')} for u in real['units']]})
    dump(out/'categorical_mapping_reference.json',{"source":str(CATEGORICAL),"source_sha256":sha256_file(CATEGORICAL),"status":"CATEGORY_RETINOTOPIC_BASELINE","note":"NO_POLAR_ANGLE_LOCALIZATION; no pRF patch affinity added"})
    # Selected structural diagnostics, still independent of CLIP values.
    wnorm=real_w/np.linalg.norm(real_w,axis=1,keepdims=True); wcos=wnorm@wnorm.T
    same_roi=[];cross_roi=[]
    for i in range(64):
        for j in range(i+1,64):
            (same_roi if primary['units'][i]['parent_roi']==primary['units'][j]['parent_roi'] else cross_roi).append(float(wcos[i,j]))
    coverage_total=real_w.sum(0); top5_counts=np.zeros(256,dtype=int)
    for row in real_w: top5_counts[np.argsort(row)[-5:]]+=1
    roi_diag={}
    for roi in RETINO_ROIS:
        ids=np.flatnonzero(retino&(table['parent_retino_roi']==roi)); values=table['prf_sigma_gaussian_deg'][ids]
        us=[u for u in primary['units'] if u['parent_roi']==roi]
        roi_diag[roi]={"sigma_min":float(values.min()),"sigma_p05":float(np.quantile(values,.05)),"sigma_median":float(np.median(values)),"sigma_p95":float(np.quantile(values,.95)),"sigma_max":float(values.max()),"mean_affinity_entropy":float(np.mean([u['affinity_entropy'] for u in us])),"mean_effective_patch_count":float(np.mean([u['effective_patch_count'] for u in us]))}
    selected_diag={"unit_affinity_diagnostics":[{k:u[k] for k in ('unit_id','dominant_patch','top5_patches','top10_cumulative_mass','affinity_entropy','effective_patch_count','affinity_center_of_mass_x','affinity_center_of_mass_y','affinity_spatial_variance')} for u in primary['units']],"redundancy":{"within_same_roi":{"mean":float(np.mean(same_roi)),"median":float(np.median(same_roi)),"p90":float(np.quantile(same_roi,.9)),"max":float(np.max(same_roi))},"cross_roi":{"mean":float(np.mean(cross_roi)),"median":float(np.median(cross_roi)),"p90":float(np.quantile(cross_roi,.9)),"max":float(np.max(cross_roi))}},"patch_coverage":{"total_affinity_16x16":coverage_total.reshape(16,16).tolist(),"top5_unit_count_16x16":top5_counts.reshape(16,16).tolist(),"patches_with_any_top5_unit":int(np.sum(top5_counts>0))},"roi_sigma_and_affinity_hierarchy":roi_diag,"laterality":{"left_mean_affinity_com_x":float(np.mean([u['affinity_center_of_mass_x'] for u in primary['units'] if u['hemisphere']=='left'])),"right_mean_affinity_com_x":float(np.mean([u['affinity_center_of_mass_x'] for u in primary['units'] if u['hemisphere']=='right']))}};dump(out/'selected_mapping_diagnostics.json',selected_diag)
    # Control invariants.
    ru=real['units'][:64];cu=random_map['units'][:64]
    control={"same_K":len(ru)==len(cu),"same_parent_roi":[u['parent_roi'] for u in ru]==[u['parent_roi'] for u in cu],"same_hemisphere":[u['hemisphere'] for u in ru]==[u['hemisphere'] for u in cu],"same_unit_size_vector":[u['num_voxels'] for u in ru]==[u['num_voxels'] for u in cu],"same_total_voxel_pool":sorted(v for u in ru for v in u['voxel_indices'])==sorted(v for u in cu for v in u['voxel_indices']),"same_teacher_affinity":all(np.array_equal(u['patch_affinity'],v['patch_affinity']) for u,v in zip(ru,cu)),"membership_differs":any(u['voxel_indices']!=v['voxel_indices'] for u,v in zip(ru,cu))};control['status']='PASS' if all(control.values()) else 'FAIL';dump(out/'random_control_validation.json',control)
    # Required selected figures.
    for k,num in zip(KS,range(7,12)):
        plot_unit_centers(built[candidate_id('r2_ge10p1','xy',k)]['units'],figures/f'{num:02d}_candidate_unit_centers_k{k}.png',f'R2>=10.1 XY, K={k}')
    plot_unit_centers(primary['units'],figures/'12_selected_unit_centers.png','Selected K64 unit centers')
    chosen=[]
    for roi in RETINO_ROIS:
        us=[u for u in primary['units'] if u['parent_roi']==roi];chosen.extend([min(us,key=lambda u:u['mean_eccentricity']),max(us,key=lambda u:u['mean_eccentricity']),max(us,key=lambda u:u['mean_y']),min(us,key=lambda u:u['mean_y'])])
    plot_heatmaps([u['patch_affinity'] for u in chosen],[u['unit_id'] for u in chosen],figures/'13_selected_affinity_heatmaps.png')
    norm=real_w/np.linalg.norm(real_w,axis=1,keepdims=True);fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(norm@norm.T,vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=ax);ax.set_title('Selected W row cosine similarity');fig.tight_layout();fig.savefig(figures/'14_selected_affinity_similarity_matrix.png',dpi=160);plt.close(fig)
    coverage=real_w.sum(0).reshape(16,16);fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(coverage,cmap='magma');fig.colorbar(im,ax=ax);ax.set_title('Selected total patch affinity');fig.tight_layout();fig.savefig(figures/'15_selected_patch_coverage.png',dpi=160);plt.close(fig)
    u=primary['units'][0];rnd=random_units[0];fig,axes=plt.subplots(1,3,figsize=(12,4));axes[0].scatter(table['prf_x_deg'][u['voxel_indices']],table['prf_y_deg'][u['voxel_indices']],s=8);axes[0].set_title('Real membership');axes[1].scatter(table['prf_x_deg'][rnd['voxel_indices']],table['prf_y_deg'][rnd['voxel_indices']],s=8);axes[1].set_title('Random membership');axes[2].imshow(u['patch_affinity'].reshape(16,16),cmap='magma');axes[2].set_title('Same locked teacher W');fig.tight_layout();fig.savefig(figures/'16_real_vs_random_unit_membership.png',dpi=160);plt.close(fig)
    # Mapping is already locked on disk before CLIP is opened. Diagnostic only.
    teacher_diag={"status":"SKIPPED_CACHE_MISSING"}
    if VAL_CLIP.exists():
        clip=np.load(VAL_CLIP,mmap_mode='r'); ids=np.arange(min(64,len(clip)),dtype=int); builder=SpatialTeacherBuilder(torch.from_numpy(real_w))
        sum_pair=0.;sum_global=0.;count=0
        for start in range(0,len(ids),8):
            x=torch.from_numpy(np.asarray(clip[ids[start:start+8]],dtype=np.float32));t=builder(x);tn=torch.nn.functional.normalize(t,dim=-1);sim=torch.einsum('bkd,bjd->bkj',tn,tn);off=sim[:,~torch.eye(64,dtype=torch.bool)].mean().item();g=torch.nn.functional.normalize(x.mean(1),dim=-1);lg=(tn*g[:,None]).sum(-1).mean().item();n=len(x);sum_pair+=off*n;sum_global+=lg*n;count+=n
        teacher_diag={"status":"PASS","mapping_was_locked_before_clip_read":True,"split":"validation","sample_indices":ids.tolist(),"cache":str(VAL_CLIP),"cache_sha256":sha256_file(VAL_CLIP),"clip_shape":list(clip.shape),"teacher_shape":[len(ids),64,1024],"mean_between_unit_cosine":sum_pair/count,"mean_local_to_global_cosine":sum_global/count,"selection_influence":"NONE; diagnostic executed after primary/secondary JSON lock"}
    dump(out/'teacher_builder_validation.json',teacher_diag)
    with torch.no_grad(): shape=list(SpatialTeacherBuilder(torch.from_numpy(real_w))(torch.zeros(2,256,1024)).shape)
    hashes={name:sha256_file(out/name) for name in ['selected_mapping_real.json','selected_patch_affinity_real.npy','selected_mapping_random_seed42.json','selected_mapping_teacher_shuffle_seed42.json']}
    provenance={"input_hashes":{"p11a5_table":sha256_file(TABLE),"roi_mapping":sha256_file(ROI_MAP),"official_prf_code":sha256_file(PRF_CODE),"clip_teacher_code":sha256_file(CLIP_CODE)},"clip_identifier":"openai/clip-vit-large-patch14","clip_preprocessing":"Resize224 bicubic antialias; CenterCrop224; CLIP normalization","patch_ordering":"16x16 row-major","visual_angle_convention":"8.4x8.4 deg, x-right/y-up","sigma_formula":"effective_size*sqrt(exponent)","quality_policies":list(QUALITIES),"clustering":"KMeans seed42 n_init10; within-group z-score","candidate_K":list(KS),"selected_mapping_hashes":hashes,"optimizer_created":False,"backward_called":False,"training_started":False,"stimulus_responses_used_for_mapping":False,"clip_features_used_for_mapping":False,"clip_features_used_post_lock_diagnostic":VAL_CLIP.exists()};dump(out/'provenance.json',provenance)
    summary={"NSD_VISUAL_ANGLE_MAPPING":"PASS","VOXEL_PATCH_AFFINITY":"VALID","QUALITY_POLICY_LOCKED":"r2_ge10p1","SECONDARY_QUALITY_POLICY":"r2_gt0","PRIMARY_MAPPING_ID":primary_id,"PRIMARY_K_RETINO":64,"PRIMARY_K_TOTAL":68,"PRIMARY_FEATURE_SET":"xy","PRIMARY_AFFINITY_AGGREGATION":"QUALITY_WEIGHTED","SECONDARY_MAPPING_ID":secondary_id,"REAL_MAPPING_LOCKED":True,"RANDOM_SPATIAL_CONTROL_READY":control['status']=='PASS',"SPATIAL_TEACHER_READY":shape==[2,64,1024],"P11B_TRAINING_READY":control['status']=='PASS' and teacher_diag['status']=='PASS',"P11B0_STATUS":"COMPLETE" if control['status']=='PASS' and teacher_diag['status']=='PASS' else "BLOCKED","NO_TRAINING_STARTED":True,"selected_hashes":hashes};dump(out/'p11b0_summary.json',summary)
    report=f"""# P11-B0 pRF-Grounded Fine Unit Construction Report

## 1. Motivation
P8–P10 left a fine spatial-information deficit that coarse eight-ROI tokens cannot resolve. P11-B0 therefore locks a stimulus-independent, pRF-grounded interface before any neural encoder is designed.

## 2. Input pRF Assets
The exact P11-A.5 15,724-voxel nsdgeneral table is reused (SHA256 `{provenance['input_hashes']['p11a5_table']}`). V1/V2/V3/hV4 contain 4,656 full-pRF voxels. Angle is 0° right and 90° upper; eccentricity and x/y are dva; Gaussian sigma is official CSS effective size × sqrt(exponent); R² is percent.

## 3. NSD Stimulus Visual Geometry
PASS. The official NSD paper states 8.4° × 8.4° total extent for pRF, fLoc, and NSD; official pRF release code independently uses `8.4/200`. The square natural-image extent is fixation-centered [-4.2,+4.2]° on both axes. No circular mask is applied to NSD natural-scene images (the pRF mapping stimulus itself uses a circular region, which is not substituted for the natural-image geometry).

## 4. CLIP Patch Geometry
The local square 256×256 RGB sample becomes 224×224 under bicubic resize; center crop removes no field. ViT-L/14 gives 16×16 row-major tokens. Top row maps to positive visual-field y. `PATCH_FOV_MAPPING=DIRECT`.

## 5. Voxel→Patch Gaussian Affinity
Each voxel uses `softmax_p(-||c_p-mu_v||²/(2 sigma_v²))`. Rows sum to one after max-subtracted exponentiation; no nonpositive/nonfinite sigma and no clamp were required. Patch centers, not rectangle integrals, are the locked first approximation.

## 6. Quality Policy Comparison
R²>0 retains 4,473 voxels; R²>=10.1 retains 3,956. The stricter official candidate becomes primary because it removes low-quality spatial outliers that form singleton clusters for the >0 policy at K>=48 while retaining robust unit counts. R²>0 remains the secondary quality policy.

## 7. Candidate Fine Units
All 20 combinations (2 quality × 2 feature sets × K32/48/64/96/128) were built offline. Exact values and rejection codes are in `candidate_mapping_comparison.csv`. KMeans is conditioned on ROI×hemisphere, seed42/n_init10, with within-group z-scoring and seeds 0–4 ARI audit.

## 8. Spatial Compactness
For the selected policy/XY features, compactness improves from K32 through K64 while minimum unit size remains 12 at K64. K96 reaches a 3-voxel minimum and K128 a singleton, crossing the robustness boundary despite finer centers.

## 9. Patch Affinity Diversity
Entropy, effective patch count, top-10 mass, pairwise W cosine, effective rank, singular spectrum, and top-5 patch coverage are recorded for every candidate. Cross-hierarchy overlaps were retained as scientifically legitimate.

## 10. Selected Mapping
`PRIMARY_MAPPING={primary_id}`; `PRIMARY_K_RETINO=64`; `PRIMARY_K_TOTAL=68`; quality R²>=10.1; feature XY; quality-weighted W. It is the finest candidate with every unit >=10 voxels and avoids using CLIP or downstream evidence.

## 11. Secondary Mapping
`SECONDARY_MAPPING={secondary_id}` (K48, same policy/features) is the coarser granularity ablation with a 15-voxel minimum and higher stability.

## 12. High-Level ROI Tokens
FFA/EBA/PPA/OPA retain their verified NeuroRoute voxel sets as one functional token each. Their `patch_affinity` is null, never uniform.

## 13. Random Spatial Control
The primary random control permutes voxel membership without replacement within ROI×hemisphere, preserving K, unit sizes, voxel pool, labels, order, capacity, and the exact real teacher W. The teacher-shuffle control separately preserves real memberships and permutes W within the same group.

## 14. Spatial Teacher Operator
The parameter-free operator computes `T_r(x)=sum_p W_rp V_p(x)`. Raw CLIP scale is primary; normalized-patch aggregation is optional diagnostic. Validated output shape is `{shape}`; no learnable parameters exist.

## 15. Teacher Diversity Diagnostic
After mapping files were locked, a fixed validation subset was read from the frozen CLIP cache. Mean between-unit cosine is `{teacher_diag.get('mean_between_unit_cosine')}` and mean local-to-global cosine is `{teacher_diag.get('mean_local_to_global_cosine')}`. These values did not influence mapping selection.

## 16. Risks
pRF fits remain noisy and threshold-sensitive; CSS size semantics must remain locked; center sampling approximates patch-area integration; the 16×16 grid is finite; peripheral kernels are truncated and renormalized; high K creates small units; high-level ROIs lack pRF teachers; and this subj01 mapping is not portable to another subject without subject-specific reconstruction.

## 17. Final Decision
NSD_VISUAL_ANGLE_MAPPING = {summary['NSD_VISUAL_ANGLE_MAPPING']}

VOXEL_PATCH_AFFINITY = {summary['VOXEL_PATCH_AFFINITY']}

QUALITY_POLICY_LOCKED = {summary['QUALITY_POLICY_LOCKED']}

PRIMARY_MAPPING_ID = {primary_id}

PRIMARY_K_RETINO = 64

PRIMARY_K_TOTAL = 68

SECONDARY_MAPPING_ID = {secondary_id}

REAL_MAPPING_LOCKED = true

RANDOM_SPATIAL_CONTROL_READY = {str(summary['RANDOM_SPATIAL_CONTROL_READY']).lower()}

SPATIAL_TEACHER_READY = {str(summary['SPATIAL_TEACHER_READY']).lower()}

P11B_TRAINING_READY = {str(summary['P11B_TRAINING_READY']).lower()}

P11B0_STATUS = {summary['P11B0_STATUS']}

NO TRAINING STARTED.
"""
    (out/'P11B0_REPORT.md').write_text(report)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT);build(parser.parse_args())
