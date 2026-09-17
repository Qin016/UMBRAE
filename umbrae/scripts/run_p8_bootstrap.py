#!/usr/bin/env python
"""Paired caption and image-cluster grounding bootstrap for completed P7."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "BrainHub"))
from metrics import tokenize
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.rouge.rouge import Rouge
from models.p8_diagnostics import bleu4_from_components, box_iou_vector, paired_summary

MODES = ("umbrae", "lora", "full_real", "full_random")
PAIRS = {
    "lora_minus_umbrae": ("lora", "umbrae"),
    "full_real_minus_umbrae": ("full_real", "umbrae"),
    "full_random_minus_umbrae": ("full_random", "umbrae"),
    "full_real_minus_lora": ("full_real", "lora"),
    "full_real_minus_full_random": ("full_real", "full_random"),
}


def caption_components(root, mode):
    rows = json.loads((root / mode / "caption_predictions.json").read_text())
    refs = [row["references"] for row in rows]
    cands = [row["caption"] for row in rows]
    java = Path(sys.executable).resolve().parent / "java"
    if java.is_file(): os.environ["PATH"] = str(java.parent) + os.pathsep + os.environ.get("PATH", "")
    refs, cands = tokenize(refs, cands)
    bleu = BleuScorer(n=4); cider = CiderScorer(n=4); rouge = Rouge(); rouge_scores=[]
    for key in sorted(refs):
        hypothesis, references = cands[key][0], refs[key]
        bleu += (hypothesis, references); cider += (hypothesis, references)
        rouge_scores.append(rouge.calc_score([hypothesis], references))
    cider.compute_doc_freq(); cider_scores=np.asarray(cider.compute_cider())
    comps=[]
    for item in bleu.ctest:
        testlen=item["testlen"]; reflen=bleu._single_reflen(item["reflen"], "closest", testlen)
        comps.append([testlen,reflen,*item["guess"],*item["correct"]])
    return np.asarray(comps,dtype=np.float64), cider_scores, np.asarray(rouge_scores)


def run_caption(root, output, iterations, seed):
    components={}; cider={}; rouge={}
    for mode in MODES: components[mode],cider[mode],rouge[mode]=caption_components(root,mode)
    rng=np.random.default_rng(seed); draws=rng.integers(0,982,size=(iterations,982)); scores={mode:{k:np.empty(iterations) for k in ("CIDEr","BLEU-4","ROUGE-L")} for mode in MODES}
    for start in range(0,iterations,200):
        index=draws[start:start+200]
        for mode in MODES:
            scores[mode]["CIDEr"][start:start+len(index)]=cider[mode][index].mean(1)
            scores[mode]["ROUGE-L"][start:start+len(index)]=rouge[mode][index].mean(1)
            summed=components[mode][index].sum(1)
            scores[mode]["BLEU-4"][start:start+len(index)]=[bleu4_from_components(x[0],x[1],x[2:6],x[6:10]) for x in summed]
    result={"seed":seed,"iterations":iterations,"sample_count":982,"paired_indices":True,"CIDEr_IDF_policy":"fixed locked-test reference-corpus IDF; resampled per-sample CIDEr contributions","comparisons":{}}
    for name,(left,right) in PAIRS.items(): result["comparisons"][name]={metric:paired_summary(scores[left][metric]-scores[right][metric]) for metric in scores[left]}
    (output/"p8_bootstrap_caption.json").write_text(json.dumps(result,indent=2)+"\n"); return result


def grounding_vectors(root, mode):
    rows=json.loads((root/mode/"grounding_predictions.json").read_text()); ids=np.asarray([int(r["sample_id"]) for r in rows]); preds=[None if not r["boxes"] else r["boxes"][0][0] for r in rows]; targets=[r["ground_truth_boxes"][0] for r in rows]
    return ids,box_iou_vector(preds,targets)


def run_grounding(root, output, iterations, seed):
    ids={};ious={}
    for mode in MODES: ids[mode],ious[mode]=grounding_vectors(root,mode)
    if not all(np.array_equal(ids[m],ids["umbrae"]) for m in MODES[1:]): raise ValueError("grounding sample order mismatch")
    clusters=[np.where(ids["umbrae"]==sample)[0] for sample in range(982)]; rng=np.random.default_rng(seed); scores={m:{"mean_iou":np.empty(iterations),"accuracy_at_0_5":np.empty(iterations)} for m in MODES}
    for iteration in range(iterations):
        chosen=rng.integers(0,982,size=982); index=np.concatenate([clusters[x] for x in chosen])
        for mode in MODES: scores[mode]["mean_iou"][iteration]=ious[mode][index].mean();scores[mode]["accuracy_at_0_5"][iteration]=(ious[mode][index]>.5).mean()
    result={"seed":seed,"iterations":iterations,"cluster_count":982,"query_count":2419,"cluster_unit":"fMRI/image sample_id","paired_cluster_indices":True,"comparisons":{}}
    for name,(left,right) in PAIRS.items(): result["comparisons"][name]={metric:paired_summary(scores[left][metric]-scores[right][metric]) for metric in scores[left]}
    (output/"p8_bootstrap_grounding.json").write_text(json.dumps(result,indent=2)+"\n"); return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--p7-dir",default="dual_branch_outputs/p7_downstream");p.add_argument("--output-dir",default="dual_branch_outputs/p8_diagnosis");p.add_argument("--iterations",type=int,default=10000);p.add_argument("--seed",type=int,default=8042);a=p.parse_args()
    if a.iterations<10000: raise ValueError("P8 requires at least 10000 bootstrap iterations")
    root=Path(a.p7_dir);output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True);run_caption(root,output,a.iterations,a.seed);run_grounding(root,output,a.iterations,a.seed);print(json.dumps({"caption":"complete","grounding":"complete","iterations":a.iterations}))


if __name__=="__main__": main()
