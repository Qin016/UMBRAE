#!/usr/bin/env python
"""Run locked P7 caption and REC evaluation for four frozen brain encoders."""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "BrainHub"))

from metrics import tokenize
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from eval_bbox_rec import calculate_metric
from models.downstream_brain_encoder import MODES, DownstreamBrainEncoder
from models.dual_branch_cache import sha256_file
from scripts.cache_p7_downstream_features import load_projector, resolve_config
from utils import extract_boxes, extract_id_bbox_caption


SYSTEM = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
USER_IMAGE = " <im_start>" + "<im_patch>" * 256 + "<im_end> "


def full_prompt(template, expression=None):
    prompt = template.replace("<expr>", expression) if expression is not None else template
    if "<image>" in prompt: prompt = prompt.replace("<image>", USER_IMAGE)
    else: prompt = prompt + USER_IMAGE
    return SYSTEM + prompt + " ASSISTANT:"


def load_decoder(config, device):
    tokenizer = LlamaTokenizer.from_pretrained(config["shikra_model"], padding_side="left")
    model = LlamaForCausalLM.from_pretrained(config["shikra_model"]); model.to(device).eval().requires_grad_(False)
    projector = load_projector(config["mm_projector"], device)
    return tokenizer, model, projector


def generate(tokenizer, model, projected, prompts, config, device):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True); input_ids = encoded.input_ids.to(device)
    if len(set(int((row != tokenizer.pad_token_id).sum()) for row in input_ids)) != 1:
        raise ValueError("P7 batches must group prompts with the same token length")
    with torch.inference_mode():
        embeds = model.model.embed_tokens(input_ids).clone()
        for index, row in enumerate(input_ids):
            starts = torch.where(row == 32001)[0]
            if len(starts) != 1: raise ValueError("Prompt must contain exactly one <im_start>")
            start = int(starts[0]);
            if int(row[start + 257]) != 32002: raise ValueError("<im_end> must follow exactly 256 patches")
            embeds[index, start + 1:start + 257] = projected[index].to(embeds.dtype)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            output_ids = model.generate(inputs_embeds=embeds.float(), use_cache=config["use_cache"], do_sample=config["do_sample"], pad_token_id=config["pad_token_id"], bos_token_id=config["bos_token_id"], eos_token_id=config["eos_token_id"], max_new_tokens=config["max_new_tokens"])
    return tokenizer.batch_decode(output_ids)


def validation_smoke(config, device):
    tokenizer, decoder, projector = load_decoder(config, device); fmri = np.load(config["validation_fmri"], mmap_mode="r"); x = torch.from_numpy(np.array(fmri[:1], dtype=np.float32, copy=True)).to(device); projected = []
    for mode in MODES:
        encoder = DownstreamBrainEncoder(mode, config).to(device).eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")): z = encoder(x)
        projected.append(projector(z.float())[0]); del encoder; torch.cuda.empty_cache()
    responses = generate(tokenizer, decoder, torch.stack(projected), [full_prompt(config["caption_prompt"])] * 4, config, device)
    result = {mode: {"response": response, "caption": extract_id_bbox_caption(response)[1]} for mode, response in zip(MODES, responses)}
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True); (output / "validation_generation_smoke.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


def core_caption_metrics(references, candidates):
    environment_java = Path(sys.executable).resolve().parent / "java"
    if environment_java.is_file():
        os.environ["PATH"] = str(environment_java.parent) + os.pathsep + os.environ.get("PATH", "")
    refs, cands = tokenize(references, candidates); bleu, _ = Bleu(4).compute_score(refs, cands); rouge, _ = Rouge().compute_score(refs, cands); cider, _ = Cider().compute_score(refs, cands)
    return {"BLEU-1": float(bleu[0]), "BLEU-2": float(bleu[1]), "BLEU-3": float(bleu[2]), "BLEU-4": float(bleu[3]), "ROUGE-L": float(rouge), "CIDEr": float(cider)}


def projected_batch(caches, projector, entries, device):
    values = np.stack([np.array(caches[mode][sample], dtype=np.float32, copy=True) for sample, mode, _, _ in entries])
    with torch.inference_mode(): return projector(torch.from_numpy(values).to(device))


def run_task(task, jobs, caches, tokenizer, decoder, projector, config, device, output):
    record_path = output / f"{task}_joint_predictions.jsonl"; existing = {}; prior_records = []
    if record_path.exists():
        for line in record_path.read_text().splitlines():
            record=json.loads(line); prior_records.append(record); existing[(record["sample_id"], record["mode"], record.get("expression"))]=record
    if prior_records and all("batch_identifier" in record for record in prior_records):
        prior_elapsed=sum(next(record["batch_elapsed_seconds"] for record in prior_records if record["batch_identifier"]==identifier) for identifier in {record["batch_identifier"] for record in prior_records})
    elif task=="caption":
        prior_elapsed=sum(prior_records[index]["batch_elapsed_seconds"] for index in range(0,len(prior_records),config["generation_job_batch_size"]*len(MODES)))
    else: prior_elapsed=0.0
    groups=defaultdict(list)
    for sample, expression, prompt in jobs:
        length=len(tokenizer(prompt, add_special_tokens=True).input_ids); groups[length].append((sample,expression,prompt))
    started=time.time(); generated=0
    with record_path.open("a") as handle:
        for length in sorted(groups):
            group=groups[length]
            for start in range(0,len(group),config["generation_job_batch_size"]):
                chunk=group[start:start+config["generation_job_batch_size"]]; entries=[]
                for sample,expression,prompt in chunk:
                    for mode in MODES:
                        if (sample,mode,expression) not in existing: entries.append((sample,mode,expression,prompt))
                if not entries: continue
                projected=projected_batch(caches,projector,entries,device); prompts=[entry[3] for entry in entries]; batch_started=time.time()
                try: responses=generate(tokenizer,decoder,projected,prompts,config,device); error=None
                except Exception as exception: responses=[""]*len(entries); error=f"{type(exception).__name__}: {exception}"
                batch_elapsed=time.time()-batch_started; batch_identifier=f"{task}-{length}-{start}"
                for entry,response in zip(entries,responses):
                    sample,mode,expression,_=entry
                    if task=="caption":
                        caption=extract_id_bbox_caption(response)[1] if response else ""; record={"sample_id":sample,"mode":mode,"response":response,"caption":caption,"status":"failed" if error else ("empty" if not caption.strip() else "success"),"error":error,"batch_elapsed_seconds":batch_elapsed,"batch_identifier":batch_identifier}
                    else:
                        boxes=extract_boxes(response) if response else []; record={"sample_id":sample,"mode":mode,"expression":expression,"response":response,"boxes":boxes,"status":"failed" if error else ("parse_failure" if not boxes else "success"),"error":error,"batch_elapsed_seconds":batch_elapsed,"batch_identifier":batch_identifier}
                    handle.write(json.dumps(record)+"\n");handle.flush();existing[(sample,mode,expression)]=record;generated+=1
                if generated and generated % 160 < len(entries): print(json.dumps({"task":task,"generated_entries":generated,"total_entries":len(jobs)*4}),flush=True)
    return list(existing.values()),prior_elapsed+time.time()-started


def evaluate_captions(records, refs, output):
    results={}
    for mode in MODES:
        selected={int(record["sample_id"]):record for record in records if record["mode"]==mode}; candidates=[selected[index]["caption"] for index in range(982)]; references=[refs[str(index)] for index in range(982)]; metrics=core_caption_metrics(references,candidates); statuses={name:sum(record["status"]==name for record in selected.values()) for name in ("success","empty","failed")}; metrics.update({"sample_count":982,"denominator":982,"successful_generations":statuses["success"],"empty_outputs":statuses["empty"],"failed_generations":statuses["failed"],"failure_policy":"empty caption retained in denominator"}); results[mode]=metrics
        mode_dir=output/mode;mode_dir.mkdir(exist_ok=True); predictions=[{**selected[index],"references":refs[str(index)]} for index in range(982)];(mode_dir/"caption_predictions.json").write_text(json.dumps(predictions,indent=2));(mode_dir/"caption_metrics.json").write_text(json.dumps(metrics,indent=2));(mode_dir/"fmricap.json").write_text(json.dumps({str(index):candidates[index] for index in range(982)},indent=2))
    return results


def evaluate_grounding(records, annotations, categories, output):
    results={}
    for mode in MODES:
        selected={(int(record["sample_id"]),record["expression"]):record for record in records if record["mode"]==mode}; preds=[];targets=[];schema={};prediction_rows=[]
        for sample in range(982):
            schema[str(sample)]={}
            for expression,gt_boxes in annotations[str(sample)].items():
                record=selected[(sample,expression)]; parsed=record["boxes"]; pred=None if not parsed else parsed[0][0]; preds.append(pred);targets.append(gt_boxes[0]);schema[str(sample)][expression]=parsed;prediction_rows.append({**record,"ground_truth_boxes":gt_boxes})
        overall=calculate_metric(preds,targets,threshold=.5); class_metrics={}
        for name,names in categories.items():
            class_preds=[];class_targets=[]
            for sample in range(982):
                for expression,gt_boxes in annotations[str(sample)].items():
                    if expression in names:
                        parsed=selected[(sample,expression)]["boxes"];class_preds.append(None if not parsed else parsed[0][0]);class_targets.append(gt_boxes[0])
            class_metrics[name]=calculate_metric(class_preds,class_targets,threshold=.5)
        metrics={"query_count":len(targets),"threshold":.5,"mean_iou":overall["iou"],"grounding_accuracy":overall["accuracy"],"parse_failures":overall["failed"],"target_failures":overall["target_failed"],"failure_policy":"parse failure becomes [0,0,0,0] and remains in accuracy denominator","category_metrics":class_metrics};results[mode]=metrics;mode_dir=output/mode;mode_dir.mkdir(exist_ok=True);(mode_dir/"grounding_predictions.json").write_text(json.dumps(prediction_rows,indent=2));(mode_dir/"grounding_metrics.json").write_text(json.dumps(metrics,indent=2));(mode_dir/"rec_response.json").write_text(json.dumps(schema,indent=2))
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--device",default="cuda");parser.add_argument("--validation-smoke",action="store_true");args=parser.parse_args();config=resolve_config(args.config);output=Path(config["output_dir"]);output.mkdir(parents=True,exist_ok=True)
    if args.validation_smoke: validation_smoke(config,args.device);return
    metadata=json.loads((output/"feature_cache_metadata.json").read_text());
    if not metadata.get("test_evaluation_started_after_model_selection") or metadata["sample_count"]!=982: raise ValueError("P7 test policy metadata violation")
    caches={mode:np.load(metadata["representations"][mode],mmap_mode="r") for mode in MODES}; refs=json.loads(Path(config["caption_references"]).read_text());annotations=json.loads(Path(config["grounding_annotations"]).read_text());categories=json.loads(Path(config["grounding_categories"]).read_text())
    if len(refs)!=982 or len(annotations)!=982: raise ValueError("P7 references must cover all 982 test samples")
    torch.cuda.reset_peak_memory_stats(); tokenizer,decoder,projector=load_decoder(config,args.device)
    caption_jobs=[(index,None,full_prompt(config["caption_prompt"])) for index in range(982)]; caption_records,caption_time=run_task("caption",caption_jobs,caches,tokenizer,decoder,projector,config,args.device,output);caption_metrics=evaluate_captions(caption_records,refs,output)
    grounding_jobs=[(index,expression,full_prompt(config["grounding_prompt"],expression)) for index in range(982) for expression in annotations[str(index)]];ground_records,ground_time=run_task("grounding",grounding_jobs,caches,tokenizer,decoder,projector,config,args.device,output);ground_metrics=evaluate_grounding(ground_records,annotations,categories,output);peak=torch.cuda.max_memory_allocated()/1024**3
    checkpoint_keys={"umbrae":{},"lora":{"lora_checkpoint":config["p5_checkpoint"]},"full_real":{"stage_c_checkpoint":config["p6_real_checkpoint"],"structural_checkpoint":config["real_structural_checkpoint"]},"full_random":{"stage_c_checkpoint":config["p6_random_checkpoint"],"structural_checkpoint":config["random_structural_checkpoint"]}}
    for mode in MODES:
        provenance={"protocol_version":"protocol_v1","split":"test","sample_count":982,"test_evaluation_started_after_model_selection":True,"mode":mode,"role":"MAIN METHOD" if mode=="full_real" else ("RANDOM STRUCTURAL CONTROL" if mode=="full_random" else "ABLATION"),"brainx_checkpoint":config["brainx_checkpoint"],"brainx_sha256":sha256_file(config["brainx_checkpoint"]),"mm_projector":config["mm_projector"],"mm_projector_sha256":sha256_file(config["mm_projector"]),"shikra_model_identifier":config["shikra_model"],"decoding":{"caption_prompt":config["caption_prompt"],"grounding_prompt":config["grounding_prompt"],"max_new_tokens":config["max_new_tokens"],"do_sample":config["do_sample"],"use_cache":config["use_cache"],"pad_token_id":config["pad_token_id"],"bos_token_id":config["bos_token_id"],"eos_token_id":config["eos_token_id"]},"joint_caption_wall_seconds":caption_time,"joint_grounding_wall_seconds":ground_time,"effective_per_model_caption_seconds":caption_time/4,"effective_per_model_grounding_seconds":ground_time/4,"shared_peak_gpu_memory_gib":peak,**checkpoint_keys[mode]}
        for key,value in list(provenance.items()):
            if key.endswith("checkpoint") and value: provenance[key+"_sha256"]=sha256_file(value)
        (output/mode/"provenance.json").write_text(json.dumps(provenance,indent=2))
    summary={"caption_metrics":caption_metrics,"grounding_metrics":ground_metrics,"compute":{"joint_caption_wall_seconds":caption_time,"joint_grounding_wall_seconds":ground_time,"effective_per_model_caption_seconds":caption_time/4,"effective_per_model_grounding_seconds":ground_time/4,"shared_peak_gpu_memory_gib":peak}};(output/"p7_raw_metrics.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
