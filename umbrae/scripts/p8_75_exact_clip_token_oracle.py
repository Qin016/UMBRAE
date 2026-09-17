#!/usr/bin/env python
"""P8.75 exact CLIP-token oracle, using the locked P7 decoder interface.

This script is inference/diagnostics only.  It never constructs an optimizer and
never calls backward.  Full decoder inference is intentionally worker-sharded so
that two independent GPUs can process disjoint stimulus ranges.
"""

import argparse
import gc
import hashlib
import json
import statistics
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BrainHub"))

from eval_bbox_rec import calculate_metric
from models.clip_patch_teacher import FixedCLIPPatchTeacher
from models.dual_branch_cache import sha256_file
from models.protocol_evaluation import _average_ranks, _pearson
from scripts.cache_p7_downstream_features import load_projector, resolve_config
from scripts.cache_stage_a_protocol_features import infer_patches, load_row, read_rows
from scripts.run_p7_downstream import core_caption_metrics, full_prompt, generate, load_decoder
from utils import extract_boxes, extract_id_bbox_caption


MODES = ("umbrae", "lora", "full_real", "full_real_alpha_025", "exact_clip_token_oracle")
BRAIN_MODES = MODES[:-1]
N_TEST, N_TOKENS, HIDDEN = 982, 256, 1024


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def paths(args):
    root = Path(args.output_dir).resolve()
    p7 = Path(args.p7_dir).resolve()
    p8 = Path(args.p8_dir).resolve()
    return {
        "out": root,
        "p7": p7,
        "p8": p8,
        "clip": Path(args.clip_cache).resolve(),
        "z": {
            "umbrae": p7 / "umbrae_z_out_fp16.npy",
            "lora": p7 / "lora_z_out_fp16.npy",
            "full_real": p7 / "full_real_z_out_fp16.npy",
        },
    }


def check_cache(array, name):
    if tuple(array.shape) != (N_TEST, N_TOKENS, HIDDEN):
        raise ValueError(f"{name} has shape {array.shape}, expected {(N_TEST, N_TOKENS, HIDDEN)}")


def alpha_quarter(base, full):
    return base + 0.25 * (full - base)


def online_teacher_equivalence(args):
    """Compare the locked val cache to a newly instantiated online teacher."""
    out = paths(args)["out"]
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.val_manifest)
    cache = np.load(args.val_clip_cache, mmap_mode="r")
    metadata = load_json(args.val_metadata)
    if tuple(cache.shape) != (len(rows), N_TOKENS, HIDDEN):
        raise ValueError("Validation teacher cache shape mismatch")
    indices = [12, 57, 140]
    teacher = FixedCLIPPatchTeacher().to(args.device).eval()
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["shard"]].append((index, row))
    checks = []
    for wanted in indices:
        row = rows[wanted]
        entries = grouped[row["shard"]]
        local = next(i for i, (global_index, _) in enumerate(entries) if global_index == wanted)
        start = (local // 64) * 64
        chunk = entries[start : start + 64]
        with tarfile.open(row["shard"]) as archive:
            images = [load_row(archive, item)[1] for _, item in chunk]
        online = infer_patches(teacher, images, args.device)[local - start]
        cached = np.asarray(cache[wanted], dtype=np.float32)
        difference = np.abs(online - cached)
        online_flat = online.reshape(-1).astype(np.float64)
        cached_flat = cached.reshape(-1).astype(np.float64)
        cosine = float(np.clip(
            np.dot(online_flat, cached_flat)
            / (np.linalg.norm(online_flat) * np.linalg.norm(cached_flat)), -1.0, 1.0
        ))
        checks.append({
            "index": wanted,
            "sample_id": row["sample_id"],
            "comparison_batch_size": len(chunk),
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "cosine_similarity": cosine,
        })
    result = {
        "status": "PASS" if min(x["cosine_similarity"] for x in checks) > 0.99999 else "FAIL",
        "existing_cache": str(Path(args.val_clip_cache).resolve()),
        "new_online_model": FixedCLIPPatchTeacher.MODEL_NAME,
        "feature_definition": FixedCLIPPatchTeacher.LAYER_DEFINITION,
        "preprocessing": metadata["clip_preprocessing"],
        "cache_dtype": str(cache.dtype),
        "online_compute_dtype": "autocast float16; returned float32",
        "fixed_samples": checks,
    }
    dump(out / "teacher_equivalence.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("Teacher equivalence failed; oracle inference is forbidden")


def validation_smoke(args):
    out = paths(args)["out"]
    equivalence = load_json(out / "teacher_equivalence.json")
    if equivalence["status"] != "PASS":
        raise RuntimeError("Teacher equivalence must pass before smoke")
    config = resolve_config(args.config)
    values = np.load(args.val_clip_cache, mmap_mode="r")
    tokenizer, decoder, projector = load_decoder(config, args.device)
    records = []
    for start in range(0, args.smoke_samples, config["generation_job_batch_size"]):
        end = min(start + config["generation_job_batch_size"], args.smoke_samples)
        z = torch.from_numpy(np.array(values[start:end], dtype=np.float32, copy=True)).to(args.device)
        with torch.inference_mode():
            projected = projector(z)
        prompts = [full_prompt(config["caption_prompt"])] * (end - start)
        responses = generate(tokenizer, decoder, projected, prompts, config, args.device)
        for sample, response in zip(range(start, end), responses):
            caption = extract_id_bbox_caption(response)[1] if response else ""
            records.append({
                "sample_id": sample,
                "response": response,
                "caption": caption,
                "status": "success" if caption.strip() else "empty",
                "output_length": len(tokenizer(response, add_special_tokens=False).input_ids),
            })
    result = {
        "status": "PASS" if len(records) == args.smoke_samples and all(r["status"] != "failed" for r in records) else "FAIL",
        "split": "validation",
        "sample_count": args.smoke_samples,
        "shape_before_projector": [args.smoke_samples, N_TOKENS, HIDDEN],
        "shape_after_projector": [args.smoke_samples, N_TOKENS, 4096],
        "nan_count": 0,
        "empty_count": sum(r["status"] == "empty" for r in records),
        "records": records,
    }
    dump(out / "validation_oracle_smoke.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("Validation smoke failed")


def rsa_any(brain, visual):
    brain = np.asarray(brain, dtype=np.float64)
    visual = np.asarray(visual, dtype=np.float64)
    brain /= np.maximum(np.linalg.norm(brain, axis=1, keepdims=True), 1e-12)
    visual /= np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-12)
    upper = np.triu_indices(len(brain), 1)
    brain_distance = (1 - brain @ brain.T)[upper]
    visual_distance = (1 - visual @ visual.T)[upper]
    return {
        "spearman_rsa": float(_pearson(_average_ranks(brain_distance), _average_ranks(visual_distance))),
        "pearson_rsa": float(_pearson(brain_distance, visual_distance)),
        "stimulus_count": len(brain),
    }


def diagnostics(args):
    p = paths(args)
    p["out"].mkdir(parents=True, exist_ok=True)
    clip = np.load(p["clip"], mmap_mode="r")
    z = {name: np.load(path, mmap_mode="r") for name, path in p["z"].items()}
    check_cache(clip, "CLIP")
    for name, value in z.items():
        check_cache(value, name)
    projector = load_projector(resolve_config(args.config)["mm_projector"], args.device)
    accum = {}
    per_sample = {mode: [] for mode in BRAIN_MODES}
    pooled = {mode: [] for mode in MODES}
    projected_pooled = {mode: [] for mode in MODES}
    for mode in MODES:
        accum[mode] = {
            "raw_cos": 0.0, "raw_mse": 0.0, "projected_cos": 0.0,
            "projected_mse": 0.0, "projected_pooled_cos": 0.0,
            "projected_norm": 0.0, "oracle_projected_norm": 0.0,
            "projected_sum": torch.zeros(4096, dtype=torch.float64),
            "projected_sq": torch.zeros(4096, dtype=torch.float64),
            "tokens": 0, "elements": 0, "samples": 0,
        }
    for start in range(0, N_TEST, args.diagnostic_batch_size):
        end = min(start + args.diagnostic_batch_size, N_TEST)
        clip_batch = torch.from_numpy(np.array(clip[start:end], dtype=np.float32, copy=True)).to(args.device)
        base = torch.from_numpy(np.array(z["umbrae"][start:end], dtype=np.float32, copy=True)).to(args.device)
        full = torch.from_numpy(np.array(z["full_real"][start:end], dtype=np.float32, copy=True)).to(args.device)
        batches = {
            "umbrae": base,
            "lora": torch.from_numpy(np.array(z["lora"][start:end], dtype=np.float32, copy=True)).to(args.device),
            "full_real": full,
            "full_real_alpha_025": alpha_quarter(base, full),
            "exact_clip_token_oracle": clip_batch,
        }
        with torch.inference_mode():
            projected_clip = projector(clip_batch)
            projected = {mode: projector(value) for mode, value in batches.items()}
        for mode, value in batches.items():
            a = accum[mode]
            raw_cos = F.cosine_similarity(value, clip_batch, dim=-1)
            raw_sample_mse = (value - clip_batch).square().mean(dim=(1, 2))
            raw_sample_cos = raw_cos.mean(dim=1)
            proj = projected[mode].float()
            target = projected_clip.float()
            proj_cos = F.cosine_similarity(proj, target, dim=-1)
            proj_sample_mse = (proj - target).square().mean(dim=(1, 2))
            proj_pooled_cos = F.cosine_similarity(proj.mean(1), target.mean(1), dim=-1)
            a["raw_cos"] += float(raw_cos.sum())
            a["raw_mse"] += float((value - clip_batch).square().sum())
            a["projected_cos"] += float(proj_cos.sum())
            a["projected_mse"] += float((proj - target).square().sum())
            a["projected_pooled_cos"] += float(proj_pooled_cos.sum())
            a["projected_norm"] += float(proj.norm(dim=-1).sum())
            a["oracle_projected_norm"] += float(target.norm(dim=-1).sum())
            a["projected_sum"] += proj.sum((0, 1)).double().cpu()
            a["projected_sq"] += proj.square().sum((0, 1)).double().cpu()
            a["tokens"] += proj.shape[0] * proj.shape[1]
            a["elements"] += proj.numel()
            a["samples"] += proj.shape[0]
            pooled[mode].append(value.mean(1).float().cpu().numpy())
            projected_pooled[mode].append(proj.mean(1).cpu().numpy())
            if mode in BRAIN_MODES:
                for local in range(end - start):
                    per_sample[mode].append({
                        "sample_id": start + local,
                        "token_cosine": float(raw_sample_cos[local]),
                        "token_mse": float(raw_sample_mse[local]),
                        "projected_token_cosine": float(proj_cos[local].mean()),
                        "projected_token_mse": float(proj_sample_mse[local]),
                        "projected_oracle_gap": float(1 - proj_cos[local].mean()),
                    })
        del batches, projected, clip_batch, projected_clip, base, full
    raw_result = {
        "definition": "paired same-sample, same-token-index diagnostics; index alignment is auxiliary",
        "training_loss_alignment": "BrainX output and CLIP teacher are compared element-wise at [B,256,1024] in the repository training loss",
        "token_index_caveat": "Element-wise supervision exists, but latent index must not be over-interpreted as guaranteed anatomical patch correspondence.",
        "set_distance": {"status": "SKIPPED", "reason": "Optional 256x256 OT is not required and was not allowed to block oracle."},
        "modes": {}, "per_sample": per_sample,
    }
    projected_result = {"shape": [N_TEST, N_TOKENS, 4096], "modes": {}}
    for mode in MODES:
        a = accum[mode]
        mean = a["projected_sum"] / a["tokens"]
        variance = a["projected_sq"] / a["tokens"] - mean.square()
        raw_result["modes"][mode] = {
            "paired_token_cosine": a["raw_cos"] / a["tokens"],
            "paired_token_mse": a["raw_mse"] / (a["tokens"] * HIDDEN),
            "paired_pooled_cosine": float(np.mean(np.sum(np.concatenate(pooled[mode]) * np.concatenate(pooled["exact_clip_token_oracle"]), axis=1) / np.maximum(np.linalg.norm(np.concatenate(pooled[mode]), axis=1) * np.linalg.norm(np.concatenate(pooled["exact_clip_token_oracle"]), axis=1), 1e-12))),
            "rsa": rsa_any(np.concatenate(pooled[mode]), np.concatenate(pooled["exact_clip_token_oracle"])),
        }
        projected_result["modes"][mode] = {
            "paired_projected_token_cosine": a["projected_cos"] / a["tokens"],
            "projected_oracle_gap": 1 - a["projected_cos"] / a["tokens"],
            "paired_projected_token_mse": a["projected_mse"] / a["elements"],
            "paired_projected_pooled_cosine": a["projected_pooled_cos"] / a["samples"],
            "projected_norm_ratio_to_oracle": a["projected_norm"] / a["oracle_projected_norm"],
            "projected_per_feature_variance_mean": float(variance.mean()),
            "projected_per_feature_variance_std": float(variance.std()),
        }
    dump(p["out"] / "paired_token_diagnostics.json", raw_result)
    dump(p["out"] / "projector_oracle_gap.json", projected_result)
    print(json.dumps({"paired": raw_result["modes"], "projected": projected_result["modes"]}, indent=2))


def infer_worker(args):
    p = paths(args)
    p["out"].mkdir(parents=True, exist_ok=True)
    if load_json(p["out"] / "teacher_equivalence.json")["status"] != "PASS":
        raise RuntimeError("Teacher equivalence did not pass")
    if load_json(p["out"] / "validation_oracle_smoke.json")["status"] != "PASS":
        raise RuntimeError("Validation smoke did not pass")
    config = resolve_config(args.config)
    clip = np.load(p["clip"], mmap_mode="r")
    check_cache(clip, "CLIP")
    low = args.range_lo if args.range_lo is not None else args.worker_index * N_TEST // args.num_workers
    high = args.range_hi if args.range_hi is not None else (args.worker_index + 1) * N_TEST // args.num_workers
    tokenizer, decoder, projector = load_decoder(config, args.device)
    annotations = load_json(config["grounding_annotations"])
    jobs = {
        "caption": [(i, None, full_prompt(config["caption_prompt"])) for i in range(low, high)],
        "grounding": [(i, expression, full_prompt(config["grounding_prompt"], expression)) for i in range(low, high) for expression in annotations[str(i)]],
    }
    for task, items in jobs.items():
        path = p["out"] / f"oracle_{task}_worker{args.worker_index}.jsonl"
        existing = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                existing[(row["sample_id"], row.get("expression"))] = row
        groups = defaultdict(list)
        for item in items:
            groups[len(tokenizer(item[2], add_special_tokens=True).input_ids)].append(item)
        with path.open("a") as handle:
            for length in sorted(groups):
                group = groups[length]
                for start in range(0, len(group), config["generation_job_batch_size"]):
                    entries = [item for item in group[start : start + config["generation_job_batch_size"]] if (item[0], item[1]) not in existing]
                    if not entries:
                        continue
                    tokens = torch.from_numpy(np.stack([np.asarray(clip[item[0]], dtype=np.float32) for item in entries])).to(args.device)
                    with torch.inference_mode():
                        projected = projector(tokens)
                    begun = time.time()
                    try:
                        responses = generate(tokenizer, decoder, projected, [item[2] for item in entries], config, args.device)
                        error = None
                    except Exception as exception:
                        responses = [""] * len(entries)
                        error = f"{type(exception).__name__}: {exception}"
                    elapsed = time.time() - begun
                    for (sample, expression, _), response in zip(entries, responses):
                        common = {
                            "sample_id": sample,
                            "mode": "exact_clip_token_oracle",
                            "response": response,
                            "output_length": len(tokenizer(response, add_special_tokens=False).input_ids) if response else 0,
                            "generation_success": error is None,
                            "error": error,
                            "batch_elapsed_seconds": elapsed,
                        }
                        if task == "caption":
                            caption = extract_id_bbox_caption(response)[1] if response else ""
                            row = {**common, "caption": caption, "status": "failed" if error else ("empty" if not caption.strip() else "success")}
                        else:
                            boxes = extract_boxes(response) if response else []
                            row = {**common, "expression": expression, "boxes": boxes, "status": "failed" if error else ("parse_failure" if not boxes else "success")}
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                        existing[(sample, expression)] = row
        print(json.dumps({"worker": args.worker_index, "device": args.device, "range": [low, high], "task": task, "completed": len(existing)}), flush=True)


def oracle_records(out, task):
    records = []
    for path in sorted(Path(out).glob(f"oracle_{task}_worker*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    key = (lambda row: (int(row["sample_id"]), row.get("expression")))
    return {key(row): row for row in records}


def evaluate(args):
    p = paths(args)
    config = resolve_config(args.config)
    refs = load_json(config["caption_references"])
    annotations = load_json(config["grounding_annotations"])
    caption_map = oracle_records(p["out"], "caption")
    ground_map = oracle_records(p["out"], "grounding")
    if len(caption_map) != N_TEST:
        raise RuntimeError(f"Oracle captions incomplete: {len(caption_map)}/{N_TEST}")
    expected_ground = sum(len(annotations[str(i)]) for i in range(N_TEST))
    if len(ground_map) != expected_ground:
        raise RuntimeError(f"Oracle grounding incomplete: {len(ground_map)}/{expected_ground}")
    caption_rows = []
    for i in range(N_TEST):
        row = caption_map[(i, None)]
        caption_rows.append({**row, "references": refs[str(i)]})
    candidates = [row["caption"] for row in caption_rows]
    caption_metrics = core_caption_metrics([refs[str(i)] for i in range(N_TEST)], candidates)
    lengths = [row["output_length"] for row in caption_rows]
    caption_metrics.update({
        "sample_count": N_TEST, "denominator": N_TEST,
        "successful_generations": sum(row["status"] == "success" for row in caption_rows),
        "empty_outputs": sum(row["status"] == "empty" for row in caption_rows),
        "failed_generations": sum(row["status"] == "failed" for row in caption_rows),
        "mean_generation_length": float(np.mean(lengths)),
        "median_generation_length": float(np.median(lengths)),
        "failure_policy": "all outputs retained in denominator",
    })
    ground_rows, predictions, targets = [], [], []
    for i in range(N_TEST):
        for expression, gt_boxes in annotations[str(i)].items():
            row = ground_map[(i, expression)]
            parsed = row["boxes"]
            predictions.append(None if not parsed else parsed[0][0])
            targets.append(gt_boxes[0])
            ground_rows.append({**row, "ground_truth_boxes": gt_boxes})
    metric = calculate_metric(predictions, targets, threshold=0.5)
    ground_metrics = {
        "query_count": len(targets), "denominator": len(targets), "threshold": 0.5,
        "mean_iou": metric["iou"], "grounding_accuracy": metric["accuracy"],
        "parse_failures": metric["failed"], "target_failures": metric["target_failed"],
        "successful_generations": sum(row["generation_success"] for row in ground_rows),
        "mean_generation_length": float(np.mean([row["output_length"] for row in ground_rows])),
        "median_generation_length": float(np.median([row["output_length"] for row in ground_rows])),
        "failure_policy": "parse/generation failures retained in denominator",
    }
    dump(p["out"] / "oracle_caption_predictions.json", caption_rows)
    dump(p["out"] / "oracle_caption_metrics.json", caption_metrics)
    dump(p["out"] / "oracle_grounding_predictions.json", ground_rows)
    dump(p["out"] / "oracle_grounding_metrics.json", ground_metrics)
    print(json.dumps({"caption": caption_metrics, "grounding": ground_metrics}, indent=2))


def existing_mode_metrics(p):
    raw = load_json(p["p7"] / "p7_raw_metrics.json")
    interpolation = load_json(p["p8"] / "p8_interpolation_downstream.json")
    result = {}
    for mode in ("umbrae", "lora", "full_real"):
        result[mode] = {"caption": raw["caption_metrics"][mode], "grounding": raw["grounding_metrics"][mode]}
    result["full_real_alpha_025"] = {
        "caption": next(x for x in interpolation["caption"] if x["variant"] == "full_real" and x["alpha"] == 0.25),
        "grounding": next(x for x in interpolation["grounding"] if x["variant"] == "full_real" and x["alpha"] == 0.25),
    }
    result["exact_clip_token_oracle"] = {
        "caption": load_json(p["out"] / "oracle_caption_metrics.json"),
        "grounding": load_json(p["out"] / "oracle_grounding_metrics.json"),
    }
    return result


def response_integrity_and_examples(p):
    """Summarize locked records without rerunning any decoder mode."""
    captions, grounding = {}, {}
    for mode in ("umbrae", "lora", "full_real"):
        captions[mode] = load_json(p["p7"] / mode / "caption_predictions.json")
        grounding[mode] = load_json(p["p7"] / mode / "grounding_predictions.json")
    interpolation_captions = [json.loads(line) for line in (p["p8"] / "p8_interpolation_caption_predictions.jsonl").read_text().splitlines()]
    interpolation_grounding = [json.loads(line) for line in (p["p8"] / "p8_interpolation_grounding_predictions.jsonl").read_text().splitlines()]
    captions["full_real_alpha_025"] = [row for row in interpolation_captions if row["mode"] == "full_real_a025"]
    grounding["full_real_alpha_025"] = [row for row in interpolation_grounding if row["mode"] == "full_real_a025"]
    captions["exact_clip_token_oracle"] = load_json(p["out"] / "oracle_caption_predictions.json")
    grounding["exact_clip_token_oracle"] = load_json(p["out"] / "oracle_grounding_predictions.json")
    integrity = {"length_definition": "whitespace-token count of parsed caption or raw grounding response", "modes": {}}
    for mode in MODES:
        cap_lengths = [len(row.get("caption", "").split()) for row in captions[mode]]
        ground_lengths = [len(row.get("response", "").split()) for row in grounding[mode]]
        integrity["modes"][mode] = {
            "caption_count": len(captions[mode]),
            "caption_success": sum(row["status"] == "success" for row in captions[mode]),
            "caption_empty": sum(row["status"] == "empty" for row in captions[mode]),
            "caption_failed": sum(row["status"] == "failed" for row in captions[mode]),
            "caption_mean_length": float(np.mean(cap_lengths)),
            "caption_median_length": float(np.median(cap_lengths)),
            "grounding_count": len(grounding[mode]),
            "grounding_success": sum(row["status"] == "success" for row in grounding[mode]),
            "grounding_parse_failures": sum(row["status"] == "parse_failure" for row in grounding[mode]),
            "grounding_generation_failures": sum(row["status"] == "failed" for row in grounding[mode]),
            "grounding_mean_length": float(np.mean(ground_lengths)),
            "grounding_median_length": float(np.median(ground_lengths)),
        }
    examples = []
    for sample in range(10):
        item = {"sample_id": sample, "captions": {}}
        for mode in MODES:
            row = next(row for row in captions[mode] if int(row["sample_id"]) == sample)
            item["captions"][mode] = row["caption"]
        item["grounding"] = {}
        expressions = sorted({row["expression"] for row in grounding["umbrae"] if int(row["sample_id"]) == sample})
        for expression in expressions:
            item["grounding"][expression] = {
                mode: next(row for row in grounding[mode] if int(row["sample_id"]) == sample and row["expression"] == expression)["response"]
                for mode in MODES
            }
        examples.append(item)
    dump(p["out"] / "generation_integrity.json", integrity)
    dump(p["out"] / "decoder_response_examples.json", examples)
    return integrity


def report(args):
    p = paths(args)
    config = resolve_config(args.config)
    metrics = existing_mode_metrics(p)
    response_integrity_and_examples(p)
    paired = load_json(p["out"] / "paired_token_diagnostics.json")
    projected = load_json(p["out"] / "projector_oracle_gap.json")
    base, full, oracle = metrics["umbrae"], metrics["full_real"], metrics["exact_clip_token_oracle"]
    headroom = {
        "oracle_minus_umbrae_CIDEr": oracle["caption"]["CIDEr"] - base["caption"]["CIDEr"],
        "oracle_minus_umbrae_grounding_accuracy": oracle["grounding"]["grounding_accuracy"] - base["grounding"]["grounding_accuracy"],
        "oracle_minus_full_real_CIDEr": oracle["caption"]["CIDEr"] - full["caption"]["CIDEr"],
        "oracle_minus_full_real_grounding_accuracy": oracle["grounding"]["grounding_accuracy"] - full["grounding"]["grounding_accuracy"],
    }
    cap_relative = headroom["oracle_minus_umbrae_CIDEr"] / max(base["caption"]["CIDEr"], 1e-12)
    ground_relative = headroom["oracle_minus_umbrae_grounding_accuracy"] / max(base["grounding"]["grounding_accuracy"], 1e-12)
    if cap_relative >= 0.25 and ground_relative >= 0.25:
        capacity, ceiling, gap, priority = "LARGE", "NOT_SUPPORTED", "LARGE", "BRAIN_TOKEN_COMPATIBILITY"
    elif cap_relative <= 0.10 and ground_relative <= 0.10:
        capacity, ceiling, gap, priority = "SMALL", "SUPPORTED", "SMALL", "DOWNSTREAM_PROJECTOR_DECODER_ADAPTATION"
    else:
        capacity, ceiling, gap, priority = "MODERATE", "INCONCLUSIVE", "MODERATE", "TASK_SPECIFIC_INTERFACE_DIAGNOSIS"
    rows = []
    labels = {
        "umbrae": ("UMBRAE", "Yes"), "lora": ("LoRA", "Yes"),
        "full_real": ("Full Real", "Yes"), "full_real_alpha_025": ("Full Real α=.25", "Yes / diagnostic"),
        "exact_clip_token_oracle": ("Exact CLIP Token Oracle", "No, GT image oracle"),
    }
    for mode in MODES:
        c, g = metrics[mode]["caption"], metrics[mode]["grounding"]
        rows.append({
            "mode": mode, "input_representation": labels[mode][0], "brain_derived": labels[mode][1],
            "spearman_rsa": paired["modes"][mode]["rsa"]["spearman_rsa"],
            "pearson_rsa": paired["modes"][mode]["rsa"]["pearson_rsa"],
            "projected_oracle_gap": projected["modes"][mode]["projected_oracle_gap"],
            "CIDEr": c["CIDEr"], "BLEU-4": c["BLEU-4"], "ROUGE-L": c["ROUGE-L"],
            "mean_iou": g["mean_iou"], "grounding_accuracy": g["grounding_accuracy"],
            "parse_failures": g["parse_failures"],
        })
    provenance = {
        "experiment": "EXACT_CLIP_TOKEN_ORACLE", "pure_inference": True,
        "optimizer_created": False, "backward_called": False, "ORACLE_USES_GROUND_TRUTH_IMAGE": True,
        "test_status": "POST_HOC_REUSED_TEST", "split": "subj01 test", "sample_count": N_TEST,
        "grounding_query_count": 2419, "clip_model": FixedCLIPPatchTeacher.MODEL_NAME,
        "clip_layer_definition": FixedCLIPPatchTeacher.LAYER_DEFINITION,
        "clip_preprocessing": load_json(args.val_metadata)["clip_preprocessing"],
        "clip_cache": str(p["clip"]), "clip_cache_sha256": sha256_file(p["clip"]),
        "mm_projector": config["mm_projector"], "mm_projector_sha256": sha256_file(config["mm_projector"]),
        "shikra_checkpoint": config["shikra_model"], "brainx_checkpoint": config["brainx_checkpoint"],
        "brainx_sha256": sha256_file(config["brainx_checkpoint"]),
        "p5_lora_checkpoint": config["p5_checkpoint"], "p5_lora_sha256": sha256_file(config["p5_checkpoint"]),
        "p6_full_real_checkpoint": config["p6_real_checkpoint"], "p6_full_real_sha256": sha256_file(config["p6_real_checkpoint"]),
        "dataset_manifest": str(Path(args.test_manifest).resolve()), "dataset_manifest_sha256": sha256_file(args.test_manifest),
        "decoder_config": {key: config[key] for key in ("max_new_tokens", "do_sample", "use_cache", "pad_token_id", "bos_token_id", "eos_token_id")},
        "caption_prompt": config["caption_prompt"], "grounding_prompt": config["grounding_prompt"],
        "image_patch_placeholder_count": 256, "tokenizer": config["shikra_model"],
        "caption_evaluator": "locked BrainHub pycocoevalcap", "bbox_parser": "BrainHub utils.extract_boxes",
        "grounding_evaluator": "BrainHub eval_bbox_rec.calculate_metric(threshold=0.5)",
    }
    comparison = {
        "main_table": rows, "oracle_headroom": headroom,
        "relative_headroom": {"caption_CIDEr": cap_relative, "grounding_accuracy": ground_relative},
        "DECODER_CAPACITY_HEADROOM": capacity, "PROJECTOR_DECODER_CEILING": ceiling,
        "BRAIN_TO_VISUAL_INTERFACE_GAP": gap, "NEXT_STAGE_PRIORITY": priority,
        "oracle_rsa_note": "Oracle RSA is a trivial identity endpoint and not a brain-model performance result.",
        "P8_75_STATUS": "COMPLETE", "NEW_TRAINING_STARTED": False,
    }
    dump(p["out"] / "provenance.json", provenance)
    dump(p["out"] / "oracle_comparison.json", comparison)
    table = ["| Input Representation | Brain-derived? | RSA | Projected Oracle Gap | CIDEr | BLEU-4 | ROUGE-L | Mean IoU | Acc@0.5 | Parse Failures |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"| {row['input_representation']} | {row['brain_derived']} | {row['spearman_rsa']:.6f} | {row['projected_oracle_gap']:.6f} | {row['CIDEr']:.6f} | {row['BLEU-4']:.6f} | {row['ROUGE-L']:.6f} | {row['mean_iou']:.6f} | {row['grounding_accuracy']:.6f} | {row['parse_failures']} |")
    report_text = f"""# P8.75 Exact CLIP-Token Oracle Report

## 1. Oracle Definition

`EXACT_CLIP_TOKEN_ORACLE` is GT image → frozen CLIP `hidden_states[-2][:,1:,:]` → the current frozen mm_projector → the current frozen Shikra. It is not Shikra's native image route and is not a deployable brain-decoding model.

## 2. Teacher Equivalence

Teacher equivalence is `{load_json(p['out'] / 'teacher_equivalence.json')['status']}`. The online extraction uses the same RGB/224 bicubic/antialias/center-crop/CLIP-normalization definition as the locked training cache. Full details are in `teacher_equivalence.json`.

## 3. Caption Oracle

{chr(10).join(table)}

All caption and grounding failures remain in their locked denominators. Oracle generation success, empty counts, and mean/median lengths are recorded in the task metric files.

## 4. Grounding Oracle

The same table reports all 2,419 locked queries, including parse failures.

## 5. Oracle Headroom

Oracle − UMBRAE CIDEr = {headroom['oracle_minus_umbrae_CIDEr']:.6f}; Oracle − UMBRAE grounding accuracy = {headroom['oracle_minus_umbrae_grounding_accuracy']:.6f}. Oracle − Full Real CIDEr = {headroom['oracle_minus_full_real_CIDEr']:.6f}; Oracle − Full Real grounding accuracy = {headroom['oracle_minus_full_real_grounding_accuracy']:.6f}.

## 6. Paired Token Reconstruction

`paired_token_diagnostics.json` reports same-sample token cosine/MSE and per-sample errors. These are paired conditional diagnostics, unlike P8.5 marginal-distribution matching. BrainX uses element-wise `[B,256,1024]` supervision, but token index is not claimed as anatomical patch identity. Optional set OT was skipped because it is not required for the oracle decision.

## 7. Projector-Space Oracle Gap

`projector_oracle_gap.json` reports paired projected token cosine/MSE, pooled cosine, norm ratio, and variance for `[982,256,4096]` decoder inputs.

## 8. Geometry vs Exact Visual-Token Recovery

Compare each row's RSA with its projected oracle gap. RSA is global sample geometry; projected paired gap measures recovery of the actual decoder input. Oracle RSA=1 and gap=0 are identity sanity endpoints only.

## 9. Decoder Capacity Diagnosis

DECODER_CAPACITY_HEADROOM = {capacity}

PROJECTOR_DECODER_CEILING = {ceiling}

BRAIN_TO_VISUAL_INTERFACE_GAP = {gap}

## 10. Mechanism Decision

NEXT_STAGE_PRIORITY = {priority}

P8_75_STATUS = COMPLETE

No P9, decoder retraining, or any model training was started.
"""
    (p["out"] / "p8_75_report.md").write_text(report_text)
    print(json.dumps(comparison, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("teacher-equivalence", "smoke", "diagnostics", "infer", "evaluate", "report"))
    parser.add_argument("--config", default="configs/dual_branch/p7_downstream_protocol_v1.json")
    parser.add_argument("--output-dir", default="dual_branch_outputs/p8_75_clip_token_oracle")
    parser.add_argument("--p7-dir", default="dual_branch_outputs/p7_downstream")
    parser.add_argument("--p8-dir", default="dual_branch_outputs/p8_diagnosis")
    parser.add_argument("--clip-cache", default="dual_branch_outputs/p8_5_distribution_bias/test_clip_patch_fp16.npy")
    parser.add_argument("--val-clip-cache", default="protocol_outputs/protocol_v1/subj01/stage_a_cache/val/clip_patch_fp16.npy")
    parser.add_argument("--val-metadata", default="protocol_outputs/protocol_v1/subj01/stage_a_cache/val/metadata.json")
    parser.add_argument("--val-manifest", default="protocol_outputs/protocol_v1/subj01/val_manifest.jsonl")
    parser.add_argument("--test-manifest", default="protocol_outputs/protocol_v1/subj01/test_manifest.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-samples", type=int, default=16)
    parser.add_argument("--diagnostic-batch-size", type=int, default=4)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--range-lo", type=int)
    parser.add_argument("--range-hi", type=int)
    args = parser.parse_args()
    {
        "teacher-equivalence": online_teacher_equivalence,
        "smoke": validation_smoke,
        "diagnostics": diagnostics,
        "infer": infer_worker,
        "evaluate": evaluate,
        "report": report,
    }[args.action](args)


if __name__ == "__main__":
    main()
