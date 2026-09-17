#!/usr/bin/env python
"""Locked validation downstream evaluation and selection for P10."""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BrainHub"))
from eval_bbox_rec import calculate_metric
from losses.interface_distillation_loss import InterfaceDistillationLoss
from models.token_mixing_interface_adapter import TokenMixingInterfaceAdapter
from scripts.cache_p7_downstream_features import load_projector, resolve_config
from scripts.run_p7_downstream import core_caption_metrics, full_prompt, generate, load_decoder
from scripts.train_p10_tokenmix_adapter import evaluate as evaluate_interface, load_config
from utils import extract_boxes, extract_id_bbox_caption


def load_json(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def lock_p9_baseline(cfg):
    out = Path(cfg["output_dir"])
    p9 = Path(cfg["p9_output_dir"])
    interface = load_json(p9 / "source_interface_baseline.json")
    interface["reused_from"] = str((p9 / "source_interface_baseline.json").resolve())
    interface["reuse_reason"] = "identical frozen source cache and frozen mm_projector"
    downstream = load_json(p9 / "source_downstream_baseline.json")
    downstream["reused_from"] = str((p9 / "source_downstream_baseline.json").resolve())
    downstream["reuse_reason"] = "P9 locked validation baseline; downstream path unchanged in P10"
    dump(out / "source_interface_baseline.json", interface)
    dump(out / "source_downstream_baseline.json", downstream)
    path = out / "validation_downstream.jsonl"
    existing = [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]
    if not any(row["label"] == "source_baseline" for row in existing):
        with path.open("a") as handle:
            handle.write(json.dumps(downstream) + "\n")
    print(json.dumps({"experiment": cfg["experiment_name"], "baseline_locked_from": str(p9), "test_used": False}))


def checkpoint_labels(out, every):
    checkpoints = []
    for path in sorted(out.glob("epoch_*.pth")):
        epoch = int(path.stem.split("_")[-1])
        if epoch % every == 0:
            checkpoints.append((f"epoch_{epoch:03d}", path, epoch))
    last = torch.load(out / "last.pth", map_location="cpu")["epoch"]
    last_path = out / f"epoch_{last:03d}.pth"
    if not any(epoch == last for _, _, epoch in checkpoints):
        checkpoints.append((f"epoch_{last:03d}", last_path, last))
    return checkpoints


def adapter_for(checkpoint, cfg, device):
    adapter = TokenMixingInterfaceAdapter(
        hidden_dim=1024, num_heads=cfg["num_heads"], bottleneck_dim=cfg["ffn_bottleneck"],
        gate_logit=cfg["gate_logit_init"], dropout=cfg["dropout"],
    ).to(device).eval()
    adapter.load_state_dict(torch.load(checkpoint, map_location="cpu")["adapter"], strict=True)
    adapter.requires_grad_(False)
    return adapter


def projected(cache, indices, adapter, projector, device):
    source = torch.from_numpy(np.stack([np.asarray(cache[index], dtype=np.float32) for index in indices])).to(device)
    with torch.inference_mode():
        return projector(adapter(source))


def run_label(label, checkpoint, epoch, cfg, args):
    out = Path(cfg["output_dir"])
    downstream = resolve_config(cfg["p7_config"])
    source = np.load(cfg["source_val"], mmap_mode="r")
    references = load_json(args.caption_references)
    annotations = load_json(args.grounding_annotations)
    tokenizer, decoder, projector = load_decoder(downstream, args.device)
    adapter = adapter_for(checkpoint, cfg, args.device)
    jobs = {
        "caption": [(index, None, full_prompt(downstream["caption_prompt"])) for index in range(300)],
        "grounding": [(index, expression, full_prompt(downstream["grounding_prompt"], expression)) for index in range(300) for expression in annotations[str(index)]],
    }
    all_records = {}
    for task, items in jobs.items():
        path = out / f"validation_{task}_{label}.jsonl"
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
                for start in range(0, len(groups[length]), downstream["generation_job_batch_size"]):
                    entries = [item for item in groups[length][start:start + downstream["generation_job_batch_size"]] if (item[0], item[1]) not in existing]
                    if not entries:
                        continue
                    features = projected(source, [item[0] for item in entries], adapter, projector, args.device)
                    try:
                        responses = generate(tokenizer, decoder, features, [item[2] for item in entries], downstream, args.device)
                        error = None
                    except Exception as exc:
                        responses, error = [""] * len(entries), f"{type(exc).__name__}: {exc}"
                    for (sample, expression, _), response in zip(entries, responses):
                        common = {
                            "sample_id": sample, "label": label, "epoch": epoch,
                            "response": response, "generation_success": error is None,
                            "output_length": len(tokenizer(response, add_special_tokens=False).input_ids) if response else 0,
                            "error": error,
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
        all_records[task] = existing
        print(json.dumps({"experiment": cfg["experiment_name"], "label": label, "task": task, "completed": len(existing)}), flush=True)
    caption_rows = [all_records["caption"][(index, None)] for index in range(300)]
    caption = core_caption_metrics([references[str(index)] for index in range(300)], [row["caption"] for row in caption_rows])
    caption.update({
        "sample_count": 300, "success": sum(row["status"] == "success" for row in caption_rows),
        "empty": sum(row["status"] == "empty" for row in caption_rows),
        "failed": sum(row["status"] == "failed" for row in caption_rows),
        "mean_output_length": float(np.mean([row["output_length"] for row in caption_rows])),
        "median_output_length": float(np.median([row["output_length"] for row in caption_rows])),
    })
    predictions, targets = [], []
    for index in range(300):
        for expression, boxes in annotations[str(index)].items():
            parsed = all_records["grounding"][(index, expression)]["boxes"]
            predictions.append(None if not parsed else parsed[0][0])
            targets.append(boxes[0])
    value = calculate_metric(predictions, targets, 0.5)
    grounding = {
        "query_count": len(targets), "mean_iou": value["iou"], "grounding_accuracy": value["accuracy"],
        "parse_failures": value["failed"], "target_failures": value["target_failed"],
    }
    row = {"label": label, "epoch": epoch, "checkpoint": str(Path(checkpoint).resolve()), "caption": caption, "grounding": grounding}
    with (out / "validation_downstream.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    del adapter, projector, decoder
    torch.cuda.empty_cache()


def finalize(cfg):
    out = Path(cfg["output_dir"])
    rows = [json.loads(line) for line in (out / "validation_downstream.jsonl").read_text().splitlines()]
    unique = {row["label"]: row for row in rows}
    baseline = unique["source_baseline"]
    interface = {int(row["epoch"]): row for row in map(json.loads, (out / "metrics_per_epoch.jsonl").read_text().splitlines())}
    candidates = []
    for label, row in unique.items():
        if label == "source_baseline":
            continue
        epoch = int(row["epoch"])
        compatible = row["caption"]["CIDEr"] >= baseline["caption"]["CIDEr"] and row["grounding"]["grounding_accuracy"] >= baseline["grounding"]["grounding_accuracy"]
        candidates.append({**row, "projected_oracle_gap": interface[epoch]["projected_oracle_gap"], "downstream_compatible": compatible})
    eligible = [row for row in candidates if row["downstream_compatible"]]
    if eligible:
        selected = min(eligible, key=lambda row: (row["projected_oracle_gap"], -row["caption"]["CIDEr"]))
        status = "DOWNSTREAM_COMPATIBLE_CHECKPOINT_FOUND"
        shutil.copy2(selected["checkpoint"], out / "best.pth")
        best = torch.load(out / "best.pth", map_location="cpu")
        best["metadata"].update({"checkpoint_role": "downstream_selected", "downstream_compatible": True})
        torch.save(best, out / "best.pth")
    else:
        selected, status = None, "PROJECTED_ORACLE_GAP_NOT_SUFFICIENT"
        # Preserve the required artifact without misrepresenting it as selected.
        # Consumers must check final_val_metrics.selected and this explicit role.
        shutil.copy2(out / "best_interface.pth", out / "best.pth")
        best = torch.load(out / "best.pth", map_location="cpu")
        best["metadata"].update({"checkpoint_role": "diagnostic_interface_best_not_selected", "downstream_compatible": False})
        torch.save(best, out / "best.pth")
    result = {
        "selection_rule": "CIDEr >= source AND grounding accuracy >= source; then lowest projected oracle gap, CIDEr tie-break",
        "source_baseline": baseline, "candidates": candidates, "selected": selected,
        "selection_status": status, "test_used": False,
    }
    dump(out / "final_val_metrics.json", result)
    selected_metrics = None if selected is None else interface[selected["epoch"]]
    dump(out / "adapter_diagnostics.json", {
        "source_interface": load_json(out / "source_interface_baseline.json"),
        "selected_interface": selected_metrics,
        "selected_attention": None if selected_metrics is None else selected_metrics["attention"],
    })
    print(json.dumps({"experiment": cfg["experiment_name"], "selection_status": status, "selected": None if selected is None else selected["label"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("lock-baseline", "checkpoints", "finalize"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--caption-references", default="protocol_outputs/protocol_v1/subj01/p9_validation_targets/caption_references.json")
    parser.add_argument("--grounding-annotations", default="protocol_outputs/protocol_v1/subj01/p9_validation_targets/grounding_annotations.json")
    parser.add_argument("--downstream-every", type=int, default=2)
    args = parser.parse_args()
    cfg = load_config(args.config)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    if args.action == "lock-baseline":
        lock_p9_baseline(cfg)
        return
    if args.action == "finalize":
        finalize(cfg)
        return
    path = Path(cfg["output_dir"]) / "validation_downstream.jsonl"
    if not path.exists() or not any(json.loads(line)["label"] == "source_baseline" for line in path.read_text().splitlines()):
        raise RuntimeError("Run lock-baseline before checkpoint evaluation")
    existing = {json.loads(line)["label"] for line in path.read_text().splitlines()}
    for label, checkpoint, epoch in checkpoint_labels(Path(cfg["output_dir"]), args.downstream_every):
        if label not in existing:
            run_label(label, checkpoint, epoch, cfg, args)


if __name__ == "__main__":
    main()
