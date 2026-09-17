#!/usr/bin/env python
"""Inspect Stage-1 checkpoint completeness and retrieval compatibility."""

import argparse
import json
from pathlib import Path

import torch


REQUIRED_RETRIEVAL_CONFIG = (
    "router_type",
    "selected_clip_layers",
    "roi_token_dim",
    "clip_layer_target_dim",
    "use_brain_clip_projector",
    "contrastive_loss_weight",
    "contrastive_temperature",
    "contrastive_brain_pooling",
    "contrastive_image_target",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--brain-pooling",
        choices=["mean", "attention", "roi_flatten_mlp"],
        default="mean",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def inspect_checkpoint(checkpoint_path: Path, requested_pooling: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    state = checkpoint.get("model", checkpoint)
    config = checkpoint.get("config", {})
    prefixes = {
        "roi_tokenizer": "roi_tokenizer.",
        "brain_clip_projector": "brain_clip_projector.",
        "roi_layer_router": "router.",
        "retrieval_pooler": "retrieval_pooler.",
        "attention_pooler": "retrieval_pooler.attention_score.",
        "flatten_mlp_pooler": "retrieval_pooler.flatten_mlp.",
    }
    module_status = {
        name: {
            "present": any(key.startswith(prefix) for key in state),
            "num_state_entries": sum(
                key.startswith(prefix) for key in state
            ),
        }
        for name, prefix in prefixes.items()
    }
    run_dir = checkpoint_path.parent
    train_metrics = load_jsonl(run_dir / "metrics_train.jsonl")
    val_metrics = load_jsonl(run_dir / "metrics_val.jsonl")
    best_epoch = checkpoint.get("epoch")
    if val_metrics and "total_loss" in val_metrics[0]:
        best_epoch_from_metrics = min(
            val_metrics, key=lambda row: row["total_loss"]
        ).get("epoch")
    else:
        best_epoch_from_metrics = None
    missing_config = [
        key for key in REQUIRED_RETRIEVAL_CONFIG if key not in config
    ]
    warnings = []
    if requested_pooling == "attention" and not module_status[
        "attention_pooler"
    ]["present"]:
        warnings.append(
            "Requested attention pooling but no attention pooler weights exist."
        )
    if requested_pooling == "roi_flatten_mlp" and not module_status[
        "flatten_mlp_pooler"
    ]["present"]:
        warnings.append(
            "Requested roi_flatten_mlp but no flatten MLP weights exist."
        )
    contrastive_weight = float(config.get("contrastive_loss_weight", 0.0))
    if contrastive_weight > 0 and not module_status[
        "retrieval_pooler"
    ]["present"]:
        pooling = config.get("contrastive_brain_pooling", "mean")
        if pooling != "mean":
            warnings.append(
                "Contrastive loss was enabled but no parameterized retrieval "
                "pooler weights were found."
            )
    if missing_config:
        warnings.append(
            "Checkpoint lacks retrieval config fields: "
            + ", ".join(missing_config)
        )
    if config.get("debug_max_steps", 0):
        warnings.append(
            f"Training used debug_max_steps={config['debug_max_steps']}; "
            "this may be a smoke run rather than a stable checkpoint."
        )
    compatible = requested_pooling == "mean" or (
        requested_pooling == config.get("contrastive_brain_pooling")
        and contrastive_weight > 0
        and (
            module_status["attention_pooler"]["present"]
            if requested_pooling == "attention"
            else module_status["flatten_mlp_pooler"]["present"]
        )
    )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_keys": list(checkpoint.keys()),
        "num_model_state_entries": len(state),
        "modules": module_status,
        "config": config,
        "missing_retrieval_config_fields": missing_config,
        "metrics_train_path": str(run_dir / "metrics_train.jsonl"),
        "metrics_val_path": str(run_dir / "metrics_val.jsonl"),
        "train_metric_keys": (
            sorted(train_metrics[-1].keys()) if train_metrics else []
        ),
        "val_metric_keys": (
            sorted(val_metrics[-1].keys()) if val_metrics else []
        ),
        "num_train_epochs_logged": len(train_metrics),
        "num_val_epochs_logged": len(val_metrics),
        "best_epoch_checkpoint": best_epoch,
        "best_epoch_from_val_total": best_epoch_from_metrics,
        "final_epoch": (
            val_metrics[-1].get("epoch") if val_metrics else best_epoch
        ),
        "requested_brain_pooling": requested_pooling,
        "requested_pooling_compatible": compatible,
        "warnings": warnings,
    }


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    report = inspect_checkpoint(checkpoint_path, args.brain_pooling)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "checkpoint_diagnostic.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    for warning in report["warnings"]:
        print(f"[WARN] {warning}")
    print(f"Saved diagnostic report to {output_path}")


if __name__ == "__main__":
    main()
