#!/usr/bin/env python
"""Protocol-V1 formal Stage-A structural warm-up (no BrainX/Fusion/LoRA)."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.semantic_uot_loss import SemanticUOTLoss
from models.dual_branch_cache import sha256_file
from models.protocol_evaluation import retrieval_metrics, rsa_metrics
from models.protocol_metadata import git_commit_or_unavailable
from models.roi_mapping import EXPECTED_NSDGENERAL_VOXELS, load_roi_indices
from models.structural_branch import StructuralBranch


class StageACacheDataset(Dataset):
    def __init__(self, cache_dir):
        root = Path(cache_dir).expanduser().resolve()
        self.metadata = json.loads((root / "metadata.json").read_text())
        self.sample_ids = json.loads(Path(self.metadata["sample_id_index"]).read_text())
        self.fmri = np.load(self.metadata["fmri_path"], mmap_mode="r")
        self.patches = np.load(self.metadata["clip_path"], mmap_mode="r")
        if len(self.sample_ids) != len(self.fmri) or len(self.fmri) != len(self.patches):
            raise ValueError("Cache sample counts disagree")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        return {
            "sample_id": self.sample_ids[index],
            "fmri": np.array(self.fmri[index], dtype=np.float32, copy=True),
            "v_patch": np.array(self.patches[index], dtype=np.float16, copy=True),
        }


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args()
    defaults = json.loads(Path(known.config).read_text()) if known.config else {}
    parser = argparse.ArgumentParser(parents=[pre])
    required = not bool(defaults)
    parser.add_argument("--protocol", required=required)
    parser.add_argument("--mapping-kind", choices=["real", "random"], required=required)
    parser.add_argument("--roi-mapping", required=required)
    parser.add_argument("--train-cache", required=required)
    parser.add_argument("--val-cache", required=required)
    parser.add_argument("--output-dir", required=required)
    parser.add_argument("--experiment-name", required=required)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--sinkhorn-iters", type=int, default=100)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    missing = [name for name in ("protocol", "mapping_kind", "roi_mapping", "train_cache", "val_cache", "output_dir", "experiment_name") if not getattr(args, name, None)]
    if missing:
        parser.error("missing configuration: " + ", ".join(missing))
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def active_mapping(args):
    path = Path(args.roi_mapping).expanduser().resolve()
    if args.mapping_kind == "real":
        values, _ = load_roi_indices(
            str(path), expected_voxel_count=EXPECTED_NSDGENERAL_VOXELS[1],
            expected_subject="subj01", strict=True,
        )
    else:
        payload = json.loads(path.read_text())
        values = payload["rois"]
    return path, values


def parameter_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def token_diagnostics(tokens):
    values = tokens.float()
    mean_norm = values.norm(dim=-1).mean()
    normalized = F.normalize(values, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    count = similarity.shape[1]
    off_diagonal = (similarity.sum(dim=(1, 2)) - similarity.diagonal(dim1=1, dim2=2).sum(dim=1)) / (count * (count - 1))
    return mean_norm, off_diagonal.mean()


def accumulator():
    return {key: 0.0 for key in (
        "uot_loss", "transport_mass", "normalized_transport_entropy",
        "max_transport_fraction", "mean_roi_token_norm", "mean_pairwise_roi_cosine",
    )}


def update_accumulator(total, uot, tokens, batch_size):
    norm, cosine = token_diagnostics(tokens)
    values = {
        "uot_loss": uot["loss"],
        "transport_mass": uot["transport_mass"],
        "normalized_transport_entropy": uot["normalized_transport_entropy"],
        "max_transport_fraction": uot["max_transport_fraction"],
        "mean_roi_token_norm": norm,
        "mean_pairwise_roi_cosine": cosine,
    }
    for key, value in values.items():
        scalar = float(value.detach())
        if not np.isfinite(scalar):
            raise RuntimeError(f"Non-finite {key}: {scalar}")
        total[key] += scalar * batch_size


def finish_accumulator(total, count):
    return {key: value / count for key, value in total.items()}


def evaluate(model, criterion, loader, device, amp):
    model.eval()
    total, count = accumulator(), 0
    structural, visual = [], []
    with torch.inference_mode():
        for batch in loader:
            fmri = batch["fmri"].to(device, non_blocking=True)
            patches = batch["v_patch"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                h_struct = model(fmri)["h_struct"]
            uot = criterion(h_struct, patches)
            size = fmri.shape[0]
            update_accumulator(total, uot, h_struct, size)
            count += size
            structural.append(F.normalize(h_struct.float().mean(dim=1), dim=-1).cpu())
            visual.append(F.normalize(patches.float().mean(dim=1), dim=-1).cpu())
    q_struct = torch.cat(structural).numpy()
    q_visual = torch.cat(visual).numpy()
    result = finish_accumulator(total, count)
    result["sample_count"] = count
    result["retrieval"] = retrieval_metrics(q_struct, q_visual)
    result["rsa"] = rsa_metrics(q_struct, q_visual)
    return result


def save_checkpoint(path, model, optimizer, scaler, epoch, best, best_epoch, bad_epochs, elapsed):
    torch.save({
        "structural_branch": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "best_val_uot": best,
        "best_epoch": best_epoch, "bad_epochs": bad_epochs,
        "elapsed_seconds": elapsed,
    }, path)


def run(args):
    seed_everything(args.seed)
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_version"] != "protocol_v1":
        raise ValueError("Stage A is locked to protocol_v1")
    if args.seed != protocol["primary_seed"]:
        raise ValueError("Stage A seed must match Protocol V1")
    locked = {
        "lr": 1e-4, "weight_decay": 1e-4, "max_epochs": 20,
        "min_epochs": 5, "patience": 5, "gradient_clip_norm": 1.0,
        "epsilon": 0.05, "tau": 1.0, "sinkhorn_iters": 100,
        "sinkhorn_tol": 1e-5,
    }
    mismatches = {key: (getattr(args, key), value) for key, value in locked.items() if getattr(args, key) != value}
    if mismatches or args.batch_size not in {32, 64}:
        raise ValueError(f"P3 locked training configuration mismatch: {mismatches}, batch_size={args.batch_size}")
    if args.experiment_name not in {
        "dualbranch_s1_stageA_real_seed42_protocolv1",
        "dualbranch_s1_stageA_random42_seed42_protocolv1",
    }:
        raise ValueError("Experiment name is not a locked P3 identity")
    output = Path(args.output_dir).expanduser().resolve() / args.experiment_name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pth").exists() and not args.resume:
        raise FileExistsError(f"Existing run requires --resume: {output}")
    mapping_path, roi_indices = active_mapping(args)
    # Re-seeding immediately before construction guarantees identical learned
    # parameter initialization for real and random mappings.
    seed_everything(args.seed)
    model = StructuralBranch(token_dim=1024, roi_indices=roi_indices).to(args.device)
    report = model.parameter_report()
    print(json.dumps({"structural_total_params": report["total"], "structural_trainable_params": report["trainable"]}), flush=True)
    if report["total"] != report["trainable"]:
        raise RuntimeError("All and only StructuralBranch parameters must be trainable")
    init_hash = parameter_sha256(model)
    criterion = SemanticUOTLoss(args.epsilon, args.tau, args.sinkhorn_iters, args.sinkhorn_tol)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = bool(args.amp and str(args.device).startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    train_data, val_data = StageACacheDataset(args.train_cache), StageACacheDataset(args.val_cache)
    if len(train_data) != protocol["train_split"]["sample_count"] or len(val_data) != protocol["val_split"]["sample_count"]:
        raise ValueError("Cache sample counts do not match Protocol V1")
    expected_cache = {
        "train": (train_data, protocol["train_split"]["manifest_sha256"]),
        "val": (val_data, protocol["val_split"]["manifest_sha256"]),
    }
    for split, (dataset, manifest_hash) in expected_cache.items():
        metadata = dataset.metadata
        if (
            metadata.get("protocol_version") != protocol["protocol_version"]
            or metadata.get("split") != split
            or metadata.get("manifest_sha256") != manifest_hash
            or metadata.get("clip_model") != protocol["clip_model"]
            or metadata.get("clip_feature_definition") != protocol["clip_feature_definition"]
            or metadata.get("repeat_policy") != protocol["repeat_policy"]["train" if split == "train" else "validation"]
        ):
            raise ValueError(f"{split} cache violates Protocol V1 metadata")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=amp)
    config = vars(args).copy()
    for key in ("protocol", "roi_mapping", "train_cache", "val_cache", "output_dir"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    config.update({
        "objective": "semantic_uot_only", "repeat_policy": "mean_over_num_uniques_valid_repeats",
        "structural_parameter_report": report, "initial_parameter_sha256": init_hash,
        "brainx_used": False, "fusion_used": False, "lora_used": False, "test_used": False,
    })
    (output / "config.json").write_text(json.dumps(config, indent=2))
    start_epoch, best, best_epoch, bad_epochs, elapsed_before = 1, float("inf"), 0, 0, 0.0
    if args.resume:
        state = torch.load(output / "last.pth", map_location=args.device)
        model.load_state_dict(state["structural_branch"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best, best_epoch, bad_epochs = state["best_val_uot"], state["best_epoch"], state["bad_epochs"]
        elapsed_before = state.get("elapsed_seconds", 0.0)
    initial_path = output / "initial_val_metrics.json"
    if initial_path.exists():
        initial = json.loads(initial_path.read_text())
    else:
        initial = evaluate(model, criterion, val_loader, args.device, amp)
        initial_path.write_text(json.dumps(initial, indent=2))
    print(json.dumps({"initial_validation": initial}), flush=True)
    run_start = time.time()
    records_path = output / "metrics_per_epoch.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() and args.resume else []
    for epoch in range(start_epoch, args.max_epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, pin_memory=amp)
        model.train()
        total, grad_total, count, steps = accumulator(), 0.0, 0, 0
        for batch in train_loader:
            fmri = batch["fmri"].to(args.device, non_blocking=True)
            patches = batch["v_patch"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                h_struct = model(fmri)["h_struct"]
            uot = criterion(h_struct, patches)
            scaler.scale(uot["loss"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}")
            scaler.step(optimizer)
            scaler.update()
            size = fmri.shape[0]
            update_accumulator(total, uot, h_struct, size)
            grad_total += float(grad_norm) * size
            count += size
            steps += 1
        train_metrics = finish_accumulator(total, count)
        train_metrics["structural_grad_norm"] = grad_total / count
        validation = evaluate(model, criterion, val_loader, args.device, amp)
        improved = validation["uot_loss"] < best
        if improved:
            best, best_epoch, bad_epochs = validation["uot_loss"], epoch, 0
        else:
            bad_epochs += 1
        elapsed = elapsed_before + time.time() - run_start
        record = {
            "epoch": epoch, "train_uot_loss": train_metrics["uot_loss"],
            "val_uot_loss": validation["uot_loss"], "learning_rate": optimizer.param_groups[0]["lr"],
            "structural_grad_norm": train_metrics["structural_grad_norm"],
            "transport_mass": validation["transport_mass"],
            "normalized_transport_entropy": validation["normalized_transport_entropy"],
            "max_transport_fraction": validation["max_transport_fraction"],
            "mean_roi_token_norm": validation["mean_roi_token_norm"],
            "mean_pairwise_roi_cosine": validation["mean_pairwise_roi_cosine"],
            "val_spearman_rsa": validation["rsa"]["spearman_rsa"],
            "val_pearson_rsa": validation["rsa"]["pearson_rsa"],
            "val_retrieval": validation["retrieval"], "checkpoint_selection_metric": "val_uot_loss",
        }
        records.append(record)
        records_path.write_text("".join(json.dumps(value) + "\n" for value in records))
        save_checkpoint(output / "last.pth", model, optimizer, scaler, epoch, best, best_epoch, bad_epochs, elapsed)
        if improved:
            save_checkpoint(output / "best.pth", model, optimizer, scaler, epoch, best, best_epoch, bad_epochs, elapsed)
        print(json.dumps(record), flush=True)
        if epoch >= args.min_epochs and bad_epochs >= args.patience:
            print(json.dumps({"early_stop": epoch, "patience": args.patience}), flush=True)
            break
    total_elapsed = elapsed_before + time.time() - run_start
    best_state = torch.load(output / "best.pth", map_location=args.device)
    model.load_state_dict(best_state["structural_branch"])
    final = evaluate(model, criterion, val_loader, args.device, amp)
    final.update({"best_epoch": best_state["best_epoch"], "best_val_uot": best_state["best_val_uot"]})
    (output / "final_val_metrics.json").write_text(json.dumps(final, indent=2))
    provenance = {
        "protocol_version": protocol["protocol_version"], "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(str(protocol_path)),
        "git_commit_hash": git_commit_or_unavailable(str(Path(__file__).resolve().parents[2])),
        "brain_checkpoint_sha256_locked_but_not_used": protocol["brain_checkpoint_sha256"],
        "active_roi_mapping": str(mapping_path), "active_roi_mapping_sha256": sha256_file(str(mapping_path)),
        "real_roi_mapping_sha256": protocol["roi_mapping_sha256"], "seed": args.seed,
        "initial_parameter_sha256": init_hash, "gpu": torch.cuda.get_device_name(0) if amp else "CPU",
        "training_seconds": total_elapsed,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "metrics.json").write_text(json.dumps({"initial": initial, "final": final}, indent=2))
    print(json.dumps({"final_validation": final, "provenance": provenance}), flush=True)
    return final


if __name__ == "__main__":
    run(parse_args())
