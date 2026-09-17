#!/usr/bin/env python
"""P3-R ROI-specific relational training under locked Protocol V1."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.roi_relational_loss import ROIRelationalLoss, SymmetricInfoNCELoss
from models.dual_branch_cache import sha256_file
from models.protocol_evaluation import _average_ranks, _pearson, retrieval_metrics, rsa_metrics
from models.protocol_metadata import git_commit_or_unavailable
from models.roi_specific_structural_branch import ROISpecificStructuralBranch, StructuralAttentionPool
from scripts.train_stage_a_structural import active_mapping, parameter_sha256, seed_everything, token_diagnostics


ROI_NAMES = ["V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA"]
CLIP_LAYERS = [4, 8, 12, 16, 20, 24]


class P3RDataset(Dataset):
    def __init__(self, stage_a_cache, relational_cache):
        base = json.loads((Path(stage_a_cache).resolve() / "metadata.json").read_text())
        relation = json.loads((Path(relational_cache).resolve() / "metadata.json").read_text())
        self.metadata, self.relational_metadata = base, relation
        self.sample_ids = json.loads(Path(base["sample_id_index"]).read_text())
        relation_ids = json.loads(Path(relation["sample_ids"]).read_text())
        if self.sample_ids != relation_ids:
            raise ValueError("Stage-A and P3-R cache sample IDs disagree")
        self.fmri = np.load(base["fmri_path"], mmap_mode="r")
        self.layer_pools = np.load(relation["layer_pools_path"], mmap_mode="r")
        self.global_visual = np.load(relation["global_visual_path"], mmap_mode="r")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        return {
            "fmri": np.array(self.fmri[index], dtype=np.float32, copy=True),
            "layer_pools": np.array(self.layer_pools[index], dtype=np.float16, copy=True),
            "global_visual": np.array(self.global_visual[index], dtype=np.float16, copy=True),
        }


class P3RModel(nn.Module):
    def __init__(self, roi_indices):
        super().__init__()
        self.structural = ROISpecificStructuralBranch(roi_indices, hidden_dim=256, token_dim=1024)
        self.pool = StructuralAttentionPool(1024)

    def forward(self, fmri):
        tokens = self.structural(fmri)["h_struct"]
        pooled = self.pool(tokens)
        return {"h_struct": tokens, "q_struct": pooled["pooled"], "attention_weights": pooled["attention_weights"]}


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args()
    defaults = json.loads(Path(known.config).read_text()) if known.config else {}
    parser = argparse.ArgumentParser(parents=[pre])
    required = not bool(defaults)
    for name in ("protocol", "mapping_kind", "roi_mapping", "train_cache", "val_cache", "train_relational_cache", "val_relational_cache", "routing_weights", "router_checkpoint", "output_dir", "experiment_name"):
        parser.add_argument("--" + name.replace("_", "-"), required=required)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--lambda-global", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--rdm-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(**defaults)
    return parser.parse_args()


def load_routing(args):
    path = Path(args.routing_weights).expanduser().resolve()
    weights = np.load(path, allow_pickle=False).astype(np.float32)
    root = path.parent
    names = json.loads((root / "roi_names.json").read_text())
    layers = json.loads((root / "selected_clip_layers.json").read_text())
    if names != ROI_NAMES or layers != CLIP_LAYERS or weights.shape != (8, 6):
        raise ValueError("Router prior order/shape violates P3-R")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(1), 1, atol=1e-6):
        raise ValueError("Router weights are not valid probabilities")
    return path, torch.from_numpy(weights)


def roi_rsa(tokens, layer_pools, routing):
    brain = F.normalize(tokens.float(), dim=-1).cpu().numpy().transpose(1, 0, 2)
    visual = F.normalize(layer_pools.float(), dim=-1).cpu().numpy().transpose(1, 0, 2)
    layer_rdms = np.stack([1.0 - value @ value.T for value in visual])
    target_rdms = np.einsum("rl,lij->rij", routing.cpu().numpy(), layer_rdms)
    upper = np.triu_indices(tokens.shape[0], k=1)
    per_roi = {}
    for index, name in enumerate(ROI_NAMES):
        left = (1.0 - brain[index] @ brain[index].T)[upper]
        right = target_rdms[index][upper]
        per_roi[name] = {
            "spearman": _pearson(_average_ranks(left), _average_ranks(right)),
            "pearson": _pearson(left, right),
        }
    return {
        "per_roi": per_roi,
        "mean_spearman": float(np.mean([value["spearman"] for value in per_roi.values()])),
        "mean_pearson": float(np.mean([value["pearson"] for value in per_roi.values()])),
    }


def diagnostics(tokens):
    norm, cosine = token_diagnostics(tokens)
    variance = tokens.float().var(dim=0, unbiased=False).mean()
    return norm, cosine, variance


def evaluate(model, relational, contrastive, loader, routing, device, amp, lambda_global):
    model.eval()
    sums = {key: 0.0 for key in ("total_loss", "roi_rel_loss", "global_contrastive_loss", "mean_roi_token_norm", "mean_pairwise_roi_cosine", "across_sample_variance")}
    per_roi_loss = np.zeros(8, dtype=np.float64)
    count = 0
    all_tokens, all_layers, all_struct, all_visual = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            fmri = batch["fmri"].to(device, non_blocking=True)
            layers = batch["layer_pools"].to(device, non_blocking=True)
            visual = batch["global_visual"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                output = model(fmri)
            relation = relational(output["h_struct"], layers, routing)
            global_loss = contrastive(output["q_struct"], visual)
            total = relation["loss"] + lambda_global * global_loss
            norm, cosine, variance = diagnostics(output["h_struct"])
            size = len(fmri)
            values = {"total_loss": total, "roi_rel_loss": relation["loss"], "global_contrastive_loss": global_loss, "mean_roi_token_norm": norm, "mean_pairwise_roi_cosine": cosine, "across_sample_variance": variance}
            for key, value in values.items():
                sums[key] += float(value) * size
            per_roi_loss += relation["per_roi_loss"].cpu().numpy() * size
            count += size
            all_tokens.append(output["h_struct"].float().cpu())
            all_layers.append(layers.float().cpu())
            all_struct.append(F.normalize(output["q_struct"].float(), dim=-1).cpu())
            all_visual.append(F.normalize(visual.float(), dim=-1).cpu())
    tokens, layers = torch.cat(all_tokens), torch.cat(all_layers)
    q_struct, q_visual = torch.cat(all_struct).numpy(), torch.cat(all_visual).numpy()
    result = {key: value / count for key, value in sums.items()}
    result.update({
        "sample_count": count, "per_roi_rel_loss": {name: float(per_roi_loss[i] / count) for i, name in enumerate(ROI_NAMES)},
        "global_rsa": rsa_metrics(q_struct, q_visual),
        "retrieval": retrieval_metrics(q_struct, q_visual),
        "roi_wise_rsa": roi_rsa(tokens, layers, routing),
    })
    return result


def save_checkpoint(path, model, optimizer, scaler, epoch, best, best_epoch, bad_epochs, elapsed):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best_val_total_loss": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "elapsed_seconds": elapsed}, path)


def run(args):
    locked = {"batch_size": 64, "max_epochs": 20, "min_epochs": 5, "patience": 5, "lr": 1e-4, "weight_decay": 1e-4, "gradient_clip_norm": 1.0, "lambda_global": 0.1, "temperature": 0.07, "rdm_eps": 1e-6, "seed": 42}
    mismatch = {key: (getattr(args, key), value) for key, value in locked.items() if getattr(args, key) != value}
    if mismatch:
        raise ValueError(f"P3-R locked configuration mismatch: {mismatch}")
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_version"] != "protocol_v1":
        raise ValueError("P3-R requires protocol_v1")
    allowed = {"dualbranch_s1_stageA_relational_real_seed42_protocolv1", "dualbranch_s1_stageA_relational_random42_seed42_protocolv1"}
    if args.experiment_name not in allowed:
        raise ValueError("Unexpected P3-R experiment identity")
    routing_path, routing = load_routing(args)
    routing = routing.to(args.device)
    mapping_path, mapping = active_mapping(args)
    seed_everything(args.seed)
    model = P3RModel(mapping).to(args.device)
    initial_hash = parameter_sha256(model)
    report = model.structural.parameter_report()
    report["attention_pool"] = sum(p.numel() for p in model.pool.parameters())
    report["total_with_pool"] = sum(p.numel() for p in model.parameters())
    print(json.dumps({"parameter_report": report, "initial_parameter_sha256": initial_hash}), flush=True)
    relational = ROIRelationalLoss(args.rdm_eps)
    contrastive = SymmetricInfoNCELoss(args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = bool(args.amp and str(args.device).startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    train_data = P3RDataset(args.train_cache, args.train_relational_cache)
    val_data = P3RDataset(args.val_cache, args.val_relational_cache)
    if len(train_data) != 8559 or len(val_data) != 300:
        raise ValueError("P3-R cache count violates protocol")
    for split, dataset, expected in (("train", train_data, protocol["train_split"]["manifest_sha256"]), ("val", val_data, protocol["val_split"]["manifest_sha256"])):
        if dataset.metadata["manifest_sha256"] != expected or dataset.relational_metadata["manifest_sha256"] != expected or dataset.relational_metadata["selected_clip_layers"] != CLIP_LAYERS:
            raise ValueError(f"{split} cache metadata violates protocol")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=amp)
    output_dir = Path(args.output_dir).resolve() / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "last.pth").exists() and not args.resume:
        raise FileExistsError("Existing run requires --resume")
    config = vars(args).copy()
    for key in ("protocol", "roi_mapping", "train_cache", "val_cache", "train_relational_cache", "val_relational_cache", "routing_weights", "router_checkpoint", "output_dir"):
        config[key] = str(Path(config[key]).resolve())
    config.update({"objective": "standardized_roi_relational_plus_0.1_symmetric_infonce", "uot_used": False, "brainx_used": False, "fusion_used": False, "lora_used": False, "test_used": False, "router_prior_available": True, "selected_clip_layers": CLIP_LAYERS, "parameter_report": report, "initial_parameter_sha256": initial_hash})
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    start_epoch, best, best_epoch, bad_epochs, elapsed_before = 1, float("inf"), 0, 0, 0.0
    if args.resume:
        state = torch.load(output_dir / "last.pth", map_location=args.device)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        start_epoch, best, best_epoch, bad_epochs, elapsed_before = state["epoch"] + 1, state["best_val_total_loss"], state["best_epoch"], state["bad_epochs"], state.get("elapsed_seconds", 0.0)
    initial_path = output_dir / "initial_val_metrics.json"
    initial = json.loads(initial_path.read_text()) if initial_path.exists() else evaluate(model, relational, contrastive, val_loader, routing, args.device, amp, args.lambda_global)
    initial_path.write_text(json.dumps(initial, indent=2))
    print(json.dumps({"initial_validation": initial}), flush=True)
    records_path = output_dir / "metrics_per_epoch.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if args.resume and records_path.exists() else []
    started = time.time()
    for epoch in range(start_epoch, args.max_epochs + 1):
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + epoch), num_workers=args.num_workers, pin_memory=amp)
        model.train(); sums = np.zeros(7); count = 0
        for batch in train_loader:
            fmri = batch["fmri"].to(args.device, non_blocking=True); layers = batch["layer_pools"].to(args.device, non_blocking=True); visual = batch["global_visual"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): output = model(fmri)
            relation = relational(output["h_struct"], layers, routing); global_loss = contrastive(output["q_struct"], visual); total = relation["loss"] + args.lambda_global * global_loss
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            if not torch.isfinite(grad) or not torch.isfinite(total): raise RuntimeError("Non-finite P3-R optimization")
            scaler.step(optimizer); scaler.update()
            norm, cosine, variance = diagnostics(output["h_struct"]); size = len(fmri)
            sums += np.asarray([float(total), float(relation["loss"]), float(global_loss), float(grad), float(norm), float(cosine), float(variance)]) * size; count += size
        validation = evaluate(model, relational, contrastive, val_loader, routing, args.device, amp, args.lambda_global)
        improved = validation["total_loss"] < best
        if improved: best, best_epoch, bad_epochs = validation["total_loss"], epoch, 0
        else: bad_epochs += 1
        elapsed = elapsed_before + time.time() - started
        means = sums / count
        record = {"epoch": epoch, "train_total_loss": means[0], "train_roi_rel_loss": means[1], "train_global_contrastive_loss": means[2], "structural_grad_norm": means[3], "train_mean_roi_token_norm": means[4], "train_mean_pairwise_roi_cosine": means[5], "train_across_sample_variance": means[6], "val_total_loss": validation["total_loss"], "val_roi_rel_loss": validation["roi_rel_loss"], "val_global_contrastive_loss": validation["global_contrastive_loss"], "val_global_spearman_rsa": validation["global_rsa"]["spearman_rsa"], "val_global_pearson_rsa": validation["global_rsa"]["pearson_rsa"], "val_mean_roi_spearman_rsa": validation["roi_wise_rsa"]["mean_spearman"], "val_retrieval": validation["retrieval"], "val_mean_roi_token_norm": validation["mean_roi_token_norm"], "val_mean_pairwise_roi_cosine": validation["mean_pairwise_roi_cosine"], "val_across_sample_variance": validation["across_sample_variance"], "checkpoint_selection_metric": "val_total_loss"}
        records.append(record); records_path.write_text("".join(json.dumps(value) + "\n" for value in records))
        save_checkpoint(output_dir / "last.pth", model, optimizer, scaler, epoch, best, best_epoch, bad_epochs, elapsed)
        if improved: save_checkpoint(output_dir / "best.pth", model, optimizer, scaler, epoch, best, best_epoch, bad_epochs, elapsed)
        print(json.dumps(record), flush=True)
        if epoch >= args.min_epochs and bad_epochs >= args.patience:
            print(json.dumps({"early_stop": epoch}), flush=True); break
    total_elapsed = elapsed_before + time.time() - started
    state = torch.load(output_dir / "best.pth", map_location=args.device); model.load_state_dict(state["model"])
    final = evaluate(model, relational, contrastive, val_loader, routing, args.device, amp, args.lambda_global)
    final.update({"best_epoch": state["best_epoch"], "best_val_total_loss": state["best_val_total_loss"]})
    (output_dir / "final_val_metrics.json").write_text(json.dumps(final, indent=2))
    provenance = {"protocol_version": "protocol_v1", "protocol_sha256": sha256_file(str(protocol_path)), "git_commit_hash": git_commit_or_unavailable(str(Path(__file__).resolve().parents[2])), "active_roi_mapping": str(mapping_path), "active_roi_mapping_sha256": sha256_file(str(mapping_path)), "routing_weights": str(routing_path), "routing_weights_sha256": sha256_file(str(routing_path)), "router_checkpoint": str(Path(args.router_checkpoint).resolve()), "router_checkpoint_sha256": sha256_file(args.router_checkpoint), "router_prior_available": True, "seed": args.seed, "initial_parameter_sha256": initial_hash, "gpu": torch.cuda.get_device_name(0) if amp else "CPU", "training_seconds": total_elapsed}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2)); (output_dir / "metrics.json").write_text(json.dumps({"initial": initial, "final": final}, indent=2))
    print(json.dumps({"final_validation": final, "provenance": provenance}), flush=True)
    return final


if __name__ == "__main__":
    run(parse_args())
