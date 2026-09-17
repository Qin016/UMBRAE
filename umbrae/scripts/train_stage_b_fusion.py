#!/usr/bin/env python
"""P4 frozen-branch gated fusion calibration under Protocol V1."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.fusion_calibration_loss import FusionCalibrationLoss
from models.dual_branch_cache import sha256_file
from models.gated_cross_attention import GatedCrossAttentionFusion
from models.protocol_evaluation import retrieval_metrics, rsa_metrics
from models.protocol_metadata import git_commit_or_unavailable
from scripts.train_p3r_roi_relational import P3RModel
from scripts.train_stage_a_structural import active_mapping, parameter_sha256, seed_everything


class StageBDataset(Dataset):
    def __init__(self, stage_a_cache, semantic_cache, visual_cache):
        base = json.loads((Path(stage_a_cache).resolve() / "metadata.json").read_text())
        semantic = json.loads((Path(semantic_cache).resolve() / "metadata.json").read_text())
        visual = json.loads((Path(visual_cache).resolve() / "metadata.json").read_text())
        ids = json.loads(Path(base["sample_id_index"]).read_text())
        if ids != json.loads(Path(semantic["sample_ids"]).read_text()) or ids != json.loads(Path(visual["sample_ids"]).read_text()):
            raise ValueError("Stage-B cache sample IDs disagree")
        if semantic.get("z_sem_valid_stages") != ["B"] or semantic.get("h_struct_cached"):
            raise ValueError("Invalid semantic-cache lifecycle metadata")
        if not all(x["manifest_sha256"] == base["manifest_sha256"] for x in (semantic, visual)):
            raise ValueError("Stage-B cache manifests disagree")
        if visual.get("global_visual_definition") != "hidden_states[-2][:,1:,:].mean(patch_axis)":
            raise ValueError("Visual cache is not mean frozen CLIP patch tokens")
        self.metadata, self.semantic_metadata, self.visual_metadata = base, semantic, visual
        self.fmri = np.load(base["fmri_path"], mmap_mode="r")
        self.z_sem = np.load(semantic["z_sem_path"], mmap_mode="r")
        self.q_visual = np.load(visual["global_visual_path"], mmap_mode="r")
        if self.z_sem.shape != (len(ids), 256, 1024) or self.q_visual.shape != (len(ids), 1024):
            raise ValueError("Stage-B cache tensor shape violation")

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, index):
        return {
            "fmri": np.array(self.fmri[index], dtype=np.float32, copy=True),
            "z_sem": np.array(self.z_sem[index], dtype=np.float16, copy=True),
            "q_visual": np.array(self.q_visual[index], dtype=np.float16, copy=True),
        }


def parse_args():
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--config")
    known, _ = pre.parse_known_args()
    defaults = json.loads(Path(known.config).read_text()) if known.config else {}
    parser = argparse.ArgumentParser(parents=[pre]); required = not bool(defaults)
    for name in ("protocol", "mapping_kind", "roi_mapping", "train_cache", "val_cache", "train_semantic_cache", "val_semantic_cache", "train_visual_cache", "val_visual_cache", "structural_checkpoint", "brainx_checkpoint", "output_dir", "experiment_name"):
        parser.add_argument("--" + name.replace("_", "-"), required=required)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20); parser.add_argument("--min-epochs", type=int, default=5); parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--lambda-global", type=float, default=1.0); parser.add_argument("--lambda-rel", type=float, default=1.0)
    parser.add_argument("--gate-init", type=float, default=-4.0); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--expected-structural-epoch", type=int, required=required)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True); parser.add_argument("--resume", action="store_true")
    parser.set_defaults(**defaults)
    return parser.parse_args()


def freeze(module):
    module.eval()
    for parameter in module.parameters(): parameter.requires_grad = False


def load_structural(args):
    _, mapping = active_mapping(args)
    model = P3RModel(mapping).to(args.device)
    state = torch.load(Path(args.structural_checkpoint).resolve(), map_location=args.device)
    if state["epoch"] != args.expected_structural_epoch or state["best_epoch"] != args.expected_structural_epoch:
        raise ValueError(f"Unexpected P3-R checkpoint epoch: {state['epoch']}/{state['best_epoch']}")
    model.load_state_dict(state["model"], strict=True); freeze(model)
    return model


def forward_batch(structural, fusion, batch, device, amp):
    fmri = batch["fmri"].to(device, non_blocking=True)
    z_sem = batch["z_sem"].to(device, non_blocking=True)
    q_visual = batch["q_visual"].to(device, non_blocking=True)
    # Structural is frozen, but remains online. Its outputs are constants for P4.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        h_struct = structural.structural(fmri)["h_struct"]
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        output = fusion(z_sem, h_struct, return_attention=False)
    return z_sem, q_visual, output


def diagnostics(z_sem, output):
    q_sem = F.normalize(z_sem.float().mean(1), dim=-1)
    q_cal = F.normalize(output["z_cal"].float().mean(1), dim=-1)
    preservation = (q_sem * q_cal).sum(-1).mean()
    correction = output["fusion_gate"].float() * output["delta_z"].float()
    ratio = (correction.flatten(1).norm(dim=1) / z_sem.float().flatten(1).norm(dim=1).clamp_min(1e-12)).mean()
    return preservation, ratio


def evaluate(structural, fusion, criterion, loader, device, amp):
    structural.eval(); fusion.eval(); sums = np.zeros(5); count = 0; calibrated = []; visual = []
    with torch.inference_mode():
        for batch in loader:
            z_sem, q_visual, output = forward_batch(structural, fusion, batch, device, amp)
            losses = criterion(output["z_cal"], q_visual); preservation, ratio = diagnostics(z_sem, output); size = len(z_sem)
            sums += np.asarray([float(losses["loss"]), float(losses["global_loss"]), float(losses["rel_loss"]), float(preservation), float(ratio)]) * size; count += size
            calibrated.append(losses["q_cal"].cpu()); visual.append(losses["q_visual"].cpu())
    q_cal, q_visual = torch.cat(calibrated).numpy(), torch.cat(visual).numpy()
    means = sums / count
    return {"total_loss": means[0], "global_loss": means[1], "rel_loss": means[2], "preservation_cosine": means[3], "delta_ratio": means[4], "gate": float(torch.sigmoid(fusion.gate_logit)), "sample_count": count, "rsa": rsa_metrics(q_cal, q_visual), "retrieval": retrieval_metrics(q_cal, q_visual)}


def save_checkpoint(path, fusion, optimizer, scaler, epoch, best, best_epoch, bad_epochs, elapsed):
    torch.save({"fusion": fusion.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best_val_total_loss": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "elapsed_seconds": elapsed}, path)


def run(args):
    locked = {"batch_size": 64, "max_epochs": 20, "min_epochs": 5, "patience": 5, "lr": 1e-4, "weight_decay": 1e-4, "gradient_clip_norm": 1.0, "lambda_global": 1.0, "lambda_rel": 1.0, "gate_init": -4.0, "seed": 42}
    mismatch = {k: (getattr(args, k), v) for k, v in locked.items() if getattr(args, k) != v}
    if mismatch: raise ValueError(f"P4 locked configuration mismatch: {mismatch}")
    identities = {"real": ("dualbranch_s1_stageB_real_seed42_protocolv1", 8), "random": ("dualbranch_s1_stageB_random42_seed42_protocolv1", 16)}
    if (args.experiment_name, args.expected_structural_epoch) != identities.get(args.mapping_kind): raise ValueError("P4 experiment identity/checkpoint mismatch")
    protocol_path = Path(args.protocol).resolve(); protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_version"] != "protocol_v1": raise ValueError("P4 requires protocol_v1")
    seed_everything(args.seed); structural = load_structural(args)
    seed_everything(args.seed); fusion = GatedCrossAttentionFusion(1024, 8, 0.0, args.gate_init).to(args.device)
    initial_hash = parameter_sha256(fusion); fusion_params = sum(p.numel() for p in fusion.parameters())
    if fusion_params != 4202497 or any(p.requires_grad for p in structural.parameters()) or not all(p.requires_grad for p in fusion.parameters()): raise RuntimeError("P4 trainability invariant violated")
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in fusion.parameters()}: raise RuntimeError("Optimizer contains non-fusion parameters")
    amp = bool(args.amp and str(args.device).startswith("cuda")); scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = FusionCalibrationLoss(args.lambda_global, args.lambda_rel)
    train_data = StageBDataset(args.train_cache, args.train_semantic_cache, args.train_visual_cache); val_data = StageBDataset(args.val_cache, args.val_semantic_cache, args.val_visual_cache)
    if len(train_data) != 8559 or len(val_data) != 300: raise ValueError("P4 sample counts violate protocol")
    for split, data, expected in (("train", train_data, protocol["train_split"]["manifest_sha256"]), ("val", val_data, protocol["val_split"]["manifest_sha256"])):
        if data.metadata["manifest_sha256"] != expected: raise ValueError(f"{split} manifest violates Protocol V1")
        if data.semantic_metadata["brainx_checkpoint_sha256"] != sha256_file(args.brainx_checkpoint): raise ValueError(f"{split} BrainX checkpoint hash mismatch")
    output = Path(args.output_dir).resolve() / args.experiment_name; output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pth").exists() and not args.resume: raise FileExistsError("Existing P4 run requires --resume")
    config = vars(args).copy()
    for key in ("protocol", "roi_mapping", "train_cache", "val_cache", "train_semantic_cache", "val_semantic_cache", "train_visual_cache", "val_visual_cache", "structural_checkpoint", "brainx_checkpoint", "output_dir"): config[key] = str(Path(config[key]).resolve())
    config.update({"objective": "1.0_global_cosine_plus_1.0_raw_offdiagonal_rdm_mse", "brainx_frozen": True, "structural_frozen": True, "clip_frozen": True, "fusion_trainable": True, "preserve_loss_used": False, "uot_used": False, "roi_relational_loss_used": False, "lora_used": False, "test_used": False, "h_struct_cached": False, "z_cal_cached": False, "fusion_parameter_count": fusion_params, "initial_fusion_sha256": initial_hash})
    (output / "config.json").write_text(json.dumps(config, indent=2))
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=amp)
    initial_path = output / "initial_val_metrics.json"; initial = json.loads(initial_path.read_text()) if initial_path.exists() else evaluate(structural, fusion, criterion, val_loader, args.device, amp)
    initial.update({"initial_fusion_sha256": initial_hash}); initial_path.write_text(json.dumps(initial, indent=2)); print(json.dumps({"initial_validation": initial}), flush=True)
    start_epoch, best, best_epoch, bad_epochs, elapsed_before = 1, float("inf"), 0, 0, 0.0
    if args.resume:
        state = torch.load(output / "last.pth", map_location=args.device); fusion.load_state_dict(state["fusion"]); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        start_epoch, best, best_epoch, bad_epochs, elapsed_before = state["epoch"] + 1, state["best_val_total_loss"], state["best_epoch"], state["bad_epochs"], state.get("elapsed_seconds", 0.0)
    records_path = output / "metrics_per_epoch.jsonl"; records = [json.loads(x) for x in records_path.read_text().splitlines()] if args.resume and records_path.exists() else []
    started = time.time()
    for epoch in range(start_epoch, args.max_epochs + 1):
        loader = DataLoader(train_data, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(args.seed + epoch), num_workers=args.num_workers, pin_memory=amp)
        structural.eval(); fusion.train(); sums = np.zeros(6); count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); z_sem, q_visual, fused = forward_batch(structural, fusion, batch, args.device, amp); losses = criterion(fused["z_cal"], q_visual)
            scaler.scale(losses["loss"]).backward(); scaler.unscale_(optimizer); grad = torch.nn.utils.clip_grad_norm_(fusion.parameters(), args.gradient_clip_norm)
            if not torch.isfinite(grad) or not torch.isfinite(losses["loss"]): raise RuntimeError("Non-finite P4 optimization")
            scaler.step(optimizer); scaler.update(); preservation, ratio = diagnostics(z_sem, fused); size = len(z_sem)
            sums += np.asarray([float(losses["loss"]), float(losses["global_loss"]), float(losses["rel_loss"]), float(grad), float(preservation), float(ratio)]) * size; count += size
        val = evaluate(structural, fusion, criterion, val_loader, args.device, amp); improved = val["total_loss"] < best
        if improved: best, best_epoch, bad_epochs = val["total_loss"], epoch, 0
        else: bad_epochs += 1
        elapsed = elapsed_before + time.time() - started; means = sums / count; primary = val["retrieval"]["brain_to_image"]
        record = {"epoch": epoch, "train_total_loss": means[0], "train_global_loss": means[1], "train_rel_loss": means[2], "val_total_loss": val["total_loss"], "val_global_loss": val["global_loss"], "val_rel_loss": val["rel_loss"], "val_spearman_rsa": val["rsa"]["spearman_rsa"], "val_pearson_rsa": val["rsa"]["pearson_rsa"], "val_recall_at_1": primary["recall_at_1"], "val_recall_at_5": primary["recall_at_5"], "val_recall_at_10": primary["recall_at_10"], "gate": val["gate"], "preservation_cosine": val["preservation_cosine"], "delta_ratio": val["delta_ratio"], "fusion_grad_norm": means[3], "checkpoint_selection_metric": "val_total_loss"}
        records.append(record); records_path.write_text("".join(json.dumps(x) + "\n" for x in records)); save_checkpoint(output / "last.pth", fusion, optimizer, scaler, epoch, best, best_epoch, bad_epochs, elapsed)
        if improved: save_checkpoint(output / "best.pth", fusion, optimizer, scaler, epoch, best, best_epoch, bad_epochs, elapsed)
        print(json.dumps(record), flush=True)
        if epoch >= args.min_epochs and bad_epochs >= args.patience: print(json.dumps({"early_stop": epoch}), flush=True); break
    elapsed = elapsed_before + time.time() - started; state = torch.load(output / "best.pth", map_location=args.device); fusion.load_state_dict(state["fusion"])
    final = evaluate(structural, fusion, criterion, val_loader, args.device, amp); final.update({"best_epoch": state["best_epoch"], "best_val_total_loss": state["best_val_total_loss"]})
    (output / "final_val_metrics.json").write_text(json.dumps(final, indent=2)); (output / "metrics.json").write_text(json.dumps({"initial": initial, "final": final}, indent=2))
    provenance = {"protocol_version": "protocol_v1", "protocol_sha256": sha256_file(protocol_path), "git_commit_hash": git_commit_or_unavailable(str(Path(__file__).resolve().parents[2])), "mapping_kind": args.mapping_kind, "roi_mapping": str(Path(args.roi_mapping).resolve()), "roi_mapping_sha256": sha256_file(args.roi_mapping), "brainx_checkpoint": str(Path(args.brainx_checkpoint).resolve()), "brainx_checkpoint_sha256": sha256_file(args.brainx_checkpoint), "structural_checkpoint": str(Path(args.structural_checkpoint).resolve()), "structural_checkpoint_sha256": sha256_file(args.structural_checkpoint), "structural_checkpoint_epoch": args.expected_structural_epoch, "initial_fusion_sha256": initial_hash, "seed": args.seed, "gpu": torch.cuda.get_device_name(0) if amp else "CPU", "training_seconds": elapsed}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)); print(json.dumps({"final_validation": final, "provenance": provenance}), flush=True); return final


if __name__ == "__main__": run(parse_args())
