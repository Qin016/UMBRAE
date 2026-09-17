#!/usr/bin/env python
"""P6 full Stage-C: frozen structural branch plus trainable LoRA and fusion."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.lora_calibration_loss import LoRACalibrationLoss
from models.dual_branch_cache import sha256_file
from models.gated_cross_attention import GatedCrossAttentionFusion
from models.protocol_evaluation import retrieval_metrics, rsa_metrics
from models.protocol_metadata import git_commit_or_unavailable
from models.shared_lora import SharedLoRAUMBRAEEncoder, load_lora_state_dict, lora_named_parameters, lora_state_dict
from models.umbrae_backbone import FrozenUMBRAEEncoder
from scripts.train_p3r_roi_relational import P3RModel
from scripts.train_p5_shared_lora import P5Dataset, linear_cka
from scripts.train_stage_a_structural import active_mapping, seed_everything


def parse_args():
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--config"); known, _ = pre.parse_known_args(); defaults = json.loads(Path(known.config).read_text()) if known.config else {}
    parser = argparse.ArgumentParser(parents=[pre]); required = not bool(defaults)
    for name in ("protocol", "mapping_kind", "roi_mapping", "train_cache", "val_cache", "train_visual_cache", "val_visual_cache", "train_base_reference_cache", "val_base_reference_cache", "brainx_checkpoint", "lora_checkpoint", "structural_checkpoint", "fusion_checkpoint", "output_dir", "experiment_name"):
        parser.add_argument("--" + name.replace("_", "-"), required=required)
    parser.add_argument("--subject", default="subj01"); parser.add_argument("--rank", type=int, default=8); parser.add_argument("--alpha", type=float, default=16.0); parser.add_argument("--lora-dropout", type=float, default=0.05); parser.add_argument("--expected-structural-epoch", type=int, required=required); parser.add_argument("--expected-fusion-epoch", type=int, default=4); parser.add_argument("--expected-lora-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--max-epochs", type=int, default=20); parser.add_argument("--min-epochs", type=int, default=5); parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lora-lr", type=float, default=1e-5); parser.add_argument("--fusion-lr", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--gradient-clip-norm", type=float, default=1.0); parser.add_argument("--lambda-global", type=float, default=1.0); parser.add_argument("--lambda-rel", type=float, default=1.0); parser.add_argument("--lambda-preserve", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cuda"); parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True); parser.add_argument("--resume", action="store_true"); parser.set_defaults(**defaults); return parser.parse_args()


def freeze(module):
    module.eval()
    for parameter in module.parameters(): parameter.requires_grad = False


def build_models(args):
    base = FrozenUMBRAEEncoder(args.brainx_checkpoint, args.subject).to(args.device)
    semantic = SharedLoRAUMBRAEEncoder(base, args.rank, args.alpha, args.lora_dropout).to(args.device)
    lora_checkpoint = torch.load(Path(args.lora_checkpoint).resolve(), map_location=args.device)
    if lora_checkpoint["epoch"] != args.expected_lora_epoch or lora_checkpoint["metadata"]["base_brainx_sha256"] != sha256_file(args.brainx_checkpoint): raise ValueError("P5 LoRA checkpoint provenance/epoch mismatch")
    load_lora_state_dict(semantic, lora_checkpoint["lora"])
    _, mapping = active_mapping(args); structural = P3RModel(mapping).to(args.device); structural_checkpoint = torch.load(Path(args.structural_checkpoint).resolve(), map_location=args.device)
    if structural_checkpoint["epoch"] != args.expected_structural_epoch or structural_checkpoint["best_epoch"] != args.expected_structural_epoch: raise ValueError("P3-R structural checkpoint epoch mismatch")
    structural.load_state_dict(structural_checkpoint["model"], strict=True); freeze(structural)
    fusion = GatedCrossAttentionFusion(1024, 8, 0.0, -4.0).to(args.device); fusion_checkpoint = torch.load(Path(args.fusion_checkpoint).resolve(), map_location=args.device)
    if fusion_checkpoint["epoch"] != args.expected_fusion_epoch or fusion_checkpoint["best_epoch"] != args.expected_fusion_epoch: raise ValueError("P4 Fusion checkpoint epoch mismatch")
    fusion.load_state_dict(fusion_checkpoint["fusion"], strict=True)
    return semantic, structural, fusion


def forward_batch(semantic, structural, fusion, batch, device, amp):
    fmri, visual, base = (batch[key].to(device, non_blocking=True) for key in ("fmri", "q_visual", "z_base"))
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): z_lora = semantic(fmri)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): h_struct = structural.structural(fmri)["h_struct"]
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): fused = fusion(z_lora, h_struct, return_attention=False)
    return z_lora, fused, visual, base


def diagnostics(z_lora, fused, z_base, loss):
    base, lora, calibrated = z_base.detach().float(), z_lora.float(), fused["z_cal"].float()
    q_base, q_lora, q_cal = loss["q_base"], F.normalize(lora.mean(1), dim=-1), loss["q_lora"]
    base_lora = (q_base * q_lora).sum(-1).mean(); base_calibrated = (q_base * q_cal).sum(-1).mean()
    base_norm = base.flatten(1).norm(2, 1).clamp_min(1e-12); lora_norm = lora.flatten(1).norm(2, 1).clamp_min(1e-12)
    lora_delta = ((lora - base).flatten(1).norm(2, 1) / base_norm).mean()
    correction = fused["fusion_gate"].float() * fused["delta_z"].float(); fusion_ratio = (correction.flatten(1).norm(2, 1) / lora_norm).mean()
    final_delta = ((calibrated - base).flatten(1).norm(2, 1) / base_norm).mean()
    return base_lora, base_calibrated, lora_delta, fusion_ratio, final_delta


def evaluate(semantic, structural, fusion, criterion, loader, device, amp):
    semantic.eval(); structural.eval(); fusion.eval(); sums = np.zeros(9); count = 0; calibrated_all = []; base_all = []; visual_all = []
    with torch.inference_mode():
        for batch in loader:
            z_lora, fused, visual, base = forward_batch(semantic, structural, fusion, batch, device, amp); loss = criterion(fused["z_cal"], visual, base); values = diagnostics(z_lora, fused, base, loss); size = len(z_lora)
            sums += np.asarray([float(loss["loss"]), float(loss["global_loss"]), float(loss["rel_loss"]), float(loss["preserve_loss"]), *(float(x) for x in values)]) * size; count += size
            calibrated_all.append(loss["pooled_lora"].cpu()); base_all.append(loss["pooled_base"].cpu()); visual_all.append(loss["q_visual"].cpu())
    calibrated, base, visual = torch.cat(calibrated_all), torch.cat(base_all), torch.cat(visual_all); normalized = F.normalize(calibrated.float(), dim=-1).numpy(); means = sums / count
    return {"total_loss": means[0], "global_loss": means[1], "rel_loss": means[2], "preserve_loss": means[3], "base_lora_cosine": means[4], "base_calibrated_cosine": means[5], "lora_relative_delta": means[6], "fusion_delta_ratio": means[7], "final_relative_delta": means[8], "final_pooled_linear_cka": linear_cka(base, calibrated), "gate": float(torch.sigmoid(fusion.gate_logit)), "sample_count": count, "rsa": rsa_metrics(normalized, visual.numpy()), "retrieval": retrieval_metrics(normalized, visual.numpy())}


def parameter_grad_norm(parameters):
    values = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).norm(2)) if values else 0.0


def trainability_audit(semantic, structural, fusion, optimizer):
    lora = [p for _, p in lora_named_parameters(semantic)]; fusion_parameters = list(fusion.parameters()); base = [p for name, p in semantic.named_parameters() if ".lora_a." not in name and ".lora_b." not in name]
    if any(p.requires_grad for p in base) or any(p.requires_grad for p in structural.parameters()) or not all(p.requires_grad for p in lora + fusion_parameters): raise RuntimeError("Stage-C trainability invariant violated")
    groups = optimizer.param_groups
    if len(groups) != 2 or {id(p) for p in groups[0]["params"]} != {id(p) for p in lora} or {id(p) for p in groups[1]["params"]} != {id(p) for p in fusion_parameters}: raise RuntimeError("Stage-C optimizer groups are incorrect")
    return lora, fusion_parameters, base


def preflight_gradient_audit(semantic, structural, fusion, criterion, optimizer, dataset, device, amp, lora, fusion_parameters, base):
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0))); semantic.train(); structural.eval(); fusion.train(); optimizer.zero_grad(set_to_none=True); z_lora, fused, visual, reference = forward_batch(semantic, structural, fusion, batch, device, amp); loss = criterion(fused["z_cal"], visual, reference)["loss"]; loss.backward()
    result = {"lora_grad_norm": parameter_grad_norm(lora), "fusion_grad_norm": parameter_grad_norm(fusion_parameters), "base_gradient_tensor_count": sum(p.grad is not None for p in base), "structural_gradient_tensor_count": sum(p.grad is not None for p in structural.parameters())}
    if result["lora_grad_norm"] <= 0 or result["fusion_grad_norm"] <= 0 or result["base_gradient_tensor_count"] or result["structural_gradient_tensor_count"]: raise RuntimeError(f"Stage-C gradient audit failed: {result}")
    optimizer.zero_grad(set_to_none=True); return result


def save_checkpoint(path, semantic, fusion, optimizer, scaler, metadata, epoch, best, best_epoch, bad_epochs, elapsed):
    torch.save({"lora": lora_state_dict(semantic), "fusion": fusion.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "metadata": metadata, "epoch": epoch, "best_val_total_loss": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "elapsed_seconds": elapsed}, path)


def run(args):
    locked = {"rank": 8, "alpha": 16.0, "lora_dropout": 0.05, "expected_fusion_epoch": 4, "expected_lora_epoch": 20, "max_epochs": 20, "min_epochs": 5, "patience": 5, "lora_lr": 1e-5, "fusion_lr": 1e-4, "weight_decay": 1e-4, "gradient_clip_norm": 1.0, "lambda_global": 1.0, "lambda_rel": 1.0, "lambda_preserve": 0.1, "seed": 42}
    mismatch = {key: (getattr(args, key), value) for key, value in locked.items() if getattr(args, key) != value}
    if mismatch or args.batch_size not in {32, 64}: raise ValueError(f"P6 locked configuration mismatch: {mismatch}, batch={args.batch_size}")
    identities = {"real": ("dualbranch_s1_stageC_real_lora_r8_seed42_protocolv1", 8), "random": ("dualbranch_s1_stageC_random42_lora_r8_seed42_protocolv1", 16)}
    if (args.experiment_name, args.expected_structural_epoch) != identities.get(args.mapping_kind): raise ValueError("Unexpected P6 experiment identity")
    if getattr(args, "train_semantic_cache", None) or getattr(args, "val_semantic_cache", None): raise ValueError("Stage C forbids cached Z_sem as semantic model input")
    protocol_path = Path(args.protocol).resolve(); protocol = json.loads(protocol_path.read_text()); seed_everything(args.seed); semantic, structural, fusion = build_models(args)
    lora_parameters = [p for _, p in lora_named_parameters(semantic)]; fusion_parameters = list(fusion.parameters()); optimizer = torch.optim.AdamW([{"params": lora_parameters, "lr": args.lora_lr, "name": "shared_qv_lora"}, {"params": fusion_parameters, "lr": args.fusion_lr, "name": "gated_fusion"}], weight_decay=args.weight_decay)
    lora_parameters, fusion_parameters, base_parameters = trainability_audit(semantic, structural, fusion, optimizer); amp = bool(args.amp and str(args.device).startswith("cuda")); scaler = torch.cuda.amp.GradScaler(enabled=amp); criterion = LoRACalibrationLoss(args.lambda_global, args.lambda_rel, args.lambda_preserve)
    train_data = P5Dataset(args.train_cache, args.train_visual_cache, args.train_base_reference_cache); val_data = P5Dataset(args.val_cache, args.val_visual_cache, args.val_base_reference_cache)
    if len(train_data) != 8559 or len(val_data) != 300: raise ValueError("P6 sample counts violate Protocol V1")
    checkpoint_hash = sha256_file(args.brainx_checkpoint)
    for split, data, expected in (("train", train_data, protocol["train_split"]["manifest_sha256"]), ("val", val_data, protocol["val_split"]["manifest_sha256"])):
        if data.metadata["manifest_sha256"] != expected or data.reference_metadata["brainx_checkpoint_sha256"] != checkpoint_hash: raise ValueError(f"{split} P6 cache provenance mismatch")
    output = Path(args.output_dir).resolve() / args.experiment_name; output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pth").exists() and not args.resume: raise FileExistsError("Existing P6 run requires --resume")
    audit = preflight_gradient_audit(semantic, structural, fusion, criterion, optimizer, val_data, args.device, amp, lora_parameters, fusion_parameters, base_parameters)
    parameter_report = {"brainx_base_parameters": sum(p.numel() for p in base_parameters), "lora_parameters": sum(p.numel() for p in lora_parameters), "fusion_parameters": sum(p.numel() for p in fusion_parameters), "structural_branch_parameters": sum(p.numel() for p in structural.structural.parameters()), "structural_pool_parameters_unused": sum(p.numel() for p in structural.pool.parameters()), "total_trainable_parameters": sum(p.numel() for p in lora_parameters + fusion_parameters)}; parameter_report["trainable_ratio_percent_of_brainx"] = 100 * parameter_report["total_trainable_parameters"] / parameter_report["brainx_base_parameters"]
    metadata = {"base_brainx_sha256": checkpoint_hash, "initial_lora_checkpoint_sha256": sha256_file(args.lora_checkpoint), "structural_checkpoint_sha256": sha256_file(args.structural_checkpoint), "initial_fusion_checkpoint_sha256": sha256_file(args.fusion_checkpoint), "rank": args.rank, "alpha": args.alpha, "dropout": args.lora_dropout, "lora_targets": semantic.target_modules, "fusion_config": {"dim": 1024, "num_heads": 8, "dropout": 0.0}, "protocol_version": "protocol_v1", "mapping_kind": args.mapping_kind}
    config = vars(args).copy()
    for key in ("protocol", "roi_mapping", "train_cache", "val_cache", "train_visual_cache", "val_visual_cache", "train_base_reference_cache", "val_base_reference_cache", "brainx_checkpoint", "lora_checkpoint", "structural_checkpoint", "fusion_checkpoint", "output_dir"): config[key] = str(Path(config[key]).resolve())
    config.update({"objective": "global_1.0_plus_raw_rel_1.0_plus_final_preserve_0.1", "brainx_base_frozen": True, "structural_frozen": True, "clip_frozen": True, "lora_trainable": True, "fusion_trainable": True, "z_lora_online": True, "cached_z_sem_model_input": False, "base_reference_cache_detached_only": True, "h_struct_online": True, "test_used": False, "parameter_report": parameter_report, "gradient_audit": audit, "checkpoint_metadata": metadata}); (output / "config.json").write_text(json.dumps(config, indent=2)); print(json.dumps({"parameter_report": parameter_report, "gradient_audit": audit}), flush=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=amp); initial_path = output / "initial_val_metrics.json"; initial = json.loads(initial_path.read_text()) if initial_path.exists() else evaluate(semantic, structural, fusion, criterion, val_loader, args.device, amp); initial_path.write_text(json.dumps(initial, indent=2)); print(json.dumps({"initial_validation": initial}), flush=True)
    start_epoch, best, best_epoch, bad_epochs, elapsed_before = 1, float("inf"), 0, 0, 0.0
    if args.resume:
        state = torch.load(output / "last.pth", map_location=args.device); load_lora_state_dict(semantic, state["lora"]); fusion.load_state_dict(state["fusion"]); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"]); start_epoch, best, best_epoch, bad_epochs, elapsed_before = state["epoch"] + 1, state["best_val_total_loss"], state["best_epoch"], state["bad_epochs"], state.get("elapsed_seconds", 0.0)
    records_path = output / "metrics_per_epoch.jsonl"; records = [json.loads(line) for line in records_path.read_text().splitlines()] if args.resume and records_path.exists() else []; started = time.time(); trainable = lora_parameters + fusion_parameters
    for epoch in range(start_epoch, args.max_epochs + 1):
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + epoch), num_workers=args.num_workers, pin_memory=amp); semantic.train(); structural.eval(); fusion.train(); sums = np.zeros(6); count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); z_lora, fused, visual, reference = forward_batch(semantic, structural, fusion, batch, args.device, amp); loss = criterion(fused["z_cal"], visual, reference); scaler.scale(loss["loss"]).backward(); scaler.unscale_(optimizer); lora_grad, fusion_grad = parameter_grad_norm(lora_parameters), parameter_grad_norm(fusion_parameters); total_grad = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip_norm)
            if not torch.isfinite(total_grad) or not torch.isfinite(loss["loss"]): raise RuntimeError("Non-finite P6 optimization")
            scaler.step(optimizer); scaler.update(); size = len(z_lora); sums += np.asarray([float(loss["loss"]), float(loss["global_loss"]), float(loss["rel_loss"]), float(loss["preserve_loss"]), lora_grad, fusion_grad]) * size; count += size
        val = evaluate(semantic, structural, fusion, criterion, val_loader, args.device, amp); improved = val["total_loss"] < best
        if improved: best, best_epoch, bad_epochs = val["total_loss"], epoch, 0
        else: bad_epochs += 1
        elapsed = elapsed_before + time.time() - started; means = sums / count; retrieval = val["retrieval"]["brain_to_image"]
        record = {"epoch": epoch, "train_total": means[0], "train_global": means[1], "train_rel": means[2], "train_preserve": means[3], "val_total": val["total_loss"], "val_global": val["global_loss"], "val_rel": val["rel_loss"], "val_preserve": val["preserve_loss"], "val_spearman_rsa": val["rsa"]["spearman_rsa"], "val_pearson_rsa": val["rsa"]["pearson_rsa"], "val_recall_at_1": retrieval["recall_at_1"], "val_recall_at_5": retrieval["recall_at_5"], "val_recall_at_10": retrieval["recall_at_10"], "gate": val["gate"], "base_lora_cosine": val["base_lora_cosine"], "base_calibrated_cosine": val["base_calibrated_cosine"], "lora_relative_delta": val["lora_relative_delta"], "fusion_delta_ratio": val["fusion_delta_ratio"], "final_relative_delta": val["final_relative_delta"], "lora_grad_norm": means[4], "fusion_grad_norm": means[5], "checkpoint_selection_metric": "val_total_loss"}
        records.append(record); records_path.write_text("".join(json.dumps(value) + "\n" for value in records)); save_checkpoint(output / "last.pth", semantic, fusion, optimizer, scaler, metadata, epoch, best, best_epoch, bad_epochs, elapsed)
        if improved: save_checkpoint(output / "best.pth", semantic, fusion, optimizer, scaler, metadata, epoch, best, best_epoch, bad_epochs, elapsed)
        print(json.dumps(record), flush=True)
        if epoch >= args.min_epochs and bad_epochs >= args.patience: print(json.dumps({"early_stop": epoch}), flush=True); break
    elapsed = elapsed_before + time.time() - started; state = torch.load(output / "best.pth", map_location=args.device); load_lora_state_dict(semantic, state["lora"]); fusion.load_state_dict(state["fusion"]); final = evaluate(semantic, structural, fusion, criterion, val_loader, args.device, amp); final.update({"best_epoch": state["best_epoch"], "best_val_total_loss": state["best_val_total_loss"]}); (output / "final_val_metrics.json").write_text(json.dumps(final, indent=2)); (output / "metrics.json").write_text(json.dumps({"initial": initial, "final": final}, indent=2)); provenance = {"protocol_version": "protocol_v1", "protocol_sha256": sha256_file(protocol_path), "git_commit_hash": git_commit_or_unavailable(str(Path(__file__).resolve().parents[2])), **metadata, "seed": args.seed, "gpu": torch.cuda.get_device_name(0) if amp else "CPU", "training_seconds": elapsed}; (output / "provenance.json").write_text(json.dumps(provenance, indent=2)); print(json.dumps({"final_validation": final, "provenance": provenance}), flush=True); return final


if __name__ == "__main__": run(parse_args())
