#!/usr/bin/env python
"""P5 shared Q/V LoRA-only calibration under Protocol V1."""

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

from losses.lora_calibration_loss import LoRACalibrationLoss
from models.dual_branch_cache import sha256_file
from models.protocol_evaluation import retrieval_metrics, rsa_metrics
from models.protocol_metadata import git_commit_or_unavailable
from models.shared_lora import LoRAQLinear, LoRASelectiveVLinear, SharedLoRAUMBRAEEncoder, load_lora_state_dict, lora_named_parameters, lora_state_dict
from models.umbrae_backbone import FrozenUMBRAEEncoder
from scripts.train_stage_a_structural import seed_everything


class P5Dataset(Dataset):
    """Online fMRI input plus frozen visual and base-reference caches."""
    def __init__(self, stage_a_cache, visual_cache, base_reference_cache):
        base = json.loads((Path(stage_a_cache).resolve() / "metadata.json").read_text())
        visual = json.loads((Path(visual_cache).resolve() / "metadata.json").read_text())
        reference = json.loads((Path(base_reference_cache).resolve() / "metadata.json").read_text())
        ids = json.loads(Path(base["sample_id_index"]).read_text())
        if ids != json.loads(Path(visual["sample_ids"]).read_text()) or ids != json.loads(Path(reference["sample_ids"]).read_text()):
            raise ValueError("P5 cache sample IDs disagree")
        if not all(item["manifest_sha256"] == base["manifest_sha256"] for item in (visual, reference)):
            raise ValueError("P5 cache manifests disagree")
        if visual.get("global_visual_definition") != "hidden_states[-2][:,1:,:].mean(patch_axis)":
            raise ValueError("P5 visual teacher is not frozen CLIP patch-token mean")
        if reference.get("z_sem_forbidden_stage") != "C" or reference.get("z_sem_valid_stages") != ["B"]:
            raise ValueError("P5 base reference must be clearly distinguished from online Z_lora")
        self.metadata, self.visual_metadata, self.reference_metadata = base, visual, reference
        self.fmri = np.load(base["fmri_path"], mmap_mode="r")
        self.q_visual = np.load(visual["global_visual_path"], mmap_mode="r")
        self.z_base = np.load(reference["z_sem_path"], mmap_mode="r")
        if self.z_base.shape != (len(ids), 256, 1024) or self.q_visual.shape != (len(ids), 1024):
            raise ValueError("P5 cache tensor shape violation")

    def __len__(self): return len(self.fmri)

    def __getitem__(self, index):
        return {"fmri": np.array(self.fmri[index], dtype=np.float32, copy=True), "q_visual": np.array(self.q_visual[index], dtype=np.float16, copy=True), "z_base": np.array(self.z_base[index], dtype=np.float16, copy=True)}


def parse_args():
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--config"); known, _ = pre.parse_known_args()
    defaults = json.loads(Path(known.config).read_text()) if known.config else {}
    parser = argparse.ArgumentParser(parents=[pre]); required = not bool(defaults)
    for name in ("protocol", "train_cache", "val_cache", "train_visual_cache", "val_visual_cache", "train_base_reference_cache", "val_base_reference_cache", "brainx_checkpoint", "output_dir", "experiment_name"):
        parser.add_argument("--" + name.replace("_", "-"), required=required)
    parser.add_argument("--subject", default="subj01"); parser.add_argument("--rank", type=int, default=8); parser.add_argument("--alpha", type=float, default=16.0); parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--max-epochs", type=int, default=20); parser.add_argument("--min-epochs", type=int, default=5); parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--lambda-global", type=float, default=1.0); parser.add_argument("--lambda-rel", type=float, default=1.0); parser.add_argument("--lambda-preserve", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cuda"); parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True); parser.add_argument("--resume", action="store_true")
    parser.set_defaults(**defaults); return parser.parse_args()


def forward_batch(model, batch, device, amp):
    fmri, visual, base = (batch[key].to(device, non_blocking=True) for key in ("fmri", "q_visual", "z_base"))
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): z_lora = model(fmri)
    return z_lora, visual, base


def drift(z_lora, z_base, output):
    values, reference = z_lora.float(), z_base.detach().float()
    relative = (values.sub(reference).flatten(1).norm(2, 1) / reference.flatten(1).norm(2, 1).clamp_min(1e-12)).mean()
    pooled_relative = ((output["pooled_lora"] - output["pooled_base"]).norm(2, 1) / output["pooled_base"].norm(2, 1).clamp_min(1e-12)).mean()
    cosine = (output["q_lora"] * output["q_base"]).sum(-1).mean()
    return cosine, relative, pooled_relative


def linear_cka(left, right):
    left, right = left.double() - left.double().mean(0), right.double() - right.double().mean(0)
    numerator = (left.T @ right).square().sum()
    denominator = (left.T @ left).square().sum().sqrt() * (right.T @ right).square().sum().sqrt()
    return float(numerator / denominator.clamp_min(1e-30))


def evaluate(model, criterion, loader, device, amp):
    model.eval(); sums = np.zeros(7); count = 0; lora_all = []; base_all = []; visual_all = []
    with torch.inference_mode():
        for batch in loader:
            z_lora, visual, base = forward_batch(model, batch, device, amp); out = criterion(z_lora, visual, base); cosine, relative, pooled_relative = drift(z_lora, base, out); size = len(z_lora)
            sums += np.asarray([float(out["loss"]), float(out["global_loss"]), float(out["rel_loss"]), float(out["preserve_loss"]), float(cosine), float(relative), float(pooled_relative)]) * size; count += size
            lora_all.append(out["pooled_lora"].cpu()); base_all.append(out["pooled_base"].cpu()); visual_all.append(out["q_visual"].cpu())
    pooled_lora, pooled_base, visual = torch.cat(lora_all), torch.cat(base_all), torch.cat(visual_all)
    means = sums / count; normalized_lora = F.normalize(pooled_lora.float(), dim=-1).numpy()
    return {"total_loss": means[0], "global_loss": means[1], "rel_loss": means[2], "preserve_loss": means[3], "base_lora_cosine": means[4], "relative_feature_delta": means[5], "pooled_relative_delta": means[6], "pooled_linear_cka": linear_cka(pooled_base, pooled_lora), "sample_count": count, "rsa": rsa_metrics(normalized_lora, visual.numpy()), "retrieval": retrieval_metrics(normalized_lora, visual.numpy())}


def adapter_grad_audit(model):
    groups = {"q": [], "v": []}
    for module in model.modules():
        if isinstance(module, LoRAQLinear): groups["q"].append(float(module.lora_b.weight.grad.norm()) if module.lora_b.weight.grad is not None else 0.0)
        if isinstance(module, LoRASelectiveVLinear): groups["v"].append(float(module.lora_b.weight.grad.norm()) if module.lora_b.weight.grad is not None else 0.0)
    return {"q_lora_b_grad_norms": groups["q"], "v_lora_b_grad_norms": groups["v"], "all_q_nonzero": len(groups["q"]) == 6 and all(x > 0 for x in groups["q"]), "all_v_nonzero": len(groups["v"]) == 6 and all(x > 0 for x in groups["v"])}


def initialization_and_gradient_audit(model, dataset, original, criterion, optimizer, device, amp):
    sample = torch.from_numpy(np.stack([dataset.fmri[0], dataset.fmri[1]]).astype(np.float32)).to(device)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): injected = model(sample)
    difference = injected.float() - original.float()
    equivalence = {"max_abs_error": float(difference.abs().max()), "mean_abs_error": float(difference.abs().mean()), "cosine_similarity": float(F.cosine_similarity(injected.float().flatten(1), original.float().flatten(1), dim=-1).mean())}
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0))); model.train(); optimizer.zero_grad(set_to_none=True)
    z_lora, visual, base = forward_batch(model, batch, device, amp); loss = criterion(z_lora, visual, base)["loss"]; loss.backward(); gradients = adapter_grad_audit(model)
    base_gradient_count = sum(p.grad is not None for name, p in model.named_parameters() if ".lora_a." not in name and ".lora_b." not in name)
    subject_specific_frozen = all(not p.requires_grad and p.grad is None for name, p in model.named_parameters() if ".lin1." in name or ".lin2." in name)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}; lora_ids = {id(p) for _, p in lora_named_parameters(model)}
    gradients.update({"base_gradient_tensor_count": base_gradient_count, "subject_specific_lin1_lin2_frozen": subject_specific_frozen, "optimizer_is_lora_only": optimizer_ids == lora_ids})
    if not gradients["all_q_nonzero"] or not gradients["all_v_nonzero"] or base_gradient_count or not subject_specific_frozen or optimizer_ids != lora_ids: raise RuntimeError(f"P5 gradient audit failed: {gradients}")
    optimizer.zero_grad(set_to_none=True); model.eval(); return {"initialization_equivalence": equivalence, "gradient_audit": gradients}


def save_checkpoint(path, model, optimizer, scaler, metadata, epoch, best, best_epoch, bad_epochs, elapsed):
    torch.save({"lora": lora_state_dict(model), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "metadata": metadata, "epoch": epoch, "best_val_total_loss": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "elapsed_seconds": elapsed}, path)


def run(args):
    locked = {"rank": 8, "alpha": 16.0, "lora_dropout": 0.05, "max_epochs": 20, "min_epochs": 5, "patience": 5, "lr": 1e-5, "weight_decay": 1e-4, "gradient_clip_norm": 1.0, "lambda_global": 1.0, "lambda_rel": 1.0, "lambda_preserve": 0.1, "seed": 42}
    mismatch = {key: (getattr(args, key), value) for key, value in locked.items() if getattr(args, key) != value}
    if mismatch or args.batch_size not in {32, 64}: raise ValueError(f"P5 locked configuration mismatch: {mismatch}, batch={args.batch_size}")
    if args.experiment_name != "umbrae_s1_lora_qv_r8_seed42_protocolv1" or args.subject != "subj01": raise ValueError("Unexpected P5 experiment identity")
    if getattr(args, "train_semantic_cache", None) or getattr(args, "val_semantic_cache", None): raise ValueError("P5 forbids cached Z_sem as model input; only detached base-reference cache is allowed")
    protocol_path = Path(args.protocol).resolve(); protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_version"] != "protocol_v1": raise ValueError("P5 requires Protocol V1")
    train_data = P5Dataset(args.train_cache, args.train_visual_cache, args.train_base_reference_cache); val_data = P5Dataset(args.val_cache, args.val_visual_cache, args.val_base_reference_cache)
    if len(train_data) != 8559 or len(val_data) != 300: raise ValueError("P5 sample counts violate Protocol V1")
    checkpoint_hash = sha256_file(args.brainx_checkpoint)
    for split, data, expected in (("train", train_data, protocol["train_split"]["manifest_sha256"]), ("val", val_data, protocol["val_split"]["manifest_sha256"])):
        if data.metadata["manifest_sha256"] != expected or data.reference_metadata["brainx_checkpoint_sha256"] != checkpoint_hash: raise ValueError(f"{split} P5 cache provenance mismatch")
    amp = bool(args.amp and str(args.device).startswith("cuda")); seed_everything(args.seed); backbone = FrozenUMBRAEEncoder(args.brainx_checkpoint, args.subject).to(args.device).eval()
    sample = torch.from_numpy(np.array(val_data.fmri[:2], dtype=np.float32, copy=True)).to(args.device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): original_output = backbone(sample)
    model = SharedLoRAUMBRAEEncoder(backbone, args.rank, args.alpha, args.lora_dropout).to(args.device); report = model.parameter_report(); lora_parameters = [p for _, p in lora_named_parameters(model)]
    if report["lora_parameters"] != 245760 or report["trainable_parameters"] != 245760: raise RuntimeError(f"Unexpected LoRA parameter count: {report}")
    optimizer = torch.optim.AdamW(lora_parameters, lr=args.lr, weight_decay=args.weight_decay); scaler = torch.cuda.amp.GradScaler(enabled=amp); criterion = LoRACalibrationLoss(args.lambda_global, args.lambda_rel, args.lambda_preserve)
    output = Path(args.output_dir).resolve() / args.experiment_name; output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pth").exists() and not args.resume: raise FileExistsError("Existing P5 run requires --resume")
    audit = initialization_and_gradient_audit(model, val_data, original_output, criterion, optimizer, args.device, amp)
    metadata = {"base_brainx_sha256": checkpoint_hash, "rank": args.rank, "alpha": args.alpha, "dropout": args.lora_dropout, "target_modules": model.target_modules, "protocol_version": "protocol_v1"}
    config = vars(args).copy()
    for key in ("protocol", "train_cache", "val_cache", "train_visual_cache", "val_visual_cache", "train_base_reference_cache", "val_base_reference_cache", "brainx_checkpoint", "output_dir"): config[key] = str(Path(config[key]).resolve())
    config.update({"objective": "global_1.0_plus_raw_rel_1.0_plus_preserve_0.1", "structural_used": False, "fusion_used": False, "clip_frozen": True, "brainx_base_frozen": True, "z_lora_online": True, "cached_z_sem_model_input": False, "base_reference_cache_detached_only": True, "test_used": False, "parameter_report": report, "checkpoint_metadata": metadata, **audit}); (output / "config.json").write_text(json.dumps(config, indent=2)); print(json.dumps({"parameter_report": report, **audit}), flush=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=amp)
    initial_path = output / "initial_val_metrics.json"; initial = json.loads(initial_path.read_text()) if initial_path.exists() else evaluate(model, criterion, val_loader, args.device, amp); initial_path.write_text(json.dumps(initial, indent=2)); print(json.dumps({"initial_validation": initial}), flush=True)
    start_epoch, best, best_epoch, bad_epochs, elapsed_before = 1, float("inf"), 0, 0, 0.0
    if args.resume:
        state = torch.load(output / "last.pth", map_location=args.device); load_lora_state_dict(model, state["lora"]); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"]); start_epoch, best, best_epoch, bad_epochs, elapsed_before = state["epoch"] + 1, state["best_val_total_loss"], state["best_epoch"], state["bad_epochs"], state.get("elapsed_seconds", 0.0)
    records_path = output / "metrics_per_epoch.jsonl"; records = [json.loads(x) for x in records_path.read_text().splitlines()] if args.resume and records_path.exists() else []; started = time.time()
    for epoch in range(start_epoch, args.max_epochs + 1):
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + epoch), num_workers=args.num_workers, pin_memory=amp); model.train(); sums = np.zeros(5); count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); z_lora, visual, base = forward_batch(model, batch, args.device, amp); out = criterion(z_lora, visual, base); scaler.scale(out["loss"]).backward(); scaler.unscale_(optimizer); grad = torch.nn.utils.clip_grad_norm_(lora_parameters, args.gradient_clip_norm)
            if not torch.isfinite(grad) or not torch.isfinite(out["loss"]): raise RuntimeError("Non-finite P5 optimization")
            scaler.step(optimizer); scaler.update(); size = len(z_lora); sums += np.asarray([float(out["loss"]), float(out["global_loss"]), float(out["rel_loss"]), float(out["preserve_loss"]), float(grad)]) * size; count += size
        val = evaluate(model, criterion, val_loader, args.device, amp); improved = val["total_loss"] < best
        if improved: best, best_epoch, bad_epochs = val["total_loss"], epoch, 0
        else: bad_epochs += 1
        elapsed = elapsed_before + time.time() - started; means = sums / count; retrieval = val["retrieval"]["brain_to_image"]
        record = {"epoch": epoch, "train_total": means[0], "train_global": means[1], "train_rel": means[2], "train_preserve": means[3], "val_total": val["total_loss"], "val_global": val["global_loss"], "val_rel": val["rel_loss"], "val_preserve": val["preserve_loss"], "val_spearman_rsa": val["rsa"]["spearman_rsa"], "val_pearson_rsa": val["rsa"]["pearson_rsa"], "val_recall_at_1": retrieval["recall_at_1"], "base_lora_cosine": val["base_lora_cosine"], "relative_feature_delta": val["relative_feature_delta"], "lora_grad_norm": means[4], "checkpoint_selection_metric": "val_total_loss"}
        records.append(record); records_path.write_text("".join(json.dumps(x) + "\n" for x in records)); save_checkpoint(output / "last.pth", model, optimizer, scaler, metadata, epoch, best, best_epoch, bad_epochs, elapsed)
        if improved: save_checkpoint(output / "best.pth", model, optimizer, scaler, metadata, epoch, best, best_epoch, bad_epochs, elapsed)
        print(json.dumps(record), flush=True)
        if epoch >= args.min_epochs and bad_epochs >= args.patience: print(json.dumps({"early_stop": epoch}), flush=True); break
    elapsed = elapsed_before + time.time() - started; state = torch.load(output / "best.pth", map_location=args.device); load_lora_state_dict(model, state["lora"]); final = evaluate(model, criterion, val_loader, args.device, amp); final.update({"best_epoch": state["best_epoch"], "best_val_total_loss": state["best_val_total_loss"]})
    (output / "final_val_metrics.json").write_text(json.dumps(final, indent=2)); (output / "metrics.json").write_text(json.dumps({"initial": initial, "final": final}, indent=2)); provenance = {"protocol_version": "protocol_v1", "protocol_sha256": sha256_file(protocol_path), "git_commit_hash": git_commit_or_unavailable(str(Path(__file__).resolve().parents[2])), "base_brainx_checkpoint": str(Path(args.brainx_checkpoint).resolve()), "base_brainx_sha256": checkpoint_hash, "target_modules": model.target_modules, "rank": args.rank, "alpha": args.alpha, "dropout": args.lora_dropout, "seed": args.seed, "gpu": torch.cuda.get_device_name(0) if amp else "CPU", "training_seconds": elapsed}; (output / "provenance.json").write_text(json.dumps(provenance, indent=2)); print(json.dumps({"final_validation": final, "provenance": provenance}), flush=True); return final


if __name__ == "__main__": run(parse_args())
