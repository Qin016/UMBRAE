#!/usr/bin/env python
"""Train only the frozen-representation P10 token-mixing adapter."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from losses.interface_distillation_loss import InterfaceDistillationLoss
from models.dual_branch_cache import sha256_file
from models.token_mixing_interface_adapter import (
    TokenMixingInterfaceAdapter,
    adapter_parameter_counts,
    attention_diagnostics,
)
from scripts.analyze_p8_projector import hard_retrieval, rsa_any
from scripts.cache_p7_downstream_features import load_projector, resolve_config


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    cfg["_config_path"] = str(Path(path).resolve())
    for key in (
        "source_train", "source_val", "visual_train", "visual_val",
        "projected_teacher_train", "projected_teacher_val", "output_dir",
        "p7_config", "p9_output_dir", "cache_hash_manifest",
    ):
        cfg[key] = str(Path(cfg[key]).resolve())
    return cfg


def arrays(cfg, split):
    return tuple(np.load(cfg[f"{name}_{split}"], mmap_mode="r") for name in ("source", "visual", "projected_teacher"))


def adapter_state_sha256(adapter):
    digest = hashlib.sha256()
    for name, value in sorted(adapter.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def forward_loss(adapter, projector, criterion, source, visual, teacher, amp, return_attention=False):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        output = adapter(source, return_details=True, return_attention=return_attention)
        projected = projector(output["z_hat"])
        losses = criterion(source, output["z_hat"], visual, projected, teacher)
    return output, projected, losses


def aggregate_attention(rows, weights):
    total = float(sum(weights))
    scalar_keys = (
        "mean_attention_entropy", "normalized_attention_entropy",
        "mean_max_attention_weight", "off_diagonal_attention_mass",
    )
    result = {key: sum(row[key] * weight for row, weight in zip(rows, weights)) / total for key in scalar_keys}
    result["token_count"] = rows[0]["token_count"]
    result["per_head"] = []
    for head in range(len(rows[0]["per_head"])):
        item = {"head": head}
        for key in (
            "mean_attention_entropy", "normalized_attention_entropy",
            "mean_max_attention_weight", "off_diagonal_attention_mass",
            "token_usage_entropy", "normalized_token_usage_entropy",
            "most_used_token_share",
        ):
            item[key] = sum(row["per_head"][head][key] * weight for row, weight in zip(rows, weights)) / total
        # The aggregate received-token argmax is approximated by the modal batch argmax;
        # collapse decisions use entropy/share, not this descriptive index.
        indices = [row["per_head"][head]["most_used_token"] for row in rows]
        item["most_used_token"] = max(set(indices), key=indices.count)
        result["per_head"].append(item)
    result["attention_collapse_detected"] = any(
        head["most_used_token_share"] > 0.25 or head["normalized_token_usage_entropy"] < 0.5
        for head in result["per_head"]
    )
    result["collapse_rule"] = "any head most_used_token_share > 0.25 or normalized_token_usage_entropy < 0.5"
    return result


def evaluate(adapter, projector, criterion, cfg, device):
    source, visual, teacher = arrays(cfg, "val")
    sums = {key: 0.0 for key in ("loss", "proj_cos", "proj_mse", "preproj_cos", "preserve", "norm")}
    pooled, visual_pooled, correction, projected_norm, teacher_norm = [], [], [], [], []
    attention_rows, attention_weights = [], []
    samples = 0
    adapter.eval()
    for start in range(0, len(source), cfg["batch_size"]):
        end = min(start + cfg["batch_size"], len(source))
        z = torch.from_numpy(np.array(source[start:end], dtype=np.float32, copy=True)).to(device)
        v = torch.from_numpy(np.array(visual[start:end], dtype=np.float32, copy=True)).to(device)
        h = torch.from_numpy(np.array(teacher[start:end], dtype=np.float32, copy=True)).to(device)
        with torch.inference_mode():
            out, projected, loss = forward_loss(adapter, projector, criterion, z, v, h, cfg["amp"], return_attention=True)
        size = end - start
        for key in sums:
            sums[key] += float(loss[key]) * size
        pooled.append(out["z_hat"].float().mean(1).cpu().numpy())
        visual_pooled.append(v.float().mean(1).cpu().numpy())
        correction.extend((out["z_hat"].float() - z.float()).flatten(1).norm(dim=1).div(z.float().flatten(1).norm(dim=1).clamp_min(1e-8)).cpu().tolist())
        projected_norm.append(projected.float().norm(dim=-1).cpu().numpy())
        teacher_norm.append(h.float().norm(dim=-1).cpu().numpy())
        attention_rows.append(attention_diagnostics(out["attention_weights"]))
        attention_weights.append(size)
        samples += size
    pooled = np.concatenate(pooled)
    visual_pooled = np.concatenate(visual_pooled)
    projected_norm = np.concatenate(projected_norm)
    teacher_norm = np.concatenate(teacher_norm)
    losses = {key: value / samples for key, value in sums.items()}
    retrieval = hard_retrieval(pooled, visual_pooled, k=99)
    retrieval = {key: value for key, value in retrieval.items() if key != "ranks"}
    attention = aggregate_attention(attention_rows, attention_weights)
    metrics = {
        **{f"val_{key}": value for key, value in losses.items()},
        "projected_paired_token_cosine": 1 - losses["proj_cos"],
        "projected_oracle_gap": losses["proj_cos"],
        "preprojector_paired_cosine": 1 - losses["preproj_cos"],
        "source_preservation_cosine": 1 - losses["preserve"],
        "adapter_correction_ratio": float(np.mean(correction)),
        "projected_norm_ratio": float(projected_norm.mean() / teacher_norm.mean()),
        "g_attn": float(adapter.attn_gate),
        "g_ffn": float(adapter.ffn_gate),
        "rsa": rsa_any(pooled, visual_pooled),
        "hard_retrieval": retrieval,
        "attention": attention,
        "sample_count": samples,
    }
    return metrics


def checkpoint_state(adapter, optimizer, scaler, metadata, epoch, metrics):
    return {
        "adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "metrics": metrics,
        "metadata": dict(metadata),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics_per_epoch.jsonl").exists():
        raise RuntimeError("Refusing to append to an existing P10 training run")
    (out / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=2) + "\n")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    source, visual, teacher = arrays(cfg, "train")
    val = arrays(cfg, "val")
    expected = ((8559, 256, 1024), (8559, 256, 1024), (8559, 256, 4096))
    if tuple(x.shape for x in (source, visual, teacher)) != expected or tuple(x.shape[0] for x in val) != (300, 300, 300):
        raise ValueError("P10 cache shape mismatch")

    downstream = resolve_config(cfg["p7_config"])
    projector = load_projector(downstream["mm_projector"], args.device)
    if any(parameter.requires_grad for parameter in projector.parameters()):
        raise RuntimeError("Frozen mm_projector unexpectedly has trainable parameters")
    adapter = TokenMixingInterfaceAdapter(
        hidden_dim=1024, num_heads=cfg["num_heads"], bottleneck_dim=cfg["ffn_bottleneck"],
        gate_logit=cfg["gate_logit_init"], dropout=cfg["dropout"],
    ).to(args.device)
    initial_state_hash = adapter_state_sha256(adapter)
    criterion = InterfaceDistillationLoss(**cfg["loss_weights"]).to(args.device)
    counts = adapter_parameter_counts(adapter)
    gates = [adapter.attn_gate_logit, adapter.ffn_gate_logit]
    gate_ids = {id(parameter) for parameter in gates}
    block = [parameter for parameter in adapter.parameters() if id(parameter) not in gate_ids]
    optimizer = torch.optim.AdamW([
        {"params": block, "lr": cfg["adapter_lr"], "weight_decay": cfg["weight_decay"]},
        {"params": gates, "lr": cfg["gate_lr"], "weight_decay": 0.0},
    ])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"])

    probe = torch.from_numpy(np.array(val[0][:cfg["initialization_probe_count"]], dtype=np.float32, copy=True)).to(args.device)
    with torch.inference_mode():
        initial = adapter(probe)
        difference = initial - probe
        initialization = {
            "probe_count": len(probe),
            "max_abs_error": float(difference.abs().max()),
            "mean_abs_error": float(difference.abs().mean()),
            "mean_token_cosine": float(F.cosine_similarity(initial.float(), probe.float(), dim=-1).mean()),
            "pooled_cosine": float(F.cosine_similarity(initial.float().mean(1), probe.float().mean(1), dim=-1).mean()),
            "relative_delta_norm": float(difference.float().flatten(1).norm(dim=1).div(probe.float().flatten(1).norm(dim=1).clamp_min(1e-8)).mean()),
            "g_attn": float(adapter.attn_gate), "g_ffn": float(adapter.ffn_gate),
        }
    if initialization["max_abs_error"] != 0:
        raise RuntimeError("TokenMix adapter initialization is not exact identity")

    cache_manifest = json.loads(Path(cfg["cache_hash_manifest"]).read_text())
    metadata = {
        "source_kind": cfg["source_kind"], "source_checkpoint": cfg["source_checkpoint"],
        "source_checkpoint_sha256": sha256_file(cfg["source_checkpoint"]),
        "mm_projector": downstream["mm_projector"], "mm_projector_sha256": sha256_file(downstream["mm_projector"]),
        "shikra_identifier": downstream["shikra_model"],
        "clip_teacher": "openai/clip-vit-large-patch14 hidden_states[-2][:,1:,:]",
        "teacher_cache_hashes": cache_manifest["teacher_cache_hashes"],
        "p9_adapter_reference": "LayerNorm(1024)->Linear(1024,256)->GELU->Linear(256,1024), gated token-wise residual",
        "adapter_architecture": "LN->8-head MHSA->gated residual->LN->FFN(1024,256,1024)->gated residual",
        "protocol": "protocol_v1", "test_used": False,
    }
    provenance = {
        "experiment_name": cfg["experiment_name"], "source_kind": cfg["source_kind"],
        "trainable_policy": "tokenmix_adapter_and_two_gates_only", "parameter_counts": counts,
        "trainable_over_brainx_percent": 100 * counts["total_trainable"] / cfg["brainx_parameter_count"],
        "initialization_integrity": initialization, "initial_adapter_state_sha256": initial_state_hash,
        **metadata, "seed": cfg["seed"], "batch_size": cfg["batch_size"],
        "adapter_lr": cfg["adapter_lr"], "gate_lr": cfg["gate_lr"],
        "num_heads": cfg["num_heads"], "ffn_bottleneck": cfg["ffn_bottleneck"],
        "gate_logit_init": cfg["gate_logit_init"], "dropout": cfg["dropout"],
        "train_count": 8559, "validation_count": 300,
        "brain_representation_frozen": True, "projector_frozen": True,
        "shikra_used_in_training": False,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    generator = torch.Generator().manual_seed(cfg["seed"])
    best_gap, bad, started = float("inf"), 0, time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        adapter.train()
        order = torch.randperm(len(source), generator=generator).numpy()
        sums = {key: 0.0 for key in ("loss", "proj_cos", "proj_mse", "preproj_cos", "preserve", "norm")}
        count, grad_sum = 0, 0.0
        for start in range(0, len(order), cfg["batch_size"]):
            ids = order[start:start + cfg["batch_size"]]
            z = torch.from_numpy(np.array(source[ids], dtype=np.float32, copy=True)).to(args.device)
            v = torch.from_numpy(np.array(visual[ids], dtype=np.float32, copy=True)).to(args.device)
            h = torch.from_numpy(np.array(teacher[ids], dtype=np.float32, copy=True)).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            _, _, loss = forward_loss(adapter, projector, criterion, z, v, h, cfg["amp"])
            scaler.scale(loss["loss"]).backward()
            scaler.unscale_(optimizer)
            grad = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            size = len(ids)
            count += size
            grad_sum += grad * size
            for key in sums:
                sums[key] += float(loss[key]) * size
        validation = evaluate(adapter, projector, criterion, cfg, args.device)
        record = {
            "epoch": epoch, **{f"train_{key}": value / count for key, value in sums.items()},
            "train_grad_norm": grad_sum / count, **validation,
            "epoch_wall_seconds": time.time() - started,
        }
        checkpoint = checkpoint_state(adapter, optimizer, scaler, metadata, epoch, record)
        torch.save(checkpoint, out / f"epoch_{epoch:03d}.pth")
        torch.save(checkpoint, out / "last.pth")
        with (out / "metrics_per_epoch.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        with (out / "attention_diagnostics.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch, **record["attention"]}) + "\n")
        print(json.dumps({
            "experiment": cfg["experiment_name"], "epoch": epoch,
            "gap": record["projected_oracle_gap"], "g_attn": record["g_attn"],
            "g_ffn": record["g_ffn"], "correction": record["adapter_correction_ratio"],
            "attention_entropy": record["attention"]["mean_attention_entropy"],
        }), flush=True)
        if record["projected_oracle_gap"] < best_gap - 1e-7:
            best_gap, bad = record["projected_oracle_gap"], 0
            torch.save(checkpoint, out / "best_interface.pth")
        else:
            bad += 1
        if epoch >= cfg["minimum_epochs"] and bad >= cfg["patience"]:
            break
    (out / "interface_training_summary.json").write_text(json.dumps({
        "epochs_completed": epoch, "best_interface_gap": best_gap,
        "wall_seconds": time.time() - started, "downstream_selection_pending": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
