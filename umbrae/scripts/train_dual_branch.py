#!/usr/bin/env python
"""P2 Stage-A/B skeleton. It is intentionally not a full training recipe."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.semantic_uot_loss import SemanticUOTLoss
from models.clip_patch_teacher import FixedCLIPPatchTeacher
from models.dual_branch_umbrae import DualBranchUMBRAE
from models.gated_cross_attention import GatedCrossAttentionFusion
from models.roi_mapping import EXPECTED_NSDGENERAL_VOXELS, normalize_subject_id
from models.structural_branch import StructuralBranch
from models.umbrae_backbone import FrozenUMBRAEEncoder
from scripts.train_stage1_routing import NSDTarDataset, expand_tar_paths


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    known, _ = pre_parser.parse_known_args()
    defaults = {}
    if known.config:
        defaults = json.loads(Path(known.config).expanduser().read_text())
    parser = argparse.ArgumentParser(
        description="Dual-branch UMBRAE P2 Stage-A/B skeleton",
        parents=[pre_parser],
    )
    required = not bool(defaults)
    parser.add_argument("--stage", choices=["A", "B"], required=required)
    parser.add_argument("--subject", required=required)
    parser.add_argument("--train-tar", nargs="+", required=required)
    parser.add_argument("--roi-mapping", required=required)
    parser.add_argument("--brainx-checkpoint", required=required)
    parser.add_argument("--clip-model", default=FixedCLIPPatchTeacher.MODEL_NAME)
    parser.add_argument("--token-dim", type=int, default=1024)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--sinkhorn-iters", type=int, default=100)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-5)
    parser.add_argument("--repeat-aggregation", choices=["mean", "first"], default="mean")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=required)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-fusion-connectivity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Smoke-only gradient probe; not part of the Stage-B formal objective.",
    )
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    missing = [
        name
        for name in (
            "stage", "subject", "train_tar", "roi_mapping",
            "brainx_checkpoint", "output_dir",
        )
        if not getattr(args, name, None)
    ]
    if missing:
        parser.error("missing required configuration: " + ", ".join(missing))
    return args


def gradient_norm(module: torch.nn.Module) -> float:
    squared = torch.zeros((), device=next(module.parameters()).device)
    found = False
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum()
            found = True
    return float(squared.sqrt()) if found else 0.0


def stage_b_loss_hooks(
    output: Dict[str, object], *, smoke_fusion_connectivity: bool
) -> Dict[str, Tensor]:
    """Extension point for the next phase's global/relational objective."""
    if not smoke_fusion_connectivity:
        return {}
    # This term exists only to verify the fusion gradient path during smoke
    # tests. It is not a proposed or default Stage-B scientific objective.
    return {
        "smoke_fusion_connectivity": (
            output["z_cal"] - output["z_sem"]
        ).float().square().mean()
    }


def build_components(args):
    subject_name = normalize_subject_id(args.subject)
    subject_id = int(subject_name[-2:])
    expected_voxels = EXPECTED_NSDGENERAL_VOXELS.get(subject_id)
    if expected_voxels is None:
        raise ValueError(f"No NSD voxel count registered for {subject_name}")
    semantic = FrozenUMBRAEEncoder(args.brainx_checkpoint, subject_name)
    structural = StructuralBranch(
        token_dim=args.token_dim,
        mapping_path=args.roi_mapping,
        subject=subject_name,
        expected_voxel_count=expected_voxels,
    )
    fusion = GatedCrossAttentionFusion(
        dim=args.token_dim, num_heads=args.fusion_heads, gate_init=-4.0
    )
    model = DualBranchUMBRAE(
        semantic, structural, fusion, args.repeat_aggregation
    )
    model.configure_stage(args.stage, verbose=True)
    teacher = FixedCLIPPatchTeacher(args.clip_model)
    teacher.requires_grad_(False)
    teacher.eval()
    criterion = SemanticUOTLoss(
        epsilon=args.epsilon,
        tau=args.tau,
        num_iters=args.sinkhorn_iters,
        tol=args.sinkhorn_tol,
    )
    return model, teacher, criterion


def run(args: argparse.Namespace) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    if args.max_steps < 1:
        raise ValueError("max_steps must be at least one")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    dataset = NSDTarDataset(
        expand_tar_paths(args.train_tar), args.repeat_aggregation
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    model, teacher, criterion = build_components(args)
    model = model.to(args.device)
    teacher = teacher.to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    records = []
    model.train()
    for step, batch in enumerate(loader, start=1):
        fmri = batch["fmri"].to(args.device)
        images = batch["image"].to(args.device)
        optimizer.zero_grad(set_to_none=True)
        if args.stage == "A":
            output = model(
                fmri,
                enable_semantic=False,
                enable_structural=True,
                enable_fusion=False,
            )
        else:
            output = model(fmri)
        v_patch = teacher(images)
        uot = criterion(output["h_struct"], v_patch)
        hooks = stage_b_loss_hooks(
            output,
            smoke_fusion_connectivity=(
                args.stage == "B" and args.smoke_fusion_connectivity
            ),
        )
        loss = uot["loss"] + sum(hooks.values(), uot["loss"].new_zeros(()))
        loss.backward()
        optimizer.step()
        record = {
            "step": step,
            "uot_loss": float(uot["loss"].detach()),
            "transport_mass": float(uot["transport_mass"].detach()),
            "transport_entropy": float(uot["transport_entropy"].detach()),
            "structural_grad_norm": gradient_norm(model.structural_branch),
            "fusion_grad_norm": gradient_norm(model.fusion),
            "gate": float(torch.sigmoid(model.fusion.gate_logit).detach()),
            "brainx_trainable_params": sum(
                p.numel() for p in model.semantic_backbone.parameters() if p.requires_grad
            ),
            "smoke_only_hook": bool(hooks),
        }
        if not all(
            torch.isfinite(torch.tensor(value))
            for key, value in record.items()
            if key not in {"step", "smoke_only_hook"}
        ):
            raise RuntimeError(f"Non-finite smoke diagnostics: {record}")
        print(json.dumps(record))
        records.append(record)
        if step >= args.max_steps:
            break
    peak_memory = (
        int(torch.cuda.max_memory_allocated(args.device))
        if str(args.device).startswith("cuda")
        else 0
    )
    result = {
        "stage": args.stage,
        "steps": records,
        "peak_gpu_memory_bytes": peak_memory,
        "parameter_report": model.parameter_report(),
        "formal_stage_b_losses": ["semantic_uot"],
        "stage_b_loss_hook_reserved": True,
    }
    (output_dir / "smoke_metrics.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())
