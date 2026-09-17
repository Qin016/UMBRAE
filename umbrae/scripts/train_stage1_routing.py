#!/usr/bin/env python
"""NeuroRoute Stage-1 ROI-to-CLIP-layer representation alignment."""

import argparse
import glob
import io
import json
import math
import random
import tarfile
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.alignment_loss import roi_clip_alignment_loss
from losses.routing_loss import routing_regularization_loss
from models.brain_clip_projector import BrainToCLIPProjector
from models.clip_layer_bank import CLIPLayerBank
from models.roi_layer_router import ROILayerRouter
from models.roi_tokenizer import ROITokenizer
from models.retrieval_pooler import RetrievalPooler


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    known, _ = pre_parser.parse_known_args()
    config_defaults = {}
    if known.config:
        config_path = Path(known.config).expanduser().resolve()
        config_defaults = json.loads(config_path.read_text())
    parser = argparse.ArgumentParser(
        description="Train NeuroRoute Stage-1 ROI-layer alignment",
        parents=[pre_parser],
    )
    required = not bool(config_defaults)
    parser.add_argument("--subject", required=required)
    train_group = parser.add_mutually_exclusive_group(required=required)
    train_group.add_argument("--train-tar", nargs="+")
    train_group.add_argument("--train-data", nargs="+")
    val_group = parser.add_mutually_exclusive_group(required=required)
    val_group.add_argument("--val-tar", nargs="+")
    val_group.add_argument("--val-data", nargs="+")
    parser.add_argument("--roi-mapping-json", required=required)
    parser.add_argument(
        "--selected-clip-layers",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 20, 24],
    )
    parser.add_argument(
        "--clip-model-name-or-path",
        default="openai/clip-vit-large-patch14",
    )
    parser.add_argument("--roi-token-dim", type=int, default=1024)
    parser.add_argument("--clip-layer-target-dim", type=int, default=1024)
    parser.add_argument(
        "--use-brain-clip-projector",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--projector-type",
        choices=["linear", "mlp", "residual_mlp"],
        default="mlp",
    )
    parser.add_argument("--projector-hidden-dim", type=int)
    parser.add_argument("--projector-dropout", type=float, default=0.1)
    parser.add_argument(
        "--detach-routed-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--router-hidden-dim", type=int)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument(
        "--router-type",
        choices=["soft", "uniform", "random", "hard", "single"],
        default="soft",
    )
    parser.add_argument("--router-topk", type=int, default=0)
    parser.add_argument("--single-router-layer", type=int)
    parser.add_argument(
        "--hard-intermediate-layers",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20],
    )
    parser.add_argument(
        "--hard-final-layers", type=int, nargs="+", default=[24]
    )
    parser.add_argument(
        "--fmri-repeat-mode",
        choices=["mean", "first", "random"],
        default="mean",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--cos-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument(
        "--contrastive-image-target",
        choices=["l24", "routed", "uniform_fused"],
        default="l24",
    )
    parser.add_argument(
        "--contrastive-brain-pooling",
        choices=["mean", "attention", "roi_flatten_mlp"],
        default="mean",
    )
    parser.add_argument("--router-entropy-weight", type=float, default=0.0)
    parser.add_argument("--router-balance-weight", type=float, default=0.0)
    parser.add_argument("--router-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=required)
    parser.add_argument("--debug-max-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    missing = [
        name
        for name in ("subject", "roi_mapping_json", "output_dir")
        if not getattr(args, name, None)
    ]
    if not (args.train_tar or args.train_data):
        missing.append("train_tar/train_data")
    if not (args.val_tar or args.val_data):
        missing.append("val_tar/val_data")
    if missing:
        parser.error("missing required configuration: " + ", ".join(missing))
    return args


def expand_tar_paths(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        matches = sorted(glob.glob(str(Path(value).expanduser())))
        if not matches and Path(value).is_file():
            matches = [value]
        paths.extend(Path(match).resolve() for match in matches)
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(f"No tar files matched: {values}")
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing tar files: {missing}")
    return unique


class NSDTarDataset(Dataset):
    """Map-style reader for UMBRAE sample groups stored in tar shards."""

    def __init__(
        self,
        tar_paths: Sequence[Path],
        repeat_mode: str = "mean",
        require_coco_id: bool = False,
    ) -> None:
        self.tar_paths = list(tar_paths)
        self.repeat_mode = repeat_mode
        self.require_coco_id = require_coco_id
        self.samples = []
        for tar_path in self.tar_paths:
            with tarfile.open(tar_path) as archive:
                members = {
                    member.name: member for member in archive.getmembers()
                }
                names = set(members)
            prefixes = sorted(
                name[: -len(".nsdgeneral.npy")]
                for name in names
                if name.endswith(".nsdgeneral.npy")
            )
            for prefix in prefixes:
                image_name = next(
                    (
                        f"{prefix}.{extension}"
                        for extension in ("jpg", "png")
                        if f"{prefix}.{extension}" in names
                    ),
                    None,
                )
                if image_name is None:
                    continue
                repeats_name = f"{prefix}.num_uniques.npy"
                coco_name = f"{prefix}.coco73k.npy"
                self.samples.append((
                    tar_path,
                    prefix,
                    members[f"{prefix}.nsdgeneral.npy"],
                    members.get(repeats_name),
                    members[image_name],
                    members.get(coco_name),
                ))
        if not self.samples:
            raise ValueError("No image/nsdgeneral sample pairs found")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_npy(archive: tarfile.TarFile, member_or_name) -> np.ndarray:
        member = archive.extractfile(member_or_name)
        if member is None:
            raise FileNotFoundError(
                f"Missing tar member: {member_or_name}"
            )
        return np.load(io.BytesIO(member.read()), allow_pickle=False)

    def _select_repeats(self, fmri: np.ndarray, valid_repeats: int) -> np.ndarray:
        if fmri.ndim == 1:
            return fmri
        if fmri.ndim != 2:
            raise ValueError(f"Expected nsdgeneral [V] or [R,V], got {fmri.shape}")
        count = max(1, min(int(valid_repeats), fmri.shape[0]))
        valid = fmri[:count]
        if self.repeat_mode == "mean":
            return valid.astype(np.float32).mean(axis=0)
        if self.repeat_mode == "first":
            return valid[0]
        index = int(torch.randint(count, size=(1,)).item())
        return valid[index]

    def __getitem__(self, index: int) -> Dict[str, object]:
        (
            tar_path,
            prefix,
            fmri_member,
            repeats_member,
            image_info,
            coco_member,
        ) = self.samples[index]
        with tarfile.open(tar_path) as archive:
            fmri = self._read_npy(archive, fmri_member)
            valid_repeats = (
                int(self._read_npy(archive, repeats_member).reshape(-1)[0])
                if repeats_member is not None
                else fmri.shape[0] if fmri.ndim == 2 else 1
            )
            image_member = archive.extractfile(image_info)
            if image_member is None:
                raise FileNotFoundError(
                    f"Missing tar member: {image_info.name}"
                )
            image = Image.open(io.BytesIO(image_member.read())).convert("RGB")
            if self.require_coco_id and coco_member is None:
                raise FileNotFoundError(
                    f"Retrieval sample {prefix} is missing "
                    f"{prefix}.coco73k.npy"
                )
            coco_id = (
                int(self._read_npy(archive, coco_member).reshape(-1)[0])
                if coco_member is not None
                else index
            )

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        fmri_tensor = torch.from_numpy(
            np.asarray(
                self._select_repeats(fmri, valid_repeats), dtype=np.float32
            )
        )
        return {
            "fmri": fmri_tensor,
            "image": image_tensor,
            "key": prefix,
            "coco73k_id": coco_id,
        }


def load_roi_mapping(path: str) -> tuple[List[str], Dict[str, List[int]]]:
    mapping_path = Path(path).expanduser().resolve()
    payload = json.loads(mapping_path.read_text())
    if not payload.get("roi_mapping_is_real", False):
        raise ValueError("ROI mapping is not marked as real")
    if not payload.get("voxel_order_verified", False):
        raise ValueError("ROI mapping voxel order is not verified")
    roi_names = list(payload["roi_names"])
    roi_indices = {
        name: list(payload["rois"][name]["indices"]) for name in roi_names
    }
    empty = [name for name, indices in roi_indices.items() if not indices]
    if empty:
        raise ValueError(f"Required ROI mappings are empty: {empty}")
    return roi_names, roi_indices


class FixedLayerRouter(nn.Module):
    """Uniform, random, or heuristic hard routing with the soft-router API."""

    EARLY_ROIS = {"V1", "V2", "V3", "hV4"}

    def __init__(
        self,
        router_type: str,
        roi_names: Sequence[str],
        selected_layers: Sequence[int],
        seed: int,
        intermediate_layers: Sequence[int],
        final_layers: Sequence[int],
        single_layer: Optional[int] = None,
    ) -> None:
        super().__init__()
        roi_count, layer_count = len(roi_names), len(selected_layers)
        if router_type == "uniform":
            weights = torch.full((roi_count, layer_count), 1.0 / layer_count)
        elif router_type == "random":
            generator = torch.Generator().manual_seed(seed)
            weights = torch.rand(roi_count, layer_count, generator=generator)
            weights = weights / weights.sum(dim=-1, keepdim=True)
        elif router_type == "hard":
            weights = torch.zeros(roi_count, layer_count)
            layer_to_position = {
                int(layer): position
                for position, layer in enumerate(selected_layers)
            }
            for roi_position, roi_name in enumerate(roi_names):
                requested = (
                    intermediate_layers
                    if roi_name in self.EARLY_ROIS
                    else final_layers
                )
                positions = [
                    layer_to_position[int(layer)]
                    for layer in requested
                    if int(layer) in layer_to_position
                ]
                if not positions:
                    raise ValueError(
                        f"Hard router has no selected layers for ROI {roi_name}"
                    )
                weights[roi_position, positions] = 1.0 / len(positions)
        elif router_type == "single":
            if single_layer is None:
                raise ValueError(
                    "single_router_layer is required for router_type='single'"
                )
            if int(single_layer) not in selected_layers:
                raise ValueError(
                    f"single_router_layer={single_layer} is not in "
                    f"selected_clip_layers={list(selected_layers)}"
                )
            weights = torch.zeros(roi_count, layer_count)
            position = list(selected_layers).index(int(single_layer))
            weights[:, position] = 1.0
        else:
            raise ValueError(f"Unsupported fixed router type: {router_type}")
        self.router_type = router_type
        self.register_buffer("fixed_weights", weights, persistent=True)

    def forward(
        self, roi_tokens: Tensor, clip_layer_features: Tensor
    ) -> Dict[str, object]:
        batch_size = roi_tokens.shape[0]
        weights = self.fixed_weights.unsqueeze(0).expand(batch_size, -1, -1)
        routed_targets = torch.matmul(weights, clip_layer_features)
        entropy = -(
            weights * weights.clamp_min(torch.finfo(weights.dtype).eps).log()
        ).sum(dim=-1)
        return {
            "routed_targets": routed_targets,
            "routing_weights": weights,
            "routing_logits": None,
            "diagnostics": {
                "temperature": None,
                "mean_entropy": entropy.mean().detach(),
                "mean_max_weight": weights.max(dim=-1).values.mean().detach(),
                "active_layers_per_roi": (
                    weights > 0
                ).sum(dim=-1).float().mean().detach(),
                "topk": None,
                "router_type": self.router_type,
            },
        }


class Stage1RoutingModel(nn.Module):
    def __init__(
        self,
        roi_names: Sequence[str],
        roi_indices: Mapping[str, Sequence[int]],
        selected_layers: Sequence[int],
        feature_dim: int,
        router_type: str,
        clip_model_name_or_path: str,
        router_hidden_dim: Optional[int],
        router_temperature: float,
        router_topk: int,
        hard_intermediate_layers: Sequence[int],
        hard_final_layers: Sequence[int],
        seed: int,
        single_router_layer: Optional[int] = None,
        use_brain_clip_projector: bool = True,
        projector_type: str = "mlp",
        projector_hidden_dim: Optional[int] = None,
        projector_dropout: float = 0.1,
        detach_routed_targets: bool = True,
        contrastive_brain_pooling: str = "mean",
        contrastive_image_target: str = "routed",
        clip_model=None,
    ) -> None:
        super().__init__()
        self.roi_names = list(roi_names)
        self.roi_tokenizer = ROITokenizer(
            roi_names=roi_names,
            roi_indices=roi_indices,
            token_dim=feature_dim,
            tokenizer_type="shared_mlp",
            use_roi_embeddings=True,
        )
        self.clip_layer_bank = CLIPLayerBank(
            model_name_or_path=clip_model_name_or_path,
            selected_layers=selected_layers,
            target_dim=feature_dim,
            freeze_clip=True,
            clip_model=clip_model,
        )
        # Stage-1 uses CLIP features as a stable target space.
        self.clip_layer_bank.requires_grad_(False)
        self.clip_layer_bank.eval()
        if router_type == "soft":
            self.router = ROILayerRouter(
                feature_dim=feature_dim,
                hidden_dim=router_hidden_dim,
                temperature=router_temperature,
                topk=router_topk or None,
            )
        else:
            self.router = FixedLayerRouter(
                router_type=router_type,
                roi_names=roi_names,
                selected_layers=selected_layers,
                seed=seed,
                intermediate_layers=hard_intermediate_layers,
                final_layers=hard_final_layers,
                single_layer=single_router_layer,
            )
        self.router_type = router_type
        self.use_brain_clip_projector = use_brain_clip_projector
        self.detach_routed_targets = detach_routed_targets
        self.contrastive_image_target = contrastive_image_target
        self.brain_clip_projector = (
            BrainToCLIPProjector(
                input_dim=feature_dim,
                output_dim=feature_dim,
                projector_type=projector_type,
                hidden_dim=projector_hidden_dim,
                dropout=projector_dropout,
            )
            if use_brain_clip_projector
            else nn.Identity()
        )
        self.retrieval_pooler = RetrievalPooler(
            contrastive_brain_pooling,
            len(roi_names),
            feature_dim,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_layer_bank.eval()
        return self

    def forward(self, fmri: Tensor, images: Tensor) -> Dict[str, object]:
        roi_output = self.roi_tokenizer(fmri)
        with torch.no_grad():
            clip_output = self.clip_layer_bank(images)
        raw_roi_tokens = roi_output["roi_tokens"]
        projected_roi_tokens = self.brain_clip_projector(raw_roi_tokens)
        clip_layer_features = clip_output["pooled_tokens"]
        routing_features = (
            clip_layer_features.detach()
            if self.detach_routed_targets
            else clip_layer_features
        )
        router_output = self.router(raw_roi_tokens, routing_features)
        brain_global = self.retrieval_pooler(projected_roi_tokens)
        if self.contrastive_image_target == "l24":
            if 24 not in self.clip_layer_bank.selected_layers:
                raise ValueError(
                    "contrastive-image-target=l24 requires selected layer 24"
                )
            image_global = clip_layer_features[
                :, self.clip_layer_bank.selected_layers.index(24)
            ]
        elif self.contrastive_image_target == "uniform_fused":
            image_global = clip_layer_features.mean(dim=1)
        else:
            image_global = router_output["routed_targets"].mean(dim=1)
        return {
            **roi_output,
            "projected_roi_tokens": projected_roi_tokens,
            **router_output,
            "clip_layer_features": clip_layer_features,
            "selected_layers": clip_output["selected_layers"],
            "brain_global_embedding": brain_global,
            "image_global_embedding": image_global,
        }


def compute_losses(
    output: Mapping[str, object], args: argparse.Namespace
) -> Dict[str, Tensor]:
    alignment = roi_clip_alignment_loss(
        output["projected_roi_tokens"],
        output["routed_targets"],
        mse_weight=args.mse_weight,
        cosine_weight=args.cos_weight,
    )
    routing = routing_regularization_loss(
        output["routing_weights"],
        entropy_weight=args.router_entropy_weight,
        balance_weight=args.router_balance_weight,
        smoothness_weight=args.router_smoothness_weight,
    )
    contrastive = output["projected_roi_tokens"].new_zeros(())
    if getattr(args, "contrastive_loss_weight", 0.0) > 0:
        temperature = float(args.contrastive_temperature)
        if temperature <= 0:
            raise ValueError("contrastive-temperature must be positive")
        brain_global = F.normalize(
            output["brain_global_embedding"], dim=-1
        )
        image_global = F.normalize(
            output["image_global_embedding"], dim=-1
        )
        logits = brain_global @ image_global.transpose(0, 1) / temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        contrastive = 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.transpose(0, 1), targets)
        )
    return {
        "total": (
            alignment["total"]
            + routing["total"]
            + getattr(args, "contrastive_loss_weight", 0.0) * contrastive
        ),
        "mse": alignment["mse"],
        "cosine": alignment["cosine"],
        "entropy": routing["entropy"],
        "balance": routing["balance"],
        "smoothness": routing["smoothness"],
        "contrastive": contrastive,
    }


def run_epoch(
    model: Stage1RoutingModel,
    loader: DataLoader,
    args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int = 1,
    progress=None,
) -> tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    weight_batches = []
    steps = 0
    phase = "train" if training else "val"
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            fmri = batch["fmri"].to(args.device)
            images = batch["image"].to(args.device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(fmri, images)
            losses = compute_losses(output, args)
            if training:
                losses["total"].backward()
                optimizer.step()
            diagnostics = output["diagnostics"]
            projected_norm = output["projected_roi_tokens"].norm(dim=-1).mean()
            target_norm = output["routed_targets"].norm(dim=-1).mean()
            raw_roi_norm = output["roi_tokens"].norm(dim=-1).mean()
            brain_global = F.normalize(
                output["brain_global_embedding"], dim=-1
            )
            image_global = F.normalize(
                output["image_global_embedding"], dim=-1
            )
            proxy_similarity = brain_global @ image_global.transpose(0, 1)
            proxy_targets = torch.arange(
                proxy_similarity.shape[0],
                device=proxy_similarity.device,
            )
            retrieval_proxy_r1 = (
                proxy_similarity.argmax(dim=1) == proxy_targets
            ).float().mean()
            values = {
                "total_loss": float(losses["total"].detach()),
                "mse_loss": float(losses["mse"].detach()),
                "cosine_loss": float(losses["cosine"].detach()),
                "entropy_loss": float(losses["entropy"].detach()),
                "balance_loss": float(losses["balance"].detach()),
                "smoothness_loss": float(losses["smoothness"].detach()),
                "contrastive_loss": float(
                    losses["contrastive"].detach()
                ),
                "contrastive_loss_weight": float(
                    getattr(args, "contrastive_loss_weight", 0.0)
                ),
                "routing_entropy_mean": float(diagnostics["mean_entropy"]),
                "routing_max_weight_mean": float(
                    diagnostics["mean_max_weight"]
                ),
                "projected_feature_norm": float(projected_norm.detach()),
                "routed_target_norm": float(target_norm.detach()),
                "raw_roi_token_norm": float(raw_roi_norm.detach()),
                "retrieval_proxy_r1": float(
                    retrieval_proxy_r1.detach()
                ),
                "learning_rate": (
                    float(optimizer.param_groups[0]["lr"])
                    if training
                    else float(args.lr)
                ),
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value
            if progress is not None:
                progress.set_postfix_str(
                    f"{phase} loss={values['total_loss']:.4f} "
                    f"maxw={values['routing_max_weight_mean']:.3f} "
                    f"norm={values['projected_feature_norm']:.2f}/"
                    f"{values['routed_target_norm']:.2f}"
                )
            if not training:
                weight_batches.append(
                    output["routing_weights"].detach().cpu().numpy()
                )
            steps += 1
            if progress is not None:
                progress.update(1)
            if args.debug_max_steps and steps >= args.debug_max_steps:
                break
    if steps == 0:
        raise RuntimeError("Data loader produced no batches")
    metrics = {key: value / steps for key, value in totals.items()}
    metrics["steps"] = steps
    if not weight_batches:
        return metrics, None, None
    weights = np.concatenate(weight_batches, axis=0).astype(np.float64)
    return metrics, weights.mean(axis=0), weights.std(axis=0)


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(dict(payload)) + "\n")


def build_routing_dynamics_record(
    epoch: int,
    roi_names: Sequence[str],
    selected_layers: Sequence[int],
    routing_mean: np.ndarray,
    val_metrics: Mapping[str, float],
) -> Dict[str, object]:
    """Summarize one validation epoch's ROI-to-layer routing behavior."""
    if routing_mean.shape != (len(roi_names), len(selected_layers)):
        raise ValueError(
            f"routing_mean shape {routing_mean.shape} does not match "
            f"R={len(roi_names)}, L={len(selected_layers)}"
        )
    top_positions = routing_mean.argmax(axis=-1)
    per_roi_top_layer = {
        roi_name: int(selected_layers[position])
        for roi_name, position in zip(roi_names, top_positions)
    }
    top_layer_counts = {
        str(layer): int(
            sum(position == layer_index for position in top_positions)
        )
        for layer_index, layer in enumerate(selected_layers)
    }
    layer_usage = routing_mean.mean(axis=0)
    return {
        "epoch": int(epoch),
        "selected_clip_layers": [int(layer) for layer in selected_layers],
        "roi_names": list(roi_names),
        "routing_entropy_mean": float(
            val_metrics["routing_entropy_mean"]
        ),
        "routing_max_weight_mean": float(
            val_metrics["routing_max_weight_mean"]
        ),
        "top_layer_counts": top_layer_counts,
        "per_roi_top_layer": per_roi_top_layer,
        "layer_usage_mean": {
            str(layer): float(value)
            for layer, value in zip(selected_layers, layer_usage)
        },
        "val_total_loss": float(val_metrics["total_loss"]),
        "val_mse_loss": float(val_metrics["mse_loss"]),
        "val_cosine_loss": float(val_metrics["cosine_loss"]),
    }


def save_checkpoint(
    path: Path,
    model: Stage1RoutingModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val_total_loss": best_val,
            "best_val_alignment_loss": (
                best_val
                if float(
                    getattr(args, "contrastive_loss_weight", 0.0)
                ) == 0.0
                else None
            ),
            "checkpoint_selection_metric": "total_loss",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
        },
        path,
    )


def run_training(args: argparse.Namespace, clip_model=None) -> Dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_values = args.train_tar or args.train_data
    val_values = args.val_tar or args.val_data
    train_dataset = NSDTarDataset(
        expand_tar_paths(train_values), args.fmri_repeat_mode
    )
    val_dataset = NSDTarDataset(
        expand_tar_paths(val_values), args.fmri_repeat_mode
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    roi_names, roi_indices = load_roi_mapping(args.roi_mapping_json)
    if args.roi_token_dim != args.clip_layer_target_dim:
        raise ValueError(
            "roi-token-dim and clip-layer-target-dim must match for alignment"
        )
    model = Stage1RoutingModel(
        roi_names=roi_names,
        roi_indices=roi_indices,
        selected_layers=args.selected_clip_layers,
        feature_dim=args.roi_token_dim,
        router_type=args.router_type,
        clip_model_name_or_path=args.clip_model_name_or_path,
        router_hidden_dim=args.router_hidden_dim,
        router_temperature=args.router_temperature,
        router_topk=args.router_topk,
        hard_intermediate_layers=args.hard_intermediate_layers,
        hard_final_layers=args.hard_final_layers,
        single_router_layer=getattr(args, "single_router_layer", None),
        seed=args.seed,
        use_brain_clip_projector=getattr(args, "use_brain_clip_projector", True),
        projector_type=getattr(args, "projector_type", "mlp"),
        projector_hidden_dim=getattr(args, "projector_hidden_dim", None),
        projector_dropout=getattr(args, "projector_dropout", 0.1),
        detach_routed_targets=getattr(args, "detach_routed_targets", True),
        contrastive_brain_pooling=getattr(
            args, "contrastive_brain_pooling", "mean"
        ),
        contrastive_image_target=getattr(
            args, "contrastive_image_target", "routed"
        ),
        clip_model=clip_model,
    ).to(args.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    config = dict(vars(args))
    config["train_tar_resolved"] = [str(path) for path in expand_tar_paths(train_values)]
    config["val_tar_resolved"] = [str(path) for path in expand_tar_paths(val_values)]
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "roi_names.json").write_text(json.dumps(roi_names, indent=2))
    (output_dir / "selected_clip_layers.json").write_text(
        json.dumps(list(args.selected_clip_layers), indent=2)
    )
    train_metrics_path = output_dir / "metrics_train.jsonl"
    val_metrics_path = output_dir / "metrics_val.jsonl"
    routing_dynamics_path = output_dir / "routing_dynamics.jsonl"
    train_metrics_path.write_text("")
    val_metrics_path.write_text("")
    routing_dynamics_path.write_text("")

    best_val = math.inf
    final_mean = final_std = None
    for epoch in range(1, args.epochs + 1):
        train_steps = (
            min(len(train_loader), args.debug_max_steps)
            if args.debug_max_steps
            else len(train_loader)
        )
        val_steps = (
            min(len(val_loader), args.debug_max_steps)
            if args.debug_max_steps
            else len(val_loader)
        )
        show_progress = getattr(args, "show_progress", True)
        epoch_progress = tqdm(
            total=train_steps + val_steps,
            desc=f"epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
            disable=not show_progress,
            leave=True,
        )
        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            args,
            optimizer,
            epoch=epoch,
            progress=epoch_progress if show_progress else None,
        )
        val_metrics, final_mean, final_std = run_epoch(
            model,
            val_loader,
            args,
            optimizer=None,
            epoch=epoch,
            progress=epoch_progress if show_progress else None,
        )
        train_record = {"epoch": epoch, **train_metrics}
        val_record = {"epoch": epoch, **val_metrics}
        append_jsonl(train_metrics_path, train_record)
        append_jsonl(val_metrics_path, val_record)
        np.save(
            output_dir
            / f"val_routing_weights_mean_epoch{epoch:03d}.npy",
            final_mean,
        )
        np.save(
            output_dir
            / f"val_routing_weights_std_epoch{epoch:03d}.npy",
            final_std,
        )
        append_jsonl(
            routing_dynamics_path,
            build_routing_dynamics_record(
                epoch=epoch,
                roi_names=roi_names,
                selected_layers=args.selected_clip_layers,
                routing_mean=final_mean,
                val_metrics=val_metrics,
            ),
        )
        # Select the checkpoint using the actual configured objective. With
        # contrastive_loss_weight=0 this preserves the legacy alignment-only
        # behavior; contrastive runs are no longer selected while ignoring
        # their retrieval objective.
        current_val = val_metrics["total_loss"]
        save_checkpoint(
            output_dir / "checkpoint_last.pt",
            model,
            optimizer,
            epoch,
            min(best_val, current_val),
            args,
        )
        if current_val < best_val:
            best_val = current_val
            save_checkpoint(
                output_dir / "checkpoint_best.pt",
                model,
                optimizer,
                epoch,
                best_val,
                args,
            )
        if show_progress:
            epoch_progress.set_postfix_str(
                f"done train={train_metrics['total_loss']:.4f} "
                f"val={val_metrics['total_loss']:.4f} "
                f"mse={val_metrics['mse_loss']:.4f} "
                f"cos={val_metrics['cosine_loss']:.4f}"
            )
            epoch_progress.close()
        else:
            print(
                f"epoch={epoch} train_total={train_metrics['total_loss']:.6f} "
                f"val_total={val_metrics['total_loss']:.6f} "
                f"val_mse={val_metrics['mse_loss']:.6f} "
                f"val_cos={val_metrics['cosine_loss']:.6f}"
            )

    np.save(output_dir / "val_routing_weights_mean.npy", final_mean)
    np.save(output_dir / "val_routing_weights_std.npy", final_std)
    print(f"Saved Stage-1 outputs to {output_dir}")
    return {
        "model": model,
        "best_val_total_loss": best_val,
        "best_val_alignment_loss": (
            best_val
            if float(getattr(args, "contrastive_loss_weight", 0.0)) == 0.0
            else None
        ),
        "routing_mean": final_mean,
        "routing_std": final_std,
        "output_dir": output_dir,
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
