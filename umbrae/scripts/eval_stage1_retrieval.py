#!/usr/bin/env python
"""Evaluate Stage-1 fMRI-to-image retrieval without an MLLM."""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_stage1_routing import (
    NSDTarDataset,
    Stage1RoutingModel,
    expand_tar_paths,
    load_roi_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-1 NeuroRoute fMRI/image retrieval evaluation"
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--router-type",
        choices=["soft", "uniform", "single", "random", "hard"],
        required=True,
    )
    parser.add_argument("--single-router-layer", type=int, default=24)
    parser.add_argument("--val-tar", nargs="+", required=True)
    parser.add_argument("--roi-indices-path", required=True)
    parser.add_argument(
        "--selected-clip-layers",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 20, 24],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--brain-pooling",
        choices=["mean", "attention", "roi_flatten_mlp"],
        default="mean",
    )
    parser.add_argument(
        "--image-target",
        choices=["l24", "routed", "uniform_fused"],
        default="l24",
    )
    parser.add_argument(
        "--diagonal-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--oracle-image-token-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_ranks(similarity: np.ndarray) -> np.ndarray:
    """Return one-based diagonal-match ranks for a square similarity matrix."""
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square [N,N] matrix")
    if not np.isfinite(similarity).all():
        raise ValueError("similarity contains non-finite values")
    return compute_positive_ranks(
        similarity, np.eye(similarity.shape[0], dtype=bool)
    )


def compute_positive_ranks(
    similarity: np.ndarray, positive_mask: np.ndarray
) -> np.ndarray:
    """Return the one-based best-positive rank for every query."""
    similarity = np.asarray(similarity, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    if similarity.ndim != 2 or positive_mask.shape != similarity.shape:
        raise ValueError("positive_mask must match the [Q,G] similarity shape")
    if not np.isfinite(similarity).all():
        raise ValueError("similarity contains non-finite values")
    if np.any(positive_mask.sum(axis=1) == 0):
        raise ValueError("Every query must have at least one positive")
    order = np.argsort(-similarity, axis=1, kind="stable")
    ordered_positive = np.take_along_axis(positive_mask, order, axis=1)
    return np.argmax(ordered_positive, axis=1).astype(np.int64) + 1


def retrieval_metrics(ranks: np.ndarray) -> Dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.ndim != 1 or ranks.size == 0 or np.any(ranks < 1):
        raise ValueError("ranks must be a non-empty one-based vector")
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def load_static_routing_weights(
    model: Stage1RoutingModel,
    checkpoint_path: Path,
    roi_count: int,
    layer_count: int,
) -> tuple[Tensor, str]:
    routing_path = checkpoint_path.parent / "val_routing_weights_mean.npy"
    if routing_path.is_file():
        weights = np.load(routing_path, allow_pickle=False)
        source = str(routing_path.resolve())
    elif hasattr(model.router, "fixed_weights"):
        weights = model.router.fixed_weights.detach().cpu().numpy()
        source = "checkpoint_fixed_router_weights"
    else:
        raise FileNotFoundError(
            "image-target=routed with a soft router requires "
            f"{routing_path}; per-sample fMRI-conditioned gallery weights "
            "are intentionally forbidden"
        )
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (roi_count, layer_count):
        raise ValueError(
            f"Routing matrix shape {weights.shape} != "
            f"({roi_count}, {layer_count})"
        )
    sums = weights.sum(axis=-1, keepdims=True)
    if not np.isfinite(weights).all() or np.any(sums <= 0):
        raise ValueError("Routing weights are invalid")
    weights = weights / sums
    return torch.from_numpy(weights), source


def pool_image_features(
    clip_layer_features: Tensor,
    image_target: str,
    selected_layers: Sequence[int],
    static_routing_weights: Optional[Tensor],
) -> Tensor:
    if image_target == "l24":
        if 24 not in selected_layers:
            raise ValueError("image-target=l24 requires layer 24")
        return clip_layer_features[:, list(selected_layers).index(24)]
    if image_target == "uniform_fused":
        return clip_layer_features.mean(dim=1)
    if static_routing_weights is None:
        raise ValueError("routed image target requires static routing weights")
    weights = static_routing_weights.to(
        device=clip_layer_features.device,
        dtype=clip_layer_features.dtype,
    )
    routed = torch.einsum("rl,bld->brd", weights, clip_layer_features)
    return routed.mean(dim=1)


def build_model(
    args: argparse.Namespace,
    checkpoint: Mapping[str, object],
    roi_names: Sequence[str],
    roi_indices: Mapping[str, Sequence[int]],
    clip_model=None,
) -> Stage1RoutingModel:
    config = checkpoint.get("config", {})
    checkpoint_router = config.get("router_type")
    if checkpoint_router and checkpoint_router != args.router_type:
        raise ValueError(
            f"router-type={args.router_type} does not match checkpoint "
            f"router_type={checkpoint_router}"
        )
    checkpoint_layers = config.get("selected_clip_layers")
    if checkpoint_layers and list(checkpoint_layers) != list(
        args.selected_clip_layers
    ):
        raise ValueError(
            "selected-clip-layers do not match checkpoint: "
            f"{args.selected_clip_layers} != {checkpoint_layers}"
        )
    if args.router_type == "single":
        checkpoint_single = config.get("single_router_layer")
        if checkpoint_single is not None and int(
            checkpoint_single
        ) != int(args.single_router_layer):
            raise ValueError("single-router-layer does not match checkpoint")
    if args.brain_pooling != "mean":
        trained_pooling = config.get(
            "contrastive_brain_pooling", "mean"
        )
        contrastive_weight = float(
            config.get("contrastive_loss_weight", 0.0)
        )
        if trained_pooling != args.brain_pooling or contrastive_weight <= 0:
            raise ValueError(
                f"brain-pooling={args.brain_pooling} requires a checkpoint "
                "trained with the same contrastive_brain_pooling and "
                "contrastive_loss_weight > 0"
            )
    feature_dim = int(
        config.get(
            "clip_layer_target_dim", config.get("roi_token_dim", 1024)
        )
    )
    model = Stage1RoutingModel(
        roi_names=roi_names,
        roi_indices=roi_indices,
        selected_layers=args.selected_clip_layers,
        feature_dim=feature_dim,
        router_type=args.router_type,
        clip_model_name_or_path=config.get(
            "clip_model_name_or_path", "openai/clip-vit-large-patch14"
        ),
        router_hidden_dim=config.get("router_hidden_dim"),
        router_temperature=float(config.get("router_temperature", 1.0)),
        router_topk=int(config.get("router_topk", 0)),
        hard_intermediate_layers=config.get(
            "hard_intermediate_layers", [8, 12, 16, 20]
        ),
        hard_final_layers=config.get("hard_final_layers", [24]),
        single_router_layer=(
            args.single_router_layer
            if args.router_type == "single"
            else config.get("single_router_layer")
        ),
        seed=int(config.get("seed", args.seed)),
        use_brain_clip_projector=bool(
            config.get("use_brain_clip_projector", True)
        ),
        projector_type=config.get("projector_type", "mlp"),
        projector_hidden_dim=config.get("projector_hidden_dim"),
        projector_dropout=float(config.get("projector_dropout", 0.1)),
        detach_routed_targets=True,
        contrastive_brain_pooling=args.brain_pooling,
        contrastive_image_target=args.image_target,
        clip_model=clip_model,
    )
    checkpoint_state = checkpoint["model"]
    if args.brain_pooling == "mean":
        # Mean pooling is parameter-free. A checkpoint may still contain a
        # trained attention/MLP retrieval head; ignore only those head weights
        # so the same Stage-1 representation can be evaluated with mean.
        checkpoint_state = {
            key: value
            for key, value in checkpoint_state.items()
            if not key.startswith("retrieval_pooler.")
        }
    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    pooling_missing = [
        key
        for key in incompatible.missing_keys
        if key.startswith("retrieval_pooler.")
    ]
    other_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("retrieval_pooler.")
    ]
    if incompatible.unexpected_keys or other_missing:
        raise ValueError(
            "Checkpoint/model mismatch: "
            f"missing={other_missing}, unexpected={incompatible.unexpected_keys}"
        )
    if args.brain_pooling != "mean" and pooling_missing:
        raise ValueError(
            f"brain-pooling={args.brain_pooling} requires a trained pooling "
            "head in the checkpoint; use mean for legacy/eval-only checkpoints"
        )
    model.requires_grad_(False)
    model.eval()
    return model


def evaluate(args: argparse.Namespace, clip_model=None) -> Dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if getattr(args, "oracle_image_token_mode", False):
        raise NotImplementedError(
            "Oracle image-conditioned brain queries are intentionally not "
            "implemented. Normal retrieval must remain fMRI-only."
        )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    roi_names, roi_indices = load_roi_mapping(args.roi_indices_path)
    model = build_model(
        args, checkpoint, roi_names, roi_indices, clip_model=clip_model
    ).to(args.device)

    dataset = NSDTarDataset(
        expand_tar_paths(args.val_tar),
        repeat_mode=checkpoint.get("config", {}).get(
            "fmri_repeat_mode", "mean"
        ),
        require_coco_id=True,
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    static_weights = None
    routing_source = None
    if args.image_target == "routed":
        static_weights, routing_source = load_static_routing_weights(
            model,
            checkpoint_path,
            len(roi_names),
            len(args.selected_clip_layers),
        )

    brain_batches, image_batches, sample_ids, coco_ids = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Stage-1 retrieval", dynamic_ncols=True):
            fmri = batch["fmri"].to(args.device)
            images = batch["image"].to(args.device)
            roi_tokens = model.roi_tokenizer(fmri)["roi_tokens"]
            projected = model.brain_clip_projector(roi_tokens)
            # Leakage boundary: the query branch ends here and has never seen
            # image tensors or CLIP features.
            brain = model.retrieval_pooler(projected)
            clip_layers = model.clip_layer_bank(images)["pooled_tokens"]
            image = pool_image_features(
                clip_layers,
                args.image_target,
                args.selected_clip_layers,
                static_weights,
            )
            brain_batches.append(F.normalize(brain, dim=-1).cpu())
            image_batches.append(F.normalize(image, dim=-1).cpu())
            sample_ids.extend(list(batch["key"]))
            coco_ids.extend(
                int(value) for value in batch["coco73k_id"].tolist()
            )

    brain_embeddings = torch.cat(brain_batches).numpy().astype(np.float32)
    image_embeddings = torch.cat(image_batches).numpy().astype(np.float32)
    similarity = brain_embeddings @ image_embeddings.T
    brain_ids = np.asarray(coco_ids, dtype=np.int64)
    image_ids = np.asarray(coco_ids, dtype=np.int64)
    if args.diagonal_only:
        positive_mask = np.eye(similarity.shape[0], dtype=bool)
        protocol = "diagonal_only"
    else:
        positive_mask = brain_ids[:, None] == image_ids[None, :]
        protocol = "multi_positive_coco73k"
    ranks_b2i = compute_positive_ranks(similarity, positive_mask)
    ranks_i2b = compute_positive_ranks(
        similarity.T, positive_mask.T
    )
    positive_counts_b2i = positive_mask.sum(axis=1)
    positive_counts_i2b = positive_mask.sum(axis=0)
    mask_stats = {
        "protocol": protocol,
        "num_queries": int(similarity.shape[0]),
        "num_gallery_items": int(similarity.shape[1]),
        "num_unique_coco_ids": int(np.unique(brain_ids).size),
        "average_positives_per_brain_query": float(
            positive_counts_b2i.mean()
        ),
        "average_positives_per_query": float(
            positive_counts_b2i.mean()
        ),
        "average_positives_per_image_query": float(
            positive_counts_i2b.mean()
        ),
        "min_positives_per_query": int(positive_counts_b2i.min()),
        "max_positives_per_query": int(positive_counts_b2i.max()),
    }
    metrics = {
        "subject": args.subject,
        "num_samples": int(similarity.shape[0]),
        "retrieval_protocol": protocol,
        **mask_stats,
        "brain_to_image": retrieval_metrics(ranks_b2i),
        "image_to_brain": retrieval_metrics(ranks_i2b),
        "leakage_protocol": {
            "brain_query_source": (
                "fMRI -> ROITokenizer -> BrainToCLIPProjector -> ROI pooling"
            ),
            "image_gallery_source": (
                f"frozen CLIP image features ({args.image_target})"
            ),
            "image_features_used_in_brain_query": False,
            "oracle_image_token_mode": False,
            "routed_gallery_weights_source": routing_source,
        },
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "similarity.npy", similarity.astype(np.float32))
    np.save(output_dir / "brain_embeddings.npy", brain_embeddings)
    np.save(output_dir / "image_embeddings.npy", image_embeddings)
    np.save(output_dir / "ranks_brain_to_image.npy", ranks_b2i)
    np.save(output_dir / "ranks_image_to_brain.npy", ranks_i2b)
    np.save(output_dir / "brain_ids.npy", brain_ids)
    np.save(output_dir / "image_ids.npy", image_ids)
    (output_dir / "positive_mask_stats.json").write_text(
        json.dumps(mask_stats, indent=2)
    )
    (output_dir / "metrics_retrieval.json").write_text(
        json.dumps(metrics, indent=2)
    )
    config = {
        **vars(args),
        "checkpoint": str(checkpoint_path),
        "val_tar_resolved": [
            str(path) for path in expand_tar_paths(args.val_tar)
        ],
        "roi_names": roi_names,
        "sample_ids": sample_ids,
        "routing_weights_source": routing_source,
        "retrieval_protocol": protocol,
        "pooling_head_loaded_from_checkpoint": (
            args.brain_pooling != "mean"
        ),
        "image_features_used_in_brain_query": False,
        "oracle_image_token_mode": False,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved retrieval outputs to {output_dir}")
    return {
        "metrics": metrics,
        "similarity": similarity,
        "brain_embeddings": brain_embeddings,
        "image_embeddings": image_embeddings,
        "output_dir": output_dir,
    }


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
