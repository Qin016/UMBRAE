#!/usr/bin/env python
"""Train/evaluate an fMRI-only NeuroRoute visual-prefix adapter."""

import argparse
import io
import json
import random
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.brain_clip_projector import BrainToCLIPProjector
from models.global_l24_projector import GlobalL24Projector
from models.neuroroute_mllm_adapter import NeuroRouteMLLMAdapter
from models.roi_tokenizer import ROITokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroRoute Stage-2 fMRI-only MLLM prefix training"
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--train-tar", nargs="+", required=True)
    parser.add_argument("--val-tar", nargs="+", required=True)
    parser.add_argument("--captions-json")
    parser.add_argument("--roi-indices-path", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--global-l24-checkpoint")
    parser.add_argument(
        "--router-type", choices=["soft", "uniform", "single", "hard"], default="soft"
    )
    parser.add_argument("--single-router-layer", type=int, default=24)
    parser.add_argument(
        "--fusion-mode",
        choices=[
            "l24_only",
            "routed_only",
            "concat",
            "gated_fusion",
            "cross_attention",
        ],
        default="concat",
    )
    parser.add_argument("--mllm-model-path", default="model_weights/shikra-7b")
    parser.add_argument("--mllm-dim", type=int, default=4096)
    parser.add_argument("--adapter-hidden-dim", type=int)
    parser.add_argument("--freeze-stage1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-mllm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-adapter-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--caption-loss-weight", type=float, default=1.0)
    parser.add_argument("--retrieval-loss-weight", type=float, default=0.0)
    parser.add_argument("--grounding-loss-weight", type=float, default=0.0)
    parser.add_argument("--oracle-image-token-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt", default="Describe the visual content:")
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--debug-max-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_caption_mapping(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    payload = json.loads(Path(path).expanduser().read_text())
    if isinstance(payload, list):
        result = {}
        for item in payload:
            key = item.get("coco_id", item.get("image_id", item.get("id")))
            caption = item.get("caption", item.get("captions"))
            if key is not None and caption is not None:
                result[str(key)] = caption
        return result
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    raise ValueError("captions-json must contain a JSON object or list")


class Stage2TarDataset(Dataset):
    """Read fMRI/caption pairs without decoding image pixels."""

    def __init__(
        self,
        tar_paths: Sequence[str],
        caption_mapping: Mapping[str, object],
    ) -> None:
        self.caption_mapping = caption_mapping
        self.samples = []
        for raw_path in tar_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            with tarfile.open(path) as archive:
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
                caption_member = next(
                    (
                        f"{prefix}.{suffix}"
                        for suffix in ("caption.txt", "caption.json")
                        if f"{prefix}.{suffix}" in names
                    ),
                    None,
                )
                coco_member = f"{prefix}.coco73k.npy"
                if caption_member is not None or coco_member in names:
                    self.samples.append((
                        path,
                        prefix,
                        members[f"{prefix}.nsdgeneral.npy"],
                        members.get(f"{prefix}.num_uniques.npy"),
                        members.get(caption_member) if caption_member else None,
                        members.get(coco_member),
                    ))
        if not self.samples:
            raise ValueError("No Stage-2 fMRI/caption candidates found")

    @staticmethod
    def _npy(archive: tarfile.TarFile, member_or_name) -> np.ndarray:
        member = archive.extractfile(member_or_name)
        if member is None:
            raise FileNotFoundError(member_or_name)
        return np.load(io.BytesIO(member.read()), allow_pickle=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        (
            path,
            prefix,
            fmri_member,
            repeats_member,
            caption_member,
            coco_member,
        ) = self.samples[index]
        with tarfile.open(path) as archive:
            fmri = self._npy(archive, fmri_member)
            if fmri.ndim == 2:
                count = (
                    int(self._npy(archive, repeats_member).reshape(-1)[0])
                    if repeats_member is not None
                    else fmri.shape[0]
                )
                fmri = fmri[: max(1, min(count, fmri.shape[0]))].astype(
                    np.float32
                ).mean(axis=0)
            if caption_member:
                member = archive.extractfile(caption_member)
                text = member.read().decode("utf-8")
                caption = (
                    json.loads(text).get("caption", "")
                    if caption_member.name.endswith(".json")
                    else text.strip()
                )
                coco_id = prefix
            else:
                coco_id = str(
                    int(
                        self._npy(archive, coco_member).reshape(-1)[0]
                    )
                )
                caption = self.caption_mapping.get(coco_id)
                if isinstance(caption, list):
                    caption = caption[0] if caption else None
        if not caption:
            raise ValueError(
                f"No caption for {prefix} (COCO id {coco_id}); "
                "provide --captions-json"
            )
        return {
            "fmri": torch.from_numpy(np.asarray(fmri, dtype=np.float32)),
            "caption": str(caption),
            "sample_id": prefix,
            "coco_id": coco_id,
        }


def load_roi_mapping(path: str):
    payload = json.loads(Path(path).expanduser().read_text())
    if not payload.get("roi_mapping_is_real") or not payload.get(
        "voxel_order_verified"
    ):
        raise ValueError("ROI mapping must be real and voxel-order verified")
    roi_names = list(payload["roi_names"])
    indices = {
        name: payload["rois"][name]["indices"] for name in roi_names
    }
    return roi_names, indices


class FrozenStage1FMRIEncoder(nn.Module):
    """Restore only fMRI-derived Stage-1 modules; never instantiate CLIP."""

    def __init__(
        self,
        checkpoint_path: str,
        roi_names: Sequence[str],
        roi_indices: Mapping[str, Sequence[int]],
    ) -> None:
        super().__init__()
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(), map_location="cpu"
        )
        config = checkpoint.get("config", {})
        dim = int(config.get("roi_token_dim", 1024))
        self.roi_tokenizer = ROITokenizer(
            roi_names=roi_names,
            roi_indices=roi_indices,
            token_dim=dim,
            tokenizer_type="shared_mlp",
            use_roi_embeddings=True,
        )
        use_projector = bool(config.get("use_brain_clip_projector", True))
        self.brain_clip_projector = (
            BrainToCLIPProjector(
                dim,
                dim,
                projector_type=config.get("projector_type", "mlp"),
                hidden_dim=config.get("projector_hidden_dim"),
                dropout=float(config.get("projector_dropout", 0.1)),
            )
            if use_projector
            else nn.Identity()
        )
        state = checkpoint.get("model", checkpoint)
        tokenizer_state = {
            key[len("roi_tokenizer.") :]: value
            for key, value in state.items()
            if key.startswith("roi_tokenizer.")
        }
        projector_state = {
            key[len("brain_clip_projector.") :]: value
            for key, value in state.items()
            if key.startswith("brain_clip_projector.")
        }
        self.roi_tokenizer.load_state_dict(tokenizer_state, strict=True)
        if use_projector:
            self.brain_clip_projector.load_state_dict(
                projector_state, strict=True
            )
        self.output_dim = dim
        self.stage1_config = config

    def forward(self, fmri: Tensor) -> Dict[str, Tensor]:
        raw = self.roi_tokenizer(fmri)["roi_tokens"]
        projected = self.brain_clip_projector(raw)
        return {"raw_roi_tokens": raw, "projected_roi_tokens": projected}


class NeuroRouteStage2Model(nn.Module):
    def __init__(
        self,
        stage1_encoder: FrozenStage1FMRIEncoder,
        adapter: NeuroRouteMLLMAdapter,
        global_projector: Optional[GlobalL24Projector],
        mllm: nn.Module,
        routing_weights: Optional[Tensor] = None,
        global_stage1_encoder: Optional[FrozenStage1FMRIEncoder] = None,
        global_l24_source: str = "stage2_initialized_global_l24_projector",
    ) -> None:
        super().__init__()
        self.stage1_encoder = stage1_encoder
        self.adapter = adapter
        self.global_projector = global_projector
        self.global_stage1_encoder = global_stage1_encoder
        self.global_l24_source = global_l24_source
        self.mllm = mllm
        if routing_weights is not None:
            self.register_buffer(
                "routing_weights_summary", routing_weights, persistent=True
            )
        else:
            self.routing_weights_summary = None

    def encode_prefix(
        self, fmri: Tensor, roi_names: Sequence[str]
    ) -> Dict[str, object]:
        stage1 = self.stage1_encoder(fmri)
        raw = stage1["raw_roi_tokens"]
        # These are fMRI-derived tokens aligned against image-derived routed
        # CLIP targets during Stage-1. The image targets are supervision only
        # and are unavailable/unused during Stage-2 inference.
        stage1_aligned_roi_tokens = stage1["projected_roi_tokens"]
        if self.global_stage1_encoder is not None:
            global_stage1 = self.global_stage1_encoder(fmri)
            global_l24 = global_stage1["projected_roi_tokens"].mean(dim=1)
        elif self.global_projector is not None:
            global_l24 = self.global_projector(raw)
        elif self.global_l24_source == "not_used_for_routed_only":
            global_l24 = stage1_aligned_roi_tokens.mean(dim=1).detach() * 0.0
        else:
            raise RuntimeError("No global L24 path is configured")
        routing = (
            self.routing_weights_summary.unsqueeze(0).expand(
                fmri.shape[0], -1, -1
            )
            if self.routing_weights_summary is not None
            else None
        )
        adapted = self.adapter(
            global_l24,
            stage1_aligned_roi_tokens,
            roi_names=roi_names,
            routing_weights=routing,
        )
        return {
            **adapted,
            "global_l24_token": global_l24,
            "stage1_aligned_roi_tokens": stage1_aligned_roi_tokens,
            "raw_roi_tokens": raw,
        }


def assert_no_image_feature_leakage(
    image_features_used: bool,
    phase: str,
    oracle_image_token_mode: bool,
) -> None:
    if (
        phase in ("validation", "test")
        and image_features_used
        and not oracle_image_token_mode
    ):
        raise RuntimeError(
            "Ground-truth image features are forbidden during "
            f"{phase} unless --oracle-image-token-mode is enabled"
        )


def configure_trainable_parameters(model: NeuroRouteStage2Model, args) -> None:
    if args.train_adapter_only:
        model.requires_grad_(False)
        model.adapter.requires_grad_(True)
        if (
            args.global_l24_checkpoint is None
            and model.global_projector is not None
        ):
            model.global_projector.requires_grad_(True)
        return
    model.stage1_encoder.requires_grad_(not args.freeze_stage1)
    if model.global_stage1_encoder is not None:
        model.global_stage1_encoder.requires_grad_(not args.freeze_stage1)
    if model.global_projector is not None:
        model.global_projector.requires_grad_(
            args.global_l24_checkpoint is None or not args.freeze_stage1
        )
    model.mllm.requires_grad_(not args.freeze_mllm)
    model.adapter.requires_grad_(True)


def build_global_l24_path(
    checkpoint_path: Optional[str],
    primary_stage1: FrozenStage1FMRIEncoder,
    roi_names: Sequence[str],
    roi_indices: Mapping[str, Sequence[int]],
    fusion_mode: str,
):
    """Build an explicit fMRI-derived global L24 path.

    Existing Stage-1 single-L24 checkpoints contain ROI tokenizer/projector
    modules rather than a GlobalL24Projector. For those checkpoints, the
    global token is the deterministic mean of their L24-aligned ROI tokens.
    """
    if checkpoint_path is None and fusion_mode == "routed_only":
        return None, None, "not_used_for_routed_only"
    if checkpoint_path is None:
        return (
            GlobalL24Projector(primary_stage1.output_dim),
            None,
            "stage2_initialized_global_l24_projector_from_primary_raw_roi_tokens",
        )

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    projector_state = {
        key[len("global_projector.") :]: value
        for key, value in state.items()
        if key.startswith("global_projector.")
    }
    if projector_state:
        projector = GlobalL24Projector(primary_stage1.output_dim)
        projector.load_state_dict(projector_state, strict=True)
        return (
            projector,
            None,
            "loaded_global_l24_projector_from_checkpoint",
        )

    config = checkpoint.get("config", checkpoint.get("adapter_config", {}))
    if config.get("router_type") != "single" or int(
        config.get("single_router_layer", -1)
    ) != 24:
        raise ValueError(
            "--global-l24-checkpoint must contain global_projector.* weights "
            "or be a Stage-1 router_type=single, single_router_layer=24 "
            "checkpoint"
        )
    global_stage1 = FrozenStage1FMRIEncoder(
        str(path), roi_names, roi_indices
    )
    if global_stage1.output_dim != primary_stage1.output_dim:
        raise ValueError(
            "Global-L24 and ROI Stage-1 checkpoint dimensions differ: "
            f"{global_stage1.output_dim} != {primary_stage1.output_dim}"
        )
    return (
        None,
        global_stage1,
        "stage1_single_l24_projected_roi_mean",
    )


def build_language_batch(tokenizer, captions, prompt, max_length, device):
    texts = [f"{prompt} {caption}" for caption in captions]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_ids = tokenizer(
        [prompt] * len(captions),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["attention_mask"].sum(dim=1)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for row, length in enumerate(prompt_ids):
        labels[row, : int(length)] = -100
    return input_ids, attention_mask, labels


def caption_forward(model, tokenizer, prefix, captions, args):
    input_ids, text_mask, labels = build_language_batch(
        tokenizer,
        captions,
        args.prompt,
        args.max_text_length,
        args.device,
    )
    text_embeds = model.mllm.get_input_embeddings()(input_ids)
    prefix = prefix.to(dtype=text_embeds.dtype)
    inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
    prefix_mask = torch.ones(
        prefix.shape[:2], dtype=text_mask.dtype, device=text_mask.device
    )
    attention_mask = torch.cat([prefix_mask, text_mask], dim=1)
    prefix_labels = torch.full(
        prefix.shape[:2], -100, dtype=labels.dtype, device=labels.device
    )
    labels = torch.cat([prefix_labels, labels], dim=1)
    return model.mllm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )


def generate_captions(model, tokenizer, prefix, args) -> List[str]:
    prompt = tokenizer(
        [args.prompt] * prefix.shape[0],
        padding=True,
        return_tensors="pt",
    )
    prompt_ids = prompt["input_ids"].to(args.device)
    prompt_embeds = model.mllm.get_input_embeddings()(prompt_ids)
    prefix = prefix.to(dtype=prompt_embeds.dtype)
    generated = model.mllm.generate(
        inputs_embeds=torch.cat([prefix, prompt_embeds], dim=1),
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if token == other
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def evaluate_captions(
    references: Sequence[object],
    predictions: Sequence[str],
    *,
    return_per_sample: bool = False,
):
    """Evaluate captions with one or more references per stimulus."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    normalized_references = [
        [str(item) for item in reference]
        if isinstance(reference, (list, tuple))
        else [str(reference)]
        for reference in references
    ]
    refs = [
        [reference.lower().split() for reference in reference_set]
        for reference_set in normalized_references
    ]
    hyps = [prediction.lower().split() for prediction in predictions]
    smooth = SmoothingFunction().method1
    metrics = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        metrics[f"BLEU-{n}"] = float(
            corpus_bleu(refs, hyps, weights=weights, smoothing_function=smooth)
        )
    rouge_scores = []
    for reference_set, prediction in zip(refs, hyps):
        candidates = []
        for reference in reference_set:
            lcs = lcs_length(reference, prediction)
            precision = lcs / max(len(prediction), 1)
            recall = lcs / max(len(reference), 1)
            candidates.append(
                2 * precision * recall / max(precision + recall, 1e-12)
            )
        rouge_scores.append(max(candidates, default=0.0))
    metrics["ROUGE-L"] = float(np.mean(rouge_scores))
    unavailable = {}
    gts = {
        index: reference_set
        for index, reference_set in enumerate(normalized_references)
    }
    res = {
        index: [prediction]
        for index, prediction in enumerate(predictions)
    }
    scorers = [
        ("CIDEr", "pycocoevalcap.cider.cider", "Cider"),
        ("ROUGE-L", "pycocoevalcap.rouge.rouge", "Rouge"),
        ("METEOR", "pycocoevalcap.meteor.meteor", "Meteor"),
        ("SPICE", "pycocoevalcap.spice.spice", "Spice"),
    ]
    per_sample = {}
    for metric_name, module_name, class_name in scorers:
        if metric_name in ("METEOR", "SPICE") and shutil.which("java") is None:
            metrics.setdefault(metric_name, None)
            unavailable[metric_name] = "Java runtime is unavailable"
            continue
        try:
            module = __import__(module_name, fromlist=[class_name])
            scorer = getattr(module, class_name)()
            score, scores = scorer.compute_score(gts, res)
            metrics[metric_name] = float(score)
            if metric_name in ("CIDEr", "ROUGE-L"):
                per_sample[metric_name] = [float(value) for value in scores]
            close = getattr(scorer, "close", None)
            if close is not None:
                close()
        except Exception as exc:
            metrics.setdefault(metric_name, None)
            unavailable[metric_name] = (
                f"{type(exc).__name__}: {str(exc)[:160]}"
            )
    metrics["unavailable_metrics"] = unavailable
    metrics["reference_mode"] = (
        "multi_reference"
        if any(len(items) > 1 for items in normalized_references)
        else "single_reference"
    )
    metrics["reference_count_min"] = min(map(len, normalized_references))
    metrics["reference_count_max"] = max(map(len, normalized_references))
    if return_per_sample:
        return metrics, per_sample
    return metrics


def routing_diagnostics(
    routing: Optional[np.ndarray],
    roi_names: Sequence[str],
    layers: Sequence[int],
):
    if routing is None:
        return {
            "available": False,
            "reason": "No Stage-1 routing summary found",
        }
    probabilities = routing / routing.sum(axis=-1, keepdims=True)
    expected = probabilities @ np.asarray(layers)
    return {
        "available": True,
        "selected_clip_layers": list(layers),
        "l24_usage": (
            float(probabilities[:, list(layers).index(24)].mean())
            if 24 in layers
            else None
        ),
        "expected_depth_per_roi": {
            name: float(value) for name, value in zip(roi_names, expected)
        },
        "early_roi_mean_depth": float(expected[:4].mean()),
        "high_level_roi_mean_depth": float(expected[4:].mean()),
    }


def run_training(args, mllm=None, tokenizer=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.oracle_image_token_mode:
        raise NotImplementedError(
            "Oracle image-token input is intentionally not implemented in "
            "the fMRI-only Stage-2 path. Disable --oracle-image-token-mode."
        )
    image_features_used = False
    if args.retrieval_loss_weight or args.grounding_loss_weight:
        raise NotImplementedError(
            "Stage-2 currently implements caption loss first; "
            "retrieval/grounding weights must remain zero"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_names, roi_indices = load_roi_mapping(args.roi_indices_path)
    caption_mapping = load_caption_mapping(args.captions_json)
    train_dataset = Stage2TarDataset(args.train_tar, caption_mapping)
    val_dataset = Stage2TarDataset(args.val_tar, caption_mapping)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    stage1 = FrozenStage1FMRIEncoder(
        args.stage1_checkpoint, roi_names, roi_indices
    )
    checkpoint_router_type = stage1.stage1_config.get("router_type")
    if checkpoint_router_type and checkpoint_router_type != args.router_type:
        raise ValueError(
            f"Requested router_type={args.router_type!r}, but Stage-1 "
            f"checkpoint was trained with {checkpoint_router_type!r}"
        )
    if args.router_type == "single":
        checkpoint_single_layer = stage1.stage1_config.get(
            "single_router_layer"
        )
        if (
            checkpoint_single_layer is not None
            and int(checkpoint_single_layer) != args.single_router_layer
        ):
            raise ValueError(
                "single-router layer does not match Stage-1 checkpoint: "
                f"{args.single_router_layer} != {checkpoint_single_layer}"
            )
    checkpoint_dir = Path(args.stage1_checkpoint).expanduser().resolve().parent
    routing_path = checkpoint_dir / "val_routing_weights_mean.npy"
    routing_np = (
        np.load(routing_path, allow_pickle=False)
        if routing_path.is_file()
        else None
    )
    routing_tensor = (
        torch.from_numpy(routing_np.astype(np.float32))
        if routing_np is not None
        else None
    )
    if mllm is None:
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            args.mllm_model_path
        )
        model_dtype = (
            torch.float16
            if str(args.device).startswith("cuda")
            else torch.float32
        )
        try:
            mllm = AutoModelForCausalLM.from_pretrained(
                args.mllm_model_path,
                torch_dtype=model_dtype,
            )
            mllm_loader_type = "auto_causal_lm"
        except (KeyError, ValueError) as error:
            config_path = (
                Path(args.mllm_model_path).expanduser() / "config.json"
            )
            config_payload = (
                json.loads(config_path.read_text())
                if config_path.is_file()
                else {}
            )
            if config_payload.get("model_type") != "shikra":
                raise error
            mllm = LlamaForCausalLM.from_pretrained(
                args.mllm_model_path,
                torch_dtype=model_dtype,
            )
            mllm_loader_type = "llama_direct_for_shikra_weights"
    else:
        mllm_loader_type = "injected_test_model"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    embedding_dim = int(mllm.get_input_embeddings().embedding_dim)
    if embedding_dim != args.mllm_dim:
        raise ValueError(
            f"mllm-dim={args.mllm_dim} does not match model embedding "
            f"dimension {embedding_dim}"
        )
    adapter = NeuroRouteMLLMAdapter(
        stage1.output_dim,
        args.mllm_dim,
        fusion_mode=args.fusion_mode,
        hidden_dim=args.adapter_hidden_dim,
    )
    (
        global_projector,
        global_stage1_encoder,
        global_l24_source,
    ) = build_global_l24_path(
        args.global_l24_checkpoint,
        stage1,
        roi_names,
        roi_indices,
        args.fusion_mode,
    )
    model = NeuroRouteStage2Model(
        stage1,
        adapter,
        global_projector,
        mllm,
        routing_tensor,
        global_stage1_encoder=global_stage1_encoder,
        global_l24_source=global_l24_source,
    ).to(args.device)
    configure_trainable_parameters(model, args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    adapter_config = {
        **vars(args),
        "roi_names": roi_names,
        "fusion_mode": args.fusion_mode,
        "stage1_checkpoint": str(
            Path(args.stage1_checkpoint).expanduser().resolve()
        ),
        "global_l24_checkpoint": (
            str(Path(args.global_l24_checkpoint).expanduser().resolve())
            if args.global_l24_checkpoint
            else None
        ),
        "global_l24_source": global_l24_source,
        "roi_prefix_source": (
            "fMRI-derived Stage-1 projected ROI tokens aligned to routed "
            "image-CLIP targets during Stage-1; no image targets at inference"
        ),
        "uses_image_clip_tokens_at_eval": False,
        "oracle_image_token_mode": False,
        "mllm_bridge_type": "generic_inputs_embeds_prefix",
        "mllm_loader_type": mllm_loader_type,
        "freeze_stage1": bool(args.freeze_stage1),
        "freeze_mllm": bool(args.freeze_mllm),
        "train_adapter_only": bool(args.train_adapter_only),
        "ground_truth_image_features_used": image_features_used,
        "fMRI_only_validation": True,
        "stage2_roi_representation": (
            "Stage-1 aligned fMRI ROI tokens; image-derived routed CLIP "
            "targets were Stage-1 supervision only"
        ),
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2)
    )
    selected_layers = stage1.stage1_config.get(
        "selected_clip_layers", [4, 8, 12, 16, 20, 24]
    )
    routing_summary = routing_diagnostics(
        routing_np, roi_names, selected_layers
    )
    routing_summary["ground_truth_image_features_used"] = image_features_used
    (output_dir / "routing_summary.json").write_text(
        json.dumps(routing_summary, indent=2)
    )
    predictions_path = output_dir / "predictions.jsonl"
    captions_path = output_dir / "generated_captions.jsonl"
    predictions_path.write_text("")
    captions_path.write_text("")

    best_val = float("inf")
    final_metrics = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_stage1:
            model.stage1_encoder.eval()
        if args.freeze_mllm:
            model.mllm.eval()
        train_losses = []
        train_norms = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"stage2 epoch {epoch} train", leave=False)):
            fmri = batch["fmri"].to(args.device)
            prefix_out = model.encode_prefix(fmri, roi_names)
            result = caption_forward(
                model,
                tokenizer,
                prefix_out["visual_prefix_tokens"],
                batch["caption"],
                args,
            )
            loss = args.caption_loss_weight * result.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
            diagnostics = prefix_out["diagnostics"]
            train_norms.append({
                key: (
                    float(value) if value is not None else None
                )
                for key, value in diagnostics.items()
                if key.endswith("_norm") or key.startswith("gate_")
            })
            if args.debug_max_steps and step + 1 >= args.debug_max_steps:
                break

        model.eval()
        val_losses, references, predictions, records = [], [], [], []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(val_loader, desc=f"stage2 epoch {epoch} val", leave=False)):
                if not args.oracle_image_token_mode:
                    # Dataset/model path contains no image tensor or CLIP call.
                    assert_no_image_feature_leakage(
                        image_features_used,
                        phase="validation",
                        oracle_image_token_mode=args.oracle_image_token_mode,
                    )
                fmri = batch["fmri"].to(args.device)
                prefix_out = model.encode_prefix(fmri, roi_names)
                result = caption_forward(
                    model,
                    tokenizer,
                    prefix_out["visual_prefix_tokens"],
                    batch["caption"],
                    args,
                )
                val_losses.append(float(result.loss))
                generated = generate_captions(
                    model,
                    tokenizer,
                    prefix_out["visual_prefix_tokens"],
                    args,
                )
                references.extend(batch["caption"])
                predictions.extend(generated)
                for sample_id, reference, prediction in zip(
                    batch["sample_id"], batch["caption"], generated
                ):
                    records.append({
                        "sample_id": sample_id,
                        "reference": reference,
                        "prediction": prediction,
                        "oracle_image_token_mode": bool(
                            args.oracle_image_token_mode
                        ),
                    })
                if args.debug_max_steps and step + 1 >= args.debug_max_steps:
                    break
        caption_metrics = evaluate_captions(references, predictions)
        final_metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "caption_metrics": caption_metrics,
            "ground_truth_image_features_used": image_features_used,
            "global_l24_token_norm": float(
                prefix_out["diagnostics"]["global_l24_token_norm"]
            ),
            "stage1_aligned_roi_token_norm": float(
                prefix_out["diagnostics"][
                    "stage1_aligned_roi_token_norm"
                ]
            ),
            "visual_prefix_token_norm": float(
                prefix_out["diagnostics"]["visual_prefix_token_norm"]
            ),
            "gate_mean": (
                float(prefix_out["diagnostics"]["gate_mean"])
                if prefix_out["diagnostics"]["gate_mean"] is not None
                else None
            ),
            **routing_summary,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(final_metrics, indent=2)
        )
        with predictions_path.open("w") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")
        with captions_path.open("w") as file:
            for record in records:
                file.write(json.dumps({
                    "sample_id": record["sample_id"],
                    "caption": record["prediction"],
                }) + "\n")
        checkpoint_state = {
            key: value
            for key, value in model.state_dict().items()
            if key.startswith("adapter.")
            or key.startswith("global_projector.")
            or (
                not args.freeze_stage1
                and key.startswith("global_stage1_encoder.")
            )
            or (
                not args.freeze_stage1
                and key.startswith("stage1_encoder.")
            )
            or (not args.freeze_mllm and key.startswith("mllm."))
        }
        checkpoint = {
            "epoch": epoch,
            "model": checkpoint_state,
            "optimizer": optimizer.state_dict(),
            "adapter_config": adapter_config,
            "val_loss": final_metrics["val_loss"],
        }
        torch.save(checkpoint, output_dir / "checkpoint_last.pt")
        if final_metrics["val_loss"] < best_val:
            best_val = final_metrics["val_loss"]
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
        print(
            f"epoch={epoch} train={final_metrics['train_loss']:.6f} "
            f"val={final_metrics['val_loss']:.6f} "
            f"BLEU-4={caption_metrics['BLEU-4']:.6f}"
        )
    return {"model": model, "metrics": final_metrics, "output_dir": output_dir}


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
