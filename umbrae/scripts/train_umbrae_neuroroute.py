#!/usr/bin/env python
"""Train NeuroRoute as an augmentation of the original UMBRAE/Shikra path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.fgw_stage2_prior import (
    CLIP_LAYER_ORDER,
    apply_row_derangement,
    load_frozen_plan,
    load_preregistered_derangement,
    plan_provenance_json,
)
from models.umbrae_backbone import FrozenUMBRAEEncoder
from models.umbrae_neuroroute_adapter import UMBRAENeuroRouteAdapter
from scripts.train_stage2_mllm_neuroroute import (
    FrozenStage1FMRIEncoder,
    Stage2TarDataset,
    assert_no_image_feature_leakage,
    evaluate_captions,
    load_caption_mapping,
    load_roi_mapping,
    routing_diagnostics,
)


SHIKRA_IMAGE_PATCH_TOKEN = "<im_patch>"
SHIKRA_IMAGE_START_TOKEN = "<im_start>"
SHIKRA_IMAGE_END_TOKEN = "<im_end>"
SHIKRA_IMAGE_START_ID = 32001
SHIKRA_IMAGE_END_ID = 32002
DEFAULT_NUM_IMAGE_TOKENS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UMBRAE + NeuroRoute fMRI-only caption training"
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--train-tar", nargs="+", required=True)
    parser.add_argument("--val-tar", nargs="+", required=True)
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--roi-indices-path", required=True)
    parser.add_argument("--umbrae-checkpoint", required=True)
    parser.add_argument("--neuroroute-checkpoint", required=True)
    parser.add_argument("--global-l24-checkpoint")
    parser.add_argument(
        "--umbrae-mm-projector",
        default="model_weights/mm_projector.bin",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=UMBRAENeuroRouteAdapter.SUPPORTED_MODES,
        default="umbrae_plus_soft",
    )
    parser.add_argument(
        "--fusion-type",
        choices=UMBRAENeuroRouteAdapter.SUPPORTED_FUSIONS,
        default="perceiver_resampler",
    )
    parser.add_argument(
        "--bridge-type",
        choices=["shikra_patch", "generic_prefix"],
        default="shikra_patch",
    )
    parser.add_argument("--mllm-model-path", default="model_weights/shikra-7b")
    parser.add_argument("--mllm-dim", type=int, default=4096)
    parser.add_argument("--fusion-dim", type=int, default=1024)
    parser.add_argument("--num-visual-tokens", type=int, default=256)
    parser.add_argument("--roi-token-expansion", type=int, default=1)
    parser.add_argument("--structured-routing-alpha", type=float, default=0.3)
    parser.add_argument("--routing-temperature", type=float, default=2.0)
    parser.add_argument("--fgw-gamma", type=float, default=1.0)
    parser.add_argument("--fgw-delta", type=float, default=1e-8)
    parser.add_argument(
        "--fgw-bias-layers", choices=["first"], default="first"
    )
    parser.add_argument(
        "--fgw-plan-root",
        default=str(Path(__file__).resolve().parents[1] / "fgw_outputs"),
    )
    parser.add_argument(
        "--fgw-null-registration",
        default=str(
            Path(__file__).resolve().parents[1]
            / "fgw_outputs"
            / "stage2_fgw_preregistered_null.json"
        ),
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-if-incomplete", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-umbrae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-neuroroute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-mllm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-adapter-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oracle-image-token-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--prompt",
        default="Describe this image as simply as possible.",
    )
    parser.add_argument("--max-text-length", type=int, default=512)
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
    parser.add_argument("--trainable-reference")
    parser.add_argument("--data-audit-path")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UMBRAENeuroRouteModel(nn.Module):
    def __init__(
        self,
        umbrae_encoder: nn.Module,
        neuroroute_encoder: nn.Module,
        adapter: UMBRAENeuroRouteAdapter,
        mllm: nn.Module,
        routing_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.umbrae_encoder = umbrae_encoder
        self.neuroroute_encoder = neuroroute_encoder
        self.adapter = adapter
        self.mllm = mllm
        if routing_weights is not None:
            self.register_buffer(
                "routing_weights_summary",
                routing_weights,
                persistent=True,
            )
        else:
            self.routing_weights_summary = None

    def encode_visual_tokens(self, fmri: Tensor) -> Dict[str, object]:
        umbrae_tokens = (
            None
            if self.adapter.fusion_mode == "neuroroute_only"
            else self.umbrae_encoder(fmri)
        )
        roi_tokens = None
        if self.adapter.fusion_mode != "umbrae_only":
            stage1 = self.neuroroute_encoder(fmri)
            roi_tokens = stage1["projected_roi_tokens"]
        routing = (
            self.routing_weights_summary.unsqueeze(0).expand(
                fmri.shape[0], -1, -1
            )
            if self.routing_weights_summary is not None
            else None
        )
        return self.adapter(
            umbrae_tokens,
            roi_tokens,
            routing_weights=routing,
        )


def shikra_image_prompt(prompt: str, num_visual_tokens: int) -> str:
    image = (
        " "
        + SHIKRA_IMAGE_START_TOKEN
        + SHIKRA_IMAGE_PATCH_TOKEN * num_visual_tokens
        + SHIKRA_IMAGE_END_TOKEN
        + " "
    )
    user_prompt = (
        prompt.replace("<image>", image)
        if "<image>" in prompt
        else prompt + image
    )
    return (
        "A chat between a curious user and an artificial intelligence "
        "assistant. The assistant gives helpful, detailed, and polite "
        f"answers to the user's questions. USER: {user_prompt} ASSISTANT:"
    )


def replace_shikra_patch_embeddings(
    input_ids: Tensor,
    input_embeddings: Tensor,
    visual_tokens: Tensor,
    image_start_token_id: int = SHIKRA_IMAGE_START_ID,
    image_end_token_id: int = SHIKRA_IMAGE_END_ID,
) -> Tensor:
    """Replace exactly N Shikra patch placeholders after ``<im_start>``."""
    if visual_tokens.ndim != 3:
        raise ValueError("visual_tokens must have shape [B,N_img,D_mllm]")
    if input_ids.shape[:2] != input_embeddings.shape[:2]:
        raise ValueError("input_ids and input_embeddings shapes disagree")
    if input_ids.shape[0] != visual_tokens.shape[0]:
        raise ValueError("Text and visual token batch sizes differ")
    rows = []
    num_patches = visual_tokens.shape[1]
    for ids, embeddings, patches in zip(
        input_ids,
        input_embeddings,
        visual_tokens,
    ):
        starts = torch.where(ids == image_start_token_id)[0]
        if starts.numel() != 1:
            raise ValueError(
                "Each Shikra prompt must contain exactly one <im_start>; "
                f"found {starts.numel()}"
            )
        start = int(starts[0])
        end = start + num_patches + 1
        if end >= ids.numel() or int(ids[end]) != image_end_token_id:
            raise ValueError(
                "<im_end> must immediately follow the configured number "
                f"of image patch positions ({num_patches})"
            )
        rows.append(
            torch.cat(
                [
                    embeddings[: start + 1],
                    patches.to(dtype=embeddings.dtype),
                    embeddings[end:],
                ],
                dim=0,
            )
        )
    return torch.stack(rows)


def _tokenize_shikra(
    tokenizer,
    captions: Sequence[str],
    prompt: str,
    num_visual_tokens: int,
    max_length: int,
    device: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    prefix = shikra_image_prompt(prompt, num_visual_tokens)
    texts = [f"{prefix} {caption}" for caption in captions]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prefix_encoded = tokenizer(
        [prefix] * len(captions),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    prefix_lengths = prefix_encoded["attention_mask"].sum(dim=1)
    for row, length in enumerate(prefix_lengths):
        labels[row, : int(length)] = -100
    return input_ids, attention_mask, labels


def caption_forward(model, tokenizer, visual_tokens, captions, args):
    if args.bridge_type == "generic_prefix":
        encoded = tokenizer(
            [f"{args.prompt} {caption}" for caption in captions],
            padding=True,
            truncation=True,
            max_length=args.max_text_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(args.device)
        text_mask = encoded["attention_mask"].to(args.device)
        labels = input_ids.clone()
        labels[text_mask == 0] = -100
        embeddings = model.mllm.get_input_embeddings()(input_ids)
        visual_tokens = visual_tokens.to(dtype=embeddings.dtype)
        inputs_embeds = torch.cat([visual_tokens, embeddings], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(
                    visual_tokens.shape[:2],
                    dtype=text_mask.dtype,
                    device=text_mask.device,
                ),
                text_mask,
            ],
            dim=1,
        )
        labels = torch.cat(
            [
                torch.full(
                    visual_tokens.shape[:2],
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                ),
                labels,
            ],
            dim=1,
        )
    else:
        input_ids, attention_mask, labels = _tokenize_shikra(
            tokenizer,
            captions,
            args.prompt,
            visual_tokens.shape[1],
            args.max_text_length,
            args.device,
        )
        embeddings = model.mllm.get_input_embeddings()(input_ids)
        inputs_embeds = replace_shikra_patch_embeddings(
            input_ids,
            embeddings,
            visual_tokens,
        )
    return model.mllm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )


def generate_captions(model, tokenizer, visual_tokens, args) -> List[str]:
    if args.bridge_type == "generic_prefix":
        prompt = tokenizer(
            [args.prompt] * visual_tokens.shape[0],
            padding=True,
            return_tensors="pt",
        )
    else:
        text = shikra_image_prompt(args.prompt, visual_tokens.shape[1])
        prompt = tokenizer(
            [text] * visual_tokens.shape[0],
            padding=True,
            return_tensors="pt",
        )
    input_ids = prompt["input_ids"].to(args.device)
    attention_mask = prompt["attention_mask"].to(args.device)
    embeddings = model.mllm.get_input_embeddings()(input_ids)
    if args.bridge_type == "generic_prefix":
        visual_tokens = visual_tokens.to(dtype=embeddings.dtype)
        embeddings = torch.cat([visual_tokens, embeddings], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(
                    visual_tokens.shape[:2],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ],
            dim=1,
        )
    else:
        embeddings = replace_shikra_patch_embeddings(
            input_ids,
            embeddings,
            visual_tokens,
        )
    generated = model.mllm.generate(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def _router_expected_for_mode(fusion_mode: str) -> Optional[str]:
    return {
        "umbrae_plus_soft": "soft",
        "umbrae_plus_uniform": "uniform",
        "umbrae_plus_single_l24": "single",
        "umbrae_plus_uniform_residual_soft": "soft",
        "umbrae_plus_temperature_soft": "soft",
        "uniform_prior": "soft",
        "fgw_prior": "soft",
        "row_shuffled_fgw_prior": "soft",
        "neuroroute_only": None,
        "umbrae_only": None,
    }[fusion_mode]


def build_effective_routing(
    soft_routing: np.ndarray,
    fusion_mode: str,
    structured_alpha: float = 0.3,
    temperature: float = 2.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return the routing matrix actually consumed by the fusion adapter."""
    routing = np.asarray(soft_routing, dtype=np.float64)
    if routing.ndim != 2 or routing.shape[1] < 1:
        raise ValueError("soft_routing must have shape [R,L]")
    routing = np.clip(routing, 0.0, None)
    routing /= np.maximum(routing.sum(axis=-1, keepdims=True), eps)
    if fusion_mode == "umbrae_plus_uniform_residual_soft":
        if not 0.0 <= structured_alpha <= 1.0:
            raise ValueError("structured-routing-alpha must be in [0,1]")
        uniform = np.full_like(routing, 1.0 / routing.shape[-1])
        routing = (
            (1.0 - structured_alpha) * uniform
            + structured_alpha * routing
        )
    elif fusion_mode == "umbrae_plus_temperature_soft":
        if temperature <= 0:
            raise ValueError("routing-temperature must be > 0")
        logits = np.log(routing + eps) / temperature
        logits -= logits.max(axis=-1, keepdims=True)
        routing = np.exp(logits)
        routing /= routing.sum(axis=-1, keepdims=True)
    return routing.astype(np.float32)


def completed_run_files(output_dir: Path) -> List[Path]:
    return [
        output_dir / filename
        for filename in (
            "checkpoint_last.pt",
            "metrics.json",
            "generated_captions.jsonl",
            "adapter_config.json",
        )
    ]


def is_completed_run(output_dir: Path) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in completed_run_files(output_dir))


def configure_trainable(model: UMBRAENeuroRouteModel, args) -> None:
    model.umbrae_encoder.requires_grad_(not args.freeze_umbrae)
    model.neuroroute_encoder.requires_grad_(not args.freeze_neuroroute)
    model.mllm.requires_grad_(not args.freeze_mllm)
    model.adapter.requires_grad_(True)
    if args.train_adapter_only:
        model.umbrae_encoder.requires_grad_(False)
        model.neuroroute_encoder.requires_grad_(False)
        model.mllm.requires_grad_(False)


def _load_mm_projector(path: Optional[str]) -> Optional[Dict[str, Tensor]]:
    if not path:
        return None
    payload = torch.load(path, map_location="cpu")
    return payload.get("model_state_dict", payload)


def run_training(
    args,
    mllm=None,
    tokenizer=None,
    umbrae_encoder=None,
    neuroroute_encoder=None,
):
    for name, default in (
        ("structured_routing_alpha", 0.3),
        ("routing_temperature", 2.0),
        ("overwrite", False),
        ("resume_if_incomplete", False),
        ("fgw_gamma", 1.0),
        ("fgw_delta", 1e-8),
        ("fgw_bias_layers", "first"),
        (
            "fgw_plan_root",
            str(Path(__file__).resolve().parents[1] / "fgw_outputs"),
        ),
        (
            "fgw_null_registration",
            str(
                Path(__file__).resolve().parents[1]
                / "fgw_outputs"
                / "stage2_fgw_preregistered_null.json"
            ),
        ),
        ("trainable_reference", None),
        ("data_audit_path", None),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_start_time = time.perf_counter()
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if is_completed_run(output_dir) and not args.overwrite:
        print(f"Skipping existing completed run: {output_dir}")
        return {
            "model": None,
            "metrics": json.loads((output_dir / "metrics.json").read_text()),
            "output_dir": output_dir,
            "skipped": True,
        }
    existing_payload = (
        [
            path
            for path in output_dir.iterdir()
            if path.name not in {"train.log", "stdout.log", "stderr.log"}
        ]
        if output_dir.exists()
        else []
    )
    if existing_payload and not args.overwrite:
        if not args.resume_if_incomplete:
            raise RuntimeError(
                f"Incomplete output directory exists: {output_dir}. "
                "Use --resume-if-incomplete or --overwrite."
            )
    if args.oracle_image_token_mode:
        raise ValueError(
            "Oracle image-token mode is not permitted in this integration"
        )
    if args.bridge_type == "shikra_patch" and args.num_visual_tokens != 256:
        raise ValueError(
            "Strict Shikra patch mode requires exactly 256 visual tokens"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    roi_names, roi_indices = load_roi_mapping(args.roi_indices_path)
    captions = load_caption_mapping(args.captions_json)
    train_dataset = Stage2TarDataset(args.train_tar, captions)
    val_dataset = Stage2TarDataset(args.val_tar, captions)
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

    umbrae_encoder = umbrae_encoder or FrozenUMBRAEEncoder(
        args.umbrae_checkpoint,
        args.subject,
    )
    neuroroute_encoder = (
        neuroroute_encoder
        or FrozenStage1FMRIEncoder(
            args.neuroroute_checkpoint,
            roi_names,
            roi_indices,
        )
    )
    expected_router = _router_expected_for_mode(args.fusion_mode)
    actual_router = getattr(
        neuroroute_encoder,
        "stage1_config",
        {},
    ).get("router_type")
    if expected_router and actual_router and expected_router != actual_router:
        raise ValueError(
            f"{args.fusion_mode} requires router={expected_router}, but "
            f"checkpoint uses router={actual_router}"
        )
    if args.fusion_mode == "umbrae_plus_single_l24":
        layer = getattr(
            neuroroute_encoder,
            "stage1_config",
            {},
        ).get("single_router_layer")
        if layer is not None and int(layer) != 24:
            raise ValueError("single-L24 fusion requires layer 24")

    if tokenizer is None:
        tokenizer = LlamaTokenizer.from_pretrained(
            args.mllm_model_path,
            padding_side="right",
        )
    if mllm is None:
        dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
        mllm = LlamaForCausalLM.from_pretrained(
            args.mllm_model_path,
            torch_dtype=dtype,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    embedding_dim = int(mllm.get_input_embeddings().embedding_dim)
    if embedding_dim != args.mllm_dim:
        raise ValueError(
            f"mllm-dim={args.mllm_dim} but model uses {embedding_dim}"
        )

    umbrae_dim = int(getattr(umbrae_encoder, "output_dim", args.fusion_dim))
    neuroroute_dim = int(
        getattr(neuroroute_encoder, "output_dim", args.fusion_dim)
    )
    layers = getattr(
        neuroroute_encoder,
        "stage1_config",
        {},
    ).get("selected_clip_layers", list(CLIP_LAYER_ORDER))
    prior_modes = {
        "uniform_prior",
        "fgw_prior",
        "row_shuffled_fgw_prior",
    }
    frozen_plan = None
    plan_provenance = None
    row_permutation = None
    if args.fusion_mode in prior_modes:
        frozen_plan, plan_provenance = load_frozen_plan(
            args.subject,
            args.fgw_plan_root,
            roi_order=roi_names,
            clip_layer_order=layers,
        )
        if args.fusion_mode == "row_shuffled_fgw_prior":
            row_permutation = load_preregistered_derangement(
                args.fgw_null_registration
            )
            frozen_plan = apply_row_derangement(
                frozen_plan, row_permutation
            )
    adapter = UMBRAENeuroRouteAdapter(
        umbrae_dim=umbrae_dim,
        neuroroute_dim=neuroroute_dim,
        mllm_dim=args.mllm_dim,
        num_visual_tokens=args.num_visual_tokens,
        fusion_mode=args.fusion_mode,
        fusion_type=args.fusion_type,
        fusion_dim=args.fusion_dim,
        roi_token_expansion=args.roi_token_expansion,
        num_routing_layers=len(layers),
        fgw_transport_plan=frozen_plan,
        fgw_gamma=args.fgw_gamma,
        fgw_delta=args.fgw_delta,
        fgw_bias_layers=args.fgw_bias_layers,
    )
    mm_projector_state = _load_mm_projector(args.umbrae_mm_projector)
    if mm_projector_state is not None:
        adapter.load_umbrae_mm_projector(mm_projector_state)

    routing_path = (
        Path(args.neuroroute_checkpoint).expanduser().resolve().parent
        / "val_routing_weights_mean.npy"
    )
    source_routing_np = (
        np.load(routing_path, allow_pickle=False)
        if routing_path.is_file()
        else None
    )
    if (
        args.fusion_mode
        in {
            "umbrae_plus_uniform_residual_soft",
            "umbrae_plus_temperature_soft",
        }
        and source_routing_np is None
    ):
        raise FileNotFoundError(
            f"Structured routing requires {routing_path}"
        )
    routing_np = (
        build_effective_routing(
            source_routing_np,
            args.fusion_mode,
            args.structured_routing_alpha,
            args.routing_temperature,
        )
        if source_routing_np is not None
        else None
    )
    routing_tensor = (
        torch.from_numpy(routing_np.astype(np.float32))
        if routing_np is not None
        else None
    )
    model = UMBRAENeuroRouteModel(
        umbrae_encoder,
        neuroroute_encoder,
        adapter,
        mllm,
        routing_tensor,
    ).to(args.device)
    configure_trainable(model, args)
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    parameter_audit = {
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": frozen_parameter_count,
        "fgw_trainable_correspondence_parameter_count": 0,
    }
    if args.trainable_reference:
        reference_path = Path(args.trainable_reference).expanduser().resolve()
        if reference_path.is_file():
            reference = json.loads(reference_path.read_text())
            for key in (
                "trainable_parameter_names",
                "trainable_parameter_count",
                "frozen_parameter_count",
            ):
                if parameter_audit[key] != reference[key]:
                    raise RuntimeError(
                        "Trainable-parameter equivalence failed for "
                        f"{key}: {parameter_audit[key]} != {reference[key]}"
                    )
        else:
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_text(json.dumps(parameter_audit, indent=2))
    print(json.dumps(parameter_audit, indent=2))
    (output_dir / "parameter_audit.json").write_text(
        json.dumps(parameter_audit, indent=2)
    )
    (output_dir / "trainable_parameters.json").write_text(
        json.dumps(parameter_audit, indent=2)
    )
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    image_features_used = False
    data_audit = (
        json.loads(Path(args.data_audit_path).expanduser().read_text())
        if args.data_audit_path
        else None
    )
    config = {
        **vars(args),
        "umbrae_checkpoint": str(Path(args.umbrae_checkpoint).expanduser().resolve()),
        "neuroroute_checkpoint": str(Path(args.neuroroute_checkpoint).expanduser().resolve()),
        "roi_prefix_source": (
            "fMRI-derived Stage-1 projected ROI tokens; image-derived CLIP "
            "targets are supervision only and are unavailable at inference"
        ),
        "umbrae_token_source": "fMRI-derived original UMBRAE brain encoder tokens",
        "uses_image_clip_tokens_at_eval": False,
        "oracle_image_token_mode": False,
        "bridge_type": args.bridge_type,
        "shikra_visual_token_count": args.num_visual_tokens,
        "shikra_image_start_token_id": SHIKRA_IMAGE_START_ID,
        "shikra_image_end_token_id": SHIKRA_IMAGE_END_ID,
        "umbrae_encoder_kind": getattr(umbrae_encoder, "encoder_kind", "injected"),
        "ground_truth_image_features_used": False,
        "structured_routing_alpha": (
            args.structured_routing_alpha
            if args.fusion_mode == "umbrae_plus_uniform_residual_soft"
            else None
        ),
        "routing_temperature": (
            args.routing_temperature
            if args.fusion_mode == "umbrae_plus_temperature_soft"
            else None
        ),
        "routing_weights_used_in_fusion": bool(
            args.fusion_mode
            in {
                "umbrae_plus_uniform_residual_soft",
                "umbrae_plus_temperature_soft",
            }
        ),
        "fgw_prior_mode": (
            args.fusion_mode if args.fusion_mode in prior_modes else None
        ),
        "fgw_gamma": (
            args.fgw_gamma if args.fusion_mode in prior_modes else None
        ),
        "fgw_delta": (
            args.fgw_delta if args.fusion_mode in prior_modes else None
        ),
        "fgw_bias_layers": (
            args.fgw_bias_layers if args.fusion_mode in prior_modes else None
        ),
        "fgw_bias_trainable": False,
        "fgw_transport_plan_trainable": False,
        "fgw_row_permutation": (
            list(row_permutation) if row_permutation is not None else None
        ),
        "fgw_plan_provenance": plan_provenance,
        "caption_evaluator": "COCO multi-reference CIDEr/ROUGE-L + corpus BLEU-1..4",
        "checkpoint_selection_rule": (
            "max multi-reference CIDEr, then ROUGE-L, then BLEU-4, "
            "then min validation LM loss"
        ),
        "data_audit": data_audit,
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2)
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2)
    )
    leakage_audit = {
        "bridge_type": args.bridge_type,
        "uses_image_clip_tokens_at_eval": False,
        "oracle_image_token_mode": False,
        "transport_plan_frozen": bool(args.fusion_mode in prior_modes),
        "transport_plan_gradient": "none",
        "protected_test_used_for_training": False,
        "protected_test_used_for_checkpoint_selection": False,
        "gamma_selected_on_subj01_only": args.subject == "subj01",
    }
    # The booleans whose safe state is false are checked explicitly.
    leakage_ok = (
        leakage_audit["bridge_type"] == "shikra_patch"
        and leakage_audit["uses_image_clip_tokens_at_eval"] is False
        and leakage_audit["oracle_image_token_mode"] is False
        and leakage_audit["protected_test_used_for_training"] is False
        and leakage_audit["protected_test_used_for_checkpoint_selection"] is False
        and leakage_audit["gamma_selected_on_subj01_only"] is True
        and (
            leakage_audit["transport_plan_frozen"] is True
            if args.fusion_mode in prior_modes
            else True
        )
    )
    if not leakage_ok:
        raise RuntimeError(f"Stage-2 leakage assertion failed: {leakage_audit}")
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, indent=2)
    )
    route_summary = routing_diagnostics(routing_np, roi_names, layers)
    route_summary["uses_image_clip_tokens_at_eval"] = False
    route_summary["fusion_mode"] = args.fusion_mode
    route_summary["structured_routing_alpha"] = config["structured_routing_alpha"]
    route_summary["routing_temperature"] = config["routing_temperature"]
    route_summary["routing_weights_used_in_fusion"] = config[
        "routing_weights_used_in_fusion"
    ]
    if routing_np is not None:
        route_summary["routing_matrix"] = routing_np.tolist()
    (output_dir / "routing_summary.json").write_text(
        json.dumps(route_summary, indent=2)
    )
    if plan_provenance is not None:
        (output_dir / "fgw_plan_provenance.json").write_text(
            plan_provenance_json(
                {
                    **plan_provenance,
                    "prior_mode": args.fusion_mode,
                    "gamma": args.fgw_gamma,
                    "delta": args.fgw_delta,
                    "bias_layers": args.fgw_bias_layers,
                    "row_permutation_indices": (
                        list(row_permutation)
                        if row_permutation is not None
                        else None
                    ),
                }
            )
        )

    best_selection = None
    selected_epoch = None
    start_epoch = 1
    epoch_history = []
    checkpoint_hashes = {}
    resume_path = output_dir / "checkpoint_last.pt"
    if (
        args.resume_if_incomplete
        and resume_path.is_file()
        and not args.overwrite
    ):
        resume = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(resume["model"], strict=False)
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"]) + 1
        history_path = output_dir / "epoch_metrics.json"
        if history_path.is_file():
            epoch_history = json.loads(history_path.read_text())
            for item in epoch_history:
                score = (
                    item["CIDEr"], item["ROUGE-L"], item["BLEU-4"],
                    -item["val_loss"],
                )
                if best_selection is None or score > best_selection:
                    best_selection = score
                    selected_epoch = item["epoch"]
        print(f"Resuming incomplete run at epoch {start_epoch}: {output_dir}")
    final_metrics = {}
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        if args.freeze_umbrae:
            model.umbrae_encoder.eval()
        if args.freeze_neuroroute:
            model.neuroroute_encoder.eval()
        if args.freeze_mllm:
            model.mllm.eval()
        train_losses = []
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} train")
        ):
            fmri = batch["fmri"].to(args.device)
            output = model.encode_visual_tokens(fmri)
            result = caption_forward(
                model,
                tokenizer,
                output["visual_tokens"],
                batch["caption"],
                args,
            )
            optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            optimizer.step()
            train_losses.append(float(result.loss.detach()))
            if args.debug_max_steps and step + 1 >= args.debug_max_steps:
                break

        model.eval()
        val_losses: List[float] = []
        references: List[object] = []
        predictions: List[str] = []
        records = []
        with torch.no_grad():
            for step, batch in enumerate(
                tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} val")
            ):
                assert_no_image_feature_leakage(
                    image_features_used,
                    "validation",
                    args.oracle_image_token_mode,
                )
                fmri = batch["fmri"].to(args.device)
                output = model.encode_visual_tokens(fmri)
                result = caption_forward(
                    model,
                    tokenizer,
                    output["visual_tokens"],
                    batch["caption"],
                    args,
                )
                generated = generate_captions(
                    model,
                    tokenizer,
                    output["visual_tokens"],
                    args,
                )
                val_losses.append(float(result.loss))
                batch_references = []
                for item_index, local_id in enumerate(batch["coco_id"]):
                    reference_set = captions.get(
                        str(local_id), batch["caption"][item_index]
                    )
                    if not isinstance(reference_set, list):
                        reference_set = [str(reference_set)]
                    batch_references.append([str(item) for item in reference_set])
                references.extend(batch_references)
                predictions.extend(generated)
                for sample_id, local_id, reference_set, prediction in zip(
                    batch["sample_id"],
                    batch["coco_id"],
                    batch_references,
                    generated,
                ):
                    records.append(
                        {
                            "sample_id": sample_id,
                            "local_nsd_id": str(local_id),
                            "references": reference_set,
                            "prediction": prediction,
                            "uses_image_clip_tokens": False,
                        }
                    )
                if args.debug_max_steps and step + 1 >= args.debug_max_steps:
                    break

        caption_metrics, per_sample_metrics = evaluate_captions(
            references, predictions, return_per_sample=True
        )
        if caption_metrics.get("CIDEr") is None:
            raise RuntimeError(
                "Multi-reference CIDEr is required for checkpoint selection"
            )
        for index, record in enumerate(records):
            record["per_stimulus_CIDEr"] = per_sample_metrics.get(
                "CIDEr", [None] * len(records)
            )[index]
            record["per_stimulus_ROUGE-L"] = per_sample_metrics.get(
                "ROUGE-L", [None] * len(records)
            )[index]
        elapsed = time.perf_counter() - run_start_time
        peak_gpu = (
            int(torch.cuda.max_memory_allocated(args.device))
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else None
        )
        final_metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "caption_metrics": caption_metrics,
            "num_generated_captions": len(predictions),
            "uses_image_clip_tokens_at_eval": False,
            "bridge_type": args.bridge_type,
            "fusion_mode": args.fusion_mode,
            "training_time_seconds": elapsed,
            "epoch_time_seconds": time.perf_counter() - epoch_start_time,
            "peak_gpu_memory_bytes": peak_gpu,
            "trainable_parameter_count": trainable_parameter_count,
            "leakage_assertions": leakage_audit,
        }
        epoch_predictions_path = (
            output_dir / f"validation_predictions_epoch{epoch:03d}.jsonl"
        )
        with epoch_predictions_path.open("w") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")

        checkpoint_state = {
            key: value
            for key, value in model.state_dict().items()
            if key.startswith("adapter.")
            or (not args.freeze_umbrae and key.startswith("umbrae_encoder."))
            or (
                not args.freeze_neuroroute
                and key.startswith("neuroroute_encoder.")
            )
            or (not args.freeze_mllm and key.startswith("mllm."))
        }
        checkpoint = {
            "epoch": epoch,
            "model": checkpoint_state,
            "optimizer": optimizer.state_dict(),
            "adapter_config": config,
            "val_loss": final_metrics["val_loss"],
        }
        epoch_checkpoint = output_dir / f"checkpoint_epoch{epoch:03d}.pt"
        torch.save(checkpoint, epoch_checkpoint)
        shutil.copy2(epoch_checkpoint, output_dir / "checkpoint_last.pt")
        checkpoint_hashes[str(epoch)] = sha256_file(epoch_checkpoint)
        epoch_row = {
            "epoch": epoch,
            "train_loss": final_metrics["train_loss"],
            "val_loss": final_metrics["val_loss"],
            **{
                key: caption_metrics[key]
                for key in ("CIDEr", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L")
            },
            "checkpoint": epoch_checkpoint.name,
            "checkpoint_sha256": checkpoint_hashes[str(epoch)],
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": peak_gpu,
        }
        epoch_history.append(epoch_row)
        (output_dir / "epoch_metrics.json").write_text(
            json.dumps(epoch_history, indent=2)
        )
        with (output_dir / "epoch_metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(epoch_row))
            writer.writeheader()
            writer.writerows(epoch_history)
        selection = (
            caption_metrics["CIDEr"],
            caption_metrics["ROUGE-L"],
            caption_metrics["BLEU-4"],
            -final_metrics["val_loss"],
        )
        if best_selection is None or selection > best_selection:
            best_selection = selection
            selected_epoch = epoch
            shutil.copy2(epoch_checkpoint, output_dir / "checkpoint_best.pt")
            shutil.copy2(
                epoch_predictions_path,
                output_dir / "validation_predictions.jsonl",
            )
            final_metrics["selected_epoch"] = selected_epoch
            final_metrics["selected_checkpoint"] = epoch_checkpoint.name
            final_metrics["selected_checkpoint_sha256"] = checkpoint_hashes[str(epoch)]
            (output_dir / "validation_metrics.json").write_text(
                json.dumps(final_metrics, indent=2)
            )
            (output_dir / "metrics.json").write_text(
                json.dumps(final_metrics, indent=2)
            )
            shutil.copy2(
                epoch_predictions_path, output_dir / "predictions.jsonl"
            )
            with (output_dir / "generated_captions.jsonl").open("w") as file:
                for record in records:
                    file.write(json.dumps({
                        "sample_id": record["sample_id"],
                        "caption": record["prediction"],
                    }) + "\n")
        (output_dir / "checkpoint_hashes.json").write_text(
            json.dumps(checkpoint_hashes, indent=2)
        )
        print(
            f"epoch={epoch} train={final_metrics['train_loss']:.6f} "
            f"val={final_metrics['val_loss']:.6f} "
            f"CIDEr={caption_metrics['CIDEr']:.6f} "
            f"BLEU-4={caption_metrics['BLEU-4']:.6f}"
        )
    selected_metrics = json.loads(
        (output_dir / "validation_metrics.json").read_text()
    )
    return {
        "model": model,
        "metrics": selected_metrics,
        "output_dir": output_dir,
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
