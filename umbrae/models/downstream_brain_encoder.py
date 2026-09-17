"""Unified frozen brain representation interface for P7 downstream evaluation."""

from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import torch
from torch import Tensor, nn

from models.gated_cross_attention import GatedCrossAttentionFusion
from models.shared_lora import SharedLoRAUMBRAEEncoder, load_lora_state_dict
from models.umbrae_backbone import FrozenUMBRAEEncoder
from scripts.train_p3r_roi_relational import P3RModel
from scripts.train_stage_a_structural import active_mapping


MODES = ("umbrae", "lora", "full_real", "full_random")


class DownstreamBrainEncoder(nn.Module):
    def __init__(self, mode: str, config: Mapping[str, object]) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown downstream mode: {mode}")
        self.mode = mode
        self.config = dict(config)
        self.semantic = FrozenUMBRAEEncoder(config["brainx_checkpoint"], config.get("subject", "subj01"))
        self.structural = None
        self.fusion = None
        if mode != "umbrae":
            self.semantic = SharedLoRAUMBRAEEncoder(self.semantic, 8, 16.0, 0.05)
            checkpoint_path = config["p5_checkpoint"] if mode == "lora" else config[f"p6_{mode[5:]}_checkpoint"]
            state = torch.load(Path(checkpoint_path).resolve(), map_location="cpu")
            load_lora_state_dict(self.semantic, state["lora"])
            if mode.startswith("full_"):
                kind = mode[5:]
                args = SimpleNamespace(mapping_kind=kind, roi_mapping=config[f"{kind}_roi_mapping"])
                _, mapping = active_mapping(args)
                self.structural = P3RModel(mapping)
                structural_state = torch.load(Path(config[f"{kind}_structural_checkpoint"]).resolve(), map_location="cpu")
                self.structural.load_state_dict(structural_state["model"], strict=True)
                self.fusion = GatedCrossAttentionFusion(1024, 8, 0.0, -4.0)
                self.fusion.load_state_dict(state["fusion"], strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, fmri: Tensor) -> Tensor:
        with torch.inference_mode():
            semantic = self.semantic(fmri)
            if self.mode.startswith("full_"):
                structural = self.structural.structural(fmri)["h_struct"]
                semantic = self.fusion(semantic, structural, return_attention=False)["z_cal"]
        if semantic.shape != (len(fmri), 256, 1024) or not torch.isfinite(semantic).all():
            raise RuntimeError(f"Invalid downstream representation for {self.mode}: {tuple(semantic.shape)}")
        return semantic


def encode_brain_for_downstream(mode: str, fmri: Tensor, config: Mapping[str, object], device: str = "cuda") -> Tensor:
    model = DownstreamBrainEncoder(mode, config).to(device).eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
        return model(fmri.to(device))
