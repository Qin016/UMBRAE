"""Strict, frozen restoration of the pretrained UMBRAE brain backbone."""

from pathlib import Path
from typing import Mapping, Union

import torch
from torch import Tensor, nn

from model import BrainX, BrainXS
from .roi_mapping import normalize_subject_id


def _load_checkpoint(path: Union[str, Path]) -> Mapping[str, object]:
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("UMBRAE checkpoint must be a mapping")
    return checkpoint


class FrozenUMBRAEEncoder(nn.Module):
    """Restore BrainX/BrainXS with strict loading and keep it frozen."""

    def __init__(self, checkpoint_path: Union[str, Path], subject: object) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = _load_checkpoint(checkpoint_path)
        state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        if not isinstance(state, Mapping):
            raise ValueError("UMBRAE checkpoint does not contain a state dict")

        subject_name = normalize_subject_id(subject)
        subject_id = int(subject_name[-2:])
        projector_weight = state.get("perceiver.llm_proj.weight")
        latent_weight = state.get("perceiver.perceiver.latents")
        if not isinstance(projector_weight, Tensor) or not isinstance(latent_weight, Tensor):
            raise ValueError("Checkpoint is missing UMBRAE Perceiver shape metadata")
        out_dim, hidden_dim = map(int, projector_weight.shape)
        num_latents = int(latent_weight.shape[0])
        is_cross_subject = any(key.startswith("lin1.fmri") for key in state)

        if is_cross_subject:
            modal = f"fmri{subject_id}"
            if f"lin2.{modal}.weight" not in state:
                raise ValueError(
                    f"Checkpoint has no subject-specific tokenizer for {subject_name}"
                )
            use_norm = any(key.startswith(f"lin1.{modal}.0") for key in state)
            use_token = any(key.startswith(f"token.{modal}") for key in state)
            encoder = BrainX(
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_latents=num_latents,
                use_norm=use_norm,
                use_token=use_token,
            )
            self.modal = modal
        else:
            lin2_weight = state.get("lin2.weight")
            if not isinstance(lin2_weight, Tensor):
                raise ValueError("Single-subject checkpoint is missing lin2.weight")
            encoder = BrainXS(
                in_dim=int(lin2_weight.shape[1]),
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_latents=num_latents,
            )
            self.modal = None

        encoder.load_state_dict(state, strict=True)
        encoder.requires_grad_(False)
        encoder.eval()
        self.encoder = encoder
        self.output_dim = out_dim
        self.num_visual_tokens = num_latents
        self.encoder_kind = "BrainX" if is_cross_subject else "BrainXS"
        self.subject = subject_name
        self.checkpoint_path = str(checkpoint_path)

    def train(self, mode: bool = True) -> "FrozenUMBRAEEncoder":
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, fmri: Tensor) -> Tensor:
        if fmri.ndim != 2:
            raise ValueError(f"fmri must have shape [B,V], got {tuple(fmri.shape)}")
        output = (
            self.encoder(fmri)
            if self.modal is None
            else self.encoder(fmri, modal=self.modal)
        )
        expected = (fmri.shape[0], self.num_visual_tokens, self.output_dim)
        if tuple(output.shape) != expected:
            raise RuntimeError(
                f"Unexpected UMBRAE output shape {tuple(output.shape)}; expected {expected}"
            )
        return output
