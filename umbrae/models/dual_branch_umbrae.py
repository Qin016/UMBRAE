"""Foundation wrapper for frozen semantic and trainable structural branches."""

from typing import Dict, Optional

from torch import Tensor, nn

from .gated_cross_attention import GatedCrossAttentionFusion
from .structural_branch import StructuralBranch


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    """Set each parameter explicitly; avoids future whole-model broad freezes."""
    for parameter in module.parameters():
        parameter.requires_grad = bool(trainable)


class DualBranchUMBRAE(nn.Module):
    def __init__(
        self,
        semantic_backbone: nn.Module,
        structural_branch: StructuralBranch,
        fusion: Optional[GatedCrossAttentionFusion] = None,
        repeat_aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if repeat_aggregation not in {"mean", "first"}:
            raise ValueError("repeat_aggregation must be 'mean' or 'first'")
        self.semantic_backbone = semantic_backbone
        self.structural_branch = structural_branch
        self.fusion = fusion or GatedCrossAttentionFusion(
            dim=structural_branch.token_dim
        )
        self.repeat_aggregation = repeat_aggregation
        self.stage = "A"
        self.configure_stage("A", verbose=False)

    def _semantic_input(self, fmri: Tensor) -> Tensor:
        if fmri.ndim == 3:
            return fmri.mean(dim=1) if self.repeat_aggregation == "mean" else fmri[:, 0]
        if fmri.ndim != 2:
            raise ValueError(f"fmri must have shape [B,V] or [B,R,V], got {tuple(fmri.shape)}")
        return fmri

    def configure_stage(self, stage: str, verbose: bool = True) -> Dict[str, object]:
        stage = stage.upper()
        if stage not in {"A", "B"}:
            raise ValueError("Only Stage A and Stage B are implemented in P2")
        self.stage = stage
        _set_module_trainable(self.semantic_backbone, False)
        _set_module_trainable(self.structural_branch, True)
        _set_module_trainable(self.fusion, stage == "B")
        self.semantic_backbone.eval()
        if stage == "A":
            self.fusion.eval()
        report = self.parameter_report()
        if verbose:
            names = report["trainable_parameter_names"]
            summary = names[:20] + ([f"... ({len(names) - 20} more)"] if len(names) > 20 else [])
            print(
                f"Stage {stage}: total_params={report['total']:,} "
                f"trainable_params={report['trainable']:,} "
                f"trainable_ratio={report['trainable_ratio']:.6f}"
            )
            print("Trainable parameter names:", summary)
        return report

    def train(self, mode: bool = True) -> "DualBranchUMBRAE":
        super().train(mode)
        self.semantic_backbone.eval()
        if self.stage == "A":
            self.fusion.eval()
        return self

    def parameter_report(self) -> Dict[str, object]:
        def counts(module: nn.Module) -> Dict[str, int]:
            return {
                "total": sum(parameter.numel() for parameter in module.parameters()),
                "trainable": sum(
                    parameter.numel()
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ),
            }

        total = counts(self)["total"]
        trainable = counts(self)["trainable"]
        return {
            "total": total,
            "trainable": trainable,
            "trainable_ratio": trainable / total if total else 0.0,
            "semantic_backbone": counts(self.semantic_backbone),
            "structural_branch": counts(self.structural_branch),
            "fusion": counts(self.fusion),
            "trainable_parameter_names": [
                name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            ],
        }

    def forward(
        self,
        fmri: Tensor,
        modal: Optional[str] = None,
        enable_semantic: bool = True,
        enable_structural: bool = True,
        enable_fusion: bool = True,
        return_attention: bool = True,
    ) -> Dict[str, object]:
        del modal  # The strict backbone wrapper owns its checkpoint-derived modal.
        z_sem = None
        h_struct = None
        roi_names = None
        if enable_semantic:
            z_sem = self.semantic_backbone(self._semantic_input(fmri))
        if enable_structural:
            structural = self.structural_branch(fmri)
            h_struct = structural["h_struct"]
            roi_names = structural["roi_names"]

        fusion_output = None
        if enable_fusion:
            if z_sem is None or h_struct is None:
                raise ValueError("Fusion requires both semantic and structural branches")
            fusion_output = self.fusion(
                z_sem, h_struct, return_attention=return_attention
            )
        return {
            "z_sem": z_sem,
            "h_struct": h_struct,
            "z_cal": fusion_output["z_cal"] if fusion_output is not None else None,
            "fusion_gate": (
                fusion_output["fusion_gate"] if fusion_output is not None else None
            ),
            "fusion_attention": (
                fusion_output["fusion_attention"] if fusion_output is not None else None
            ),
            "roi_names": roi_names,
        }
