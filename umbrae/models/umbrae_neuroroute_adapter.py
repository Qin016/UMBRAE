"""Fuse original UMBRAE tokens with fMRI-derived NeuroRoute ROI tokens."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .fgw_stage2_prior import row_normalize_transport, tempered_attention_bias


def build_source_attention_bias(
    *,
    num_source_keys: int,
    roi_key_start: int,
    roi_layer_bias: Tensor,
    roi_token_expansion: int = 1,
) -> Tensor:
    """Place [R,L] bias at the derived ROI-layer source-key positions."""
    if roi_layer_bias.ndim != 2:
        raise ValueError("roi_layer_bias must have shape [R,L]")
    if roi_token_expansion < 1:
        raise ValueError("roi_token_expansion must be >= 1")
    num_rois, num_layers = roi_layer_bias.shape
    flattened = (
        roi_layer_bias[:, None, :]
        .expand(num_rois, roi_token_expansion, num_layers)
        .reshape(-1)
    )
    roi_key_end = roi_key_start + flattened.numel()
    if roi_key_start < 0 or roi_key_end > num_source_keys:
        raise ValueError(
            f"ROI key range [{roi_key_start},{roi_key_end}) is outside "
            f"the {num_source_keys} source keys"
        )
    result = roi_layer_bias.new_zeros(num_source_keys)
    result[roi_key_start:roi_key_end] = flattened
    return result


class _PerceiverBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def raw_content_logits(self, queries: Tensor, source: Tensor) -> Tensor:
        """Return pre-bias QK/sqrt(d) as [B,H,Q,K] for diagnostics."""
        query = self.query_norm(queries)
        key = self.source_norm(source)
        weight = self.attention.in_proj_weight
        bias = self.attention.in_proj_bias
        dim = self.attention.embed_dim
        q = F.linear(query, weight[:dim], None if bias is None else bias[:dim])
        k = F.linear(
            key,
            weight[dim : 2 * dim],
            None if bias is None else bias[dim : 2 * dim],
        )
        batch, num_queries, _ = q.shape
        num_keys = k.shape[1]
        heads = self.attention.num_heads
        head_dim = dim // heads
        q = q.reshape(batch, num_queries, heads, head_dim).transpose(1, 2)
        k = k.reshape(batch, num_keys, heads, head_dim).transpose(1, 2)
        return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    def forward(
        self,
        queries: Tensor,
        source: Tensor,
        key_attention_bias: Optional[Tensor] = None,
    ) -> Tensor:
        attention_mask = None
        if key_attention_bias is not None:
            if key_attention_bias.shape != (source.shape[1],):
                raise ValueError(
                    "key_attention_bias must have shape [K], got "
                    f"{tuple(key_attention_bias.shape)}"
                )
            attention_mask = key_attention_bias.unsqueeze(0).expand(
                queries.shape[1], -1
            )
        attended, _ = self.attention(
            self.query_norm(queries),
            self.source_norm(source),
            self.source_norm(source),
            attn_mask=attention_mask,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(queries)


class UMBRAENeuroRouteAdapter(nn.Module):
    """Produce the fixed-length visual token sequence expected by Shikra.

    The original UMBRAE path supplies 256 tokens with dimension 1024. The
    strict Shikra bridge expects 256 replacement embeddings with dimension
    4096. NeuroRoute tokens augment that sequence but never change the number
    of ``<im_patch>`` placeholders.
    """

    SUPPORTED_MODES = (
        "umbrae_only",
        "neuroroute_only",
        "umbrae_plus_soft",
        "umbrae_plus_uniform",
        "umbrae_plus_single_l24",
        "umbrae_plus_uniform_residual_soft",
        "umbrae_plus_temperature_soft",
        "uniform_prior",
        "fgw_prior",
        "row_shuffled_fgw_prior",
    )
    SUPPORTED_FUSIONS = (
        "concat_then_project",
        "cross_attention",
        "perceiver_resampler",
    )

    def __init__(
        self,
        umbrae_dim: int = 1024,
        neuroroute_dim: int = 1024,
        mllm_dim: int = 4096,
        num_visual_tokens: int = 256,
        fusion_mode: str = "umbrae_plus_soft",
        fusion_type: str = "perceiver_resampler",
        fusion_dim: Optional[int] = None,
        roi_token_expansion: int = 1,
        num_attention_heads: int = 8,
        perceiver_depth: int = 2,
        num_routing_layers: int = 6,
        dropout: float = 0.1,
        fgw_transport_plan: Optional[Tensor] = None,
        fgw_gamma: float = 1.0,
        fgw_delta: float = 1e-8,
        fgw_bias_layers: str = "first",
    ) -> None:
        super().__init__()
        if fusion_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}; expected one of "
                f"{self.SUPPORTED_MODES}"
            )
        if fusion_type not in self.SUPPORTED_FUSIONS:
            raise ValueError(
                f"Unsupported fusion_type={fusion_type!r}; expected one of "
                f"{self.SUPPORTED_FUSIONS}"
            )
        if roi_token_expansion < 1:
            raise ValueError("roi_token_expansion must be >= 1")
        fusion_dim = int(fusion_dim or umbrae_dim)
        if fusion_dim % num_attention_heads:
            raise ValueError(
                "fusion_dim must be divisible by num_attention_heads"
            )

        self.umbrae_dim = int(umbrae_dim)
        self.neuroroute_dim = int(neuroroute_dim)
        self.mllm_dim = int(mllm_dim)
        self.num_visual_tokens = int(num_visual_tokens)
        self.fusion_mode = fusion_mode
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim
        self.roi_token_expansion = int(roi_token_expansion)
        self.num_routing_layers = int(num_routing_layers)
        self.fgw_gamma = float(fgw_gamma)
        self.fgw_delta = float(fgw_delta)
        self.fgw_bias_layers = fgw_bias_layers
        self.uses_attention_prior = fusion_mode in {
            "uniform_prior",
            "fgw_prior",
            "row_shuffled_fgw_prior",
        }
        if self.uses_attention_prior and fusion_type == "concat_then_project":
            raise ValueError(
                "FGW attention-prior modes require an attention fusion type"
            )
        if self.uses_attention_prior and fgw_bias_layers != "first":
            raise ValueError("Prompt 6A supports only --fgw-bias-layers first")
        self.uses_structured_routing = fusion_mode in {
            "umbrae_plus_uniform_residual_soft",
            "umbrae_plus_temperature_soft",
        }
        self.expands_roi_layers = (
            self.uses_structured_routing or self.uses_attention_prior
        )

        if self.uses_attention_prior:
            if fusion_mode == "uniform_prior":
                plan = torch.full((8, self.num_routing_layers), 1.0 / 48.0)
            else:
                if fgw_transport_plan is None:
                    raise ValueError(f"{fusion_mode} requires a frozen FGW plan")
                plan = torch.as_tensor(fgw_transport_plan).detach().clone()
            if tuple(plan.shape) != (8, self.num_routing_layers):
                raise ValueError(
                    "FGW transport plan must match [8,num_routing_layers], "
                    f"got {tuple(plan.shape)}"
                )
            correspondence = row_normalize_transport(plan)
            attention_bias = tempered_attention_bias(
                correspondence,
                gamma=self.fgw_gamma,
                delta=self.fgw_delta,
            )
        else:
            plan = torch.empty(0)
            attention_bias = torch.empty(0)
        self.register_buffer("fgw_transport_plan", plan, persistent=True)
        self.register_buffer(
            "fgw_attention_bias", attention_bias, persistent=True
        )

        self.umbrae_input = nn.Sequential(
            nn.LayerNorm(umbrae_dim),
            nn.Linear(umbrae_dim, fusion_dim),
        )
        self.roi_expander = nn.Sequential(
            nn.LayerNorm(neuroroute_dim),
            nn.Linear(
                neuroroute_dim,
                fusion_dim * self.roi_token_expansion,
            ),
            nn.GELU(),
        )
        self.source_type_embedding = nn.Parameter(
            torch.zeros(2, fusion_dim)
        )
        self.routing_layer_embeddings = nn.Parameter(
            torch.empty(self.num_routing_layers, fusion_dim)
        )
        nn.init.normal_(self.source_type_embedding, std=0.02)
        nn.init.normal_(self.routing_layer_embeddings, std=0.02)

        self.output_queries = nn.Parameter(
            torch.empty(1, num_visual_tokens, fusion_dim)
        )
        nn.init.normal_(self.output_queries, std=0.02)
        self.perceiver_blocks = nn.ModuleList(
            [
                _PerceiverBlock(
                    fusion_dim,
                    num_attention_heads,
                    dropout,
                )
                for _ in range(max(1, perceiver_depth))
            ]
        )
        self.cross_attention = nn.MultiheadAttention(
            fusion_dim,
            num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projector = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, mllm_dim),
        )
        self.umbrae_direct_projector = nn.Linear(
            umbrae_dim,
            mllm_dim,
        )

    def load_umbrae_mm_projector(
        self,
        state_dict: Dict[str, Tensor],
    ) -> None:
        """Load the original UMBRAE/Shikra 1024->4096 projector."""
        normalized = {
            key.split(".")[-1]: value
            for key, value in state_dict.items()
            if key.split(".")[-1] in {"weight", "bias"}
        }
        self.umbrae_direct_projector.load_state_dict(
            normalized,
            strict=True,
        )

    def _expand_roi_tokens(
        self,
        roi_tokens: Tensor,
        reliability: Optional[Tensor],
        routing_weights: Optional[Tensor],
    ) -> Tensor:
        batch, num_rois, _ = roi_tokens.shape
        expanded = self.roi_expander(roi_tokens).reshape(
            batch,
            num_rois * self.roi_token_expansion,
            self.fusion_dim,
        )
        if reliability is not None:
            if reliability.shape != (batch, num_rois):
                raise ValueError(
                    "reliability must have shape [B,R], got "
                    f"{tuple(reliability.shape)}"
                )
            reliability = reliability.repeat_interleave(
                self.roi_token_expansion,
                dim=1,
            )
            expanded = expanded * reliability.unsqueeze(-1)
        expanded = expanded + self.source_type_embedding[1]
        if not self.expands_roi_layers:
            return expanded
        if self.uses_structured_routing and routing_weights is None:
            raise ValueError(
                "Structured routing modes require routing_weights [B,R,L]"
            )
        if self.uses_structured_routing and routing_weights.shape != (
            batch,
            num_rois,
            self.num_routing_layers,
        ):
            raise ValueError(
                "routing_weights must have shape "
                f"[B,R,{self.num_routing_layers}], got "
                f"{tuple(routing_weights.shape)}"
            )
        if self.uses_structured_routing and not torch.isfinite(routing_weights).all():
            raise ValueError("routing_weights contains NaN or Inf")
        if self.uses_structured_routing and not torch.allclose(
            routing_weights.sum(dim=-1),
            torch.ones(
                batch,
                num_rois,
                device=routing_weights.device,
                dtype=routing_weights.dtype,
            ),
            atol=1e-5,
        ):
            raise ValueError("routing_weights must sum to one over layers")

        expanded = expanded.reshape(
            batch,
            num_rois,
            self.roi_token_expansion,
            self.fusion_dim,
        )
        layer_tokens = (
            expanded.unsqueeze(3)
            + self.routing_layer_embeddings.view(
                1,
                1,
                1,
                self.num_routing_layers,
                self.fusion_dim,
            )
        )
        if self.uses_structured_routing:
            # Legacy structured path: this scalar multiplication occurs
            # before the Perceiver's token-wise source LayerNorm.  The new
            # FGW paths intentionally do not encode correspondence this way.
            weights = routing_weights[:, :, None, :, None]
            layer_tokens = layer_tokens * weights * self.num_routing_layers
        return layer_tokens.reshape(
            batch,
            num_rois * self.roi_token_expansion * self.num_routing_layers,
            self.fusion_dim,
        )

    def _fixed_length(
        self,
        source: Tensor,
        key_attention_bias: Optional[Tensor] = None,
    ) -> Tensor:
        if self.fusion_type == "concat_then_project":
            return F.adaptive_avg_pool1d(
                source.transpose(1, 2),
                self.num_visual_tokens,
            ).transpose(1, 2)

        queries = self.output_queries.expand(source.shape[0], -1, -1)
        if self.fusion_type == "cross_attention":
            attention_mask = (
                key_attention_bias.unsqueeze(0).expand(
                    self.num_visual_tokens, -1
                )
                if key_attention_bias is not None
                else None
            )
            output, _ = self.cross_attention(
                queries,
                source,
                source,
                attn_mask=attention_mask,
                need_weights=False,
            )
            return queries + output

        for block_index, block in enumerate(self.perceiver_blocks):
            queries = block(
                queries,
                source,
                key_attention_bias=(
                    key_attention_bias if block_index == 0 else None
                ),
            )
        return queries

    def forward(
        self,
        umbrae_tokens: Optional[Tensor],
        neuroroute_roi_tokens: Optional[Tensor],
        routing_weights: Optional[Tensor] = None,
        reliability: Optional[Tensor] = None,
    ) -> Dict[str, object]:
        use_umbrae = self.fusion_mode != "neuroroute_only"
        use_roi = self.fusion_mode != "umbrae_only"

        if use_umbrae:
            if umbrae_tokens is None or umbrae_tokens.ndim != 3:
                raise ValueError(
                    "umbrae_tokens must have shape [B,N_u,D_u]"
                )
            if umbrae_tokens.shape[-1] != self.umbrae_dim:
                raise ValueError(
                    f"Expected UMBRAE D={self.umbrae_dim}, got "
                    f"{umbrae_tokens.shape[-1]}"
                )
            umbrae = (
                self.umbrae_input(umbrae_tokens)
                + self.source_type_embedding[0]
            )
        else:
            umbrae = None

        if use_roi:
            if (
                neuroroute_roi_tokens is None
                or neuroroute_roi_tokens.ndim != 3
            ):
                raise ValueError(
                    "neuroroute_roi_tokens must have shape [B,R,D_r]"
                )
            if neuroroute_roi_tokens.shape[-1] != self.neuroroute_dim:
                raise ValueError(
                    f"Expected NeuroRoute D={self.neuroroute_dim}, got "
                    f"{neuroroute_roi_tokens.shape[-1]}"
                )
            roi = self._expand_roi_tokens(
                neuroroute_roi_tokens,
                reliability,
                routing_weights,
            )
        else:
            roi = None

        if umbrae is not None and roi is not None:
            if umbrae.shape[0] != roi.shape[0]:
                raise ValueError("UMBRAE and ROI batch sizes must match")
            source = torch.cat([umbrae, roi], dim=1)
        else:
            source = umbrae if umbrae is not None else roi
        if source is None:
            raise RuntimeError("No visual source tokens were supplied")

        key_attention_bias = None
        roi_key_start = None
        roi_key_end = None
        if self.uses_attention_prior:
            if umbrae is None or roi is None:
                raise RuntimeError(
                    "FGW Stage-2 primary modes require UMBRAE and ROI keys"
                )
            roi_key_start = int(umbrae.shape[1])
            roi_key_end = roi_key_start + int(roi.shape[1])
            key_attention_bias = build_source_attention_bias(
                num_source_keys=int(source.shape[1]),
                roi_key_start=roi_key_start,
                roi_layer_bias=self.fgw_attention_bias.to(
                    device=source.device, dtype=source.dtype
                ),
                roi_token_expansion=self.roi_token_expansion,
            )

        # Preserve original 256 UMBRAE positions for the baseline whenever
        # possible. Augmented modes and ROI-only mode require resampling.
        if (
            self.fusion_mode == "umbrae_only"
            and umbrae_tokens is not None
            and umbrae_tokens.shape[1] == self.num_visual_tokens
        ):
            visual_tokens = self.umbrae_direct_projector(umbrae_tokens)
        else:
            fused = self._fixed_length(source, key_attention_bias)
            visual_tokens = self.output_projector(fused)
        diagnostics = {
            "fusion_mode": self.fusion_mode,
            "fusion_type": self.fusion_type,
            "num_umbrae_tokens": (
                int(umbrae_tokens.shape[1])
                if umbrae_tokens is not None
                else 0
            ),
            "num_roi_tokens": (
                int(neuroroute_roi_tokens.shape[1])
                if neuroroute_roi_tokens is not None
                else 0
            ),
            "num_expanded_roi_tokens": (
                int(roi.shape[1]) if roi is not None else 0
            ),
            "num_visual_tokens": int(visual_tokens.shape[1]),
            "num_source_keys": int(source.shape[1]),
            "roi_layer_key_start": roi_key_start,
            "roi_layer_key_end": roi_key_end,
            "latent_self_key_count": 0,
            "attention_logit_shape": (
                [
                    int(source.shape[0]),
                    int(self.perceiver_blocks[0].attention.num_heads),
                    self.num_visual_tokens,
                    int(source.shape[1]),
                ]
                if self.fusion_type == "perceiver_resampler"
                else None
            ),
            "fgw_bias_layers": (
                self.fgw_bias_layers if self.uses_attention_prior else None
            ),
            "visual_token_norm": (
                visual_tokens.norm(dim=-1).mean().detach()
            ),
        }
        return {
            "visual_tokens": visual_tokens,
            "diagnostics": diagnostics,
            "routing_weights": routing_weights,
            "reliability": reliability,
        }
