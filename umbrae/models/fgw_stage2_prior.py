"""Frozen FGW transport plans and Stage-2 attention-prior utilities.

This module deliberately contains no solver code.  Stage-2 may load and
validate the frozen correspondence plans, but it must never refit them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


ROI_ORDER: Tuple[str, ...] = (
    "V1",
    "V2",
    "V3",
    "hV4",
    "FFA",
    "EBA",
    "PPA",
    "OPA",
)
CLIP_LAYER_ORDER: Tuple[int, ...] = (4, 8, 12, 16, 20, 24)
SUPPORTED_STAGE2_SUBJECTS: Tuple[str, ...] = ("subj01", "subj02", "subj05")
DEFAULT_DELTA = 1e-8


@dataclass(frozen=True)
class FrozenPlanSpec:
    relative_path: str
    sha256: str


FROZEN_PLAN_SPECS: Dict[str, FrozenPlanSpec] = {
    "subj01": FrozenPlanSpec(
        "subj01/correspondence_offline_test_v1/"
        "final_transport_plan_pretest.npy",
        "51e65527f3bbc1a3951ec376dd8be3662cab9e6b5f682d2a0da26fb347e20760",
    ),
    "subj02": FrozenPlanSpec(
        "subj02/correspondence_offline_test_replication_v1/"
        "final_transport_plan_pretest.npy",
        "a3d3ab31770ecc0fc62599597538a7b931df2721fdd22aabb13e8e7e8f7c8da0",
    ),
    "subj05": FrozenPlanSpec(
        "subj05/correspondence_offline_test_replication_v1/"
        "final_transport_plan_pretest.npy",
        "a3d3ab31770ecc0fc62599597538a7b931df2721fdd22aabb13e8e7e8f7c8da0",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plan(
    plan: np.ndarray,
    *,
    roi_order: Sequence[str] = ROI_ORDER,
    clip_layer_order: Sequence[int] = CLIP_LAYER_ORDER,
    atol: float = 1e-8,
) -> None:
    """Validate values and the caller-declared semantic axis order."""
    if tuple(roi_order) != ROI_ORDER:
        raise ValueError(
            f"ROI order must be exactly {ROI_ORDER}, got {tuple(roi_order)}"
        )
    if tuple(int(layer) for layer in clip_layer_order) != CLIP_LAYER_ORDER:
        raise ValueError(
            "CLIP layer order must be exactly "
            f"{CLIP_LAYER_ORDER}, got {tuple(clip_layer_order)}"
        )
    array = np.asarray(plan)
    if array.shape != (len(ROI_ORDER), len(CLIP_LAYER_ORDER)):
        raise ValueError(f"Frozen plan must have shape [8,6], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Frozen plan contains NaN or Inf")
    if (array < 0).any():
        raise ValueError("Frozen plan must be nonnegative")
    if not np.isclose(array.sum(), 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"Frozen plan total mass is {array.sum()}, expected 1")
    expected_rows = np.full(len(ROI_ORDER), 1.0 / len(ROI_ORDER))
    if not np.allclose(array.sum(axis=1), expected_rows, atol=atol, rtol=0.0):
        raise ValueError("Every frozen-plan row must have mass 1/8")


def load_frozen_plan(
    subject: str,
    plan_root: Path | str,
    *,
    roi_order: Sequence[str] = ROI_ORDER,
    clip_layer_order: Sequence[int] = CLIP_LAYER_ORDER,
) -> Tuple[Tensor, Dict[str, object]]:
    """Load one allow-listed frozen plan and verify its exact file hash."""
    if subject not in FROZEN_PLAN_SPECS:
        raise ValueError(
            f"FGW Stage-2 permits only {SUPPORTED_STAGE2_SUBJECTS}; "
            f"received {subject!r}"
        )
    root = Path(plan_root).expanduser().resolve()
    spec = FROZEN_PLAN_SPECS[subject]
    path = root / spec.relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = _sha256(path)
    if actual_hash != spec.sha256:
        raise ValueError(
            f"Frozen plan hash mismatch for {subject}: "
            f"{actual_hash} != {spec.sha256}"
        )
    plan = np.load(path, allow_pickle=False)
    validate_plan(
        plan,
        roi_order=roi_order,
        clip_layer_order=clip_layer_order,
    )
    provenance = {
        "subject": subject,
        "path": str(path),
        "sha256": actual_hash,
        "hash_algorithm": "SHA256 over exact .npy file bytes",
        "roi_order": list(ROI_ORDER),
        "clip_layer_order": [f"L{layer}" for layer in CLIP_LAYER_ORDER],
        "shape": list(plan.shape),
        "method": "sr_fgw",
        "structure_weight_beta": 0.5,
        "lambda_cov": 0.0,
        "entropy": 0.0,
    }
    return torch.from_numpy(plan.copy()), provenance


def row_normalize_transport(plan: Tensor) -> Tensor:
    if plan.ndim != 2 or tuple(plan.shape) != (8, 6):
        raise ValueError(f"transport plan must have shape [8,6], got {tuple(plan.shape)}")
    if not torch.isfinite(plan).all() or torch.any(plan < 0):
        raise ValueError("transport plan must be finite and nonnegative")
    row_mass = plan.sum(dim=-1, keepdim=True)
    if torch.any(row_mass <= 0):
        raise ValueError("transport plan contains an empty ROI row")
    return plan / row_mass


def tempered_attention_bias(
    correspondence: Tensor,
    gamma: float,
    delta: float = DEFAULT_DELTA,
) -> Tensor:
    """Compute log(6*q_gamma), preserving zero bias for uniform/gamma=0."""
    if correspondence.ndim != 2 or correspondence.shape[-1] != 6:
        raise ValueError("correspondence must have shape [R,6]")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if not torch.isfinite(correspondence).all() or torch.any(correspondence < 0):
        raise ValueError("correspondence must be finite and nonnegative")
    if gamma == 0:
        return torch.zeros_like(correspondence)
    powered = (correspondence + delta).pow(float(gamma))
    q_gamma = powered / powered.sum(dim=-1, keepdim=True)
    return torch.log(q_gamma * correspondence.shape[-1])


def load_preregistered_derangement(path: Path | str) -> Tuple[int, ...]:
    payload = json.loads(Path(path).expanduser().read_text())
    permutation = tuple(int(index) for index in payload["row_permutation_indices"])
    if sorted(permutation) != list(range(len(ROI_ORDER))):
        raise ValueError("Preregistered row permutation is not a permutation")
    if any(source == target for source, target in enumerate(permutation)):
        raise ValueError("Preregistered row permutation must be a derangement")
    if tuple(payload.get("roi_order", ())) != ROI_ORDER:
        raise ValueError("Preregistered null ROI order does not match Stage-2")
    return permutation


def apply_row_derangement(plan: Tensor, permutation: Sequence[int]) -> Tensor:
    permutation = tuple(int(index) for index in permutation)
    if sorted(permutation) != list(range(plan.shape[0])):
        raise ValueError("row permutation does not cover all ROI rows")
    if any(index == value for index, value in enumerate(permutation)):
        raise ValueError("row permutation has a fixed point")
    return plan[list(permutation)]


def plan_provenance_json(provenance: Dict[str, object]) -> str:
    """Stable rendering used by run artifacts and tests."""
    return json.dumps(provenance, indent=2, sort_keys=True)
