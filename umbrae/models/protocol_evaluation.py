"""Locked pooling, retrieval and RSA definitions for protocol v1."""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def mean_pool_and_normalize(tokens: Tensor) -> tuple[Tensor, Tensor]:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [B,N,D], got {tuple(tokens.shape)}")
    pooled = tokens.mean(dim=1)
    return pooled, F.normalize(pooled, dim=-1)


def _ranks(similarity: np.ndarray) -> np.ndarray:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Protocol retrieval requires a square shared candidate pool")
    if not np.isfinite(similarity).all():
        raise ValueError("similarity contains non-finite values")
    order = np.argsort(-similarity, axis=1, kind="stable")
    return np.argmax(order == np.arange(len(similarity))[:, None], axis=1) + 1


def _rank_metrics(ranks: np.ndarray) -> Dict[str, float]:
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def retrieval_metrics(brain: np.ndarray, visual: np.ndarray) -> Dict[str, object]:
    brain = np.asarray(brain, dtype=np.float64)
    visual = np.asarray(visual, dtype=np.float64)
    if brain.shape != visual.shape or brain.ndim != 2:
        raise ValueError("brain and visual must have the same [N,D] shape")
    brain /= np.maximum(np.linalg.norm(brain, axis=1, keepdims=True), 1e-12)
    visual /= np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-12)
    similarity = brain @ visual.T
    return {
        "primary_direction": "brain_to_image",
        "candidate_count": int(brain.shape[0]),
        "brain_to_image": _rank_metrics(_ranks(similarity)),
        "image_to_brain": _rank_metrics(_ranks(similarity.T)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else float("nan")


def rsa_metrics(brain: np.ndarray, visual: np.ndarray) -> Dict[str, object]:
    brain = np.asarray(brain, dtype=np.float64)
    visual = np.asarray(visual, dtype=np.float64)
    if brain.shape != visual.shape or brain.ndim != 2 or len(brain) < 3:
        raise ValueError("RSA requires matching [N,D] arrays with N >= 3")
    brain /= np.maximum(np.linalg.norm(brain, axis=1, keepdims=True), 1e-12)
    visual /= np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-12)
    upper = np.triu_indices(len(brain), k=1)
    brain_rdm = (1.0 - brain @ brain.T)[upper]
    visual_rdm = (1.0 - visual @ visual.T)[upper]
    return {
        "primary_metric": "spearman_upper_triangle_cosine_rdm",
        "spearman_rsa": _pearson(_average_ranks(brain_rdm), _average_ranks(visual_rdm)),
        "pearson_rsa": _pearson(brain_rdm, visual_rdm),
        "stimulus_count": int(len(brain)),
        "rdm_upper_triangle_count": int(len(brain_rdm)),
    }
