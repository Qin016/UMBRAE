"""Pure, deterministic statistical helpers for P8 diagnosis."""

import math
from typing import Mapping

import numpy as np


def interpolate(base: np.ndarray, variant: np.ndarray, alpha: float) -> np.ndarray:
    base, variant = np.asarray(base), np.asarray(variant)
    if base.shape != variant.shape or not (0.0 <= alpha <= 1.0):
        raise ValueError("matching shapes and alpha in [0,1] are required")
    if alpha == 0.0:
        return base.copy()
    if alpha == 1.0:
        return variant.copy()
    result = base.astype(np.float32) + alpha * (variant.astype(np.float32) - base.astype(np.float32))
    if not np.isfinite(result).all():
        raise ValueError("interpolation produced NaN/Inf")
    return result


def paired_summary(deltas: np.ndarray) -> Mapping[str, float]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("bootstrap deltas must be a finite vector")
    return {
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "standard_deviation": float(values.std(ddof=1)),
        "percentile_2_5": float(np.percentile(values, 2.5)),
        "percentile_97_5": float(np.percentile(values, 97.5)),
        "confidence_interval_95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "probability_delta_gt_0": float(np.mean(values > 0)),
        "probability_delta_lt_0": float(np.mean(values < 0)),
    }


def box_iou_vector(predictions, targets) -> np.ndarray:
    pred = np.asarray([[0, 0, 0, 0] if value is None else value for value in predictions], dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("boxes must have matching [N,4] shapes")
    left = np.maximum(pred[:, :2], target[:, :2])
    right = np.minimum(pred[:, 2:], target[:, 2:])
    intersection = np.maximum(right - left, 0).prod(axis=1)
    pred_area = np.maximum(pred[:, 2:] - pred[:, :2], 0).prod(axis=1)
    target_area = np.maximum(target[:, 2:] - target[:, :2], 0).prod(axis=1)
    union = pred_area + target_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def bleu4_from_components(testlen, reflen, guesses, correct) -> float:
    tiny, small = 1e-15, 1e-9
    bleu = 1.0
    for index in range(4):
        bleu *= (float(correct[index]) + tiny) / (float(guesses[index]) + small)
    bleu = bleu ** 0.25
    ratio = (float(testlen) + tiny) / (float(reflen) + small)
    if ratio < 1:
        bleu *= math.exp(1.0 - 1.0 / ratio)
    return float(bleu)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64); right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("CKA inputs must match [N,D]")
    left -= left.mean(0, keepdims=True); right -= right.mean(0, keepdims=True)
    # Dual form avoids materializing a D x D cross-covariance for the
    # projector's 4096-dimensional output.
    left_gram, right_gram = left @ left.T, right @ right.T
    cross = float(np.sum(left_gram * right_gram))
    denom = float(np.sqrt(np.sum(left_gram ** 2) * np.sum(right_gram ** 2)))
    return float(cross / denom) if denom > 0 else float("nan")
