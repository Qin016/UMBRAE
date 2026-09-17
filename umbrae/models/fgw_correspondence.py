"""Offline semi-relaxed FGW objectives and constrained solvers.

This module has no dependency on UMBRAE, Shikra, or NeuroRoute training code.
Rows are source ROIs with fixed mass; columns are CLIP layers with a free target
marginal unless the balanced outer-OT baseline is requested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.stats import spearmanr


@dataclass(frozen=True)
class LossCalibration:
    feature_scale: float
    gw_scale: float
    eps_scale: float = 1e-12


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """Numerically safe natural-log JS divergence for probability vectors."""
    p = np.clip(np.asarray(left, dtype=np.float64), 0.0, None)
    q = np.clip(np.asarray(right, dtype=np.float64), 0.0, None)
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0.0 or q_sum <= 0.0:
        raise ValueError("JS inputs must have positive total mass")
    p /= p_sum
    q /= q_sum
    midpoint = 0.5 * (p + q)
    p_term = np.zeros_like(p)
    q_term = np.zeros_like(q)
    p_mask = p > 0.0
    q_mask = q > 0.0
    p_term[p_mask] = p[p_mask] * np.log(p[p_mask] / midpoint[p_mask])
    q_term[q_mask] = q[q_mask] * np.log(q[q_mask] / midpoint[q_mask])
    # Round-off can make the exact-zero case slightly negative.
    return max(0.0, float(0.5 * (p_term.sum() + q_term.sum())))


def validate_problem(
    feature_cost: np.ndarray, brain_geometry: np.ndarray, clip_geometry: np.ndarray
) -> Tuple[int, int]:
    feature_cost = np.asarray(feature_cost)
    brain_geometry = np.asarray(brain_geometry)
    clip_geometry = np.asarray(clip_geometry)
    if feature_cost.ndim != 2:
        raise ValueError("feature_cost must be [R,L]")
    roi_count, layer_count = feature_cost.shape
    if brain_geometry.shape != (roi_count, roi_count):
        raise ValueError("brain_geometry shape does not match feature_cost")
    if clip_geometry.shape != (layer_count, layer_count):
        raise ValueError("clip_geometry shape does not match feature_cost")
    for name, matrix in (
        ("feature_cost", feature_cost),
        ("brain_geometry", brain_geometry),
        ("clip_geometry", clip_geometry),
    ):
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if not np.allclose(brain_geometry, brain_geometry.T, atol=1e-10):
        raise ValueError("brain_geometry must be symmetric")
    if not np.allclose(clip_geometry, clip_geometry.T, atol=1e-10):
        raise ValueError("clip_geometry must be symmetric")
    if not np.allclose(np.diag(brain_geometry), 0.0, atol=1e-10):
        raise ValueError("brain_geometry diagonal must be zero")
    if not np.allclose(np.diag(clip_geometry), 0.0, atol=1e-10):
        raise ValueError("clip_geometry diagonal must be zero")
    return roi_count, layer_count


def project_vector_to_simplex(vector: torch.Tensor, mass: float) -> torch.Tensor:
    """Euclidean projection onto {x >= 0, sum(x) = mass}."""
    if mass <= 0:
        raise ValueError("simplex mass must be positive")
    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    sorted_values, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - mass
    positions = torch.arange(
        1, vector.numel() + 1, dtype=vector.dtype, device=vector.device
    )
    support = sorted_values - cumulative / positions > 0
    rho = torch.nonzero(support, as_tuple=False)[-1, 0]
    threshold = cumulative[rho] / positions[rho]
    return torch.clamp(vector - threshold, min=0.0)


def project_rows_to_simplex(plan: torch.Tensor, row_mass: float) -> torch.Tensor:
    if plan.ndim != 2:
        raise ValueError("plan must be a matrix")
    return torch.stack(
        [project_vector_to_simplex(row, row_mass) for row in plan], dim=0
    )


def feature_loss(plan: torch.Tensor, feature_cost: torch.Tensor) -> torch.Tensor:
    return torch.sum(plan * feature_cost)


def gw_loss(
    plan: torch.Tensor, brain_geometry: torch.Tensor, clip_geometry: torch.Tensor
) -> torch.Tensor:
    squared_difference = (
        brain_geometry[:, :, None, None] - clip_geometry[None, None, :, :]
    ).square()
    return torch.einsum("ijkl,ik,jl->", squared_difference, plan, plan)


def coverage_loss(
    plan: torch.Tensor, eps_cov: float = 1e-12
) -> torch.Tensor:
    target = plan.sum(dim=0)
    uniform = torch.full_like(target, 1.0 / target.numel())
    return torch.sum(target * torch.log((target + eps_cov) / uniform))


def entropy_loss(plan: torch.Tensor, eps_entropy: float = 1e-12) -> torch.Tensor:
    """Negative Shannon entropy term, sum T log(T), used only if weighted."""
    return torch.sum(plan * torch.log(plan + eps_entropy))


def objective_components(
    plan: torch.Tensor,
    feature_cost_matrix: torch.Tensor,
    brain_geometry: torch.Tensor,
    clip_geometry: torch.Tensor,
    calibration: LossCalibration,
    structure_weight: float,
    coverage_weight: float = 0.0,
    entropy_weight: float = 0.0,
    eps_cov: float = 1e-12,
    eps_entropy: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    if not 0.0 <= structure_weight <= 1.0:
        raise ValueError("structure_weight must be in [0,1]")
    raw_feature = feature_loss(plan, feature_cost_matrix)
    raw_gw = gw_loss(plan, brain_geometry, clip_geometry)
    normalized_feature = raw_feature / (calibration.feature_scale + calibration.eps_scale)
    normalized_gw = raw_gw / (calibration.gw_scale + calibration.eps_scale)
    cov = coverage_loss(plan, eps_cov)
    negative_entropy = entropy_loss(plan, eps_entropy)
    total = (
        (1.0 - structure_weight) * normalized_feature
        + structure_weight * normalized_gw
        + coverage_weight * cov
        + entropy_weight * negative_entropy
    )
    return {
        "total_objective": total,
        "feature_raw": raw_feature,
        "feature_normalized": normalized_feature,
        "gw_raw": raw_gw,
        "gw_normalized": normalized_gw,
        "coverage_loss": cov,
        "entropy_loss": negative_entropy,
        "shannon_plan_entropy": -negative_entropy,
    }


def calibration_from_uniform_plan(
    feature_cost_matrix: np.ndarray,
    brain_geometry: np.ndarray,
    clip_geometry: np.ndarray,
    eps_scale: float = 1e-12,
) -> LossCalibration:
    roi_count, layer_count = validate_problem(
        feature_cost_matrix, brain_geometry, clip_geometry
    )
    dtype = torch.float64
    plan = torch.full((roi_count, layer_count), 1.0 / (roi_count * layer_count), dtype=dtype)
    feature = torch.as_tensor(feature_cost_matrix, dtype=dtype)
    brain = torch.as_tensor(brain_geometry, dtype=dtype)
    clip = torch.as_tensor(clip_geometry, dtype=dtype)
    return LossCalibration(
        feature_scale=float(feature_loss(plan, feature)),
        gw_scale=float(gw_loss(plan, brain, clip)),
        eps_scale=eps_scale,
    )


def initialization_plan(
    kind: str,
    feature_cost_matrix: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    roi_count, layer_count = feature_cost_matrix.shape
    row_mass = 1.0 / roi_count
    if kind == "uniform":
        return np.full((roi_count, layer_count), row_mass / layer_count, dtype=np.float64)
    if kind == "random":
        rng = np.random.default_rng(seed)
        return rng.dirichlet(np.ones(layer_count), size=roi_count) * row_mass
    if kind == "feature_informed":
        plan = np.zeros((roi_count, layer_count), dtype=np.float64)
        plan[np.arange(roi_count), np.argmin(feature_cost_matrix, axis=1)] = row_mass
        return plan
    raise ValueError(f"Unsupported initialization kind: {kind}")


def _float_components(components: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in components.items()}


def solve_projected_simplex(
    feature_cost_matrix: np.ndarray,
    brain_geometry_matrix: np.ndarray,
    clip_geometry_matrix: np.ndarray,
    calibration: LossCalibration,
    structure_weight: float,
    coverage_weight: float,
    initial_plan: np.ndarray,
    entropy_weight: float = 0.0,
    learning_rate: float = 0.1,
    max_iterations: int = 3000,
    tolerance: float = 1e-11,
    patience: int = 80,
    eps_cov: float = 1e-12,
    eps_entropy: float = 1e-12,
    device: str = "cpu",
) -> Dict[str, Any]:
    roi_count, layer_count = validate_problem(
        feature_cost_matrix, brain_geometry_matrix, clip_geometry_matrix
    )
    row_mass = 1.0 / roi_count
    dtype = torch.float64
    feature = torch.as_tensor(feature_cost_matrix, dtype=dtype, device=device)
    brain = torch.as_tensor(brain_geometry_matrix, dtype=dtype, device=device)
    clip = torch.as_tensor(clip_geometry_matrix, dtype=dtype, device=device)
    plan = project_rows_to_simplex(
        torch.as_tensor(initial_plan, dtype=dtype, device=device), row_mass
    ).detach()
    trace: List[Dict[str, float]] = []
    small_improvement_count = 0
    step_size = float(learning_rate)
    converged = False
    previous_objective: Optional[float] = None

    for iteration in range(max_iterations + 1):
        plan = plan.detach().requires_grad_(True)
        components = objective_components(
            plan,
            feature,
            brain,
            clip,
            calibration,
            structure_weight,
            coverage_weight,
            entropy_weight,
            eps_cov,
            eps_entropy,
        )
        objective = components["total_objective"]
        gradient = torch.autograd.grad(objective, plan)[0]
        current_value = float(objective.detach().cpu())
        gradient_norm = float(torch.linalg.vector_norm(gradient).detach().cpu())
        if iteration == max_iterations:
            trace.append(
                {
                    "iteration": iteration,
                    **_float_components(components),
                    "gradient_norm": gradient_norm,
                    "step_norm": 0.0,
                    "source_marginal_feasibility_error": float(
                        torch.max(torch.abs(plan.sum(dim=1) - row_mass)).detach().cpu()
                    ),
                    "objective_improvement": 0.0,
                    "learning_rate": step_size,
                }
            )
            break

        trial_step = step_size
        accepted_plan = None
        accepted_components = None
        for _ in range(30):
            candidate = project_rows_to_simplex(
                plan.detach() - trial_step * gradient.detach(), row_mass
            )
            candidate_components = objective_components(
                candidate,
                feature,
                brain,
                clip,
                calibration,
                structure_weight,
                coverage_weight,
                entropy_weight,
                eps_cov,
                eps_entropy,
            )
            if float(candidate_components["total_objective"].detach().cpu()) <= current_value + 1e-14:
                accepted_plan = candidate
                accepted_components = candidate_components
                break
            trial_step *= 0.5
        if accepted_plan is None or accepted_components is None:
            accepted_plan = plan.detach()
            accepted_components = components
            trial_step = 0.0
        accepted_value = float(accepted_components["total_objective"].detach().cpu())
        improvement = current_value - accepted_value
        step_norm = float(torch.linalg.vector_norm(accepted_plan - plan.detach()).cpu())
        feasibility = float(
            torch.max(torch.abs(accepted_plan.sum(dim=1) - row_mass)).cpu()
        )
        trace.append(
            {
                "iteration": iteration,
                **_float_components(accepted_components),
                "gradient_norm": gradient_norm,
                "step_norm": step_norm,
                "source_marginal_feasibility_error": feasibility,
                "objective_improvement": improvement,
                "learning_rate": trial_step,
            }
        )
        if improvement < tolerance:
            small_improvement_count += 1
        else:
            small_improvement_count = 0
        plan = accepted_plan.detach()
        previous_objective = accepted_value
        step_size = min(float(learning_rate), max(trial_step * 1.05, 1e-12))
        if small_improvement_count >= patience:
            converged = True
            break

    final_plan = plan.detach().cpu().numpy()
    final_torch = torch.as_tensor(final_plan, dtype=dtype, device=device)
    final_components = _float_components(
        objective_components(
            final_torch,
            feature,
            brain,
            clip,
            calibration,
            structure_weight,
            coverage_weight,
            entropy_weight,
            eps_cov,
            eps_entropy,
        )
    )
    return {
        "plan": final_plan,
        "components": final_components,
        "trace": trace,
        "converged": converged,
        "iterations": int(trace[-1]["iteration"]) if trace else 0,
        "stopping_reason": "objective_patience" if converged else "maximum_iterations",
        "source_marginal_feasibility_error": float(
            np.max(np.abs(final_plan.sum(axis=1) - row_mass))
        ),
    }


def solve_multiple_initializations(
    feature_cost_matrix: np.ndarray,
    brain_geometry_matrix: np.ndarray,
    clip_geometry_matrix: np.ndarray,
    calibration: LossCalibration,
    structure_weight: float,
    coverage_weight: float,
    initialization_specs: Sequence[Tuple[str, int]],
    **solver_kwargs: Any,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for kind, seed in initialization_specs:
        initial = initialization_plan(kind, feature_cost_matrix, seed)
        result = solve_projected_simplex(
            feature_cost_matrix,
            brain_geometry_matrix,
            clip_geometry_matrix,
            calibration,
            structure_weight,
            coverage_weight,
            initial,
            **solver_kwargs,
        )
        result["initialization_kind"] = kind
        result["initialization_seed"] = seed
        results.append(result)
    objectives = np.asarray(
        [result["components"]["total_objective"] for result in results]
    )
    winner = int(np.argmin(objectives))
    return {
        "winner_index": winner,
        "winner": results[winner],
        "all_results": results,
        "selection_metric": "lowest offline_discovery total objective for fixed config",
        "objective_range": [float(objectives.min()), float(objectives.max())],
    }


def solve_balanced_outer_ot(feature_cost_matrix: np.ndarray) -> Dict[str, Any]:
    feature_cost_matrix = np.asarray(feature_cost_matrix, dtype=np.float64)
    roi_count, layer_count = feature_cost_matrix.shape
    source = np.full(roi_count, 1.0 / roi_count)
    target = np.full(layer_count, 1.0 / layer_count)
    variable_count = roi_count * layer_count
    constraints = []
    values = []
    for roi in range(roi_count):
        row = np.zeros(variable_count)
        row[roi * layer_count : (roi + 1) * layer_count] = 1.0
        constraints.append(row)
        values.append(source[roi])
    for layer in range(layer_count):
        row = np.zeros(variable_count)
        row[layer::layer_count] = 1.0
        constraints.append(row)
        values.append(target[layer])
    result = linprog(
        feature_cost_matrix.reshape(-1),
        A_eq=np.asarray(constraints),
        b_eq=np.asarray(values),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Balanced outer OT linear program failed: {result.message}")
    plan = result.x.reshape(roi_count, layer_count)
    return {
        "plan": plan,
        "raw_feature_objective": float(np.sum(plan * feature_cost_matrix)),
        "solver": "scipy.optimize.linprog(method='highs')",
        "converged": True,
        "iterations": int(result.nit),
        "source_marginal_feasibility_error": float(np.max(np.abs(plan.sum(axis=1) - source))),
        "target_marginal_feasibility_error": float(np.max(np.abs(plan.sum(axis=0) - target))),
    }


def _safe_row_correlation(rows: np.ndarray) -> np.ndarray:
    count = rows.shape[0]
    result = np.full((count, count), np.nan, dtype=np.float64)
    for left in range(count):
        for right in range(count):
            if np.std(rows[left]) > 0 and np.std(rows[right]) > 0:
                result[left, right] = float(np.corrcoef(rows[left], rows[right])[0, 1])
    return result


def _spearman_diagnostic(left: Sequence[float], right: Sequence[float]) -> Dict[str, float]:
    result = spearmanr(np.asarray(left), np.asarray(right))
    return {"spearman": float(result.statistic), "two_sided_p": float(result.pvalue)}


def coupling_diagnostics(
    plan: np.ndarray,
    roi_names: Sequence[str],
    layer_labels: Sequence[str],
    layer_depths: Sequence[float],
    roi_feature_variance: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    plan = np.asarray(plan, dtype=np.float64)
    roi_count, layer_count = plan.shape
    row_mass = 1.0 / roi_count
    if np.min(plan) < -1e-10:
        raise ValueError("Coupling contains negative mass")
    if np.max(np.abs(plan.sum(axis=1) - row_mass)) > 1e-8:
        raise ValueError("Coupling violates source marginal")
    target = plan.sum(axis=0)
    normalized_rows = plan / row_mass
    target_entropy = float(-np.sum(target * np.log(target + 1e-12)))
    row_entropies = -np.sum(normalized_rows * np.log(normalized_rows + 1e-12), axis=1)
    row_max = normalized_rows.max(axis=1)
    preferred = normalized_rows.argmax(axis=1)
    depths = np.asarray(layer_depths, dtype=np.float64)
    expected_depth = normalized_rows @ depths
    js_matrix = np.zeros((roi_count, roi_count), dtype=np.float64)
    for left in range(roi_count):
        for right in range(roi_count):
            js_matrix[left, right] = jensen_shannon_divergence(
                normalized_rows[left], normalized_rows[right]
            )
    pair_values = js_matrix[np.triu_indices(roi_count, k=1)]
    diagnostics: Dict[str, Any] = {
        "target_utilization": {
            "target_marginal": {label: float(value) for label, value in zip(layer_labels, target)},
            "target_entropy": target_entropy,
            "effective_target_layer_count": float(np.exp(target_entropy)),
            "maximum_target_mass": float(target.max()),
            "target_mass_sum": float(target.sum()),
        },
        "roi_rows": {
            roi_name: {
                "row_distribution": {
                    label: float(value) for label, value in zip(layer_labels, normalized_rows[index])
                },
                "row_entropy": float(row_entropies[index]),
                "row_max_mass": float(row_max[index]),
                "preferred_layer_descriptive": layer_labels[int(preferred[index])],
                "expected_clip_depth_under_coupling": float(expected_depth[index]),
            }
            for index, roi_name in enumerate(roi_names)
        },
        "mean_row_entropy": float(row_entropies.mean()),
        "pairwise_jensen_shannon_divergence": js_matrix,
        "mean_pairwise_jensen_shannon_divergence": float(pair_values.mean()),
        "row_correlation_matrix": _safe_row_correlation(normalized_rows),
        "specialization_caution": (
            "High pairwise JS alone is not evidence of meaningful or stable ROI specialization."
        ),
        "depth_caution": "Expected CLIP depth under coupling is descriptive, not biological hierarchy.",
    }
    if roi_feature_variance is not None:
        variance = np.asarray(roi_feature_variance, dtype=np.float64)
        diagnostics["feature_variance_confound_diagnostics"] = {
            "registered_prompt2r_rho": 0.7760,
            "registered_prompt2r_p": 0.0236,
            "feature_variance_vs_row_entropy": _spearman_diagnostic(variance, row_entropies),
            "feature_variance_vs_row_max_mass": _spearman_diagnostic(variance, row_max),
            "feature_variance_vs_expected_clip_depth": _spearman_diagnostic(
                variance, expected_depth
            ),
            "used_for_candidate_selection": False,
        }
    return diagnostics
