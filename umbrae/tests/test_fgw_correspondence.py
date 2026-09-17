#!/usr/bin/env python
"""Unit tests for the offline constrained FGW solver."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.fgw_correspondence import (
    calibration_from_uniform_plan,
    coverage_loss,
    gw_loss,
    initialization_plan,
    objective_components,
    project_rows_to_simplex,
    solve_balanced_outer_ot,
    solve_multiple_initializations,
    solve_projected_simplex,
)


def controlled_problem():
    feature = np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64)
    geometry = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    calibration = calibration_from_uniform_plan(feature, geometry, geometry)
    return feature, geometry, calibration


def test_simplex_projection_and_marginals() -> None:
    raw = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 4.0, 1.0]], dtype=torch.float64)
    projected = project_rows_to_simplex(raw, 1.0 / 2.0).numpy()
    assert np.min(projected) >= 0.0
    np.testing.assert_allclose(projected.sum(axis=1), 0.5, atol=1e-12)
    np.testing.assert_allclose(projected.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(projected.sum(axis=0).sum(), 1.0, atol=1e-12)


def test_balanced_outer_ot_target_marginal() -> None:
    rng = np.random.default_rng(2)
    result = solve_balanced_outer_ot(rng.uniform(size=(8, 6)))
    plan = result["plan"]
    np.testing.assert_allclose(plan.sum(axis=1), 1.0 / 8.0, atol=1e-10)
    np.testing.assert_allclose(plan.sum(axis=0), 1.0 / 6.0, atol=1e-10)
    np.testing.assert_allclose(plan.sum(), 1.0, atol=1e-10)


def test_coverage_uniform_minimum_and_zero_safety() -> None:
    uniform = torch.full((8, 6), 1.0 / 48.0, dtype=torch.float64)
    concentrated = torch.zeros((8, 6), dtype=torch.float64)
    concentrated[:, 0] = 1.0 / 8.0
    uniform_loss = float(coverage_loss(uniform))
    concentrated_loss = float(coverage_loss(concentrated))
    assert np.isfinite(uniform_loss) and np.isfinite(concentrated_loss)
    assert uniform_loss < concentrated_loss


def test_feature_routing_recovers_obvious_assignment() -> None:
    feature, geometry, calibration = controlled_problem()
    result = solve_projected_simplex(
        feature,
        geometry,
        geometry,
        calibration,
        structure_weight=0.0,
        coverage_weight=0.0,
        initial_plan=initialization_plan("uniform", feature),
        learning_rate=0.2,
        max_iterations=500,
        tolerance=1e-12,
        patience=20,
    )
    assert result["converged"]
    np.testing.assert_allclose(result["plan"], np.diag([0.5, 0.5]), atol=1e-8)


def test_matched_structures_lower_gw() -> None:
    geometry = np.asarray(
        [[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]], dtype=np.float64
    )
    identity = torch.eye(3, dtype=torch.float64) / 3.0
    uniform = torch.full((3, 3), 1.0 / 9.0, dtype=torch.float64)
    matrix = torch.as_tensor(geometry, dtype=torch.float64)
    assert float(gw_loss(identity, matrix, matrix)) < float(gw_loss(uniform, matrix, matrix))


def test_direct_solver_controlled_and_multiple_initializations() -> None:
    feature, geometry, calibration = controlled_problem()
    results = solve_multiple_initializations(
        feature,
        geometry,
        geometry,
        calibration,
        structure_weight=0.5,
        coverage_weight=0.0,
        initialization_specs=[("uniform", 0), ("random", 1), ("random", 2), ("feature_informed", 0)],
        learning_rate=0.1,
        max_iterations=600,
        tolerance=1e-11,
        patience=30,
    )
    objectives = [item["components"]["total_objective"] for item in results["all_results"]]
    assert results["winner"]["components"]["total_objective"] == min(objectives)
    assert results["selection_metric"] == "lowest offline_discovery total objective for fixed config"
    assert results["winner"]["source_marginal_feasibility_error"] < 1e-10


def test_normalized_scales_deterministic() -> None:
    feature, geometry, first = controlled_problem()
    second = calibration_from_uniform_plan(feature.copy(), geometry.copy(), geometry.copy())
    assert first == second
    plan = torch.full((2, 2), 0.25, dtype=torch.float64)
    components = objective_components(
        plan,
        torch.as_tensor(feature),
        torch.as_tensor(geometry),
        torch.as_tensor(geometry),
        first,
        structure_weight=0.5,
    )
    np.testing.assert_allclose(float(components["feature_normalized"]), 1.0, atol=1e-12)
    np.testing.assert_allclose(float(components["gw_normalized"]), 1.0, atol=1e-12)


def test_pot_reference_when_available() -> None:
    try:
        import ot  # noqa: F401
    except ImportError:
        print("POT reference test skipped: package 'ot' is unavailable")
        return
    # POT API/version-specific reference tests belong here when POT is installed.
    assert hasattr(ot, "gromov")


def run_all() -> None:
    test_simplex_projection_and_marginals()
    test_balanced_outer_ot_target_marginal()
    test_coverage_uniform_minimum_and_zero_safety()
    test_feature_routing_recovers_obvious_assignment()
    test_matched_structures_lower_gw()
    test_direct_solver_controlled_and_multiple_initializations()
    test_normalized_scales_deterministic()
    test_pot_reference_when_available()


if __name__ == "__main__":
    run_all()
    print("FGW correspondence tests passed")
