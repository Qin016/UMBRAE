#!/usr/bin/env python3
"""Dependency-light checks for the Prompt-4A offline validation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_fgw_correspondence.py"
SPEC = importlib.util.spec_from_file_location("validate_fgw_prompt4a_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_unique_permutations() -> None:
    permutations = MODULE.unique_permutations(200, 8, np.random.default_rng(4))
    tuples = [tuple(item.tolist()) for item in permutations]
    assert len(tuples) == len(set(tuples)) == 200
    assert tuple(range(8)) not in tuples


def test_fixed_validation_score_and_identifiability() -> None:
    feature = np.arange(48, dtype=np.float64).reshape(8, 6) / 48.0 + 0.1
    brain = np.abs(np.subtract.outer(np.arange(8), np.arange(8))).astype(np.float64)
    clip = np.abs(np.subtract.outer(np.arange(6), np.arange(6))).astype(np.float64)
    calibration = MODULE.calibration_from_uniform_plan(feature, brain, clip)
    plan = np.full((8, 6), 1.0 / 48.0)
    metrics = MODULE.validation_metrics(plan, feature, brain, clip, calibration)
    np.testing.assert_allclose(metrics["validation_feature_normalized"], 1.0)
    np.testing.assert_allclose(metrics["validation_gw_normalized"], 1.0)
    np.testing.assert_allclose(metrics["validation_composite"], 1.0)
    plans = np.stack(
        [MODULE.initialization_plan("random", feature, seed) for seed in range(5)]
    )
    result = MODULE.plan_set_identifiability(plans)
    assert result["plan_count"] == 5
    assert result["pair_count"] == 10


def run_all() -> None:
    test_unique_permutations()
    test_fixed_validation_score_and_identifiability()


if __name__ == "__main__":
    run_all()
    print("Prompt-4A validation helper tests passed")
