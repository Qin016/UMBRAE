#!/usr/bin/env python3
"""Dependency-light checks for the Prompt-4B one-shot boundary helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_fgw_offline_test.py"
SPEC = importlib.util.spec_from_file_location("run_fgw_offline_test_unit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_anchor_selection_is_deterministic() -> None:
    first = MODULE.fixed_anchor_positions(100, 40, 42)
    second = MODULE.fixed_anchor_positions(100, 40, 42)
    np.testing.assert_array_equal(first, second)
    assert len(first) == len(set(first.tolist())) == 40


def test_probe_refit_and_prediction() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 5)).astype(np.float32)
    weights = rng.normal(size=(5, 7)).astype(np.float32)
    y = x @ weights + 0.1
    model = MODULE.fit_full_probe(x, y, alpha=0.01, std_epsilon=1e-6, device="cpu")
    prediction = MODULE.predict_full_probe(model, x)
    assert prediction.shape == y.shape
    assert np.mean(np.square(prediction - y)) < 1e-5
    assert len(MODULE.hash_probe_models([model])) == 64


def test_durable_marker() -> None:
    with TemporaryDirectory() as temporary:
        marker = Path(temporary) / "final_plan_frozen.marker"
        MODULE.durable_marker(marker, {"final_plan_sha256": "abc", "frozen": True})
        assert json.loads(marker.read_text())["frozen"] is True
        try:
            MODULE.durable_marker(marker, {"final_plan_sha256": "changed"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Freeze marker must be create-only")


def run_all() -> None:
    test_anchor_selection_is_deterministic()
    test_probe_refit_and_prediction()
    test_durable_marker()


if __name__ == "__main__":
    run_all()
    print("Prompt-4B offline-test boundary tests passed")
