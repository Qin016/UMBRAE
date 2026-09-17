#!/usr/bin/env python3
"""Resume Prompt-5B after interruption without refitting the frozen real plans.

The interrupted directory must predate ``final_plan_frozen.marker`` and must
not contain any offline-test-derived files.  The normal one-shot driver is run
from the beginning to reconstruct probes and null plans, but its two calls that
would fit the primary and baseline plans are replaced by verified plans from
the interrupted pretest directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_fgw_offline_test as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume locked one-shot FGW test after a pretest interruption"
    )
    parser.add_argument("--stimulus-manifest", required=True)
    parser.add_argument("--locked-configuration", required=True)
    parser.add_argument("--validation-summary", required=True)
    parser.add_argument("--feature-cost-dir", required=True)
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-device", default="cuda")
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--interrupted-dir", required=True)
    return parser.parse_args()


def load_saved_fit(root: Path, stem: str) -> dict[str, Any]:
    plan = np.load(root / f"{stem}_transport_plan_pretest.npy", allow_pickle=False)
    diagnostics = json.loads((root / f"{stem}_refit_diagnostics.json").read_text())
    convergence = diagnostics["convergence"]
    if convergence.get("converged") is not True:
        raise RuntimeError(f"Interrupted {stem} plan was not converged")
    if not np.isfinite(plan).all() or np.min(plan) < -1e-12:
        raise RuntimeError(f"Interrupted {stem} plan is invalid")
    if plan.shape != (8, 6) or not np.allclose(plan.sum(axis=1), 1 / 8, atol=1e-10):
        raise RuntimeError(f"Interrupted {stem} plan violates source marginals")
    return {
        "plan": plan,
        "components": convergence["training_components"],
        "converged": True,
        "iterations": convergence["iterations"],
        "stopping_reason": convergence["stopping_reason"],
        "source_marginal_feasibility_error": convergence[
            "source_marginal_feasibility_error"
        ],
    }


def main() -> None:
    args = parse_args()
    interrupted = Path(args.interrupted_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if (interrupted / "final_plan_frozen.marker").exists():
        raise RuntimeError("Interrupted directory already crossed the freeze boundary")
    forbidden = [
        interrupted / name
        for name in (
            "offline_test_metrics.json",
            "offline_test_feature_cost.npy",
            "offline_test_brain_geometry.npy",
            "offline_test_clip_geometry.npy",
        )
        if (interrupted / name).exists()
    ]
    if forbidden:
        raise RuntimeError(f"Interrupted directory contains test-derived files: {forbidden}")
    if output.exists():
        raise FileExistsError(output)

    saved_primary = load_saved_fit(interrupted, "final")
    saved_baseline = load_saved_fit(interrupted, "baseline")
    saved_hashes = {
        "primary": base.sha256_file(interrupted / "final_transport_plan_pretest.npy"),
        "baseline": base.sha256_file(interrupted / "baseline_transport_plan_pretest.npy"),
        "feature": base.sha256_file(interrupted / "final_train_feature_cost_oof.npy"),
        "brain_geometry": base.sha256_file(interrupted / "final_train_brain_geometry.npy"),
        "clip_geometry": base.sha256_file(interrupted / "final_train_clip_geometry.npy"),
    }
    original_fit_one_plan = base.fit_one_plan
    call_count = 0

    def reuse_real_plan(
        feature: np.ndarray,
        brain_geometry: np.ndarray,
        clip_geometry: np.ndarray,
        calibration: base.LossCalibration,
        beta: float,
        coverage: float,
        init_kind: str,
        init_seed: int,
        solver: Mapping[str, Any],
        device: str,
    ) -> dict[str, Any]:
        nonlocal call_count
        expected_feature = np.load(
            interrupted / "final_train_feature_cost_oof.npy", allow_pickle=False
        )
        expected_brain = np.load(
            interrupted / "final_train_brain_geometry.npy", allow_pickle=False
        )
        expected_clip = np.load(
            interrupted / "final_train_clip_geometry.npy", allow_pickle=False
        )
        if not (
            np.array_equal(feature, expected_feature)
            and np.array_equal(brain_geometry, expected_brain)
            and np.array_equal(clip_geometry, expected_clip)
        ):
            raise RuntimeError("Reconstructed final-train matrices differ from interrupted run")
        if call_count == 0:
            expected = saved_primary
            if (beta, coverage) != (0.5, 0.0):
                raise RuntimeError("Primary recovery call has unexpected configuration")
        elif call_count == 1:
            expected = saved_baseline
            if (beta, coverage) != (0.0, 1.0):
                raise RuntimeError("Baseline recovery call has unexpected configuration")
        else:
            raise RuntimeError("Unexpected extra real-plan fit call during recovery")
        call_count += 1
        return expected

    base.fit_one_plan = reuse_real_plan
    try:
        result_dir = base.run(args)
    finally:
        base.fit_one_plan = original_fit_one_plan
    if call_count != 2:
        raise RuntimeError(f"Expected two recovered real plans, observed {call_count}")
    recovered_primary_hash = base.sha256_file(
        result_dir / "final_transport_plan_pretest.npy"
    )
    recovered_baseline_hash = base.sha256_file(
        result_dir / "baseline_transport_plan_pretest.npy"
    )
    if recovered_primary_hash != saved_hashes["primary"]:
        raise RuntimeError("Recovered primary plan hash changed")
    if recovered_baseline_hash != saved_hashes["baseline"]:
        raise RuntimeError("Recovered baseline plan hash changed")
    base.write_json(
        result_dir / "interruption_recovery_provenance.json",
        {
            "recovery_reason": "interactive status query interrupted pretest feature-pairing null generation",
            "interrupted_directory": str(interrupted),
            "interrupted_before_final_plan_frozen_marker": True,
            "interrupted_offline_test_accessed": False,
            "primary_plan_refit_during_recovery": False,
            "baseline_plan_refit_during_recovery": False,
            "final_train_matrices_reconstructed_bitwise_identically": True,
            "saved_hashes": saved_hashes,
            "recovered_primary_plan_sha256": recovered_primary_hash,
            "recovered_baseline_plan_sha256": recovered_baseline_hash,
            "normal_driver_sha256": base.sha256_file(Path(base.__file__).resolve()),
            "recovery_driver_sha256": base.sha256_file(Path(__file__).resolve()),
        },
    )


if __name__ == "__main__":
    main()
