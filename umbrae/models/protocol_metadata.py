"""Locked experiment identity and provenance helpers."""

import json
import subprocess
from pathlib import Path
from typing import Mapping


def experiment_name(
    method: str, subject: str, stage: str, seed: int, protocol_version: str
) -> str:
    method = method.strip().lower().replace("+", "_").replace(" ", "_")
    subject = subject.strip().lower().replace("subj", "s")
    stage = stage.strip().lower()
    protocol = protocol_version.strip().lower().replace("_", "")
    if not all((method, subject, stage, protocol)):
        raise ValueError("Experiment identity fields must be non-empty")
    return f"{method}_{subject}_{stage}_seed{int(seed)}_{protocol}"


def git_commit_or_unavailable(worktree: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def save_experiment_snapshot(
    output_dir: str,
    *,
    config: Mapping[str, object],
    protocol_version: str,
    checkpoint_sha256: str,
    roi_mapping_sha256: str,
    seed: int,
    metrics: Mapping[str, object],
    worktree: str,
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(json.dumps(dict(config), indent=2))
    provenance = {
        "protocol_version": protocol_version,
        "git_commit_hash": git_commit_or_unavailable(worktree),
        "checkpoint_sha256": checkpoint_sha256,
        "roi_mapping_sha256": roi_mapping_sha256,
        "seed": int(seed),
    }
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (target / "metrics.json").write_text(json.dumps(dict(metrics), indent=2))
