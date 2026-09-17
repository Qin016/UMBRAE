#!/usr/bin/env python3
"""Recover the minimal official subjXX NSD continuous-pRF volume assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


OFFICIAL_BASE = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata"
ASSETS = {
    "angle": {"filename": "prf_angle.nii.gz", "subj01_bytes": 1352843},
    "eccentricity": {"filename": "prf_eccentricity.nii.gz", "subj01_bytes": 1387701},
    "size": {"filename": "prf_size.nii.gz", "subj01_bytes": 1393612},
    "R2": {"filename": "prf_R2.nii.gz", "subj01_bytes": 1384083},
    "exponent": {"filename": "prf_exponent.nii.gz", "subj01_bytes": 1046162},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="subj01")
    parser.add_argument(
        "--recovery-root",
        type=Path,
        default=Path("/opt/data/private/BA/NSD/nsd_prf_recovery"),
    )
    args = parser.parse_args()
    subject_root = args.recovery_root / args.subject
    source = subject_root / "source" / "func1pt8mm"
    metadata = subject_root / "metadata"
    source.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    records = []
    for parameter, spec in ASSETS.items():
        filename = spec["filename"]
        url = f"{OFFICIAL_BASE}/ppdata/{args.subject}/func1pt8mm/{filename}"
        destination = source / filename
        expected = spec.get(f"{args.subject}_bytes")
        already_valid = destination.exists() and (expected is None or destination.stat().st_size == expected)
        if not already_valid:
            temporary = destination.with_suffix(destination.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if expected is not None and temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"Downloaded size mismatch for {url}: {temporary.stat().st_size} != {expected}"
                )
            temporary.replace(destination)
        records.append(
            {
                "parameter": parameter,
                "url": url,
                "provider": "NSD official public S3 bucket",
                "file": filename,
                "bytes": destination.stat().st_size,
                "expected_bytes_from_official_listing": expected,
                "sha256": sha256(destination),
                "download_success": True,
                "reused_verified_local_copy": already_valid,
                "relative_destination": str(destination.relative_to(args.recovery_root)),
                "license_access_notes": "Public official NSD object; no credential required. Dataset terms remain applicable.",
            }
        )
    evidence = {
        "nsddatapaper_analysis_prf.m": "https://raw.githubusercontent.com/cvnlab/nsddatapaper/master/main/analysis_prf.m",
        "nsddatapaper_analysis_prf_maps.m": "https://raw.githubusercontent.com/cvnlab/nsddatapaper/master/main/analysis_prf_maps.m",
        "analyzePRF.m": "https://raw.githubusercontent.com/kendrickkay/analyzePRF/master/analyzePRF.m",
    }
    for filename, url in evidence.items():
        destination = metadata / filename
        if not destination.exists():
            with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    manifest = {
        "subject": args.subject,
        "download_time_utc": datetime.now(timezone.utc).isoformat(),
        "official_base": OFFICIAL_BASE,
        "manual_download_required": False,
        "assets": records,
        "official_code_evidence": [
            {"file": name, "url": url, "sha256": sha256(metadata / name)}
            for name, url in evidence.items()
        ],
    }
    (metadata / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
