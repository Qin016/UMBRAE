#!/usr/bin/env python
"""Emit deterministic ROI and random-control audit artifacts; no training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dual_branch_cache import sha256_file
from models.random_structure_control import (
    generate_random_structure_control,
    roi_overlap_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(args):
    mapping_path = Path(args.roi_mapping).expanduser().resolve()
    payload = json.loads(mapping_path.read_text())
    order = list(payload["roi_names"])
    real = {name: payload["rois"][name]["indices"] for name in order}
    control = generate_random_structure_control(real, order, args.seed)
    real_overlap = roi_overlap_matrix(real, order)
    control_overlap = roi_overlap_matrix(control, order)
    real_union = {voxel for values in real.values() for voxel in values}
    control_union = {voxel for values in control.values() for voxel in values}
    audit = {
        "subject": payload["subject"],
        "roi_set": payload["roi_set"],
        "roi_mapping_path": str(mapping_path),
        "roi_mapping_sha256": sha256_file(str(mapping_path)),
        "roi_order": order,
        "roi_counts": {name: len(real[name]) for name in order},
        "total_roi_assignments": sum(len(real[name]) for name in order),
        "union_voxel_count": len(real_union),
        "overlapping_assignments": sum(len(real[name]) for name in order) - len(real_union),
        "overlap_matrix": real_overlap.tolist(),
        "random_control": {
            "seed": args.seed,
            "method": "single_seeded_bijection_over_exact_real_roi_union",
            "uses_visual_or_clip_information": False,
            "uses_fmri_response_magnitude": False,
            "same_voxel_universe": control_union == real_union,
            "same_roi_sizes": all(len(control[n]) == len(real[n]) for n in order),
            "same_pairwise_overlap_matrix": bool((control_overlap == real_overlap).all()),
        },
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "roi_audit.json").write_text(json.dumps(audit, indent=2))
    control_payload = {
        "subject": payload["subject"],
        "seed": args.seed,
        "roi_order": order,
        "rois": control,
    }
    (output_dir / f"random_structure_control_seed{args.seed}.json").write_text(
        json.dumps(control_payload, indent=2)
    )
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    run(parse_args())
