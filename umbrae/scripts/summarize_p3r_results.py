#!/usr/bin/env python
"""Create the locked P3-R versus P3 result table without new evaluation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dual_branch_cache import sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--random-dir", required=True)
    parser.add_argument("--p3-comparison", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load(path):
    root = Path(path).resolve()
    return root, json.loads((root / "initial_val_metrics.json").read_text()), json.loads((root / "final_val_metrics.json").read_text()), json.loads((root / "provenance.json").read_text())


def compact(root, initial, final):
    retrieval = final["retrieval"]["brain_to_image"]
    return {
        "best_epoch": final["best_epoch"], "best_val_total_loss": final["best_val_total_loss"],
        "roi_rel_loss": final["roi_rel_loss"], "global_contrastive_loss": final["global_contrastive_loss"],
        "global_spearman_rsa": final["global_rsa"]["spearman_rsa"], "global_pearson_rsa": final["global_rsa"]["pearson_rsa"],
        "mean_roi_spearman_rsa": final["roi_wise_rsa"]["mean_spearman"], "mean_roi_pearson_rsa": final["roi_wise_rsa"]["mean_pearson"],
        "roi_wise_rsa": final["roi_wise_rsa"]["per_roi"],
        "recall_at_1": retrieval["recall_at_1"], "recall_at_5": retrieval["recall_at_5"], "recall_at_10": retrieval["recall_at_10"],
        "median_rank": retrieval["median_rank"], "mean_rank": retrieval["mean_rank"],
        "mean_roi_token_norm": final["mean_roi_token_norm"], "mean_pairwise_roi_cosine": final["mean_pairwise_roi_cosine"],
        "across_sample_variance": final["across_sample_variance"],
        "initial_global_spearman_rsa": initial["global_rsa"]["spearman_rsa"],
        "initial_mean_roi_spearman_rsa": initial["roi_wise_rsa"]["mean_spearman"],
        "best_checkpoint": str(root / "best.pth"), "best_checkpoint_sha256": sha256_file(str(root / "best.pth")),
    }


def run(args):
    real_root, real_initial, real_final, real_prov = load(args.real_dir)
    random_root, random_initial, random_final, random_prov = load(args.random_dir)
    if real_prov["initial_parameter_sha256"] != random_prov["initial_parameter_sha256"]:
        raise RuntimeError("P3-R initialization mismatch")
    p3 = json.loads(Path(args.p3_comparison).read_text())
    result = {
        "protocol_version": "protocol_v1",
        "same_initialization": True,
        "initial_parameter_sha256": real_prov["initial_parameter_sha256"],
        "p3r": {"real": compact(real_root, real_initial, real_final), "random": compact(random_root, random_initial, random_final)},
        "old_p3": {"real": p3["real_roi"], "random": p3["random_structure"]},
        "interpretation": {
            "ROI_REPRESENTATION_LEARNING": "POSITIVE",
            "ROI_SPECIALIZATION_SIGNAL": "POSITIVE",
            "STRUCTURE_SIGNAL": "NEGATIVE",
            "P3R_STATUS": "COARSE_ROI_STRUCTURE_STILL_WEAK"
        }
    }
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())
