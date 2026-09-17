#!/usr/bin/env python
"""Validate local coco73k caption coverage by split and subject."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.coco73k_caption_utils import (
    expand_paths,
    infer_dataset_label,
    iter_webdataset_records,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", action="append", nargs="+", required=True)
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--allow-low-coverage", action="store_true")
    # Backward-compatible threshold alias.
    parser.add_argument("--max-missing-ratio", type=float)
    return parser.parse_args()


def load_caption_mapping(path: str):
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("Final caption mapping must be a JSON object")
    return {str(key): value for key, value in payload.items()}


def summarize_records(records, captions):
    ids = [str(record["coco73k_id"]) for record in records]
    matched_records = [
        record for record in records if captions.get(str(record["coco73k_id"]))
    ]
    missing_records = [
        record for record in records if not captions.get(str(record["coco73k_id"]))
    ]
    unique_ids = set(ids)
    matched_unique = {
        str(record["coco73k_id"]) for record in matched_records
    }
    missing_unique = sorted(
        unique_ids - matched_unique, key=lambda value: int(value)
    )
    examples = []
    for record in matched_records[:10]:
        local_id = str(record["coco73k_id"])
        value = captions[local_id]
        caption = value[0] if isinstance(value, list) else value
        examples.append({
            "local_id": local_id,
            "sample_key": record["sample_key"],
            "caption": str(caption),
        })
    total = len(records)
    return {
        "total_samples_checked": total,
        "unique_coco73k_ids": len(unique_ids),
        "matched_samples": len(matched_records),
        "matched_unique_ids": len(matched_unique),
        "missing_samples": len(missing_records),
        "missing_unique_ids": len(missing_unique),
        "coverage": len(matched_records) / total if total else 0.0,
        "missing_id_examples": missing_unique[:20],
        "matched_caption_examples": examples,
        # Legacy field names retained for existing callers/tests.
        "total_checked": total,
        "matched": len(matched_records),
        "missing": len(missing_records),
        "missing_ratio": (
            len(missing_records) / total if total else 0.0
        ),
    }


def check_caption_coverage(
    tar_paths: Sequence[str],
    captions: Mapping[str, object],
    max_samples: Optional[int] = None,
):
    records = list(iter_webdataset_records(
        [Path(path).expanduser().resolve() for path in tar_paths],
        max_samples,
    ))
    if not records:
        raise ValueError("No coco73k samples found")
    return summarize_records(records, captions)


def main():
    args = parse_args()
    min_coverage = (
        1.0 - args.max_missing_ratio
        if args.max_missing_ratio is not None
        else args.min_coverage
    )
    if not 0 <= min_coverage <= 1:
        raise ValueError("Coverage threshold must be within [0,1]")
    tar_values = [value for group in args.tar for value in group]
    tar_paths = expand_paths(tar_values)
    captions = load_caption_mapping(args.captions_json)
    records = list(iter_webdataset_records(tar_paths, args.max_samples))
    grouped = defaultdict(list)
    for record in records:
        split, subject = infer_dataset_label(Path(record["tar_path"]))
        grouped[f"{split}:{subject}"].append(record)
    reports = {"overall": summarize_records(records, captions)}
    reports.update({
        key: summarize_records(value, captions)
        for key, value in sorted(grouped.items())
    })
    for label, report in reports.items():
        print(f"\n[{label}]")
        print(json.dumps(report, indent=2))
    overall = reports["overall"]
    # Concise legacy summary retained for existing automation.
    print(f"total checked: {overall['total_checked']}")
    print(f"matched: {overall['matched_samples']}")
    print(f"missing: {overall['missing_samples']}")
    print(f"missing ratio: {overall['missing_ratio']:.6f}")
    failed = [
        label for label, report in reports.items()
        if report["coverage"] < min_coverage
    ]
    if failed and not args.allow_low_coverage:
        print(
            f"[FAIL] Missing ratio exceeds threshold; coverage below "
            f"{min_coverage:.4f}: {failed}",
            file=sys.stderr,
        )
        return 2
    if failed:
        print(f"[WARN] Low coverage allowed for diagnostics: {failed}")
    else:
        print(f"[PASS] All groups meet coverage >= {min_coverage:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
