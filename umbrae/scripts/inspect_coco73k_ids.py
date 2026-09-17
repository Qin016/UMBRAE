#!/usr/bin/env python
"""Inspect coco73k IDs and metadata stored in UMBRAE WebDataset shards."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.coco73k_caption_utils import expand_paths, iter_webdataset_records


def classify_ids(ids):
    unique = sorted(set(ids))
    if unique == list(range(len(unique))):
        return "0-based contiguous"
    if unique == list(range(1, len(unique) + 1)):
        return "1-based contiguous"
    if unique and min(unique) >= 0 and max(unique) < 73000:
        return "sparse subset of 0-based NSD/COCO73k indices"
    return "possibly raw COCO image IDs or another sparse ID system"


def inspect_ids(tar_values, max_samples=None):
    records = list(iter_webdataset_records(
        expand_paths(tar_values), max_samples=max_samples
    ))
    if not records:
        raise ValueError("No samples containing coco73k.npy were found")
    ids = [record["coco73k_id"] for record in records]
    counts = Counter(ids)
    duplicates = {str(key): value for key, value in counts.items() if value > 1}
    stats = {
        "num_samples_inspected": len(records),
        "min_coco73k_id": min(ids),
        "max_coco73k_id": max(ids),
        "first_20_coco73k_ids": ids[:20],
        "num_unique_coco73k_ids": len(counts),
        "id_system_heuristic": classify_ids(ids),
        "num_duplicate_ids": len(duplicates),
        "max_occurrences_per_id": max(counts.values()),
        "duplicate_id_examples": dict(list(duplicates.items())[:20]),
        "member_suffixes": sorted({
            suffix for record in records
            for suffix in record["member_suffixes"]
        }),
    }
    return records, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", nargs="+", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    records, stats = inspect_ids(args.tar, args.max_samples)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "inspected_ids.jsonl").open("w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    (output_dir / "id_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Saved inspection to {output_dir}")


if __name__ == "__main__":
    main()
