#!/usr/bin/env python
"""Build local NSD coco73k-ID to COCO-caption mappings."""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.coco73k_caption_utils import (
    expand_paths,
    iter_webdataset_records,
    load_coco_captions,
)


LOCAL_FIELDS = ("nsdId", "nsdid", "nsd_id", "index", "idx", "id")
COCO_FIELDS = ("cocoId", "cocoid", "coco_id", "image_id", "imageId")


def load_metadata_records(path: str):
    metadata_path = Path(path).expanduser().resolve()
    suffix = metadata_path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        import pandas as pd
        frame = pd.read_csv(
            metadata_path, sep="\t" if suffix == ".tsv" else ","
        )
        return frame.to_dict("records")
    if suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(metadata_path).to_dict("records")
    if suffix == ".json":
        payload = json.loads(metadata_path.read_text())
    elif suffix == ".pkl":
        with metadata_path.open("rb") as file:
            payload = pickle.load(file)
    elif suffix == ".npy":
        payload = np.load(metadata_path, allow_pickle=True)
    elif suffix == ".npz":
        with np.load(metadata_path, allow_pickle=True) as archive:
            payload = {key: archive[key] for key in archive.files}
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path}")
    if isinstance(payload, Mapping):
        if all(
            hasattr(value, "__len__") and not isinstance(value, str)
            for value in payload.values()
        ):
            keys = list(payload)
            length = min(len(payload[key]) for key in keys)
            return [
                {key: payload[key][index] for key in keys}
                for index in range(length)
            ]
        return list(payload.values())
    if hasattr(payload, "dtype") and payload.dtype.names:
        return [
            {name: row[name].item() for name in payload.dtype.names}
            for row in payload
        ]
    return list(payload)


def find_field(record, candidates):
    lower = {str(key).lower(): key for key in record}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def metadata_hypotheses(metadata_paths: Sequence[str]):
    hypotheses = []
    for path in metadata_paths:
        records = [
            record for record in load_metadata_records(path)
            if isinstance(record, Mapping)
        ]
        if not records:
            continue
        local_field = find_field(records[0], LOCAL_FIELDS)
        coco_field = find_field(records[0], COCO_FIELDS)
        if coco_field is None:
            continue
        if local_field is not None:
            direct = {}
            one_based = {}
            for record in records:
                try:
                    local_id = int(record[local_field])
                    coco_id = int(record[coco_field])
                except (KeyError, TypeError, ValueError):
                    continue
                direct[local_id] = coco_id
                one_based[local_id + 1] = coco_id
            hypotheses.extend([
                {
                    "name": "B_0_based_metadata_index",
                    "mapping": direct,
                    "metadata": str(Path(path).resolve()),
                    "local_field": str(local_field),
                    "coco_field": str(coco_field),
                },
                {
                    "name": "C_1_based_metadata_index",
                    "mapping": one_based,
                    "metadata": str(Path(path).resolve()),
                    "local_field": str(local_field),
                    "coco_field": str(coco_field),
                },
            ])
        # Row order is a common NSD table convention and is tested separately
        # from named ID columns.
        row_mapping = {}
        for index, record in enumerate(records):
            try:
                row_mapping[index] = int(record[coco_field])
            except (KeyError, TypeError, ValueError):
                continue
        hypotheses.append({
            "name": "D_metadata_row_index",
            "mapping": row_mapping,
            "metadata": str(Path(path).resolve()),
            "local_field": "row_index",
            "coco_field": str(coco_field),
        })
    return hypotheses


def evaluate_hypothesis(local_ids, coco_captions, hypothesis):
    mapping = hypothesis["mapping"]
    matched, missing, output = [], [], {}
    for local_id in sorted(set(local_ids)):
        raw_coco_id = mapping.get(local_id)
        captions = coco_captions.get(raw_coco_id, [])
        if captions:
            output[str(local_id)] = captions
            matched.append({
                "local_id": local_id,
                "raw_coco_id": raw_coco_id,
                "caption": captions[0],
            })
        else:
            missing.append({
                "local_id": local_id,
                "raw_coco_id": raw_coco_id,
            })
    total = len(set(local_ids))
    return {
        **{key: value for key, value in hypothesis.items() if key != "mapping"},
        "total_unique_local_ids": total,
        "matched_ids": len(matched),
        "missing_ids": len(missing),
        "coverage": len(matched) / total if total else 0.0,
        "matched_examples": matched[:10],
        "missing_examples": missing[:20],
        "output_mapping": output,
    }


def build_mapping(local_ids, coco_captions, metadata_paths=()):
    hypotheses = [{
        "name": "A_raw_coco_image_id",
        "mapping": {local_id: local_id for local_id in set(local_ids)},
        "metadata": None,
        "local_field": "coco73k.npy",
        "coco_field": "same_value",
    }]
    hypotheses.extend(metadata_hypotheses(metadata_paths))
    results = [
        evaluate_hypothesis(local_ids, coco_captions, hypothesis)
        for hypothesis in hypotheses
    ]
    best = max(results, key=lambda item: item["coverage"])
    return best["output_mapping"], best, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--webdataset-tar", action="append", required=True
    )
    parser.add_argument(
        "--captions-json", action="append", required=True
    )
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    tar_paths = expand_paths(args.webdataset_tar)
    records = list(iter_webdataset_records(tar_paths, args.max_samples))
    local_ids = [record["coco73k_id"] for record in records]
    captions = load_coco_captions(args.captions_json)
    output_mapping, best, results = build_mapping(
        local_ids, captions, args.metadata
    )
    report = {
        "selected_hypothesis": best["name"],
        "selection_reason": (
            "Highest coverage; ties prefer the explicit NSD nsdId->cocoId "
            "field mapping over 1-based or row-order alternatives."
        ),
        "total_samples_checked": len(records),
        "total_unique_local_ids": best["total_unique_local_ids"],
        "matched_ids": best["matched_ids"],
        "missing_ids": best["missing_ids"],
        "coverage": best["coverage"],
        "metadata_file_used": best["metadata"],
        "local_field": best["local_field"],
        "coco_field": best["coco_field"],
        "coco_caption_files": [
            str(Path(path).expanduser().resolve())
            for path in args.captions_json
        ],
        "matched_examples": best["matched_examples"],
        "missing_examples": best["missing_examples"],
        "all_hypotheses": [
            {key: value for key, value in result.items() if key != "output_mapping"}
            for result in results
        ],
    }
    output_path = Path(args.output_json).expanduser().resolve()
    report_path = Path(args.output_report).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_mapping, indent=2))
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if best["coverage"] < 0.95:
        print(
            "[FAIL] Best mapping coverage is below 95%; diagnostics were saved.",
            file=sys.stderr,
        )
        return 2
    print(f"[PASS] Saved caption mapping to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
