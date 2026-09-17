#!/usr/bin/env python
"""Audit Stage-2 IDs and caption coverage without decoding images or fMRI."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tar", nargs="+", required=True)
    parser.add_argument("--val-tar", nargs="+", required=True)
    parser.add_argument("--protected-test-tar", nargs="+", required=True)
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--protected-test-captions-json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def digest_ids(values):
    payload = "".join(f"{value}\n" for value in sorted(set(values))).encode()
    return hashlib.sha256(payload).hexdigest()


def inspect_split(paths, captions):
    sample_ids = []
    local_ids = []
    forbidden_members = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with tarfile.open(path) as archive:
            members = {member.name: member for member in archive.getmembers()}
            prefixes = sorted(
                name[: -len(".nsdgeneral.npy")]
                for name in members
                if name.endswith(".nsdgeneral.npy")
            )
            for prefix in prefixes:
                id_name = f"{prefix}.coco73k.npy"
                if id_name not in members:
                    raise ValueError(f"Missing {id_name} in {path}")
                handle = archive.extractfile(members[id_name])
                local_id = str(int(np.load(io.BytesIO(handle.read())).reshape(-1)[0]))
                sample_ids.append(prefix)
                local_ids.append(local_id)
            forbidden_members.extend(
                name for name in members
                if name.endswith((".jpg", ".jpeg", ".png", ".clip.npy"))
            )
    reference_counts = []
    missing = 0
    duplicate_references = 0
    for local_id in sorted(set(local_ids)):
        references = captions.get(local_id)
        if references is None:
            missing += 1
            continue
        if not isinstance(references, list):
            references = [references]
        normalized = [str(item).strip() for item in references if str(item).strip()]
        reference_counts.append(len(normalized))
        duplicate_references += len(normalized) - len(set(normalized))
    histogram = Counter(reference_counts)
    return {
        "sample_count": len(sample_ids),
        "unique_sample_ids": len(set(sample_ids)),
        "unique_stimuli": len(set(local_ids)),
        "stimulus_id_sha256": digest_ids(local_ids),
        "sample_id_sha256": digest_ids(sample_ids),
        "missing_reference_count": missing,
        "duplicate_reference_count": duplicate_references,
        "reference_count_min": min(reference_counts) if reference_counts else 0,
        "reference_count_max": max(reference_counts) if reference_counts else 0,
        "reference_count_mean": (
            float(np.mean(reference_counts)) if reference_counts else 0.0
        ),
        "reference_count_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
        "image_or_clip_members_present_but_not_loaded": len(forbidden_members),
    }, set(local_ids)


def main():
    args = parse_args()
    captions = json.loads(Path(args.captions_json).expanduser().read_text())
    protected_captions = (
        json.loads(Path(args.protected_test_captions_json).expanduser().read_text())
        if args.protected_test_captions_json
        else captions
    )
    split_paths = {
        "train": args.train_tar,
        "validation": args.val_tar,
        "protected_test": args.protected_test_tar,
    }
    result = {
        "audit_version": "stage2_fgw_data_audit_v1",
        "subject": "subj01",
        "protected_test_performance_accessed": False,
        "protected_test_images_or_fmri_decoded": False,
        "caption_mapping": str(Path(args.captions_json).expanduser().resolve()),
        "splits": {},
    }
    ids = {}
    for split, paths in split_paths.items():
        split_captions = protected_captions if split == "protected_test" else captions
        result["splits"][split], ids[split] = inspect_split(paths, split_captions)
        result["splits"][split]["tar_paths"] = [
            str(Path(path).expanduser().resolve()) for path in paths
        ]
    result["overlap_counts"] = {
        "train_validation": len(ids["train"] & ids["validation"]),
        "train_protected_test": len(ids["train"] & ids["protected_test"]),
        "validation_protected_test": len(ids["validation"] & ids["protected_test"]),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
