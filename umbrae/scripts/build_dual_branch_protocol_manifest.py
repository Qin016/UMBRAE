#!/usr/bin/env python
"""Build split manifests from actual WebDataset tar metadata and image bytes."""

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def scalar_npy(archive: tarfile.TarFile, member: tarfile.TarInfo) -> int:
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member.name)
    value = np.load(io.BytesIO(handle.read()), allow_pickle=False)
    if value.size != 1:
        raise ValueError(f"Expected scalar metadata in {member.name}, got {value.shape}")
    return int(value.reshape(-1)[0])


def subject_from_shard(path: Path) -> str:
    match = re.search(r"subj(\d{2})", path.name)
    if match is None:
        raise ValueError(f"Cannot determine subject from shard {path}")
    return f"subj{match.group(1)}"


def records_from_shard(path: Path, split: str, expected_subject: str) -> List[Dict[str, object]]:
    actual_subject = subject_from_shard(path)
    if actual_subject != expected_subject:
        raise ValueError(
            f"Shard subject mismatch: {path.name} contains {actual_subject}, "
            f"expected {expected_subject}"
        )
    with tarfile.open(path) as archive:
        grouped: Dict[str, Dict[str, tarfile.TarInfo]] = {}
        for member in archive.getmembers():
            if not member.isfile() or "." not in member.name:
                continue
            prefix, suffix = member.name.split(".", 1)
            grouped.setdefault(prefix, {})[suffix] = member
        records = []
        for sample_id in sorted(grouped):
            members = grouped[sample_id]
            required = {"nsdgeneral.npy", "trial.npy", "coco73k.npy", "num_uniques.npy"}
            missing = sorted(required - set(members))
            if missing:
                raise ValueError(f"{path.name}:{sample_id} missing metadata {missing}")
            image_suffixes = [suffix for suffix in ("jpg", "png") if suffix in members]
            if len(image_suffixes) != 1:
                raise ValueError(
                    f"{path.name}:{sample_id} requires exactly one image, got {image_suffixes}"
                )
            image_member = members[image_suffixes[0]]
            image_handle = archive.extractfile(image_member)
            if image_handle is None:
                raise FileNotFoundError(image_member.name)
            image_bytes = image_handle.read()
            records.append(
                {
                    "sample_id": sample_id,
                    "subject": actual_subject,
                    "split": split,
                    "shard": str(path.resolve()),
                    "trial_id": scalar_npy(archive, members["trial.npy"]),
                    "stimulus_id": scalar_npy(archive, members["coco73k.npy"]),
                    "coco73k_id": scalar_npy(archive, members["coco73k.npy"]),
                    "image_id": image_member.name,
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "num_fmri_repeats": scalar_npy(archive, members["num_uniques.npy"]),
                }
            )
    return records


def duplicates(values: Iterable[object]) -> List[object]:
    seen, repeated = set(), set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)


def overlap(left: List[Mapping[str, object]], right: List[Mapping[str, object]], key: str):
    return sorted({item[key] for item in left} & {item[key] for item in right})


def build_manifests(data_root: Path, subject: str, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: Dict[str, List[Dict[str, object]]] = {}
    shard_manifest = {}
    for split in ("train", "val", "test"):
        shards = sorted((data_root / split).glob(f"{split}_{subject}_*.tar"), key=natural_key)
        if not shards:
            raise FileNotFoundError(f"No {split} shards found for {subject}")
        shard_manifest[split] = [str(path.resolve()) for path in shards]
        records = []
        for shard in shards:
            records.extend(records_from_shard(shard, split, subject))
        manifests[split] = records
        with (output_dir / f"{split}_manifest.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    pair_audit = {}
    for left, right in pairs:
        name = f"{left}_vs_{right}"
        sample_overlap = overlap(manifests[left], manifests[right], "sample_id")
        image_overlap = overlap(manifests[left], manifests[right], "image_sha256")
        coco_overlap = overlap(manifests[left], manifests[right], "coco73k_id")
        pair_audit[name] = {
            "sample_id_overlap_count": len(sample_overlap),
            "sample_id_overlap_examples": sample_overlap[:20],
            "image_sha256_overlap_count": len(image_overlap),
            "image_sha256_overlap_examples": image_overlap[:20],
            "coco73k_id_overlap_count": len(coco_overlap),
            "coco73k_id_overlap_examples": coco_overlap[:20],
        }
    summary = {
        "subject": subject,
        "data_root": str(data_root.resolve()),
        "shards": shard_manifest,
        "splits": {
            split: {
                "sample_count": len(records),
                "unique_sample_id_count": len({item["sample_id"] for item in records}),
                "unique_stimulus_count_image_sha256": len(
                    {item["image_sha256"] for item in records}
                ),
                "unique_stimulus_count_coco73k": len(
                    {item["coco73k_id"] for item in records}
                ),
                "duplicate_sample_ids_within_split": duplicates(
                    item["sample_id"] for item in records
                ),
                "duplicate_image_hashes_within_split": duplicates(
                    item["image_sha256"] for item in records
                ),
                "repeat_count_histogram": {
                    str(count): sum(item["num_fmri_repeats"] == count for item in records)
                    for count in sorted({item["num_fmri_repeats"] for item in records})
                },
            }
            for split, records in manifests.items()
        },
        "cross_split_overlap": pair_audit,
        "stimulus_identity_primary": "image_sha256",
        "stimulus_identity_secondary": "coco73k_id",
    }
    (output_dir / "split_integrity_audit.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_manifests(
        Path(args.data_root).expanduser(), args.subject, Path(args.output_dir).expanduser()
    )
    print(json.dumps(result, indent=2))
