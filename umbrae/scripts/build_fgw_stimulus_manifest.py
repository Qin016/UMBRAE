#!/usr/bin/env python
"""Build leakage-safe stimulus splits over an existing FGW feature cache.

This script never computes features.  Its index arrays address a canonical
cache-row table that maps each row back to an existing cache split and local row.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


OFFLINE_SPLITS = ("offline_discovery", "offline_validation", "offline_test")
DEFAULT_RATIOS = (0.6, 0.2, 0.2)
FEATURE_FILES = (
    "brain_roi_features.npy",
    "projected_roi_features.npy",
    "clip_layer_features.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unique-stimulus FGW splits without recomputing features"
    )
    parser.add_argument("--representation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--downstream-train-tar", nargs="+", required=True)
    parser.add_argument("--downstream-validation-tar", nargs="+", required=True)
    parser.add_argument("--protected-downstream-test-tar", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratios", nargs=3, type=float, default=DEFAULT_RATIOS)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def sha256_ids(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _natural_key(path: Path) -> List[Any]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path))
    ]


def expand_paths(values: Sequence[str]) -> List[Path]:
    result: List[Path] = []
    for value in values:
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            matches = list(candidate.rglob("*.tar"))
        else:
            matches = [Path(item) for item in glob.glob(str(candidate))]
            if not matches and candidate.is_file():
                matches = [candidate]
        result.extend(item.resolve() for item in matches)
    result = sorted(dict.fromkeys(result), key=_natural_key)
    if not result:
        raise FileNotFoundError(f"No tar files matched {values}")
    return result


def _load_scalar(archive: tarfile.TarFile, member: tarfile.TarInfo) -> int:
    handle = archive.extractfile(member)
    if handle is None:
        raise IOError(f"Could not read tar member {member.name}")
    return int(np.asarray(np.load(io.BytesIO(handle.read()), allow_pickle=False)).reshape(-1)[0])


def scan_source_split(
    paths: Sequence[Path], subject: str, source_split: str
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for tar_path in paths:
        with tarfile.open(tar_path) as archive:
            grouped: Dict[str, Dict[str, tarfile.TarInfo]] = defaultdict(dict)
            for member in archive.getmembers():
                if member.isfile() and "." in member.name:
                    prefix, suffix = member.name.split(".", 1)
                    grouped[prefix][suffix] = member
            for sample_key, members in grouped.items():
                if "nsdgeneral.npy" not in members or "coco73k.npy" not in members:
                    continue
                coco_id = _load_scalar(archive, members["coco73k.npy"])
                trial = (
                    _load_scalar(archive, members["trial.npy"])
                    if "trial.npy" in members
                    else None
                )
                repeats = (
                    _load_scalar(archive, members["num_uniques.npy"])
                    if "num_uniques.npy" in members
                    else None
                )
                records.append(
                    {
                        "stable_stimulus_id": f"coco73k:{coco_id}",
                        "coco73k_id": coco_id,
                        "sample_key": sample_key,
                        "trial_id": trial,
                        "number_of_repeats": repeats,
                        "subject": subject,
                        "source_split": source_split,
                        "source_tar": str(tar_path),
                    }
                )
    records.sort(
        key=lambda item: (
            int(re.search(r"(\d+)$", item["sample_key"]).group(1))
            if re.search(r"(\d+)$", item["sample_key"])
            else math.inf,
            item["source_tar"],
        )
    )
    return records


def load_cache_rows(cache_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root_config = json.loads((cache_root / "cache_config.json").read_text())
    root_metadata = json.loads((cache_root / "metadata.json").read_text())
    rows: List[Dict[str, Any]] = []
    split_names = list(root_metadata["splits"])
    for cache_split in split_names:
        split_dir = cache_root / root_metadata["splits"][cache_split]["relative_path"]
        metadata = json.loads((split_dir / "metadata.json").read_text())
        for feature_file in FEATURE_FILES:
            if not (split_dir / feature_file).is_file():
                raise FileNotFoundError(split_dir / feature_file)
        for item in metadata["samples"]:
            rows.append(
                {
                    "global_cache_row": int(item["sample_index"]),
                    "cache_split": cache_split,
                    "cache_split_row": int(item["split_sample_index"]),
                    "stable_stimulus_id": str(item["stable_stimulus_id"]),
                    "coco73k_id": item.get("coco73k_id"),
                    "sample_key": str(item["sample_key"]),
                    "trial_id": item.get("trial_id"),
                    "number_of_repeats": item.get("number_of_repeats"),
                    "subject": item.get("subject"),
                    "original_source_split": item.get("dataset_split"),
                    "source_tar": item.get("source_tar"),
                }
            )
    rows.sort(key=lambda item: item["global_cache_row"])
    if [item["global_cache_row"] for item in rows] != list(range(len(rows))):
        raise ValueError("Cache sample_index values are not exactly 0..N-1")
    if len(rows) != int(root_metadata["sample_count"]):
        raise ValueError("Cache row count disagrees with root metadata")
    for item in rows:
        if item["subject"] != root_config["subject"]:
            raise ValueError("Cache contains a mismatched subject")
    return rows, root_config


def split_counts(count: int, ratios: Sequence[float]) -> List[int]:
    values = np.asarray(ratios, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("split_ratios must be three finite non-negative values")
    if values.sum() <= 0:
        raise ValueError("split_ratios must have positive mass")
    values /= values.sum()
    expected = values * count
    counts = np.floor(expected).astype(np.int64)
    for index in np.argsort(-(expected - counts), kind="stable")[: count - counts.sum()]:
        counts[index] += 1
    return counts.tolist()


def assign_unique_ids(
    eligible_ids: Sequence[str], seed: int, ratios: Sequence[float]
) -> Dict[str, List[str]]:
    unique_ids = sorted(set(eligible_ids))
    generator = np.random.default_rng(seed)
    shuffled = [unique_ids[index] for index in generator.permutation(len(unique_ids))]
    counts = split_counts(len(unique_ids), ratios)
    result: Dict[str, List[str]] = {}
    offset = 0
    for name, count in zip(OFFLINE_SPLITS, counts):
        result[name] = sorted(shuffled[offset : offset + count])
        offset += count
    return result


def summarize_source(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ids = [str(item["stable_stimulus_id"]) for item in records]
    repeats = Counter(item.get("number_of_repeats") for item in records)
    return {
        "row_count": len(records),
        "unique_stimulus_count": len(set(ids)),
        "duplicate_stimulus_row_count": len(ids) - len(set(ids)),
        "ordered_id_sha256": sha256_ids(ids),
        "ids": sorted(set(ids)),
        "trial_id_available_for_all_rows": all(item.get("trial_id") is not None for item in records),
        "repeat_count_available_for_all_rows": all(
            item.get("number_of_repeats") is not None for item in records
        ),
        "number_of_repeats_histogram": {
            str(key): value for key, value in sorted(repeats.items(), key=lambda pair: str(pair[0]))
        },
    }


def validate_feature_addressing(cache_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    by_split: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["cache_split"])].append(row)
    for split, selected in by_split.items():
        metadata = json.loads((cache_root / split / "metadata.json").read_text())
        for row in selected:
            original = metadata["samples"][int(row["cache_split_row"])]
            if original["stable_stimulus_id"] != row["stable_stimulus_id"]:
                raise ValueError("Index manifest does not reproduce cache metadata rows")
        for feature_file in FEATURE_FILES:
            array = np.load(cache_root / split / feature_file, mmap_mode="r")
            local = np.asarray([row["cache_split_row"] for row in selected], dtype=np.int64)
            if len(local) and (local.min() < 0 or local.max() >= len(array)):
                raise ValueError(f"Out-of-range rows for {split}/{feature_file}")


def run(args: argparse.Namespace) -> Path:
    cache_root = Path(args.representation_cache).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite manifest output: {output_dir}")
    if not (cache_root / "cache_config.json").is_file():
        raise FileNotFoundError(f"Invalid cache root: {cache_root}")
    rows, cache_config = load_cache_rows(cache_root)
    if cache_config["subject"] != args.subject:
        raise ValueError("--subject does not match representation cache")

    source_records = {
        "downstream_training": scan_source_split(
            expand_paths(args.downstream_train_tar), args.subject, "train"
        ),
        "downstream_validation_evaluation": scan_source_split(
            expand_paths(args.downstream_validation_tar), args.subject, "val"
        ),
        "protected_final_downstream_test": scan_source_split(
            expand_paths(args.protected_downstream_test_tar), args.subject, "test"
        ),
    }
    source_summaries = {
        name: summarize_source(records) for name, records in source_records.items()
    }
    training_ids = set(source_summaries["downstream_training"]["ids"])
    validation_ids = set(source_summaries["downstream_validation_evaluation"]["ids"])
    test_ids = set(source_summaries["protected_final_downstream_test"]["ids"])
    if training_ids & validation_ids or training_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Original downstream train/val/test unique-ID sets overlap")

    cache_ids = [row["stable_stimulus_id"] for row in rows]
    cached_unique = set(cache_ids)
    excluded_protected = validation_ids | test_ids
    eligible_ids = sorted(cached_unique - excluded_protected)
    assignments = assign_unique_ids(eligible_ids, args.seed, args.split_ratios)
    id_to_split = {
        stimulus_id: split for split, ids in assignments.items() for stimulus_id in ids
    }
    index_arrays: Dict[str, np.ndarray] = {}
    for split in OFFLINE_SPLITS:
        index_arrays[split] = np.asarray(
            [
                row["global_cache_row"]
                for row in rows
                if id_to_split.get(row["stable_stimulus_id"]) == split
            ],
            dtype=np.int64,
        )

    output_dir.mkdir(parents=True)
    for split, indices in index_arrays.items():
        np.save(output_dir / f"{split}_indices.npy", indices)

    validate_feature_addressing(cache_root, rows)
    split_sets = {name: set(ids) for name, ids in assignments.items()}
    overlaps = {
        "discovery_validation": len(split_sets[OFFLINE_SPLITS[0]] & split_sets[OFFLINE_SPLITS[1]]),
        "discovery_test": len(split_sets[OFFLINE_SPLITS[0]] & split_sets[OFFLINE_SPLITS[2]]),
        "validation_test": len(split_sets[OFFLINE_SPLITS[1]] & split_sets[OFFLINE_SPLITS[2]]),
    }
    protected_overlaps = {
        name: {
            "downstream_validation_evaluation": len(set(ids) & validation_ids),
            "protected_final_downstream_test": len(set(ids) & test_ids),
        }
        for name, ids in assignments.items()
    }
    repeated_row_groups: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        repeated_row_groups[row["stable_stimulus_id"]].append(row["global_cache_row"])
    repeated_cross_split = 0
    for stimulus_id, group in repeated_row_groups.items():
        assigned = {id_to_split.get(stimulus_id) for _ in group}
        repeated_cross_split += int(len(assigned) > 1)

    aliases = {
        "brain_roi_features": {
            "canonical_semantic_name": "roi_tokens_before_projector",
            "storage_file": "brain_roi_features.npy",
            "meaning": "fixed-dimensional ROITokenizer output",
            "is_raw_roi_voxel_vector": False,
        },
        "roi_tokens_before_projector": {
            "alias_of": "brain_roi_features",
            "storage_file": "brain_roi_features.npy",
        },
        "projected_roi_features": {
            "canonical_semantic_name": "projected_roi_features",
            "storage_file": "projected_roi_features.npy",
            "meaning": "BrainToCLIPProjector output",
        },
    }
    manifest = {
        "manifest_version": 1,
        "subject": args.subject,
        "representation_cache": str(cache_root),
        "cache_sample_count": len(rows),
        "cache_unique_stimulus_count": len(cached_unique),
        "cache_duplicate_stimulus_row_count": len(rows) - len(cached_unique),
        "cache_rows_can_share_unique_stimulus": len(rows) != len(cached_unique),
        "cache_row_table": rows,
        "row_metadata_availability": {
            key: all(row.get(key) is not None for row in rows)
            for key in (
                "stable_stimulus_id",
                "coco73k_id",
                "global_cache_row",
                "trial_id",
                "subject",
                "original_source_split",
                "number_of_repeats",
            )
        },
        "feature_semantic_aliases": aliases,
        "downstream_id_sets": source_summaries,
        "downstream_usage_provenance": {
            "training": "used by completed Stage-2 adapter configs via train_tar",
            "validation_evaluation": (
                "the existing 300-sample val set was repeatedly used for validation, "
                "checkpoint selection, and caption metrics; it is not a final test set"
            ),
            "protected_final_test": (
                "the 982-sample NSD test split is protected; no completed adapter_config "
                "under stage2_outputs or umbrae_neuroroute_outputs references test tar files"
            ),
            "genuinely_untouched_final_caption_test_status": (
                "PROTECTED_CANDIDATE_NOT_YET_EVALUATED"
            ),
        },
        "eligibility_policy": {
            "source": "existing cached downstream-training stimuli only",
            "excluded_from_all_offline_splits": [
                "downstream_validation_evaluation",
                "protected_final_downstream_test",
            ],
            "note": (
                "Offline splits are held out for FGW method development only; their "
                "representations came from a Stage-1 checkpoint trained on downstream train."
            ),
        },
        "split_generation": {
            "unit": "unique stable stimulus ID",
            "method": (
                "lexically sorted eligible IDs, NumPy default_rng(seed) permutation, "
                "largest-remainder 60/20/20 counts"
            ),
            "seed": args.seed,
            "ratios": list(args.split_ratios),
        },
        "offline_splits": {
            name: {
                "unique_stimulus_count": len(ids),
                "cache_row_count": len(index_arrays[name]),
                "ids": ids,
                "sorted_id_sha256": sha256_ids(ids),
                "index_file": f"{name}_indices.npy",
                "index_semantics": "indices into cache_row_table/global_cache_row",
                "use": {
                    "offline_discovery": "candidate geometry/coupling discovery",
                    "offline_validation": "FGW hyperparameter selection only",
                    "offline_test": "one-shot offline scientific evaluation only",
                }[name],
            }
            for name, ids in assignments.items()
        },
        "overlap_checks": {
            "offline_unique_stimulus_overlaps": overlaps,
            "protected_set_overlaps": protected_overlaps,
            "repeated_stimulus_groups_crossing_offline_splits": repeated_cross_split,
            "eligible_ids_covered_exactly_once": (
                set().union(*split_sets.values()) == set(eligible_ids)
                and sum(len(value) for value in split_sets.values()) == len(eligible_ids)
            ),
        },
        "feature_recomputation": False,
    }
    write_json(output_dir / "fgw_stimulus_manifest.json", manifest)
    write_json(output_dir / "cache_feature_aliases.json", aliases)
    print(f"Saved stimulus manifest to {output_dir}")
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
