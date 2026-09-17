#!/usr/bin/env python
"""Merge real label-level NSD mappings into NeuroRoute v1 ROIs."""

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_COLLECTIONS = [
    "prf-visualrois",
    "floc-faces",
    "floc-bodies",
    "floc-places",
]
DEFAULT_REQUIRED_ROIS = [
    "V1",
    "V2",
    "V3",
    "hV4",
    "FFA",
    "EBA",
    "PPA",
    "OPA",
    "RSC",
]
ROI_RULES = {
    "V1": [("prf-visualrois", "V1v"), ("prf-visualrois", "V1d")],
    "V2": [("prf-visualrois", "V2v"), ("prf-visualrois", "V2d")],
    "V3": [("prf-visualrois", "V3v"), ("prf-visualrois", "V3d")],
    "hV4": [("prf-visualrois", "hV4")],
    "FFA": [("floc-faces", "FFA-1"), ("floc-faces", "FFA-2")],
    "EBA": [("floc-bodies", "EBA")],
    "PPA": [("floc-places", "PPA")],
    "OPA": [("floc-places", "OPA")],
    "RSC": [("floc-places", "RSC")],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final merged NeuroRoute v1 ROI mapping"
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required-rois", nargs="+", default=DEFAULT_REQUIRED_ROIS)
    parser.add_argument(
        "--roi-set-config",
        help="JSON config defining required_rois, optional_rois, and excluded_rois",
    )
    parser.add_argument("--validation-json")
    parser.add_argument("--collections", nargs="+", default=DEFAULT_COLLECTIONS)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_roi_set_config(path: str, fallback_required_rois: Sequence[str]) -> Dict[str, object]:
    if not path:
        return {
            "name": None,
            "required_rois": list(fallback_required_rois),
            "optional_rois": [],
            "excluded_rois": {},
            "path": None,
        }
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing ROI-set config: {config_path}")
    config = json.loads(config_path.read_text())
    required = list(config.get("required_rois", []))
    optional = list(config.get("optional_rois", []))
    excluded = config.get("excluded_rois", {})
    if not required or len(required) != len(set(required)):
        raise ValueError("ROI-set required_rois must be a non-empty unique list")
    if not isinstance(excluded, Mapping):
        raise ValueError("ROI-set excluded_rois must be a JSON object")
    unknown = [
        name for name in required + optional + list(excluded) if name not in ROI_RULES
    ]
    if unknown:
        raise ValueError(f"ROI-set contains unknown ROIs: {unknown}")
    conflicting = sorted(set(required).intersection(excluded))
    if conflicting:
        raise ValueError(f"Required ROIs cannot also be excluded: {conflicting}")
    return {
        "name": config.get("name"),
        "required_rois": required,
        "optional_rois": optional,
        "excluded_rois": dict(excluded),
        "path": str(config_path),
    }


def normalize_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def load_collection(path: Path) -> Dict[int, Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label-level mapping: {path}")
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Mapping must be a dictionary: {path}")
    result = {}
    for raw_label, raw_entry in payload.items():
        label = int(raw_label)
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"{path}: label {label} entry is not a dictionary")
        name = str(raw_entry.get("name", "")).strip()
        indices = np.asarray(
            raw_entry.get("nsdgeneral_indices", []), dtype=np.int64
        )
        result[label] = {**raw_entry, "name": name, "indices": indices}
    return dict(sorted(result.items()))


def index_sources(
    collections: Mapping[str, Mapping[int, Mapping[str, object]]]
) -> Dict[Tuple[str, str], List[Tuple[int, Mapping[str, object]]]]:
    sources: Dict[
        Tuple[str, str], List[Tuple[int, Mapping[str, object]]]
    ] = {}
    for collection, labels in collections.items():
        for label, entry in labels.items():
            key = (collection, normalize_name(entry["name"]))
            sources.setdefault(key, []).append((label, entry))
    return sources


def merge_preserving_order(arrays: Sequence[np.ndarray]) -> np.ndarray:
    seen = set()
    merged = []
    for array in arrays:
        for value in np.asarray(array, dtype=np.int64):
            integer = int(value)
            if integer not in seen:
                seen.add(integer)
                merged.append(integer)
    return np.asarray(merged, dtype=np.int64)


def resolve_roi(
    roi_name: str,
    source_index: Mapping[
        Tuple[str, str], List[Tuple[int, Mapping[str, object]]]
    ],
) -> Dict[str, object]:
    required_sources = ROI_RULES.get(roi_name)
    if required_sources is None:
        return {
            "indices": [],
            "source_labels": [],
            "num_voxels": 0,
            "resolved": False,
            "reason": "No merge rule is defined",
        }

    source_labels = []
    arrays = []
    unresolved_reasons = []
    for collection, expected_name in required_sources:
        matches = source_index.get(
            (collection, normalize_name(expected_name)), []
        )
        if len(matches) != 1:
            unresolved_reasons.append(
                f"{collection}:{expected_name} matched {len(matches)} labels"
            )
            continue
        label, entry = matches[0]
        if len(entry["indices"]) == 0:
            unresolved_reasons.append(
                f"{collection}:{expected_name} is empty"
            )
            continue
        arrays.append(entry["indices"])
        source_labels.append(
            {
                "collection": collection,
                "label": int(label),
                "name": entry["name"],
            }
        )

    resolved = not unresolved_reasons and len(arrays) == len(required_sources)
    indices = merge_preserving_order(arrays) if resolved else np.empty(0, np.int64)
    return {
        "indices": indices.tolist(),
        "source_labels": source_labels,
        "num_voxels": int(len(indices)),
        "resolved": bool(resolved and len(indices) > 0),
        **(
            {"reason": "; ".join(unresolved_reasons)}
            if unresolved_reasons
            else {}
        ),
    }


def build_overlap_report(
    rois: Mapping[str, Mapping[str, object]]
) -> Dict[str, object]:
    names = list(rois)
    pairs = {}
    unexpected_pairs = []
    for left_index, left_name in enumerate(names):
        left = rois[left_name]
        left_indices = set(left["indices"])
        left_collections = {
            source["collection"] for source in left["source_labels"]
        }
        for right_name in names[left_index + 1 :]:
            right = rois[right_name]
            overlap = sorted(left_indices.intersection(right["indices"]))
            if not overlap:
                continue
            right_collections = {
                source["collection"] for source in right["source_labels"]
            }
            # Labels in one atlas collection are expected to be mutually
            # exclusive. Independent atlas collections may overlap anatomically.
            unexpected = bool(left_collections.intersection(right_collections))
            pair_name = f"{left_name}__{right_name}"
            pairs[pair_name] = {
                "count": len(overlap),
                "fraction_of_left": len(overlap) / left["num_voxels"],
                "fraction_of_right": len(overlap) / right["num_voxels"],
                "unexpected": unexpected,
                "shared_source_collections": sorted(
                    left_collections.intersection(right_collections)
                ),
            }
            if unexpected:
                unexpected_pairs.append(pair_name)
    return {
        "pairs_with_overlap": pairs,
        "unexpected_overlap_pairs": unexpected_pairs,
        "has_unexpected_overlap": bool(unexpected_pairs),
        "policy": (
            "Within-collection overlap is unexpected; overlap between "
            "independent NSD atlas collections is reported as expected."
        ),
    }


def load_validation(path: str) -> Dict[str, object]:
    if not path:
        return {
            "num_samples_checked": 0,
            "num_repeats_checked": 0,
            "mean_pearson_correlation": None,
            "max_abs_error": None,
            "mean_abs_error": None,
            "strict_allclose": False,
            "relaxed_allclose": False,
            "voxel_order_verified": False,
        }
    validation_path = Path(path).expanduser().resolve()
    validation = json.loads(validation_path.read_text())
    validation["voxel_order_verified"] = bool(
        validation.get("num_samples_checked", 0) > 0
        and validation.get("relaxed_allclose", False)
    )
    return validation


def main() -> None:
    args = parse_args()
    roi_set = load_roi_set_config(args.roi_set_config, args.required_rois)
    required_rois = roi_set["required_rois"]
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    collections = {}
    for collection in args.collections:
        path = (
            input_dir
            / f"{args.subject}_{collection}_nsdgeneral_indices.pkl"
        )
        collections[collection] = load_collection(path)

    sources = index_sources(collections)
    rois = {}
    unresolved = []
    for roi_name in required_rois:
        rois[roi_name] = resolve_roi(roi_name, sources)
        if not rois[roi_name]["resolved"]:
            unresolved.append(roi_name)
            print(
                f"[WARN] Unresolved ROI {roi_name}: "
                f"{rois[roi_name].get('reason', 'empty')}"
            )
        else:
            print(f"{roi_name}: {rois[roi_name]['num_voxels']} voxels")

    overlap_report = build_overlap_report(rois)
    for pair, report in overlap_report["pairs_with_overlap"].items():
        classification = "UNEXPECTED" if report["unexpected"] else "expected"
        print(f"Overlap {pair}: {report['count']} ({classification})")

    validation = load_validation(args.validation_json)
    payload = {
        "subject": args.subject,
        "space": "func1pt8mm",
        "roi_mapping_is_real": True,
        "voxel_order_verified": validation["voxel_order_verified"],
        "mapping_source": (
            "NSD func1pt8mm ROI labels + verified "
            "wholebrain_3d/nsdgeneral mapping"
        ),
        "roi_set": roi_set["name"],
        "roi_set_config": roi_set["path"],
        "roi_names": list(required_rois),
        "required_rois": list(required_rois),
        "optional_rois": list(roi_set["optional_rois"]),
        "excluded_rois": list(roi_set["excluded_rois"]),
        "excluded_roi_details": roi_set["excluded_rois"],
        "roi_counts": {
            name: rois[name]["num_voxels"] for name in required_rois
        },
        "rois": rois,
        "unresolved_rois": unresolved,
        "overlap_report": overlap_report,
        "validation": {
            key: validation.get(key)
            for key in (
                "num_samples_checked",
                "num_repeats_checked",
                "mean_pearson_correlation",
                "max_abs_error",
                "mean_abs_error",
                "strict_allclose",
                "relaxed_allclose",
            )
        },
        "collections": list(args.collections),
        "created_time": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved NeuroRoute mapping: {output_path}")
    print(
        f"voxel_order_verified={payload['voxel_order_verified']}, "
        f"unresolved={unresolved}, "
        f"unexpected_overlap={overlap_report['has_unexpected_overlap']}"
    )

    strict_failed = bool(
        unresolved
        or overlap_report["has_unexpected_overlap"]
        or not payload["voxel_order_verified"]
    )
    if args.strict and strict_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
