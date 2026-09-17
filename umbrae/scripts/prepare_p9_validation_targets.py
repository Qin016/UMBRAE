#!/usr/bin/env python
"""Build and lock P9 validation references from NSD/COCO metadata.

The bbox transform is audited against BrainHub's existing 982-image test lock
before validation targets are emitted.  No test value is used as a train/val
target or selection signal.
"""

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dual_branch_cache import sha256_file


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_coco(paths, needed):
    images, annotations, categories, schemas = {}, defaultdict(list), {}, {}
    for split, path in paths.items():
        payload = json.loads(Path(path).read_text())
        schemas[split] = {
            "images": len(payload["images"]),
            "annotations": len(payload["annotations"]),
            "categories": len(payload["categories"]),
            "description": payload.get("info", {}).get("description"),
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
        }
        categories.update({int(row["id"]): row["name"] for row in payload["categories"]})
        split_ids = needed.get(split, set())
        images.update({int(row["id"]): row for row in payload["images"] if int(row["id"]) in split_ids})
        for row in payload["annotations"]:
            if int(row["image_id"]) in split_ids:
                annotations[int(row["image_id"])].append(row)
    return images, annotations, categories, schemas


def sample_metadata(rows, metadata):
    result, needed = [], defaultdict(set)
    for sample_index, row in enumerate(rows):
        local_id = int(row["coco73k_id"])
        record = metadata.loc[local_id]
        item = {
            "sample_index": sample_index,
            "sample_id": row["sample_id"],
            "coco73k_id": local_id,
            "coco_id": int(record["cocoId"]),
            "coco_split": str(record["cocoSplit"]),
            "crop_box": str(record["cropBox"]),
        }
        result.append(item)
        needed[item["coco_split"]].add(item["coco_id"])
    return result, needed


def transformed_boxes(item, image, annotations, category_names):
    top, bottom, left, right = ast.literal_eval(item["crop_box"])
    width, height = float(image["width"]), float(image["height"])
    x0, y0 = left * width, top * height
    crop_width = width * (1.0 - left - right)
    crop_height = height * (1.0 - top - bottom)
    grouped = defaultdict(list)
    for annotation in annotations:
        x, y, box_width, box_height = map(float, annotation["bbox"])
        box = [
            max(0.0, min(1.0, (x - x0) / crop_width)),
            max(0.0, min(1.0, (y - y0) / crop_height)),
            max(0.0, min(1.0, (x + box_width - x0) / crop_width)),
            max(0.0, min(1.0, (y + box_height - y0) / crop_height)),
        ]
        if box[2] > box[0] and box[3] > box[1]:
            grouped[category_names[int(annotation["category_id"])]].append(box)
    return dict(grouped)


def build_targets(items, images, annotations, categories):
    output = {}
    for item in items:
        image_id = item["coco_id"]
        if image_id not in images:
            raise KeyError(f"Missing COCO image {image_id}")
        output[str(item["sample_index"])] = transformed_boxes(
            item, images[image_id], annotations[image_id], categories
        )
    return output


def audit_existing_test(generated, existing):
    exact_sets, coordinate_differences, count_mismatches = 0, [], 0
    extra_categories, missing_categories = defaultdict(int), defaultdict(int)
    for key in existing:
        expected, actual = existing[key], generated[key]
        expected_set, actual_set = set(expected), set(actual)
        exact_sets += expected_set == actual_set
        for name in actual_set - expected_set:
            extra_categories[name] += 1
        for name in expected_set - actual_set:
            missing_categories[name] += 1
        for name in expected_set & actual_set:
            if len(expected[name]) != len(actual[name]):
                count_mismatches += 1
                continue
            for left, right in zip(expected[name], actual[name]):
                coordinate_differences.extend(abs(float(a) - float(b)) for a, b in zip(left, right))
    values = np.asarray(coordinate_differences, dtype=np.float64)
    return {
        "test_images": len(existing),
        "exact_category_set_images": exact_sets,
        "exact_category_set_fraction": exact_sets / len(existing),
        "missing_category_occurrences": dict(missing_categories),
        "extra_category_occurrences": dict(extra_categories),
        "box_count_mismatches_on_shared_categories": count_mismatches,
        "matched_coordinate_count": len(values),
        "coordinate_absolute_error_mean": float(values.mean()),
        "coordinate_absolute_error_median": float(np.median(values)),
        "coordinate_absolute_error_p99": float(np.percentile(values, 99)),
        "coordinate_absolute_error_max": float(values.max()),
        "status": "PASS" if not missing_categories and exact_sets / len(existing) >= 0.99 and np.percentile(values, 99) < 1e-4 else "FAIL",
        "interpretation": "Small extras are attributed to COCO annotation snapshot differences; no existing BrainHub query is missing.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="/opt/data/private/BA/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.csv")
    parser.add_argument("--caption-map", default="stage2_outputs/caption_mapping/coco73k_captions.json")
    parser.add_argument("--train-instances", default="protocol_outputs/protocol_v1/coco_annotations/instances_train2017.json")
    parser.add_argument("--val-instances", default="protocol_outputs/protocol_v1/coco_annotations/instances_val2017.json")
    parser.add_argument("--val-manifest", default="protocol_outputs/protocol_v1/subj01/val_manifest.jsonl")
    parser.add_argument("--test-manifest", default="protocol_outputs/protocol_v1/subj01/test_manifest.jsonl")
    parser.add_argument("--test-grounding", default="../BrainHub/data/bbox/coco_bbox_categorized.json")
    parser.add_argument("--output-dir", default="protocol_outputs/protocol_v1/subj01/p9_validation_targets")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).set_index("nsdId")
    val_rows, test_rows = read_manifest(args.val_manifest), read_manifest(args.test_manifest)
    val_items, val_needed = sample_metadata(val_rows, metadata)
    test_items, test_needed = sample_metadata(test_rows, metadata)
    needed = {split: val_needed[split] | test_needed[split] for split in set(val_needed) | set(test_needed)}
    instance_paths = {"train2017": args.train_instances, "val2017": args.val_instances}
    images, annotations, categories, schemas = load_coco(instance_paths, needed)
    test_generated = build_targets(test_items, images, annotations, categories)
    audit = audit_existing_test(test_generated, json.loads(Path(args.test_grounding).read_text()))
    if audit["status"] != "PASS":
        raise RuntimeError(f"Test bbox transform audit failed: {audit}")
    val_grounding = build_targets(val_items, images, annotations, categories)
    caption_map = json.loads(Path(args.caption_map).read_text())
    val_captions = {str(item["sample_index"]): caption_map[str(item["coco73k_id"])] for item in val_items}
    if len(val_captions) != 300 or any(len(value) < 5 for value in val_captions.values()):
        raise ValueError("Validation captions must contain at least five references for all 300 images")
    query_count = sum(len(value) for value in val_grounding.values())
    if not query_count:
        raise ValueError("Validation grounding contains no queries")
    (output / "caption_references.json").write_text(json.dumps(val_captions, indent=2) + "\n")
    (output / "grounding_annotations.json").write_text(json.dumps(val_grounding, indent=2) + "\n")
    (output / "sample_metadata.json").write_text(json.dumps(val_items, indent=2) + "\n")
    provenance = {
        "protocol": "protocol_v1", "subject": "subj01", "split": "validation",
        "sample_count": len(val_items), "grounding_query_count": query_count,
        "caption_reference_count": sum(len(value) for value in val_captions.values()),
        "caption_map": str(Path(args.caption_map).resolve()), "caption_map_sha256": sha256_file(args.caption_map),
        "metadata": str(Path(args.metadata).resolve()), "metadata_sha256": sha256_file(args.metadata),
        "instance_schemas": schemas, "test_transform_audit": audit,
        "test_values_used_for_training_or_selection": False,
        "bbox_rule": "COCO xywh transformed through NSD cropBox then clipped to [0,1]; each visible category is one query and all same-category boxes are retained",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
