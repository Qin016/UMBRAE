#!/usr/bin/env python
"""Synthetic checks for unique-stimulus FGW manifest logic."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_fgw_stimulus_manifest import (
    FEATURE_FILES,
    OFFLINE_SPLITS,
    assign_unique_ids,
    validate_feature_addressing,
)


def test_unique_stimulus_split_and_cache_addressing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        split = root / "geometry_fit"
        split.mkdir()
        ids = ["coco73k:10", "coco73k:10", "coco73k:11", "coco73k:12", "coco73k:13"]
        samples = []
        rows = []
        for index, stimulus_id in enumerate(ids):
            samples.append(
                {
                    "stable_stimulus_id": stimulus_id,
                    "split_sample_index": index,
                }
            )
            rows.append(
                {
                    "global_cache_row": index,
                    "cache_split": "geometry_fit",
                    "cache_split_row": index,
                    "stable_stimulus_id": stimulus_id,
                }
            )
        (split / "metadata.json").write_text(json.dumps({"samples": samples}))
        for feature_file in FEATURE_FILES:
            values = np.arange(len(ids) * 4, dtype=np.float32).reshape(len(ids), 1, 4)
            np.save(split / feature_file, values)

        protected_test = {"coco73k:99"}
        assignments = assign_unique_ids(
            sorted(set(ids) - protected_test), seed=17, ratios=(0.5, 0.25, 0.25)
        )
        split_sets = {name: set(values) for name, values in assignments.items()}
        assert all(
            not split_sets[left] & split_sets[right]
            for index, left in enumerate(OFFLINE_SPLITS)
            for right in OFFLINE_SPLITS[index + 1 :]
        )
        assert all(not values & protected_test for values in split_sets.values())
        containing = [name for name, values in split_sets.items() if "coco73k:10" in values]
        assert len(containing) == 1

        validate_feature_addressing(root, rows)
        original = np.load(split / FEATURE_FILES[0])
        reconstructed = np.stack(
            [original[row["cache_split_row"]] for row in rows], axis=0
        )
        np.testing.assert_array_equal(reconstructed, original)


if __name__ == "__main__":
    test_unique_stimulus_split_and_cache_addressing()
    print("Synthetic FGW stimulus-manifest tests passed")
