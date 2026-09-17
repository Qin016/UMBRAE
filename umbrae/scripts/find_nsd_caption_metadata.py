#!/usr/bin/env python
"""Find and preview likely NSD/COCO caption and stimulus metadata files."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


EXTENSIONS = {".json", ".csv", ".tsv", ".npy", ".npz", ".pkl", ".parquet"}
KEYWORDS = (
    "caption", "coco", "coco73k", "nsd_stim", "stimuli", "image_info",
    "annotation", "instance", "karpathy", "train2017", "val2017",
)


def preview_file(path: Path):
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return {"type": "dict", "keys": list(payload)[:20], "length": len(payload)}
            return {"type": type(payload).__name__, "length": len(payload), "first": payload[:1]}
        if suffix in (".csv", ".tsv"):
            import pandas as pd
            frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=5)
            return {"columns": frame.columns.tolist(), "preview": frame.head(2).to_dict("records")}
        if suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            return {"shape": list(array.shape), "dtype": str(array.dtype)}
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                return {"keys": list(payload.files), "shapes": {key: list(payload[key].shape) for key in payload.files[:20]}}
        if suffix == ".parquet":
            import pandas as pd
            frame = pd.read_parquet(path).head(5)
            return {"columns": frame.columns.tolist(), "preview": frame.head(2).to_dict("records")}
        if suffix == ".pkl" and path.stat().st_size < 50_000_000:
            with path.open("rb") as file:
                payload = pickle.load(file)
            return {"type": type(payload).__name__, "keys": list(payload)[:20] if isinstance(payload, dict) else None}
    except Exception as error:
        return {"preview_error": str(error)}
    return {"preview": "skipped"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.search_root).expanduser().resolve()
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        lower = path.name.lower()
        if not any(keyword in lower for keyword in KEYWORDS):
            continue
        item = {
            "path": str(path),
            "file_type": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "preview": preview_file(path),
        }
        candidates.append(item)
        print(json.dumps(item, ensure_ascii=False))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata_candidates.json").write_text(
        json.dumps(candidates, indent=2)
    )
    print(f"Found {len(candidates)} candidates; saved to {output_dir}")


if __name__ == "__main__":
    main()
