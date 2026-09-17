"""Shared helpers for NSD coco73k ID and caption mapping utilities."""

import glob
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


def expand_paths(values: Sequence[str]) -> list[Path]:
    paths = []
    for value in values:
        matches = sorted(glob.glob(str(Path(value).expanduser())))
        if not matches and Path(value).expanduser().is_file():
            matches = [str(Path(value).expanduser())]
        paths.extend(Path(match).resolve() for match in matches)
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise FileNotFoundError(f"No files matched: {values}")
    return paths


def read_npy_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo
) -> np.ndarray:
    handle = archive.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member.name)
    return np.load(io.BytesIO(handle.read()), allow_pickle=False)


def iter_webdataset_records(
    tar_paths: Sequence[Path], max_samples: Optional[int] = None
):
    count = 0
    for tar_path in tar_paths:
        with tarfile.open(tar_path) as archive:
            grouped = defaultdict(dict)
            for member in archive.getmembers():
                name = member.name
                if "." not in name or not member.isfile():
                    continue
                prefix, suffix = name.split(".", 1)
                grouped[prefix][suffix] = member
            for prefix in sorted(grouped):
                members = grouped[prefix]
                coco_member = members.get("coco73k.npy")
                if coco_member is None:
                    continue
                coco_value = read_npy_member(
                    archive, coco_member
                ).reshape(-1)
                if coco_value.size != 1:
                    raise ValueError(
                        f"{coco_member.name} must contain one ID"
                    )
                record = {
                    "tar_path": str(tar_path),
                    "sample_key": prefix,
                    "coco73k_id": int(coco_value[0]),
                    "member_suffixes": sorted(members),
                }
                for field in ("trial.npy", "num_uniques.npy"):
                    member = members.get(field)
                    if member is not None:
                        values = read_npy_member(archive, member)
                        record[field[:-4]] = np.asarray(values).tolist()
                yield record
                count += 1
                if max_samples is not None and count >= max_samples:
                    return


def load_coco_captions(paths: Sequence[str]) -> dict[int, list[str]]:
    captions = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text())
        if isinstance(payload, Mapping) and "annotations" in payload:
            records = payload["annotations"]
        elif isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping):
            for key, value in payload.items():
                values = value if isinstance(value, list) else [value]
                for caption in values:
                    if caption and str(caption) not in captions[int(key)]:
                        captions[int(key)].append(str(caption))
            continue
        else:
            raise ValueError(f"Unsupported caption format: {path}")
        for record in records:
            image_id = record.get(
                "image_id",
                record.get(
                    "coco_id",
                    record.get("cocoId", record.get("id")),
                ),
            )
            caption = record.get("caption", record.get("captions"))
            if image_id is None or caption is None:
                continue
            values = caption if isinstance(caption, list) else [caption]
            for text in values:
                if text and str(text) not in captions[int(image_id)]:
                    captions[int(image_id)].append(str(text))
    return dict(captions)


def infer_dataset_label(path: Path) -> tuple[str, str]:
    text = path.name.lower()
    split = "train" if "train" in text else "validation" if "val" in text else "unknown"
    subject = "unknown"
    for token in ("subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"):
        if token in text:
            subject = token
            break
    return split, subject
