"""Read-only helpers for P11 structural mappings.

This module deliberately has no optimizer, loss, or model-training code.  The
small NIfTI reader covers the scalar NIfTI-1 volumes shipped in the local NSD
copy, avoiding a runtime dependency on nibabel for the audit artifacts.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


NIFTI_DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_nifti1(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a scalar NIfTI-1 `.nii` or `.nii.gz` with scaling and sform."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        raw = handle.read()
    if len(raw) < 352:
        raise ValueError(f"NIfTI file is too short: {path}")
    if struct.unpack("<i", raw[:4])[0] == 348:
        endian = "<"
    elif struct.unpack(">i", raw[:4])[0] == 348:
        endian = ">"
    else:
        raise ValueError(f"Not a NIfTI-1 header: {path}")
    dim = struct.unpack(endian + "8h", raw[40:56])
    shape = tuple(int(v) for v in dim[1 : dim[0] + 1])
    datatype = int(struct.unpack(endian + "h", raw[70:72])[0])
    if datatype not in NIFTI_DTYPES:
        raise ValueError(f"Unsupported NIfTI datatype code {datatype}: {path}")
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    slope = float(struct.unpack(endian + "f", raw[112:116])[0])
    intercept = float(struct.unpack(endian + "f", raw[116:120])[0])
    dtype = np.dtype(NIFTI_DTYPES[datatype]).newbyteorder(endian)
    array = np.frombuffer(
        raw, dtype=dtype, count=int(np.prod(shape)), offset=offset
    ).reshape(shape, order="F")
    if slope not in (0.0, 1.0):
        array = array.astype(np.float64) * slope
    if intercept != 0.0:
        array = array.astype(np.float64) + intercept
    sform_code = int(struct.unpack(endian + "h", raw[254:256])[0])
    if sform_code > 0:
        affine = np.eye(4, dtype=np.float64)
        affine[:3] = np.asarray(
            struct.unpack(endian + "12f", raw[280:328]), dtype=np.float64
        ).reshape(3, 4)
        affine_source = "sform"
    else:
        affine = np.diag([pixdim[1], pixdim[2], pixdim[3], 1.0])
        affine_source = "pixdim_fallback"
    meta = {
        "shape": list(shape),
        "dtype": str(array.dtype),
        "datatype_code": datatype,
        "pixdim": [float(v) for v in pixdim[1 : 1 + len(shape)]],
        "affine": affine.tolist(),
        "affine_source": affine_source,
        "scl_slope": slope,
        "scl_inter": intercept,
    }
    return np.asarray(array), meta


def nsdgeneral_voxel_table(mask: np.ndarray, affine: np.ndarray) -> dict[str, np.ndarray]:
    """Return the C-order index/ijk/world mapping used by UMBRAE vectors."""
    selected = np.flatnonzero(np.asarray(mask).ravel(order="C") > 0)
    ijk = np.column_stack(np.unravel_index(selected, mask.shape, order="C"))
    xyz = np.column_stack([ijk, np.ones(len(ijk))]) @ affine.T
    return {
        "voxel_index": np.arange(len(selected), dtype=np.int64),
        "volume_flat_index_c": selected.astype(np.int64),
        "ijk": ijk.astype(np.int16),
        "xyz_mm": xyz[:, :3].astype(np.float32),
    }


def clip_patch_index_map(grid_size: int = 16) -> list[dict[str, Any]]:
    """CLIP image-patch map for `flatten(2).transpose(1, 2)` ordering."""
    result = []
    for row in range(grid_size):
        for column in range(grid_size):
            index = row * grid_size + column
            x = 2.0 * ((column + 0.5) / grid_size) - 1.0
            y_down = 2.0 * ((row + 0.5) / grid_size) - 1.0
            result.append(
                {
                    "patch_index": index,
                    "row": row,
                    "column": column,
                    "image_x_normalized": x,
                    "image_y_down_normalized": y_down,
                    "visual_field_y_up_normalized": -y_down,
                    "visual_angle_x_degrees": None,
                    "visual_angle_y_degrees": None,
                }
            )
    return result


def gaussian_patch_affinity(
    centers_xy: np.ndarray,
    sigma: np.ndarray,
    patch_centers_xy: np.ndarray,
) -> np.ndarray:
    """Construct normalized Gaussian affinities for verified, like-unit inputs."""
    centers_xy = np.asarray(centers_xy, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    patch_centers_xy = np.asarray(patch_centers_xy, dtype=np.float64)
    if centers_xy.ndim != 2 or centers_xy.shape[1] != 2:
        raise ValueError("centers_xy must have shape [N,2]")
    if len(sigma) != len(centers_xy) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma must be finite, positive, and match centers_xy")
    squared_distance = np.sum(
        (centers_xy[:, None, :] - patch_centers_xy[None, :, :]) ** 2,
        axis=-1,
    )
    logits = -squared_distance / (2.0 * sigma[:, None] ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    affinity = np.exp(logits)
    affinity /= affinity.sum(axis=1, keepdims=True)
    return affinity.astype(np.float32)


def within_roi_random_control(
    units: Sequence[Mapping[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Randomize voxel membership within each ROI while preserving unit sizes."""
    rng = np.random.default_rng(seed)
    result = [dict(unit) for unit in units]
    rois = sorted({str(unit["parent_roi"]) for unit in units})
    for roi in rois:
        positions = [i for i, unit in enumerate(units) if str(unit["parent_roi"]) == roi]
        sizes = [len(units[i]["voxel_indices"]) for i in positions]
        pool = np.concatenate(
            [np.asarray(units[i]["voxel_indices"], dtype=np.int64) for i in positions]
        )
        shuffled = rng.permutation(pool)
        offset = 0
        for position, size in zip(positions, sizes):
            result[position]["voxel_indices"] = sorted(
                int(v) for v in shuffled[offset : offset + size]
            )
            result[position]["num_voxels"] = size
            offset += size
    return result


class FineGrainedStructuralTokenizer:
    """Non-learnable reader for unit-specific fMRI values.

    This skeleton becomes usable only after a mapping with non-empty `units` is
    structurally locked.  It performs indexing and padding only.
    """

    def __init__(self, mapping: str | Path | Mapping[str, Any]) -> None:
        if isinstance(mapping, (str, Path)):
            mapping = json.loads(Path(mapping).read_text())
        self.mapping = dict(mapping)
        self.units = list(self.mapping.get("units", []))
        if not self.units:
            raise ValueError("Mapping contains no constructed units")
        self.indices = [
            torch.as_tensor(unit["voxel_indices"], dtype=torch.long)
            for unit in self.units
        ]

    def __call__(self, x: Tensor, padded: bool = True):
        if x.ndim != 2:
            raise ValueError(f"x must have shape [B,V], got {tuple(x.shape)}")
        values = []
        for indices in self.indices:
            if len(indices) == 0 or int(indices.max()) >= x.shape[1]:
                raise IndexError("Unit voxel index is empty or outside input V")
            values.append(x.index_select(1, indices.to(x.device)))
        if not padded:
            return values
        max_voxels = max(value.shape[1] for value in values)
        raw = x.new_zeros((x.shape[0], len(values), max_voxels))
        mask = torch.zeros(
            (len(values), max_voxels), dtype=torch.bool, device=x.device
        )
        for unit_index, value in enumerate(values):
            raw[:, unit_index, : value.shape[1]] = value
            mask[unit_index, : value.shape[1]] = True
        return raw, mask
