"""Deterministic size-and-overlap-matched random ROI controls."""

from typing import Dict, Mapping, Sequence

import numpy as np


def roi_overlap_matrix(
    roi_indices: Mapping[str, Sequence[int]], roi_order: Sequence[str]
) -> np.ndarray:
    sets = [set(map(int, roi_indices[name])) for name in roi_order]
    return np.asarray(
        [[len(left & right) for right in sets] for left in sets], dtype=np.int64
    )


def generate_random_structure_control(
    roi_indices: Mapping[str, Sequence[int]],
    roi_order: Sequence[str],
    seed: int,
) -> Dict[str, list[int]]:
    """Apply one random bijection of the real ROI union to every ROI.

    A shared bijection preserves the exact voxel universe, every ROI size, all
    pairwise overlaps, and in fact every higher-order intersection. It uses no
    visual labels, CLIP features, or fMRI response statistics.
    """
    missing = [name for name in roi_order if name not in roi_indices]
    if missing:
        raise ValueError(f"Missing ROI indices: {missing}")
    universe = sorted(
        {int(voxel) for name in roi_order for voxel in roi_indices[name]}
    )
    if not universe:
        raise ValueError("ROI voxel universe is empty")
    shuffled = np.asarray(universe, dtype=np.int64)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    permutation = dict(zip(universe, shuffled.tolist()))
    return {
        name: sorted(permutation[int(voxel)] for voxel in roi_indices[name])
        for name in roi_order
    }
