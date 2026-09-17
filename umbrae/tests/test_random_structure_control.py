import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.random_structure_control import (
    generate_random_structure_control,
    roi_overlap_matrix,
)
from models.roi_mapping import MVP_ROI_NAMES


def real_mapping():
    return {
        "V1": [0, 1, 2, 3],
        "V2": [2, 3, 4],
        "V3": [4, 5],
        "hV4": [5, 6],
        "FFA": [6, 7, 8],
        "EBA": [8, 9],
        "PPA": [1, 9, 10],
        "OPA": [0, 10, 11],
    }


def test_random_control_reproducibility_and_exact_matching():
    real = real_mapping()
    first = generate_random_structure_control(real, MVP_ROI_NAMES, seed=42)
    repeat = generate_random_structure_control(real, MVP_ROI_NAMES, seed=42)
    different = generate_random_structure_control(real, MVP_ROI_NAMES, seed=43)
    assert first == repeat
    assert first != different
    assert {v for values in first.values() for v in values} == {
        v for values in real.values() for v in values
    }
    assert [len(first[name]) for name in MVP_ROI_NAMES] == [
        len(real[name]) for name in MVP_ROI_NAMES
    ]
    assert (
        roi_overlap_matrix(first, MVP_ROI_NAMES)
        == roi_overlap_matrix(real, MVP_ROI_NAMES)
    ).all()
