import numpy as np

from models.p8_diagnostics import box_iou_vector, interpolate, paired_summary


def test_interpolation_endpoints_are_exact():
    base = np.arange(12, dtype=np.float16).reshape(3, 4)
    variant = base + np.float16(2)
    assert np.array_equal(interpolate(base, variant, 0.0), base)
    assert np.array_equal(interpolate(base, variant, 1.0), variant)


def test_box_iou_and_paired_summary():
    values = box_iou_vector([[0, 0, 1, 1], None], [[0, 0, 1, 1], [0, 0, 1, 1]])
    assert np.allclose(values, [1, 0])
    result = paired_summary(np.array([-1.0, 1.0]))
    assert result["mean_delta"] == 0
    assert result["probability_delta_gt_0"] == 0.5
