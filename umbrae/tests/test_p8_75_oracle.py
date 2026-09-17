from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.p8_75_exact_clip_token_oracle import (
    BRAIN_MODES,
    MODES,
    alpha_quarter,
    check_cache,
    rsa_any,
)


def test_exact_oracle_modes_are_locked():
    assert MODES == (
        "umbrae", "lora", "full_real", "full_real_alpha_025",
        "exact_clip_token_oracle",
    )
    assert MODES[:-1] == BRAIN_MODES


def test_alpha_quarter_is_posthoc_residual():
    base = np.array([0.0, 4.0], dtype=np.float32)
    full = np.array([4.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(alpha_quarter(base, full), [1.0, 3.0])


def test_cache_shape_guard_accepts_shape_only_object():
    check_cache(SimpleNamespace(shape=(982, 256, 1024)), "dummy")
    with pytest.raises(ValueError, match="expected"):
        check_cache(SimpleNamespace(shape=(982, 257, 1024)), "dummy")


def test_rsa_identity_endpoint_is_one():
    values = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    result = rsa_any(values, values)
    assert result["spearman_rsa"] == pytest.approx(1.0)
    assert result["pearson_rsa"] == pytest.approx(1.0)


def test_oracle_driver_contains_no_training_call():
    source = (Path(__file__).parents[1] / "scripts" / "p8_75_exact_clip_token_oracle.py").read_text()
    assert ".backward(" not in source
    assert "torch.optim" not in source
