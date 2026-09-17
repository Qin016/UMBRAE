import pytest

from models.downstream_brain_encoder import MODES, DownstreamBrainEncoder


def test_downstream_modes_are_locked():
    assert MODES == ("umbrae", "lora", "full_real", "full_random")
    with pytest.raises(ValueError, match="Unknown downstream mode"):
        DownstreamBrainEncoder("real_plus_random", {})
