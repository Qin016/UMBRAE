import io
import json
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

UMBRAE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UMBRAE_ROOT))

from models.fgw_stage2_prior import (  # noqa: E402
    CLIP_LAYER_ORDER,
    FROZEN_PLAN_SPECS,
    ROI_ORDER,
    apply_row_derangement,
    load_frozen_plan,
    load_preregistered_derangement,
    row_normalize_transport,
    tempered_attention_bias,
    validate_plan,
)
from models.umbrae_neuroroute_adapter import (  # noqa: E402
    UMBRAENeuroRouteAdapter,
    build_source_attention_bias,
)
from scripts.train_stage2_mllm_neuroroute import Stage2TarDataset  # noqa: E402


PLAN_ROOT = UMBRAE_ROOT / "fgw_outputs"
NULL_PATH = PLAN_ROOT / "stage2_fgw_preregistered_null.json"


@contextmanager
def _raises(error_type, match):
    try:
        yield
    except error_type as error:
        assert match in str(error), (match, str(error))
    else:
        raise AssertionError(f"Expected {error_type.__name__}: {match}")


def _plan(subject="subj01"):
    return load_frozen_plan(subject, PLAN_ROOT)[0]


def _adapter(mode, *, gamma=1.0, plan=None, depth=2, mllm_dim=64):
    return UMBRAENeuroRouteAdapter(
        umbrae_dim=32,
        neuroroute_dim=32,
        mllm_dim=mllm_dim,
        fusion_dim=32,
        num_visual_tokens=256,
        fusion_mode=mode,
        fusion_type="perceiver_resampler",
        roi_token_expansion=1,
        num_attention_heads=4,
        perceiver_depth=depth,
        num_routing_layers=6,
        dropout=0.0,
        fgw_transport_plan=plan,
        fgw_gamma=gamma,
    )


def _copy_parameters(source, target):
    target_parameters = dict(target.named_parameters())
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            target_parameters[name].copy_(parameter)


def test_frozen_plan_shape_order_and_hash_validation():
    for subject, spec in FROZEN_PLAN_SPECS.items():
        plan, provenance = load_frozen_plan(subject, PLAN_ROOT)
        assert plan.shape == (8, 6)
        assert provenance["sha256"] == spec.sha256
        validate_plan(plan.numpy(), roi_order=ROI_ORDER, clip_layer_order=CLIP_LAYER_ORDER)
    with _raises(ValueError, "ROI order"):
        validate_plan(_plan().numpy(), roi_order=tuple(reversed(ROI_ORDER)))
    with _raises(ValueError, "permits only"):
        load_frozen_plan("subj07", PLAN_ROOT)


def test_row_normalization_and_tempered_prior_invariants():
    correspondence = row_normalize_transport(_plan())
    torch.testing.assert_close(
        correspondence.sum(-1),
        torch.ones(8, dtype=correspondence.dtype),
    )
    uniform = torch.full((8, 6), 1.0 / 6.0, dtype=torch.float64)
    torch.testing.assert_close(
        tempered_attention_bias(uniform, 1.0), torch.zeros_like(uniform),
        atol=1e-12, rtol=0.0,
    )
    torch.testing.assert_close(
        tempered_attention_bias(correspondence, 0.0),
        torch.zeros_like(correspondence), atol=0.0, rtol=0.0,
    )
    bias = tempered_attention_bias(correspondence, 0.5)
    torch.testing.assert_close(
        bias.exp().sum(-1), torch.full((8,), 6.0, dtype=bias.dtype),
        atol=1e-10, rtol=0.0,
    )


def test_bias_key_placement_and_actual_attention_broadcasting():
    roi_bias = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    key_bias = build_source_attention_bias(
        num_source_keys=304,
        roi_key_start=256,
        roi_layer_bias=roi_bias,
    )
    assert torch.count_nonzero(key_bias[:256]) == 0
    torch.testing.assert_close(key_bias[256:], roi_bias.reshape(-1))
    # The audited strict adapter has no latent/self keys.  Emulate appended
    # latent slots to prove the placement helper leaves every other key zero.
    with_latents = build_source_attention_bias(
        num_source_keys=309,
        roi_key_start=256,
        roi_layer_bias=roi_bias,
    )
    assert torch.count_nonzero(with_latents[304:]) == 0
    adapter = _adapter("fgw_prior", plan=_plan())
    block = adapter.perceiver_blocks[0]
    queries = torch.randn(2, 256, 32)
    source = torch.randn(2, 304, 32)
    raw = block.raw_content_logits(queries, source)
    assert raw.shape == (2, 4, 256, 304)
    broadcast = raw + key_bias[None, None, None, :]
    assert broadcast.shape == raw.shape
    torch.testing.assert_close(
        broadcast[..., :256], raw[..., :256], atol=0.0, rtol=0.0
    )


def test_preregistered_row_shuffle_preserves_null_invariants():
    plan = _plan()
    permutation = load_preregistered_derangement(NULL_PATH)
    assert all(index != target for index, target in enumerate(permutation))
    shuffled = apply_row_derangement(plan, permutation)
    torch.testing.assert_close(shuffled.sum(0), plan.sum(0))
    torch.testing.assert_close(shuffled.sum(), plan.sum())
    entropy = lambda value: -(value * (value + 1e-12).log()).sum(-1).sort().values
    torch.testing.assert_close(entropy(shuffled), entropy(plan))


def test_gamma_zero_forward_matches_zero_bias_forward():
    plan = _plan().float()
    left = _adapter("fgw_prior", gamma=0.0, plan=plan).eval()
    right = _adapter("uniform_prior", gamma=1.0).eval()
    right.load_state_dict(left.state_dict(), strict=False)
    # Restore the uniform condition's buffers after parameter copying.
    right.fgw_transport_plan.fill_(1.0 / 48.0)
    right.fgw_attention_bias.zero_()
    umbrae = torch.randn(1, 256, 32)
    roi = torch.randn(1, 8, 32)
    with torch.no_grad():
        actual = left(umbrae, roi)["visual_tokens"]
        expected = right(umbrae, roi)["visual_tokens"]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_new_uniform_matches_legacy_48_token_uniform_architecture():
    old = _adapter("umbrae_plus_uniform_residual_soft").eval()
    new = _adapter("uniform_prior").eval()
    _copy_parameters(old, new)
    new.fgw_transport_plan.fill_(1.0 / 48.0)
    new.fgw_attention_bias.zero_()
    umbrae = torch.randn(1, 256, 32)
    roi = torch.randn(1, 8, 32)
    routing = torch.full((1, 8, 6), 1.0 / 6.0)
    with torch.no_grad():
        old_output = old(umbrae, roi, routing_weights=routing)["visual_tokens"]
        new_output = new(umbrae, roi)["visual_tokens"]
    torch.testing.assert_close(old_output, new_output, atol=1e-6, rtol=1e-6)


def test_strict_output_shape_first_block_only_and_frozen_transport():
    adapter = _adapter(
        "fgw_prior", plan=_plan().float(), depth=2, mllm_dim=4096
    ).eval()
    calls = []
    handles = []
    for index, block in enumerate(adapter.perceiver_blocks):
        handles.append(
            block.attention.register_forward_pre_hook(
                lambda module, args, kwargs, index=index: calls.append(
                    (index, kwargs.get("attn_mask"))
                ),
                with_kwargs=True,
            )
        )
    output = adapter(torch.randn(1, 256, 32), torch.randn(1, 8, 32))
    for handle in handles:
        handle.remove()
    assert output["visual_tokens"].shape == (1, 256, 4096)
    assert output["diagnostics"]["roi_layer_key_start"] == 256
    assert output["diagnostics"]["roi_layer_key_end"] == 304
    assert calls[0][1] is not None and calls[1][1] is None
    assert not adapter.fgw_transport_plan.requires_grad
    output["visual_tokens"].sum().backward()
    assert adapter.fgw_transport_plan.grad is None


def test_stage2_dataset_exposes_no_image_or_clip_feature(tmp_path=None):
    if tmp_path is None:
        temporary = tempfile.TemporaryDirectory()
        tmp_path = Path(temporary.name)
    tar_path = Path(tmp_path) / "sample.tar"
    fmri_buffer = io.BytesIO()
    np.save(fmri_buffer, np.ones(12, dtype=np.float32), allow_pickle=False)
    caption = b"a test caption"
    with tarfile.open(tar_path, "w") as archive:
        fmri_info = tarfile.TarInfo("000001.nsdgeneral.npy")
        fmri_info.size = len(fmri_buffer.getvalue())
        archive.addfile(fmri_info, io.BytesIO(fmri_buffer.getvalue()))
        caption_info = tarfile.TarInfo("000001.caption.txt")
        caption_info.size = len(caption)
        archive.addfile(caption_info, io.BytesIO(caption))
    sample = Stage2TarDataset([str(tar_path)], {})[0]
    assert set(sample) == {"fmri", "caption", "sample_id", "coco_id"}
    assert not any("image" in key or "clip" in key for key in sample)
    if "temporary" in locals():
        temporary.cleanup()


def test_null_registration_is_stable_and_self_describing():
    payload = json.loads(NULL_PATH.read_text())
    assert payload["created_before_stage2_results"] is True
    assert payload["permutation_seed"] == 62001
    assert payload["shared_across_subjects"] == ["subj01", "subj02", "subj05"]


def main():
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} FGW Stage-2 attention-prior tests passed")


if __name__ == "__main__":
    main()
