import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.roi_tokenizer import ROITokenizer


def run_mode(tokenizer_type: str) -> None:
    roi_names = ["V1", "V2", "V3", "hV4", "LOC", "FFA", "PPA"]
    roi_indices = {
        "V1": [0, 1, 2, 3],
        "V2": [4, 5, 6],
        "V3": [7, 8],
        "hV4": [9, 10, 11, 12, 13],
        "LOC": [14, 15, 16],
        "FFA": [17, 18],
        "PPA": [19, 20, 21, 22],
    }
    tokenizer = ROITokenizer(
        roi_names=roi_names,
        roi_indices=roi_indices,
        token_dim=32,
        tokenizer_type=tokenizer_type,
        use_roi_embeddings=True,
        use_subject_embeddings=True,
        num_subjects=8,
    )

    fmri = torch.randn(2, 3, 32)
    outputs = tokenizer(fmri, subject_ids=torch.tensor([0, 6]))
    assert outputs["roi_tokens"].shape == (2, 7, 32)
    assert outputs["roi_names"] == roi_names
    outputs["roi_tokens"].mean().backward()
    assert any(parameter.grad is not None for parameter in tokenizer.parameters())
    print(tokenizer_type, tuple(outputs["roi_tokens"].shape))


def test_mask_input() -> None:
    masks = torch.zeros(2, 10, dtype=torch.bool)
    masks[0, :4] = True
    masks[1, 4:] = True
    tokenizer = ROITokenizer(
        roi_names=["early_visual", "high_level_visual"],
        roi_masks=masks,
        token_dim=16,
        tokenizer_type="shared_mlp",
        use_roi_embeddings=False,
    )
    outputs = tokenizer(torch.randn(3, 10))
    assert outputs["roi_tokens"].shape == (3, 2, 16)
    assert outputs["roi_names"] == ["early_visual", "high_level_visual"]
    print("mask_input", tuple(outputs["roi_tokens"].shape))


if __name__ == "__main__":
    run_mode("shared_mlp")
    run_mode("roi_specific_mlp")
    test_mask_input()
    print("ROITokenizer smoke test passed")
