import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from transformers import CLIPVisionConfig, CLIPVisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import BrainEncoder
from models.clip_patch_teacher import FixedCLIPPatchTeacher


def make_tiny_clip():
    return CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            image_size=32,
            patch_size=2,
            attention_dropout=0.0,
        )
    ).eval()


def make_old_teacher(clip):
    old = BrainEncoder.__new__(BrainEncoder)
    nn.Module.__init__(old)
    old.clip = clip
    old.clip_size = (32, 32)
    old.preprocess = transforms.Compose(
        [
            transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop((32, 32)),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return old.eval()


def test_fixed_clip_patch_teacher_equivalence():
    torch.manual_seed(4)
    clip = make_tiny_clip()
    old = make_old_teacher(clip)
    new = FixedCLIPPatchTeacher(clip_model=clip).eval()
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        expected = old.encode_image(images, "image")
        actual = new(images)
    error = (expected - actual).abs()
    max_abs_error = float(error.max())
    mean_abs_error = float(error.mean())
    print({"max_abs_error": max_abs_error, "mean_abs_error": mean_abs_error})
    assert expected.shape == actual.shape == (2, 256, 32)
    assert max_abs_error <= 1e-6
    assert mean_abs_error <= 1e-7
    assert not any(parameter.requires_grad for parameter in new.parameters())
