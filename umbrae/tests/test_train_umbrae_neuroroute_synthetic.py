import io
import json
import re
import sys
import tarfile
import tempfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_umbrae_neuroroute import (
    SHIKRA_IMAGE_END_ID,
    SHIKRA_IMAGE_START_ID,
    replace_shikra_patch_embeddings,
    run_training,
)


ROI_NAMES = ["V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA"]


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def _encode(self, text):
        pieces = re.split(r"(<im_start>|<im_patch>|<im_end>)", text)
        ids = []
        for piece in pieces:
            if piece == "<im_start>":
                ids.append(SHIKRA_IMAGE_START_ID)
            elif piece == "<im_patch>":
                ids.append(32000)
            elif piece == "<im_end>":
                ids.append(SHIKRA_IMAGE_END_ID)
            else:
                ids.extend(2 + ord(char) % 29 for char in piece)
        return ids

    def __call__(
        self,
        texts,
        padding=True,
        truncation=False,
        max_length=None,
        return_tensors="pt",
    ):
        rows = [self._encode(text) for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        width = max(map(len, rows))
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["synthetic caption" for _ in range(token_ids.shape[0])]


class TinyCausalLM(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(32010, hidden_size)
        self.head = nn.Linear(hidden_size, 64)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        logits = self.head(inputs_embeds)
        safe_labels = labels.clone()
        valid = safe_labels != -100
        safe_labels[valid] %= logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            safe_labels.reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, inputs_embeds, max_new_tokens=3, **kwargs):
        return torch.full(
            (inputs_embeds.shape[0], min(max_new_tokens, 3)),
            2,
            dtype=torch.long,
            device=inputs_embeds.device,
        )


class TinyUMBRAE(nn.Module):
    output_dim = 16
    num_visual_tokens = 256
    encoder_kind = "TinyUMBRAE"

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 16)

    def forward(self, fmri):
        token = self.projection(fmri).unsqueeze(1)
        return token.expand(-1, 256, -1)


class TinyNeuroRoute(nn.Module):
    output_dim = 16
    stage1_config = {
        "router_type": "soft",
        "selected_clip_layers": [4, 8, 12, 16, 20, 24],
    }

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 16)

    def forward(self, fmri):
        token = self.projection(fmri).unsqueeze(1)
        return {
            "raw_roi_tokens": token.expand(-1, 8, -1),
            "projected_roi_tokens": token.expand(-1, 8, -1),
        }


def npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def add_bytes(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def make_tar(path):
    with tarfile.open(path, "w") as archive:
        for index in range(2):
            prefix = f"sample{index:012d}"
            fmri = np.random.default_rng(index).normal(
                size=(3, 16)
            ).astype(np.float16)
            add_bytes(
                archive,
                f"{prefix}.nsdgeneral.npy",
                npy_bytes(fmri),
            )
            add_bytes(
                archive,
                f"{prefix}.num_uniques.npy",
                npy_bytes(np.asarray([3], dtype=np.int64)),
            )
            add_bytes(
                archive,
                f"{prefix}.caption.txt",
                f"caption {index}".encode(),
            )


def make_mapping(path):
    path.write_text(
        json.dumps(
            {
                "roi_mapping_is_real": True,
                "voxel_order_verified": True,
                "roi_names": ROI_NAMES,
                "rois": {
                    name: {
                        "indices": [2 * index, 2 * index + 1],
                        "resolved": True,
                    }
                    for index, name in enumerate(ROI_NAMES)
                },
            }
        )
    )


def make_args(root, train_tar, val_tar, mapping, captions, output):
    return Namespace(
        subject="subj01",
        train_tar=[str(train_tar)],
        val_tar=[str(val_tar)],
        captions_json=str(captions),
        roi_indices_path=str(mapping),
        umbrae_checkpoint=str(root / "unused_umbrae.pt"),
        neuroroute_checkpoint=str(root / "unused_neuroroute.pt"),
        global_l24_checkpoint=None,
        umbrae_mm_projector=None,
        fusion_mode="umbrae_plus_soft",
        fusion_type="perceiver_resampler",
        bridge_type="shikra_patch",
        mllm_model_path="unused",
        mllm_dim=16,
        fusion_dim=16,
        num_visual_tokens=256,
        roi_token_expansion=2,
        freeze_umbrae=True,
        freeze_neuroroute=True,
        freeze_mllm=True,
        train_adapter_only=True,
        oracle_image_token_mode=False,
        prompt="Describe <image>",
        max_text_length=800,
        max_new_tokens=3,
        batch_size=1,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        device="cpu",
        output_dir=str(output),
        debug_max_steps=1,
        num_workers=0,
        seed=42,
    )


def main():
    ids = torch.tensor(
        [[8, SHIKRA_IMAGE_START_ID, 32000, 32000, SHIKRA_IMAGE_END_ID, 9]]
    )
    embeddings = torch.randn(1, 6, 4)
    patches = torch.randn(1, 2, 4)
    replaced = replace_shikra_patch_embeddings(ids, embeddings, patches)
    assert torch.equal(replaced[:, 2:4], patches)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train_tar = root / "train.tar"
        val_tar = root / "val.tar"
        mapping = root / "mapping.json"
        captions = root / "captions.json"
        output = root / "output"
        make_tar(train_tar)
        make_tar(val_tar)
        make_mapping(mapping)
        captions.write_text("{}")
        result = run_training(
            make_args(
                root,
                train_tar,
                val_tar,
                mapping,
                captions,
                output,
            ),
            mllm=TinyCausalLM(),
            tokenizer=TinyTokenizer(),
            umbrae_encoder=TinyUMBRAE(),
            neuroroute_encoder=TinyNeuroRoute(),
        )
        assert result["metrics"]["uses_image_clip_tokens_at_eval"] is False
        for filename in (
            "checkpoint_best.pt",
            "checkpoint_last.pt",
            "metrics.json",
            "predictions.jsonl",
            "generated_captions.jsonl",
            "routing_summary.json",
            "adapter_config.json",
        ):
            assert (output / filename).is_file(), filename
        config = json.loads((output / "adapter_config.json").read_text())
        assert config["bridge_type"] == "shikra_patch"
        assert config["uses_image_clip_tokens_at_eval"] is False
        assert config["oracle_image_token_mode"] is False
    print("Synthetic UMBRAE-NeuroRoute train/eval step passed")


if __name__ == "__main__":
    main()
