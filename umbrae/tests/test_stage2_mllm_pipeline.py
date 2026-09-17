import io
import json
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

from models.brain_clip_projector import BrainToCLIPProjector
from models.roi_tokenizer import ROITokenizer
from scripts.train_stage2_mllm_neuroroute import (
    assert_no_image_feature_leakage,
    build_global_l24_path,
    run_training,
)


ROI_NAMES = ["V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA"]


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ):
        rows = []
        for text in texts:
            ids = [2 + (ord(char) % 29) for char in text][:max_length]
            rows.append(ids or [1])
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["synthetic caption" for _ in range(token_ids.shape[0])]


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        logits = self.head(inputs_embeds)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, inputs_embeds, max_new_tokens=8, **kwargs):
        return torch.full(
            (inputs_embeds.shape[0], min(max_new_tokens, 3)),
            2,
            dtype=torch.long,
            device=inputs_embeds.device,
        )


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
        for index in range(4):
            prefix = f"sample{index:012d}"
            fmri = np.random.default_rng(index).normal(
                size=(3, 16)
            ).astype(np.float16)
            add_bytes(
                archive, f"{prefix}.nsdgeneral.npy", npy_bytes(fmri)
            )
            add_bytes(
                archive,
                f"{prefix}.num_uniques.npy",
                npy_bytes(np.asarray([3], dtype=np.int64)),
            )
            add_bytes(
                archive,
                f"{prefix}.caption.txt",
                f"caption number {index}".encode(),
            )


def make_mapping(path):
    rois = {
        name: {
            "indices": [2 * index, 2 * index + 1],
            "resolved": True,
            "num_voxels": 2,
        }
        for index, name in enumerate(ROI_NAMES)
    }
    path.write_text(json.dumps({
        "roi_mapping_is_real": True,
        "voxel_order_verified": True,
        "roi_names": ROI_NAMES,
        "rois": rois,
    }))
    return {name: rois[name]["indices"] for name in ROI_NAMES}


def make_stage1_checkpoint(
    path, roi_indices, router_type="soft", single_router_layer=None
):
    tokenizer = ROITokenizer(ROI_NAMES, 16, roi_indices=roi_indices)
    projector = BrainToCLIPProjector(16, 16, hidden_dim=16)
    state = {
        **{
            f"roi_tokenizer.{key}": value
            for key, value in tokenizer.state_dict().items()
        },
        **{
            f"brain_clip_projector.{key}": value
            for key, value in projector.state_dict().items()
        },
    }
    torch.save({
        "model": state,
        "config": {
            "roi_token_dim": 16,
            "use_brain_clip_projector": True,
            "projector_type": "mlp",
            "projector_hidden_dim": 16,
            "projector_dropout": 0.1,
            "selected_clip_layers": [4, 8, 12, 16, 20, 24],
            "router_type": router_type,
            "single_router_layer": single_router_layer,
        },
    }, path)
    routing = np.full((8, 6), 1 / 6, dtype=np.float32)
    np.save(path.parent / "val_routing_weights_mean.npy", routing)


def make_args(
    train_tar,
    val_tar,
    mapping,
    checkpoint,
    output,
    global_l24_checkpoint=None,
):
    return Namespace(
        subject="subj01",
        train_tar=[str(train_tar)],
        val_tar=[str(val_tar)],
        captions_json=None,
        roi_indices_path=str(mapping),
        stage1_checkpoint=str(checkpoint),
        global_l24_checkpoint=(
            str(global_l24_checkpoint)
            if global_l24_checkpoint is not None
            else None
        ),
        router_type="soft",
        single_router_layer=24,
        fusion_mode="concat",
        mllm_model_path="unused",
        mllm_dim=16,
        adapter_hidden_dim=16,
        freeze_stage1=True,
        freeze_mllm=True,
        train_adapter_only=True,
        caption_loss_weight=1.0,
        retrieval_loss_weight=0.0,
        grounding_loss_weight=0.0,
        oracle_image_token_mode=False,
        prompt="Describe:",
        max_text_length=32,
        max_new_tokens=4,
        batch_size=2,
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
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train_tar, val_tar = root / "train.tar", root / "val.tar"
        mapping, checkpoint = root / "mapping.json", root / "checkpoint.pt"
        global_checkpoint = root / "single_l24_checkpoint.pt"
        output = root / "output"
        make_tar(train_tar)
        make_tar(val_tar)
        indices = make_mapping(mapping)
        make_stage1_checkpoint(checkpoint, indices)
        make_stage1_checkpoint(
            global_checkpoint,
            indices,
            router_type="single",
            single_router_layer=24,
        )
        result = run_training(
            make_args(
                train_tar,
                val_tar,
                mapping,
                checkpoint,
                output,
                global_l24_checkpoint=global_checkpoint,
            ),
            mllm=TinyCausalLM(),
            tokenizer=TinyTokenizer(),
        )
        expected = [
            "metrics.json",
            "predictions.jsonl",
            "generated_captions.jsonl",
            "routing_summary.json",
            "adapter_config.json",
            "checkpoint_best.pt",
            "checkpoint_last.pt",
        ]
        assert all((output / name).is_file() for name in expected)
        config = json.loads((output / "adapter_config.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())
        assert config["ground_truth_image_features_used"] is False
        assert config["fMRI_only_validation"] is True
        required_config_fields = {
            "fusion_mode",
            "stage1_checkpoint",
            "global_l24_checkpoint",
            "global_l24_source",
            "roi_prefix_source",
            "uses_image_clip_tokens_at_eval",
            "oracle_image_token_mode",
            "mllm_bridge_type",
            "freeze_stage1",
            "freeze_mllm",
            "train_adapter_only",
        }
        assert required_config_fields.issubset(config)
        assert (
            config["global_l24_source"]
            == "stage1_single_l24_projected_roi_mean"
        )
        assert config["uses_image_clip_tokens_at_eval"] is False
        assert config["mllm_bridge_type"] == "generic_inputs_embeds_prefix"
        assert metrics["ground_truth_image_features_used"] is False
        assert result["model"].adapter.fusion_mode == "concat"
        assert result["model"].global_stage1_encoder is not None
        assert not any(
            parameter.requires_grad
            for parameter in result[
                "model"
            ].global_stage1_encoder.parameters()
        )
        fallback_projector, fallback_encoder, fallback_source = (
            build_global_l24_path(
                None,
                result["model"].stage1_encoder,
                ROI_NAMES,
                indices,
                "concat",
            )
        )
        assert fallback_projector is not None
        assert fallback_encoder is None
        assert fallback_source.startswith("stage2_initialized_")
        routed_projector, routed_encoder, routed_source = (
            build_global_l24_path(
                None,
                result["model"].stage1_encoder,
                ROI_NAMES,
                indices,
                "routed_only",
            )
        )
        assert routed_projector is None and routed_encoder is None
        assert routed_source == "not_used_for_routed_only"
        checkpoint_payload = torch.load(
            output / "checkpoint_best.pt", map_location="cpu"
        )
        assert all(
            key.startswith("adapter.")
            or key.startswith("global_projector.")
            for key in checkpoint_payload["model"]
        )
        try:
            assert_no_image_feature_leakage(
                True, "validation", oracle_image_token_mode=False
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("Leakage check did not reject image features")
        print("Stage-2 synthetic train/eval step passed")
        print("Stage-2 artifacts and leakage checks passed")


if __name__ == "__main__":
    main()
