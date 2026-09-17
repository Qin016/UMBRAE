import io
import json
import sys
import tarfile
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionConfig, CLIPVisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_stage1_retrieval import (
    compute_positive_ranks,
    compute_ranks,
    evaluate,
    retrieval_metrics,
)
from scripts.train_stage1_routing import Stage1RoutingModel, compute_losses


ROI_NAMES = ["V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA"]


def tiny_clip():
    return CLIPVisionModel(CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=16,
        patch_size=4,
    ))


def npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def image_bytes(value):
    array = np.full((16, 16, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG")
    return buffer.getvalue()


def add_bytes(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def make_tar(path):
    coco_ids = [10, 10, 20, 30]
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
                archive, f"{prefix}.jpg", image_bytes(30 + index * 20)
            )
            add_bytes(
                archive,
                f"{prefix}.coco73k.npy",
                npy_bytes(
                    np.asarray([coco_ids[index]], dtype=np.int64)
                ),
            )


def make_mapping(path):
    rois = {
        name: {
            "indices": [2 * index, 2 * index + 1],
            "num_voxels": 2,
            "resolved": True,
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


def make_checkpoint(
    path,
    roi_indices,
    clip_model,
    pooling="mean",
    contrastive_weight=0.0,
):
    model = Stage1RoutingModel(
        roi_names=ROI_NAMES,
        roi_indices=roi_indices,
        selected_layers=[1, 2],
        feature_dim=16,
        router_type="soft",
        clip_model_name_or_path="unused",
        router_hidden_dim=8,
        router_temperature=1.0,
        router_topk=0,
        hard_intermediate_layers=[1],
        hard_final_layers=[2],
        seed=42,
        use_brain_clip_projector=True,
        projector_type="mlp",
        projector_hidden_dim=16,
        projector_dropout=0.1,
        contrastive_brain_pooling=pooling,
        clip_model=clip_model,
    )
    torch.save({
        "model": model.state_dict(),
        "config": {
            "router_type": "soft",
            "selected_clip_layers": [1, 2],
            "clip_layer_target_dim": 16,
            "roi_token_dim": 16,
            "clip_model_name_or_path": "unused",
            "router_hidden_dim": 8,
            "router_temperature": 1.0,
            "router_topk": 0,
            "hard_intermediate_layers": [1],
            "hard_final_layers": [2],
            "seed": 42,
            "use_brain_clip_projector": True,
            "projector_type": "mlp",
            "projector_hidden_dim": 16,
            "projector_dropout": 0.1,
            "fmri_repeat_mode": "mean",
            "contrastive_brain_pooling": pooling,
            "contrastive_loss_weight": contrastive_weight,
        },
    }, path)
    np.save(
        path.parent / "val_routing_weights_mean.npy",
        np.full((8, 2), 0.5, dtype=np.float32),
    )


def make_args(tar_path, mapping_path, checkpoint, output):
    return Namespace(
        subject="subj01",
        checkpoint=str(checkpoint),
        router_type="soft",
        single_router_layer=24,
        val_tar=[str(tar_path)],
        roi_indices_path=str(mapping_path),
        selected_clip_layers=[1, 2],
        batch_size=2,
        max_samples=3,
        brain_pooling="mean",
        image_target="routed",
        diagonal_only=False,
        device="cpu",
        num_workers=0,
        output_dir=str(output),
        seed=42,
    )


def test_rank_metrics():
    similarity = np.asarray([
        [0.9, 0.1, 0.0],
        [0.8, 0.7, 0.2],
        [0.1, 0.8, 0.9],
    ])
    ranks = compute_ranks(similarity)
    assert np.array_equal(ranks, [1, 2, 1])
    metrics = retrieval_metrics(ranks)
    assert metrics["recall_at_1"] == 2 / 3
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["median_rank"] == 1.0
    assert metrics["mean_rank"] == 4 / 3

    duplicate_similarity = np.asarray([
        [0.1, 0.9, 0.8],
        [0.8, 0.1, 0.9],
        [0.9, 0.8, 0.1],
    ])
    ids = np.asarray([10, 10, 20])
    positive_mask = ids[:, None] == ids[None, :]
    multi_ranks = compute_positive_ranks(
        duplicate_similarity, positive_mask
    )
    diagonal_ranks = compute_ranks(duplicate_similarity)
    assert np.array_equal(multi_ranks, [1, 2, 3])
    assert np.array_equal(diagonal_ranks, [3, 3, 3])
    assert retrieval_metrics(multi_ranks)["recall_at_1"] == 1 / 3
    assert retrieval_metrics(diagonal_ranks)["recall_at_1"] == 0.0


def test_contrastive_training_branch():
    roi_indices = {
        name: [2 * index, 2 * index + 1]
        for index, name in enumerate(ROI_NAMES)
    }
    model = Stage1RoutingModel(
        roi_names=ROI_NAMES,
        roi_indices=roi_indices,
        selected_layers=[1, 2],
        feature_dim=16,
        router_type="soft",
        clip_model_name_or_path="unused",
        router_hidden_dim=8,
        router_temperature=1.0,
        router_topk=0,
        hard_intermediate_layers=[1],
        hard_final_layers=[2],
        seed=42,
        contrastive_brain_pooling="attention",
        contrastive_image_target="routed",
        clip_model=tiny_clip(),
    )
    output = model(torch.randn(3, 16), torch.rand(3, 3, 16, 16))
    args = Namespace(
        mse_weight=1.0,
        cos_weight=1.0,
        router_entropy_weight=0.0,
        router_balance_weight=0.0,
        router_smoothness_weight=0.0,
        contrastive_loss_weight=0.5,
        contrastive_temperature=0.07,
    )
    losses = compute_losses(output, args)
    assert torch.isfinite(losses["contrastive"])
    losses["total"].backward()
    assert model.retrieval_pooler.attention_score.weight.grad is not None


def main():
    test_rank_metrics()
    test_contrastive_training_branch()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tar_path = root / "val.tar"
        mapping_path = root / "mapping.json"
        checkpoint = root / "checkpoint.pt"
        output = root / "retrieval"
        make_tar(tar_path)
        indices = make_mapping(mapping_path)
        clip = tiny_clip()
        make_checkpoint(checkpoint, indices, clip)
        result = evaluate(
            make_args(tar_path, mapping_path, checkpoint, output),
            clip_model=tiny_clip(),
        )
        expected = [
            "metrics_retrieval.json",
            "similarity.npy",
            "brain_embeddings.npy",
            "image_embeddings.npy",
            "ranks_brain_to_image.npy",
            "ranks_image_to_brain.npy",
            "brain_ids.npy",
            "image_ids.npy",
            "positive_mask_stats.json",
            "config.json",
        ]
        assert all((output / name).is_file() for name in expected)
        assert result["similarity"].shape == (3, 3)
        config = json.loads((output / "config.json").read_text())
        metrics = json.loads(
            (output / "metrics_retrieval.json").read_text()
        )
        assert config["image_features_used_in_brain_query"] is False
        assert metrics["retrieval_protocol"] == "multi_positive_coco73k"
        assert metrics["num_unique_coco_ids"] == 2
        assert metrics["average_positives_per_brain_query"] > 1.0
        assert (
            metrics["leakage_protocol"][
                "image_features_used_in_brain_query"
            ]
            is False
        )
        attention_checkpoint = root / "attention_checkpoint.pt"
        attention_output = root / "attention_as_mean"
        make_checkpoint(
            attention_checkpoint,
            indices,
            tiny_clip(),
            pooling="attention",
            contrastive_weight=0.1,
        )
        attention_as_mean = make_args(
            tar_path,
            mapping_path,
            attention_checkpoint,
            attention_output,
        )
        attention_as_mean.brain_pooling = "mean"
        mean_result = evaluate(
            attention_as_mean,
            clip_model=tiny_clip(),
        )
        assert mean_result["similarity"].shape == (3, 3)
        print("Stage-1 retrieval metric checks passed")
        print("Stage-1 synthetic retrieval evaluation passed")
        print("Leakage rule: image CLIP features are gallery-only")


if __name__ == "__main__":
    main()
