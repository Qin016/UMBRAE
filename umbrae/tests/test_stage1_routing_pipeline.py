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

from scripts.train_stage1_routing import Stage1RoutingModel, run_training
from scripts.plot_routing_dynamics import load_jsonl, save_plots


ROI_NAMES = ["V1", "V2", "V3", "hV4", "FFA", "EBA", "PPA", "OPA"]


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def image_bytes(value: int) -> bytes:
    array = np.full((16, 16, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG")
    return buffer.getvalue()


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def make_tar(path: Path, count: int = 4) -> None:
    with tarfile.open(path, "w") as archive:
        for index in range(count):
            prefix = f"sample{index:012d}"
            generator = np.random.default_rng(index)
            fmri = generator.normal(size=(3, 16)).astype(np.float16)
            add_bytes(archive, f"{prefix}.nsdgeneral.npy", npy_bytes(fmri))
            add_bytes(
                archive,
                f"{prefix}.num_uniques.npy",
                npy_bytes(np.asarray([3], dtype=np.int64)),
            )
            add_bytes(archive, f"{prefix}.jpg", image_bytes(40 + index))


def make_mapping(path: Path) -> dict:
    rois = {}
    for index, name in enumerate(ROI_NAMES):
        indices = [2 * index, 2 * index + 1]
        rois[name] = {
            "indices": indices,
            "source_labels": [],
            "num_voxels": len(indices),
            "resolved": True,
        }
    mapping = {
        "subject": "subj01",
        "roi_mapping_is_real": True,
        "voxel_order_verified": True,
        "roi_names": ROI_NAMES,
        "rois": rois,
    }
    path.write_text(json.dumps(mapping))
    return mapping


def tiny_clip() -> CLIPVisionModel:
    return CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=16,
            patch_size=4,
        )
    )


def check_router_interfaces(mapping: dict) -> None:
    roi_indices = {
        name: mapping["rois"][name]["indices"] for name in ROI_NAMES
    }
    fmri = torch.randn(2, 16)
    images = torch.rand(2, 3, 16, 16)
    for router_type in ("soft", "uniform", "random", "hard"):
        model = Stage1RoutingModel(
            roi_names=ROI_NAMES,
            roi_indices=roi_indices,
            selected_layers=[1, 2],
            feature_dim=16,
            router_type=router_type,
            clip_model_name_or_path="unused",
            router_hidden_dim=8,
            router_temperature=1.0,
            router_topk=0,
            hard_intermediate_layers=[1],
            hard_final_layers=[2],
            seed=42,
            clip_model=tiny_clip(),
        )
        output = model(fmri, images)
        assert output["roi_tokens"].shape == (2, 8, 16)
        assert output["projected_roi_tokens"].shape == (2, 8, 16)
        assert output["routed_targets"].shape == (2, 8, 16)
        assert output["routing_weights"].shape == (2, 8, 2)
        assert torch.allclose(
            output["routing_weights"].sum(dim=-1),
            torch.ones(2, 8),
            atol=1e-6,
        )
        if router_type == "soft":
            (
                output["projected_roi_tokens"] - output["routed_targets"]
            ).square().mean().backward()
            assert model.router.query_projection.weight.grad is not None
            assert any(
                parameter.grad is not None
                for parameter in model.brain_clip_projector.parameters()
            )

    for selected_layer in (1, 2):
        single_model = Stage1RoutingModel(
            roi_names=ROI_NAMES,
            roi_indices=roi_indices,
            selected_layers=[1, 2],
            feature_dim=16,
            router_type="single",
            clip_model_name_or_path="unused",
            router_hidden_dim=8,
            router_temperature=1.0,
            router_topk=0,
            hard_intermediate_layers=[1],
            hard_final_layers=[2],
            seed=42,
            single_router_layer=selected_layer,
            clip_model=tiny_clip(),
        )
        single_output = single_model(fmri, images)
        single_weights = single_output["routing_weights"]
        assert single_weights.shape == (2, 8, 2)
        assert torch.allclose(single_weights.sum(dim=-1), torch.ones(2, 8))
        selected_position = [1, 2].index(selected_layer)
        assert torch.all(single_weights[..., selected_position] == 1)
        assert torch.all((single_weights > 0).sum(dim=-1) == 1)

    backward_compatible = Stage1RoutingModel(
        roi_names=ROI_NAMES,
        roi_indices=roi_indices,
        selected_layers=[1, 2],
        feature_dim=16,
        router_type="soft",
        clip_model_name_or_path="unused",
        router_hidden_dim=8,
        router_temperature=1.0,
        router_topk=0,
        single_router_layer=None,
        hard_intermediate_layers=[1],
        hard_final_layers=[2],
        seed=42,
        use_brain_clip_projector=False,
        clip_model=tiny_clip(),
    )
    compatible_output = backward_compatible(fmri, images)
    assert torch.equal(
        compatible_output["projected_roi_tokens"],
        compatible_output["roi_tokens"],
    )


def test_router_interfaces(tmp_path: Path) -> None:
    mapping = make_mapping(tmp_path / "mapping.json")
    check_router_interfaces(mapping)


def make_args(
    train_tar: Path, val_tar: Path, mapping_path: Path, output_dir: Path
) -> Namespace:
    return Namespace(
        subject="subj01",
        train_tar=[str(train_tar)],
        train_data=None,
        val_tar=[str(val_tar)],
        val_data=None,
        roi_mapping_json=str(mapping_path),
        selected_clip_layers=[1, 2],
        clip_model_name_or_path="unused",
        roi_token_dim=16,
        clip_layer_target_dim=16,
        use_brain_clip_projector=True,
        projector_type="mlp",
        projector_hidden_dim=16,
        projector_dropout=0.1,
        detach_routed_targets=True,
        router_hidden_dim=8,
        router_temperature=1.0,
        router_type="soft",
        router_topk=0,
        hard_intermediate_layers=[1],
        hard_final_layers=[2],
        fmri_repeat_mode="mean",
        batch_size=2,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        mse_weight=1.0,
        cos_weight=1.0,
        router_entropy_weight=0.0,
        router_balance_weight=0.0,
        router_smoothness_weight=0.0,
        device="cpu",
        output_dir=str(output_dir),
        debug_max_steps=1,
        num_workers=0,
        seed=42,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train_tar, val_tar = root / "train.tar", root / "val.tar"
        mapping_path, output_dir = root / "mapping.json", root / "output"
        make_tar(train_tar)
        make_tar(val_tar)
        mapping = make_mapping(mapping_path)
        check_router_interfaces(mapping)
        result = run_training(
            make_args(train_tar, val_tar, mapping_path, output_dir),
            clip_model=tiny_clip(),
        )
        expected_files = [
            "checkpoint_last.pt",
            "checkpoint_best.pt",
            "config.json",
            "val_routing_weights_mean.npy",
            "val_routing_weights_std.npy",
            "roi_names.json",
            "selected_clip_layers.json",
            "metrics_train.jsonl",
            "metrics_val.jsonl",
            "routing_dynamics.jsonl",
            "val_routing_weights_mean_epoch001.npy",
            "val_routing_weights_std_epoch001.npy",
        ]
        assert all((output_dir / name).is_file() for name in expected_files)
        mean = np.load(output_dir / "val_routing_weights_mean.npy")
        std = np.load(output_dir / "val_routing_weights_std.npy")
        assert mean.shape == (8, 2)
        assert std.shape == (8, 2)
        assert np.isfinite(result["best_val_alignment_loss"])
        checkpoint = torch.load(output_dir / "checkpoint_last.pt", map_location="cpu")
        assert any(
            key.startswith("brain_clip_projector.")
            for key in checkpoint["model"]
        )
        val_record = json.loads(
            (output_dir / "metrics_val.jsonl").read_text().strip()
        )
        assert np.isfinite(val_record["projected_feature_norm"])
        assert np.isfinite(val_record["routed_target_norm"])
        assert np.isfinite(val_record["raw_roi_token_norm"])
        dynamics = load_jsonl(output_dir / "routing_dynamics.jsonl")
        assert len(dynamics) == 1
        assert dynamics[0]["epoch"] == 1
        assert dynamics[0]["selected_clip_layers"] == [1, 2]
        assert dynamics[0]["roi_names"] == ROI_NAMES
        assert set(dynamics[0]["per_roi_top_layer"]) == set(ROI_NAMES)
        assert sum(dynamics[0]["top_layer_counts"].values()) == 8
        assert set(dynamics[0]["layer_usage_mean"]) == {"1", "2"}
        plot_paths = save_plots(dynamics, output_dir / "routing_dynamics")
        assert len(plot_paths) == 4
        assert all(path.is_file() for path in plot_paths)

        single_output_dir = root / "single_output"
        single_args = make_args(
            train_tar, val_tar, mapping_path, single_output_dir
        )
        single_args.router_type = "single"
        single_args.single_router_layer = 2
        run_training(single_args, clip_model=tiny_clip())
        assert all(
            (single_output_dir / name).is_file() for name in expected_files
        )
        single_mean = np.load(
            single_output_dir / "val_routing_weights_mean.npy"
        )
        assert np.all(single_mean[:, 0] == 0.0)
        assert np.all(single_mean[:, 1] == 1.0)
        print("router types: soft/uniform/random/hard/single passed")
        print("training step, projector checkpoint, and Stage-1 artifacts passed")
        print("routing matrix shapes:", mean.shape, std.shape)


if __name__ == "__main__":
    main()
