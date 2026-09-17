#!/usr/bin/env python
"""Visualize merged NeuroRoute ROIs in wholebrain voxel space."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from build_neuroroute_roi_mapping import DEFAULT_REQUIRED_ROIS, ROI_RULES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize merged NeuroRoute ROI mappings in 3D volume space"
    )
    parser.add_argument("--subject", required=True, help="e.g. subj01")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--mapping-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--roi-names", nargs="+", default=DEFAULT_REQUIRED_ROIS)
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def load_wholebrain_collections(
    input_dir: Path, subject: str, collections: Sequence[str]
) -> Dict[str, Dict[int, Mapping[str, object]]]:
    loaded = {}
    for collection in collections:
        path = input_dir / f"{subject}_{collection}_indices.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing wholebrain ROI mapping: {path}")
        with path.open("rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected dictionary in {path}")
        loaded[collection] = {
            int(label): entry for label, entry in payload.items()
        }
    return loaded


def resolve_source_label(
    labels: Mapping[int, Mapping[str, object]], expected_name: str
) -> Tuple[int, Mapping[str, object]]:
    matches = [
        (label, entry)
        for label, entry in labels.items()
        if normalize_name(str(entry.get("name", "")))
        == normalize_name(expected_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one label named {expected_name!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def merge_roi_coordinates(
    collections: Mapping[str, Mapping[int, Mapping[str, object]]],
    roi_names: Sequence[str],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Tuple[int, int, int]]:
    merged = {}
    maximum = np.zeros(3, dtype=np.int64)
    for roi_name in roi_names:
        if roi_name not in ROI_RULES:
            raise ValueError(f"No NeuroRoute merge rule for ROI {roi_name!r}")
        seen = set()
        x_values, y_values, z_values, flat_values = [], [], [], []
        for collection, label_name in ROI_RULES[roi_name]:
            _, entry = resolve_source_label(collections[collection], label_name)
            x = np.asarray(entry["x"], dtype=np.int64)
            y = np.asarray(entry["y"], dtype=np.int64)
            z = np.asarray(entry["z"], dtype=np.int64)
            flat = np.asarray(entry["flat_indices"], dtype=np.int64)
            for x_value, y_value, z_value, flat_value in zip(x, y, z, flat):
                integer = int(flat_value)
                if integer in seen:
                    continue
                seen.add(integer)
                x_values.append(int(x_value))
                y_values.append(int(y_value))
                z_values.append(int(z_value))
                flat_values.append(integer)
        if not flat_values:
            raise ValueError(f"Merged wholebrain ROI {roi_name!r} is empty")
        coordinates = {
            "x": np.asarray(x_values, dtype=np.int64),
            "y": np.asarray(y_values, dtype=np.int64),
            "z": np.asarray(z_values, dtype=np.int64),
            "flat_indices": np.asarray(flat_values, dtype=np.int64),
        }
        maximum = np.maximum(
            maximum,
            np.asarray(
                [
                    coordinates["x"].max(),
                    coordinates["y"].max(),
                    coordinates["z"].max(),
                ]
            ),
        )
        merged[roi_name] = coordinates
    # NSD func1pt8mm volumes used here are known to be (81, 104, 83).
    inferred_shape = tuple(int(value + 1) for value in maximum)
    return merged, inferred_shape


def visualize(
    output_path: Path,
    subject: str,
    roi_coordinates: Mapping[str, Mapping[str, np.ndarray]],
    mapping: Mapping[str, object],
    volume_shape: Sequence[int] = (81, 104, 83),
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("Visualization requires matplotlib") from exc

    roi_names = list(roi_coordinates)
    colors = plt.get_cmap("tab10", len(roi_names))
    figure = plt.figure(figsize=(17, 12))
    axis_3d = figure.add_subplot(2, 2, 1, projection="3d")
    projection_axes = [
        figure.add_subplot(2, 2, 2),
        figure.add_subplot(2, 2, 3),
        figure.add_subplot(2, 2, 4),
    ]
    legend_handles = []

    for color_index, roi_name in enumerate(roi_names):
        coordinates = roi_coordinates[roi_name]
        color = colors(color_index)
        x, y, z = coordinates["x"], coordinates["y"], coordinates["z"]
        axis_3d.scatter(
            x, y, z, s=3, alpha=0.48, color=color, depthshade=False
        )
        projection_axes[0].scatter(x, z, s=2, alpha=0.35, color=color)
        projection_axes[1].scatter(y, z, s=2, alpha=0.35, color=color)
        projection_axes[2].scatter(x, y, s=2, alpha=0.35, color=color)

        wholebrain_count = len(x)
        nsdgeneral_count = int(mapping["roi_counts"].get(roi_name, 0))
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=7,
                label=(
                    f"{roi_name}: wholebrain={wholebrain_count}, "
                    f"nsdgeneral={nsdgeneral_count}"
                ),
            )
        )

    axis_3d.set(
        xlabel="X voxel",
        ylabel="Y voxel",
        zlabel="Z voxel",
        xlim=(0, volume_shape[0] - 1),
        ylim=(0, volume_shape[1] - 1),
        zlim=(0, volume_shape[2] - 1),
    )
    axis_3d.set_box_aspect(volume_shape)
    axis_3d.view_init(elev=22, azim=-60)
    axis_3d.set_title("3D wholebrain voxel coordinates")

    projection_specs = [
        ("Coronal projection (X-Z)", "X voxel", "Z voxel", volume_shape[0], volume_shape[2]),
        ("Sagittal projection (Y-Z)", "Y voxel", "Z voxel", volume_shape[1], volume_shape[2]),
        ("Axial projection (X-Y)", "X voxel", "Y voxel", volume_shape[0], volume_shape[1]),
    ]
    for axis, (title, xlabel, ylabel, x_size, y_size) in zip(
        projection_axes, projection_specs
    ):
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, x_size - 1)
        axis.set_ylim(0, y_size - 1)
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)

    unresolved = mapping.get("unresolved_rois", [])
    figure.suptitle(
        f"{subject} NeuroRoute v1 ROI mapping | func1pt8mm {tuple(volume_shape)}\n"
        f"wholebrain atlas coordinates; unresolved in nsdgeneral: "
        f"{', '.join(unresolved) if unresolved else 'none'}",
        fontsize=15,
    )
    figure.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.82, 0.51),
        frameon=True,
    )
    figure.tight_layout(rect=(0, 0, 0.81, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved visualization: {output_path}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    mapping_path = Path(args.mapping_json).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing final mapping JSON: {mapping_path}")
    mapping = json.loads(mapping_path.read_text())
    collections = sorted(
        {
            collection
            for roi_name in args.roi_names
            for collection, _ in ROI_RULES[roi_name]
        }
    )
    source_collections = load_wholebrain_collections(
        input_dir, args.subject, collections
    )
    roi_coordinates, inferred_shape = merge_roi_coordinates(
        source_collections, args.roi_names
    )
    print(f"Subject: {args.subject}")
    print(f"Coordinate extent implies at least shape: {inferred_shape}")
    for roi_name, coordinates in roi_coordinates.items():
        print(
            f"{roi_name}: wholebrain={len(coordinates['x'])}, "
            f"nsdgeneral={mapping['roi_counts'].get(roi_name, 0)}"
        )
    visualize(output_path, args.subject, roi_coordinates, mapping)


if __name__ == "__main__":
    main()
