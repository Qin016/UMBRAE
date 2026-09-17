#!/usr/bin/env python
"""Print and plot NeuroRoute Stage-1 routing dynamics over epochs."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Stage-1 routing dynamics from routing_dynamics.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Stage-1 output directory containing routing_dynamics.jsonl",
    )
    parser.add_argument(
        "--save-dir",
        help="Plot output directory; defaults to <output-dir>/routing_dynamics",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Print tables without saving PNG plots",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing routing dynamics file: {path}")
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No routing dynamics records found in {path}")
    records.sort(key=lambda record: int(record["epoch"]))
    return records


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    text_rows = [
        [
            f"{value:.6f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        for row in rows
    ]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]
    print(
        " | ".join(
            str(header).ljust(widths[index])
            for index, header in enumerate(headers)
        )
    )
    print("-+-".join("-" * width for width in widths))
    for row in text_rows:
        print(
            " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


def load_final_matrix(
    output_dir: Path,
    final_epoch: int,
    roi_count: int,
    layer_count: int,
) -> np.ndarray:
    epoch_path = (
        output_dir
        / f"val_routing_weights_mean_epoch{final_epoch:03d}.npy"
    )
    fallback_path = output_dir / "val_routing_weights_mean.npy"
    path = epoch_path if epoch_path.is_file() else fallback_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing final routing matrix: {path}")
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (roi_count, layer_count):
        raise ValueError(
            f"Routing matrix shape {matrix.shape} != "
            f"expected {(roi_count, layer_count)}"
        )
    return matrix


def print_dynamics(records: Sequence[Dict[str, object]]) -> None:
    layers = records[0]["selected_clip_layers"]
    roi_names = records[0]["roi_names"]
    l24_key = "24"
    print("\nRouting dynamics")
    print_table(
        [
            "epoch",
            "val_total",
            "val_mse",
            "val_cos",
            "entropy",
            "max_weight",
            "L24_usage",
        ],
        [
            [
                record["epoch"],
                record["val_total_loss"],
                record["val_mse_loss"],
                record["val_cosine_loss"],
                record["routing_entropy_mean"],
                record["routing_max_weight_mean"],
                record["layer_usage_mean"].get(l24_key, float("nan")),
            ]
            for record in records
        ],
    )

    print("\nPer-ROI top layer over epochs")
    print_table(
        ["epoch", *roi_names],
        [
            [
                record["epoch"],
                *[
                    f"L{record['per_roi_top_layer'][roi_name]}"
                    for roi_name in roi_names
                ],
            ]
            for record in records
        ],
    )
    print(f"\nSelected CLIP layers: {layers}")


def print_final_matrix(
    matrix: np.ndarray,
    roi_names: Sequence[str],
    selected_layers: Sequence[int],
) -> None:
    print("\nFinal routing matrix")
    print_table(
        ["ROI", *[f"L{layer}" for layer in selected_layers]],
        [
            [roi_name, *[float(value) for value in matrix[index]]]
            for index, roi_name in enumerate(roi_names)
        ],
    )


def save_plots(
    records: Sequence[Dict[str, object]],
    save_dir: Path,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib") from exc

    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in records]
    plot_specs = [
        (
            "val_loss",
            [
                ("total", [record["val_total_loss"] for record in records]),
                ("mse", [record["val_mse_loss"] for record in records]),
                ("cosine", [record["val_cosine_loss"] for record in records]),
            ],
            "Validation loss",
        ),
        (
            "mean_max_weight",
            [
                (
                    "mean max weight",
                    [
                        record["routing_max_weight_mean"]
                        for record in records
                    ],
                )
            ],
            "Mean maximum routing weight",
        ),
        (
            "l24_usage",
            [
                (
                    "L24 usage",
                    [
                        record["layer_usage_mean"].get("24", float("nan"))
                        for record in records
                    ],
                )
            ],
            "Mean L24 routing usage",
        ),
    ]
    saved = []
    for filename, series, ylabel in plot_specs:
        figure, axis = plt.subplots(figsize=(7, 4))
        for label, values in series:
            axis.plot(epochs, values, marker="o", label=label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_xticks(epochs)
        axis.grid(alpha=0.25)
        if len(series) > 1:
            axis.legend()
        figure.tight_layout()
        path = save_dir / f"{filename}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)

    roi_names = records[0]["roi_names"]
    figure, axes = plt.subplots(
        len(roi_names), 1, figsize=(8, max(6, 1.4 * len(roi_names))),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, roi_name in zip(axes, roi_names):
        values = [
            record["per_roi_top_layer"][roi_name] for record in records
        ]
        axis.step(epochs, values, where="mid")
        axis.set_ylabel(roi_name)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Epoch")
    axes[-1].set_xticks(epochs)
    figure.suptitle("Per-ROI top CLIP layer")
    figure.tight_layout()
    path = save_dir / "per_roi_top_layer.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    saved.append(path)
    return saved


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    records = load_jsonl(output_dir / "routing_dynamics.jsonl")
    roi_names = records[0]["roi_names"]
    selected_layers = records[0]["selected_clip_layers"]
    for record in records[1:]:
        if record["roi_names"] != roi_names:
            raise ValueError("roi_names changed across epochs")
        if record["selected_clip_layers"] != selected_layers:
            raise ValueError("selected_clip_layers changed across epochs")

    print_dynamics(records)
    final_matrix = load_final_matrix(
        output_dir,
        int(records[-1]["epoch"]),
        len(roi_names),
        len(selected_layers),
    )
    print_final_matrix(final_matrix, roi_names, selected_layers)

    if not args.no_save_plots:
        save_dir = (
            Path(args.save_dir).expanduser().resolve()
            if args.save_dir
            else output_dir / "routing_dynamics"
        )
        for path in save_plots(records, save_dir):
            print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
