import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_coco73k_caption_mapping import build_mapping
from scripts.coco73k_caption_utils import load_coco_captions


ROOT = Path(__file__).resolve().parents[1]


def npy_bytes(value):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray([value], dtype=np.int64))
    return buffer.getvalue()


def add_member(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def make_tar(path, ids):
    with tarfile.open(path, "w") as archive:
        for index, local_id in enumerate(ids):
            add_member(
                archive,
                f"sample{index:04d}.coco73k.npy",
                npy_bytes(local_id),
            )


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        official = root / "captions.json"
        official.write_text(json.dumps({
            "annotations": [
                {"image_id": 100, "caption": "caption one"},
                {"image_id": 100, "caption": "caption two"},
                {"image_id": 200, "caption": "caption three"},
                {"image_id": 300, "caption": "caption four"},
            ]
        }))
        captions = load_coco_captions([str(official)])
        assert captions[100] == ["caption one", "caption two"]

        direct, best_direct, _ = build_mapping(
            [100, 200], captions, []
        )
        assert best_direct["name"] == "A_raw_coco_image_id"
        assert set(direct) == {"100", "200"}

        metadata = root / "metadata.csv"
        metadata.write_text(
            "nsdId,cocoId\n0,100\n1,200\n2,300\n"
        )
        zero, best_zero, _ = build_mapping(
            [0, 1, 2], captions, [str(metadata)]
        )
        assert best_zero["name"] in (
            "B_0_based_metadata_index",
            "D_metadata_row_index",
        )
        assert zero["0"][0] == "caption one"
        assert zero["2"][0] == "caption four"

        one, best_one, _ = build_mapping(
            [1, 2, 3], captions, [str(metadata)]
        )
        assert best_one["name"] == "C_1_based_metadata_index"
        assert one["1"][0] == "caption one"

        missing, best_missing, _ = build_mapping(
            [0, 1, 99], captions, [str(metadata)]
        )
        assert best_missing["coverage"] == 2 / 3
        assert "99" not in missing

        tar_path = root / "samples.tar"
        make_tar(tar_path, [0, 1, 99])
        output_json = root / "output.json"
        output_report = root / "report.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/build_coco73k_caption_mapping.py"),
            "--webdataset-tar",
            str(tar_path),
            "--captions-json",
            str(official),
            "--metadata",
            str(metadata),
            "--output-json",
            str(output_json),
            "--output-report",
            str(output_report),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0
        assert output_json.is_file() and output_report.is_file()
        payload = json.loads(output_json.read_text())
        assert isinstance(payload, dict)
        assert all(isinstance(value, list) for value in payload.values())
        print("COCO73K caption mapping hypotheses passed")
        print("Missing-ID and coverage-threshold checks passed")


if __name__ == "__main__":
    main()
