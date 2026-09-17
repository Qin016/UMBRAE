import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def npy_bytes(value):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray([value], dtype=np.int64))
    return buffer.getvalue()


def add_bytes(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tar_path = root / "tiny.tar"
        captions_path = root / "captions.json"
        with tarfile.open(tar_path, "w") as archive:
            for index, coco_id in enumerate((10, 11, 12)):
                add_bytes(
                    archive,
                    f"sample{index:04d}.coco73k.npy",
                    npy_bytes(coco_id),
                )
        captions_path.write_text(json.dumps({
            "10": ["caption ten"],
            "11": "caption eleven",
        }))
        command = [
            sys.executable,
            str(ROOT / "scripts/check_caption_json.py"),
            "--tar",
            str(tar_path),
            "--captions-json",
            str(captions_path),
        ]
        passed = subprocess.run(
            command + ["--max-missing-ratio", "0.34"],
            capture_output=True,
            text=True,
        )
        assert passed.returncode == 0, passed.stderr
        assert "total checked: 3" in passed.stdout
        assert "matched: 2" in passed.stdout
        failed = subprocess.run(
            command + ["--max-missing-ratio", "0.30"],
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        assert "Missing ratio exceeds threshold" in failed.stderr
        print("Caption JSON coverage smoke test passed")


if __name__ == "__main__":
    main()
