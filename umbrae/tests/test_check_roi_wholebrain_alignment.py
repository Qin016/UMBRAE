import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shape = (3, 4, 5)
        wholebrain = np.arange(2 * np.prod(shape), dtype=np.float32).reshape(
            2, *shape
        )
        roi = np.zeros(shape, dtype=np.int16)
        roi.ravel()[:5] = 1
        roi.ravel()[5:9] = 2
        nsdgeneral_mask = np.zeros(shape, dtype=np.uint8)
        nsdgeneral_mask.ravel()[:12] = 1
        nsdgeneral = wholebrain[:, nsdgeneral_mask.astype(bool)]

        wholebrain_path = tmp_path / "sample.wholebrain_3d.npy"
        roi_path = tmp_path / "prf-visualrois.nii.gz"
        nsdgeneral_npy_path = tmp_path / "sample.nsdgeneral.npy"
        nsdgeneral_nii_path = tmp_path / "nsdgeneral.nii.gz"
        np.save(wholebrain_path, wholebrain)
        np.save(nsdgeneral_npy_path, nsdgeneral)
        nib.save(nib.Nifti1Image(roi, np.eye(4)), roi_path)
        nib.save(
            nib.Nifti1Image(nsdgeneral_mask, np.eye(4)),
            nsdgeneral_nii_path,
        )

        command = [
            sys.executable,
            str(ROOT / "scripts/check_roi_wholebrain_alignment.py"),
            "--wholebrain_path",
            str(wholebrain_path),
            "--roi_path",
            str(roi_path),
            "--nsdgeneral_npy_path",
            str(nsdgeneral_npy_path),
            "--nsdgeneral_nii_path",
            str(nsdgeneral_nii_path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        assert "[PASS] Shape matched" in result.stdout
        assert "[PASS] nsdgeneral values and mask order matched" in result.stdout
        assert "max_abs_diff: 0" in result.stdout
        assert "can be used directly for ROI voxel extraction" in result.stdout
        assert "label 1: vector shape=(2, 5)" in result.stdout
        print("ROI/wholebrain alignment smoke test passed")


if __name__ == "__main__":
    main()
