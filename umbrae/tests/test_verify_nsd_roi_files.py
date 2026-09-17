import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        nsd_root = Path(tmp) / "nsddata"
        roi_dir = nsd_root / "ppdata/subj01/func1pt8mm/roi"
        roi_dir.mkdir(parents=True)
        shape = (3, 3, 2)
        affine = np.eye(4)

        for index, roi_name in enumerate(
            ["nsdgeneral", "prf-visualrois", "floc-faces"]
        ):
            data = np.zeros(shape, dtype=np.int16)
            data.ravel()[:10] = 1 if index == 0 else index
            nib.save(nib.Nifti1Image(data, affine), roi_dir / f"{roi_name}.nii.gz")

        report_path = Path(tmp) / "report.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/verify_nsd_roi_files.py"),
            "--nsd-root",
            str(nsd_root),
            "--subjects",
            "1",
            "--required-rois",
            "nsdgeneral",
            "prf-visualrois",
            "floc-faces",
            "--report-json",
            str(report_path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        assert "[FOUND] nsdgeneral" in result.stdout
        assert "mask voxels: 10 match: False" in result.stdout
        assert "0/1 subjects passed" in result.stdout
        assert report_path.exists()
        print("NSD ROI verification smoke test passed")


if __name__ == "__main__":
    main()
