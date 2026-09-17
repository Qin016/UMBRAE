import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.roi_mapping import load_roi_indices


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shape = (3, 3, 2)
        affine = np.eye(4)

        mask = np.zeros(shape, dtype=np.uint8)
        mask.ravel(order="C")[:10] = 1
        atlas = np.zeros(shape, dtype=np.int16)
        atlas.ravel(order="C")[:4] = 1
        atlas.ravel(order="C")[4:7] = 2
        atlas.ravel(order="C")[7:10] = 3

        mask_path = tmp_path / "nsdgeneral.nii.gz"
        atlas_path = tmp_path / "prf-visualrois.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), mask_path)
        nib.save(nib.Nifti1Image(atlas, affine), atlas_path)

        spec_path = tmp_path / "roi_spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "V1": {"atlas": atlas_path.name, "labels": [1]},
                    "V2": {"atlas": atlas_path.name, "labels": [2]},
                    "V3": {"atlas": atlas_path.name, "labels": [3]},
                }
            )
        )
        output_path = tmp_path / "subj01.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/build_nsd_roi_indices.py"),
            "--subject",
            "1",
            "--nsdgeneral-mask",
            str(mask_path),
            "--roi-spec",
            str(spec_path),
            "--expected-voxel-count",
            "10",
            "--output",
            str(output_path),
            "--strict-roi-check",
        ]
        subprocess.run(command, check=True)

        roi_indices, summary = load_roi_indices(
            str(output_path),
            mapping_format="json",
            expected_voxel_count=10,
            strict=True,
        )
        assert roi_indices == {
            "V1": [0, 1, 2, 3],
            "V2": [4, 5, 6],
            "V3": [7, 8, 9],
        }
        assert summary["union_voxel_count"] == 10
        assert summary["union_reasonable"]
        assert output_path.with_suffix(".summary.json").exists()

        npy_path = tmp_path / "subj01.npy"
        np.save(npy_path, json.loads(output_path.read_text()), allow_pickle=True)
        npy_indices, npy_summary = load_roi_indices(
            str(npy_path),
            mapping_format="npy",
            expected_voxel_count=10,
            strict=True,
        )
        assert npy_indices == roi_indices
        assert npy_summary["union_voxel_count"] == 10
        print("ROI mapping smoke test passed")
        print("roi_voxel_counts:", summary["roi_voxel_counts"])
        print("union_voxel_count:", summary["union_voxel_count"])


if __name__ == "__main__":
    main()
