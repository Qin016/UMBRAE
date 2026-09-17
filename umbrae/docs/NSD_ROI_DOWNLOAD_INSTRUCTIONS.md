# Obtaining NSD ROI Files

## Local status

No subject-specific NSD `func1pt8mm/roi` files were found under the audited
locations:

- `/data/nsddata`
- `/data/NSD`
- `/opt/data`
- `/opt/data/private`
- the UMBRAE repository and its parent directories

The current UMBRAE WebDataset shards contain `nsdgeneral.npy` responses but do
not include the NIfTI mask, ROI atlases, affine metadata, or verified voxel
order required to map atlas voxels to vector positions.

## Access requirement

NSD is controlled-access research data. Complete the official NSD data-access
form and accept the NSD terms before obtaining these files. Use the official
NSD data release or its documented AWS Open Data distribution with your own
authorized access setup.

This repository does not download protected NSD files or embed credentials.

Official starting points:

- NSD project: https://naturalscenesdataset.org/
- NSD data manual: https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions
- NSD data preparation reference: https://github.com/cvnlab/nsddatapaper

## Required directory structure

Place or mount the authorized NSD release as:

```text
/data/nsddata/
└── ppdata/
    ├── subj01/
    │   └── func1pt8mm/
    │       └── roi/
    │           ├── nsdgeneral.nii.gz
    │           ├── prf-visualrois.nii.gz
    │           ├── floc-faces.nii.gz
    │           ├── floc-places.nii.gz
    │           ├── floc-bodies.nii.gz
    │           └── streams.nii.gz
    ├── subj02/...
    ├── subj05/...
    └── subj07/...
```

Equivalent roots are accepted. For example, if the data is mounted at
`/mnt/nsd/nsddata/ppdata`, pass either `/mnt/nsd` or `/mnt/nsd/nsddata` to the
verification script.

## Verification

```bash
cd /opt/data/private/BA/UMBRAE/umbrae
conda activate brainx

python scripts/verify_nsd_roi_files.py \
  --nsd-root /data/nsddata \
  --subjects 1 2 5 7 \
  --space func1pt8mm \
  --required-rois \
    nsdgeneral prf-visualrois floc-faces floc-places floc-bodies streams \
  --report-json roi_indices/nsd_roi_file_report.json \
  --strict
```

Expected `nsdgeneral` nonzero voxel counts:

```text
S1: 15724
S2: 14278
S5: 13039
S7: 12682
```

All ROI files must be 3D NIfTI volumes. Their shape and affine should match the
subject's `nsdgeneral.nii.gz`.

## Next step after verification

File presence and geometry do not prove `nsdgeneral.npy` vector order. After
verification, use `scripts/build_nsd_roi_indices.py` with an explicit
subject-specific voxel-order file whenever possible. Mask-derived C/F flatten
order remains a documented assumption unless matched to the original
WebDataset-generation process.
