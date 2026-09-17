# NSD ROI Mapping Audit

## Result

The current UMBRAE checkout cannot by itself produce verified real ROI indices.
It contains pre-generated `nsdgeneral.npy` vectors but neither the
subject-specific ROI volumes nor the exact voxel-order metadata used to flatten
those vectors.

Do not treat flattened atlas-volume indices as `nsdgeneral` vector indices.

## Current UMBRAE data path

UMBRAE downloads pre-packed shards from `pscotti/naturalscenesdataset`. Each
sample contains:

- `nsdgeneral.npy`: `[3, V]`
- `wholebrain_3d.npy`: `[3, X, Y, Z]`
- image and trial metadata

Subject voxel counts:

| Subject | V |
|---|---:|
| S1 | 15724 |
| S2 | 14278 |
| S5 | 13039 |
| S7 | 12682 |

`utils.get_dataloaders` only reads `nsdgeneral.npy`. It does not create or
reorder it. Training selects one repeat; validation and inference average the
three repeats.

`download_data.sh` downloads only tar shards. It does not download preprocessing
code, ROI atlases, the `nsdgeneral` mask, affine metadata, or voxel order.

## Upstream audit

The MindEye repository uses the same pre-packed Hugging Face shards and likewise
does not create `nsdgeneral.npy`.

Official NSD preparation code maps ROI labels into subject-specific
`func1pt8mm/roi` NIfTI volumes. Relevant files are:

- `nsdgeneral.nii.gz`
- `prf-visualrois.nii.gz`
- `floc-faces.nii.gz`
- `floc-places.nii.gz`
- `floc-bodies.nii.gz`
- `streams.nii.gz`

Confirmed labels:

- `prf-visualrois`: 1 V1v, 2 V1d, 3 V2v, 4 V2d, 5 V3v, 6 V3d, 7 hV4
- `floc-faces`: 1 OFA, 2 FFA-1, 3 FFA-2, 4 mTL-faces, 5 aTL-faces
- `floc-places`: 1 OPA, 2 PPA, 3 RSC
- `floc-bodies`: 1 EBA, 2 FBA-1, 3 FBA-2, 4 mTL-bodies

No single canonical `LOC` label was identified in these audited NSD ROI groups.
A LOC definition must be supplied explicitly rather than inferred.

The Hugging Face files named `subjXX_annot.npy` were inspected. They are COCO
caption arrays of shape `[27750]`, not voxel annotations.

## Missing files and expected structure

For each subject:

```text
nsddata/
└── ppdata/
    └── subjXX/
        └── func1pt8mm/
            └── roi/
                ├── nsdgeneral.nii.gz
                ├── prf-visualrois.nii.gz
                ├── floc-faces.nii.gz
                ├── floc-places.nii.gz
                ├── floc-bodies.nii.gz
                └── streams.nii.gz
```

For fully auditable alignment, also provide:

```text
voxel_order/
└── subjXX_nsdgeneral_order.npy
```

The order array may be `[V]` linear volume indices or `[V,3]` integer voxel
coordinates, in the exact order used by `nsdgeneral.npy`.

Without an explicit order, the builder can derive one from
`nsdgeneral.nii.gz` using NumPy C- or Fortran-order flattening. This is recorded
as an assumption and cannot prove parity with the unavailable tar generator.

## Builder behavior

`scripts/build_nsd_roi_indices.py`:

1. validates the subject's `nsdgeneral` voxel count;
2. samples ROI atlas labels at ordered `nsdgeneral` voxels;
3. converts matches to vector positions in `[0,V)`;
4. validates empty, duplicate, and out-of-range indices;
5. reports ROI counts, union coverage, overlap, geometry warnings, and order
   provenance;
6. saves JSON or NPY plus a small JSON summary.

See `configs/nsd_roi_spec.example.json` for atlas label configuration.

## Commands once NSD files are available

Preferred explicit-order mode:

```bash
python scripts/build_nsd_roi_indices.py \
  --subject 1 \
  --nsdgeneral-order /data/voxel_order/subj01_nsdgeneral_order.npy \
  --reference-volume /data/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz \
  --roi-spec configs/subj01_roi_spec.json \
  --output roi_indices/subj01.json \
  --strict-roi-check
```

Mask-order fallback:

```bash
python scripts/build_nsd_roi_indices.py \
  --subject 1 \
  --nsdgeneral-mask /data/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz \
  --roi-spec configs/subj01_roi_spec.json \
  --flatten-order C \
  --output roi_indices/subj01.json \
  --strict-roi-check
```

Repeat with subjects `2`, `5`, and `7`, using subject-specific paths and ROI
spec files.
