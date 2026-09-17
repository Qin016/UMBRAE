# ROI-wise fMRI Tokenizer

## Shapes

`ROITokenizer` accepts either:

- `[B, V]`: one flattened `nsdgeneral` voxel vector per sample.
- `[B, repeats, V]`: the current UMBRAE WebDataset format. The tokenizer
  averages the repeat dimension before extracting ROIs.

It returns:

```python
{
    "roi_tokens": Tensor[B, R, roi_token_dim],
    "roi_names": list[str],
}
```

`R` follows the exact order supplied in `roi_names`.

## Tokenizer modes

- `shared_mlp`: ROI voxel vectors are zero-padded to the largest selected ROI
  and passed through one shared MLP.
- `roi_specific_mlp`: each ROI has an independent MLP sized to that ROI's
  voxel count.

ROI identity embeddings are enabled by default. Subject embeddings are
optional and require zero-based `subject_ids` plus `num_subjects` when the
module is constructed.

## Current NSD/UMBRAE data format

The downloaded WebDataset shards contain:

- `nsdgeneral.npy`: `[3, V]`, where `3` is the repeated-trial dimension.
- `wholebrain_3d.npy`: `[3, X, Y, Z]`, with subject-specific volume shapes.

The `nsdgeneral` voxel counts used by existing UMBRAE are:

| Subject | V |
|---|---:|
| S1 | 15724 |
| S2 | 14278 |
| S5 | 13039 |
| S7 | 12682 |

No ROI atlas, ROI mask, or mapping from `nsdgeneral` positions to V1/V2/etc.
is included in the current repository data. Real ROI tokenization therefore
requires subject-specific ROI metadata from the NSD preprocessing pipeline.

Atlas volume indices must not be used directly as `nsdgeneral` vector indices.
Real ROI tokenization requires subject-specific ROI indices aligned to the exact
voxel order used when `nsdgeneral.npy` was flattened.

## Connecting external ROI metadata

For each subject, convert ROI masks into positions in that subject's flattened
`nsdgeneral` vector and construct a tokenizer:

```python
roi_indices = {
    "V1": v1_indices,
    "V2": v2_indices,
    "V3": v3_indices,
    "hV4": hv4_indices,
    "LOC": loc_indices,
    "FFA": ffa_indices,
    "PPA": ppa_indices,
}

roi_tokenizer = ROITokenizer(
    roi_names=["V1", "V2", "V3", "hV4", "LOC", "FFA", "PPA"],
    roi_indices=roi_indices,
    token_dim=1024,
    tokenizer_type="shared_mlp",
)

outputs = roi_tokenizer(voxel)  # voxel may be [B, 3, V] or [B, V]
roi_tokens = outputs["roi_tokens"]
```

Alternatively, pass a boolean tensor `roi_masks` with shape `[R, V]`.
Overlapping ROIs are allowed, but each ROI must be non-empty and may not contain
duplicate indices.

Builder outputs can be loaded with:

```python
from models.roi_mapping import load_roi_indices

roi_indices, summary = load_roi_indices(
    "roi_indices/subj01.json",
    mapping_format="json",
    expected_voxel_count=15724,
    strict=True,
)
```

## Compatibility

The training scripts expose the following configuration without activating it:

```bash
--use_roi_tokenizer \
--roi_names V1 V2 V3 hV4 LOC FFA PPA \
--roi_token_dim 1024 \
--roi_tokenizer_type shared_mlp \
--use_roi_embeddings \
--no-use_subject_embeddings \
--roi_indices_path roi_indices/subj01.json \
--roi_mapping_format json \
--strict_roi_check
```

`use_roi_tokenizer` defaults to `False`. This step does not replace or modify
the existing `BrainX`, `BrainXS`, or `BrainXC` forward paths.

## Smoke test

```bash
python tests/test_roi_tokenizer.py
```
