# NSD COCO73K caption mapping

## What `coco73k.npy` represents

Inspection of UMBRAE WebDataset shards shows that `coco73k.npy` is not a
raw COCO image ID. It is the zero-based NSD stimulus identifier `nsdId`.

Examples:

- S1 train first 1000: IDs 2516–11005;
- S1 validation: IDs 13–2509;
- values are sparse subsets of the NSD range 0–72999;
- the inspected samples contain one local ID per sample.

Using these values directly as COCO image IDs is the earlier failure mode.
It produced accidental partial matches because some small NSD indices also
happen to be valid COCO image IDs.

## Authoritative metadata

The mapping table is:

```text
/opt/data/private/BA/NSD/nsddata/experiments/nsd/
  nsd_stim_info_merged.csv
```

It contains 73,000 rows and explicit fields:

```text
nsdId -> cocoId
```

`nsdId` runs from 0 to 72999 and matches the integer stored in local
`coco73k.npy`. `cocoId` is the raw COCO 2017 image ID used by caption
annotations. The selected mapping hypothesis is therefore:

```text
Hypothesis B: zero-based NSD/COCO73K index
```

Coverage alone cannot distinguish every off-by-one hypothesis because both
neighboring COCO images usually have captions. The explicit NSD field name
and zero-based range resolve the convention.

## Caption source

Standard COCO 2017 train and validation captions were obtained from a
Hugging Face parquet mirror of the COCO annotations and normalized into:

```text
raw COCO image_id -> list[str captions]
```

The local normalized source contains 123,287 images and 616,621 captions:

```text
stage2_outputs/caption_mapping/coco_annotations/
  coco2017_captions_raw_id.json
```

The final NeuroRoute mapping converts this through the NSD table:

```text
local nsdId -> raw cocoId -> one or more captions
```

Final output:

```text
stage2_outputs/caption_mapping/coco73k_captions.json
```

## Coverage

The final mapping covers all currently packaged S1/S2/S5/S7 train and
validation samples:

| Split | Subject | Matched / total | Coverage |
|---|---|---:|---:|
| train | S1 | 8559 / 8559 | 100% |
| train | S2 | 8559 / 8559 | 100% |
| train | S5 | 8181 / 8181 | 100% |
| train | S7 | 8189 / 8189 | 100% |
| validation | S1 | 300 / 300 | 100% |
| validation | S2 | 300 / 300 | 100% |
| validation | S5 | 300 / 300 | 100% |
| validation | S7 | 300 / 300 | 100% |

Overall: 34,688 unique local IDs, 34,688 matched, zero missing.

## Reproduction

Inspect IDs:

```bash
python scripts/inspect_coco73k_ids.py \
  --tar "nsd/webdataset_avg_split/train/train_subj01_*.tar" \
  --max-samples 1000 \
  --output-dir stage2_outputs/caption_mapping/inspect_train_subj01
```

Find metadata:

```bash
python scripts/find_nsd_caption_metadata.py \
  --search-root /opt/data/private/BA/NSD \
  --output-dir stage2_outputs/caption_mapping/metadata_search_nsd
```

Build a subject mapping:

```bash
python scripts/build_coco73k_caption_mapping.py \
  --webdataset-tar "nsd/webdataset_avg_split/train/train_subj01_*.tar" \
  --webdataset-tar nsd/webdataset_avg_split/val/val_subj01_0.tar \
  --captions-json \
    stage2_outputs/caption_mapping/coco_annotations/coco2017_captions_raw_id.json \
  --metadata \
    /opt/data/private/BA/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.csv \
  --output-json stage2_outputs/caption_mapping/coco73k_captions.json \
  --output-report \
    stage2_outputs/caption_mapping/caption_mapping_report.json
```

Validate:

```bash
python scripts/check_caption_json.py \
  --tar "nsd/webdataset_avg_split/train/train_subj01_*.tar" \
  --tar nsd/webdataset_avg_split/val/val_subj01_0.tar \
  --captions-json stage2_outputs/caption_mapping/coco73k_captions.json \
  --min-coverage 0.95
```

The previous `BrainHub/data/caption/fmri_cococap.json` has keys 0–981 and
represents the shared evaluation-caption ordering. It is not a complete
local `nsdId -> captions` mapping and must not be used for Stage-2 training
coverage.
