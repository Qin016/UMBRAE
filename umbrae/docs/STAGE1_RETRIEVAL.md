# NeuroRoute Stage-1 Retrieval

Stage-1 retrieval evaluates fMRI-derived embeddings directly against a
frozen CLIP image gallery. It does not use an MLLM.

## Multi-positive protocol

NSD/WebDataset samples are matched by `coco73k.npy`, not only by sample
position. For brain query `i` and image gallery item `j`:

```text
positive[i,j] = brain_coco73k_id[i] == image_coco73k_id[j]
```

Recall@K succeeds when any positive appears in the first K results. Rank is
the best-ranked positive. `--diagonal-only` remains available only as a
diagnostic and is explicitly labeled in output metadata.

## Leakage boundary

The brain query is computed only from:

```text
fMRI -> ROITokenizer -> BrainToCLIPProjector -> retrieval pooling
```

Image CLIP features are gallery embeddings only. For `image-target=routed`,
evaluation uses the fixed validation-mean routing matrix saved beside the
checkpoint. It does not use the query sample's fMRI to condition gallery
embeddings.

The default image target is `l24`; routed fusion is diagnostic.

## Pooling

- `mean` requires no learned parameters and works with legacy checkpoints.
- `attention` and `roi_flatten_mlp` must be loaded from a checkpoint trained
  with the same `--contrastive-brain-pooling` mode and a positive
  `--contrastive-loss-weight`.

Stage-1 contrastive training is disabled by default:

```bash
--contrastive-loss-weight 0
```

To train a retrieval head:

```bash
python scripts/train_stage1_routing.py ... \
  --contrastive-loss-weight 0.1 \
  --contrastive-temperature 0.07 \
  --contrastive-brain-pooling attention \
  --contrastive-image-target routed
```

The current training InfoNCE uses paired batch diagonals. The evaluated
500-sample train shards for S1/S2/S5/S7 each contain 500 unique coco73k IDs,
so diagonal training labels are valid for these runs. Multi-positive
handling remains active at evaluation.
