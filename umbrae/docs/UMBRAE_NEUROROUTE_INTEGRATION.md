# UMBRAE–NeuroRoute Integration

## Purpose

This integration augments the original UMBRAE pathway. It does not replace
UMBRAE with an ROI-only model:

```text
nsdgeneral fMRI
  ├─> original UMBRAE BrainX/BrainXS -> 256 global visual tokens
  └─> verified ROI indices -> ROITokenizer -> BrainToCLIPProjector
                              -> 8 Stage-1 aligned ROI tokens

[UMBRAE tokens + ROI tokens]
  -> UMBRAENeuroRouteAdapter
  -> 256 Shikra visual embeddings
  -> replace 256 <im_patch> positions
  -> frozen Shikra/LLaMA
```

The UMBRAE global path is preserved because it already provides a strong,
spatially distributed representation compatible with Shikra. NeuroRoute
adds ROI-aware information learned through Stage-1 multi-layer CLIP
alignment.

## Audited original interface

The original repository uses:

- `BrainX` or `BrainXS` to produce `[B, 256, 1024]`.
- `model_weights/mm_projector.bin` to project each token from 1024 to 4096.
- a prompt containing `<im_start>`, exactly 256 `<im_patch>` tokens, and
  `<im_end>`.
- Shikra token IDs 32000, 32001, and 32002 for patch, start, and end.
- replacement of the 256 placeholder embeddings between the start/end
  markers by fMRI-derived visual embeddings.

The local Shikra tokenizer was checked and produces exactly this sequence.

## Adapter

`models/umbrae_neuroroute_adapter.py` accepts:

```text
umbrae_tokens:          [B, N_u, D_u]
neuroroute_roi_tokens:  [B, 8, D_r]
routing_weights:        [B, 8, L]  (diagnostics)
reliability:            [B, 8]     (optional)
```

It always returns:

```text
visual_tokens: [B, 256, D_mllm]
```

The original UMBRAE projector can be loaded into the direct
`umbrae_only` path. ROI-augmented modes use one of:

- `perceiver_resampler` (default): learned 256 queries attend to all source
  tokens.
- `cross_attention`: one cross-attention resampling block.
- `concat_then_project`: deterministic adaptive pooling followed by
  projection.

`--roi-token-expansion K` expands each ROI representation into `K`
subtokens before fusion. For example, `K=4` changes 8 ROI tokens into 32
ROI subtokens while the final Shikra sequence remains 256 tokens.

## Fusion ablations

- `umbrae_only`: original UMBRAE tokens only.
- `neuroroute_only`: Stage-1 aligned ROI tokens only.
- `umbrae_plus_soft`: UMBRAE plus ROI tokens from the soft-router
  checkpoint.
- `umbrae_plus_uniform`: UMBRAE plus uniform-router ROI tokens.
- `umbrae_plus_single_l24`: UMBRAE plus single-L24 ROI tokens.

The latter three must use the same UMBRAE checkpoint, Shikra weights,
caption split, prompt, and decoding parameters.

## Bridge protocol

`--bridge-type shikra_patch` is the default and implements the original
placeholder protocol. It requires exactly 256 output visual tokens and
raises an error if `<im_end>` is not found at the expected position.

`--bridge-type generic_prefix` is retained only for controlled debugging.
It prepends embeddings and is not a strict UMBRAE/Shikra reproduction.

## Leakage prevention

- The dataset reads fMRI and captions, not image pixels.
- UMBRAE tokens and NeuroRoute ROI tokens are both computed from fMRI.
- Image-derived CLIP features were Stage-1 supervision only.
- Validation rejects image-feature use unless an oracle protocol is
  explicitly implemented; this script intentionally rejects oracle mode.
- Every run records `uses_image_clip_tokens_at_eval=false` and
  `oracle_image_token_mode=false`.

## Subj01 commands

Common arguments:

```bash
COMMON=(
  --subject subj01
  --train-tar nsd/webdataset_avg_split/train/train_subj01_*.tar
  --val-tar nsd/webdataset_avg_split/val/val_subj01_0.tar
  --captions-json stage2_outputs/caption_mapping/coco73k_captions.json
  --roi-indices-path ../roi_indices/subj01_neuroroute_v1.json
  --umbrae-checkpoint train_logs/brainx/last.pth
  --mllm-model-path model_weights/shikra-7b
  --umbrae-mm-projector model_weights/mm_projector.bin
  --bridge-type shikra_patch
  --fusion-type perceiver_resampler
  --freeze-umbrae
  --freeze-neuroroute
  --freeze-mllm
  --train-adapter-only
  --batch-size 2
  --epochs 1
)
```

Original baseline:

```bash
python scripts/train_umbrae_neuroroute.py "${COMMON[@]}" \
  --neuroroute-checkpoint \
    stage1_outputs/cross_subject_projector/subj01/soft/checkpoint_best.pt \
  --fusion-mode umbrae_only \
  --output-dir umbrae_neuroroute_outputs/subj01/umbrae_only
```

Main soft fusion:

```bash
python scripts/train_umbrae_neuroroute.py "${COMMON[@]}" \
  --neuroroute-checkpoint \
    stage1_outputs/cross_subject_projector/subj01/soft/checkpoint_best.pt \
  --fusion-mode umbrae_plus_soft \
  --output-dir umbrae_neuroroute_outputs/subj01/umbrae_plus_soft
```

Use the corresponding `uniform` and `single_L24` Stage-1 checkpoints for
the other two augmented baselines. After the one-epoch debug is stable,
repeat with `--epochs 3`.

## Interpretation and limitations

An improvement of `umbrae_plus_soft` over `umbrae_only` would support the
claim that ROI-aware tokens complement UMBRAE under the original patch
protocol. Comparisons with uniform and single-L24 determine whether
learnable multi-layer routing matters.

No improvement is claimed by this implementation alone. Real ablations
must be completed first. The module also does not establish an
ROI-specific CLIP hierarchy; prior Stage-1 results only support an
L24-biased multi-layer mixture.
