# NeuroRoute Stage-2 MLLM

## Purpose

Stage-2 tests whether Stage-1 NeuroRoute representations improve
downstream MLLM behavior. Stage-1 established a robust multi-layer
alignment advantage, but did not establish clear ROI-specific CLIP-depth
specialization. Stage-2 therefore evaluates downstream utility without
claiming a recovered neuroanatomical hierarchy.

## Original-style and NeuroRoute pathways

The original-style baseline supplies one fMRI-derived global semantic
token:

```text
fMRI -> global L24-style token -> MLLM prefix
```

The default NeuroRoute path preserves that semantic token and adds eight
fMRI-derived ROI tokens:

```text
fMRI -> Stage-1 ROI tokenizer/projector -> 8 ROI representations
fMRI -> GlobalL24Projector -> 1 global semantic token
[global token + ROI tokens] -> NeuroRouteMLLMAdapter -> MLLM prefix
```

The L24 path is preserved because the original method and Stage-1
single-layer experiments show that final-layer semantic information is
strong. Routed ROI representations are complementary prefix tokens; they
do not replace L24 by default.

## Explicit global L24 source

`--global-l24-checkpoint` makes the semantic path independent from the
Stage-1 checkpoint used for ROI prefixes.

- If the checkpoint contains `global_projector.*`, that projector is loaded.
- For an existing Stage-1 `single`/L24 checkpoint, its fMRI
  `ROITokenizer + BrainToCLIPProjector` are loaded and the eight
  L24-aligned ROI tokens are mean-pooled into one global token.
- If the argument is omitted, Stage-2 initializes a
  `GlobalL24Projector` from the primary Stage-1 raw ROI tokens and trains it.
- In `routed_only` mode, no global L24 module is created unless a checkpoint
  is explicitly supplied, because that branch does not consume a global
  token.

With `--freeze-stage1`, an externally loaded global L24 path is frozen.
For fair `concat` ablations, soft, uniform, hard, and single-L24 ROI
variants must all use the same `--global-l24-checkpoint`.

## Important fMRI-only interpretation

The Stage-1 router computes targets using image CLIP features during
Stage-1 training. Those targets cannot be recomputed during fMRI-only
validation or inference. Stage-2 calls the Stage-1
`projected_roi_tokens` its `stage1_aligned_roi_tokens`: they are
fMRI-derived representations trained against image-derived routed CLIP
targets. Those image-derived targets are supervision, not inference input.

Routing matrices loaded from Stage-1 output are diagnostics only. No
ground-truth image CLIP feature is passed into the Stage-2 MLLM unless an
experiment is explicitly marked as oracle.

## Fusion ablations

- `l24_only`: one global semantic prefix token.
- `routed_only`: eight Stage-1 projected ROI prefix tokens.
- `concat`: global token followed by eight ROI tokens; default method.
- `gated_fusion`: per-ROI learned mixture of global and ROI tokens.
- `cross_attention`: global query attends to ROI tokens.

Recommended comparisons:

1. L24-only + single-L24 Stage-1 checkpoint.
2. Routed-only + soft Stage-1 checkpoint.
3. L24 + soft ROI tokens using `concat`.
4. L24 + uniform ROI tokens.
5. L24 + single-L24 ROI tokens.
6. Optional L24 + hard routed ROI tokens.

All concatenation comparisons should share one global checkpoint, for
example:

```bash
--global-l24-checkpoint \
stage1_outputs/cross_subject_projector/subj01/single_L24/checkpoint_best.pt
```

## MLLM bridge

The current implementation uses a generic `inputs_embeds` bridge:
projected visual prefix embeddings are prepended to text embeddings and
masked from the language-model loss.

This is not yet a strict reproduction of Shikra's
`<im_start>/<im_patch>/<im_end>` placeholder replacement protocol. A
Shikra-specific bridge remains future work and must be validated before
claiming exact Shikra protocol equivalence. Runs record:

```text
mllm_bridge_type = generic_inputs_embeds_prefix
```

## Leakage prevention

- Validation/test datasets do not decode image pixels.
- Stage-2 restores only the Stage-1 fMRI tokenizer and projector; CLIP is
  not instantiated.
- `adapter_config.json`, `metrics.json`, and `routing_summary.json`
  record whether ground-truth image features were used.
- `--oracle-image-token-mode` is false by default. The current caption-first
  implementation rejects this flag because no oracle input path is wired;
  this prevents an fMRI-only run from being mislabeled as oracle.

## Training modes

Default adapter-only training freezes Stage-1 and the MLLM while training
the global L24 projector and NeuroRoute MLLM adapter. Optional flags allow
projector/Stage-1 tuning or full fine-tuning, but full fine-tuning is not
the default.

Caption loss is implemented first. Retrieval and grounding arguments are
reserved and must remain zero until those task-specific data paths are
implemented.

Before a real run, verify caption coverage:

```bash
python scripts/check_caption_json.py \
  --tar nsd/webdataset_avg_split/train/train_subj01_*.tar \
  --captions-json /path/to/coco73k_captions.json \
  --max-missing-ratio 0
```
