# NeuroRoute full pipeline status

> Historical status note: this file describes an earlier pipeline state.
> Caption-ID mapping, real Stage-2 caption runs, the strict Shikra patch
> bridge, UMBRAE integration, and structured-routing experiments were
> completed later. Use `docs/NEUROROUTE_ALL_EXPERIMENTS_SUMMARY.md` as the
> current status and results source (updated 2026-06-23).

## ROI mapping

Real subject-specific NSD mappings are verified for S1/S2/S5/S7 in
`func1pt8mm` / `nsdgeneral.npy` vector order. NeuroRoute-v1 uses:

`V1, V2, V3, hV4, FFA, EBA, PPA, OPA`.

Official `floc-places:RSC` has zero overlap with `nsdgeneral` for all four
subjects. RSC is excluded and must not be fabricated.

## Stage-1 alignment

Projector-enabled soft routing achieves mean best validation alignment loss
0.102807, versus 0.128561 for uniform and 0.140634 for single-L24.
The advantage is consistent across all four subjects.

Routing is an L24-biased multi-layer mixture. Early and high-level ROI
expected depths are nearly identical, so no ROI-specific CLIP hierarchy is
supported.

## Stage-1 retrieval

Retrieval is ID-based, multi-positive capable, and leakage-safe. The current
300-sample validation shards happen to contain 300 unique IDs each.

Alignment-only soft is modestly better than uniform retrieval on average,
but single-L24 remains stronger. A conservative mean-pooling/L24 InfoNCE
objective improves soft B2I R@5/R@10 and rank. The 0.05 model obtains the
best average B2I R@10 and rank, while single-L24 retains better B2I R@1 and
I2B averages. Contrastive training degrades alignment by roughly 5.5%-7.7%.

Therefore retrieval does not support an unconditional claim that soft
routing is superior. It supports a task-dependent trade-off.

## Stage-2 caption-first status

Implemented:

- explicit global L24 checkpoint path;
- fMRI-derived `stage1_aligned_roi_tokens`;
- L24-only, routed-only, concat, gated, and cross-attention fusion;
- frozen Stage-1/MLLM adapter-only default;
- validation/test leakage checks;
- caption prediction and metric artifacts;
- caption-ID coverage checker.

Fair concat variants share the same single-L24 global checkpoint.
Image-derived routed CLIP targets are Stage-1 supervision only and are not
MLLM inference inputs.

The current bridge prepends generic `inputs_embeds`. It is not the strict
Shikra `<im_start>/<im_patch>/<im_end>` replacement protocol. A
Shikra-specific bridge remains pending for exact reproduction.

## Current Stage-2 blocker

`BrainHub/data/caption/fmri_cococap.json` is not a complete training caption
map:

- S1 validation shard: 114/300 IDs matched (38%);
- first 100 S1 training samples: 0/100 matched.

A complete COCO/NSD `coco73k ID -> captions` mapping is required before a
real caption benchmark. Stage-2 debug code is runnable, but real scores from
the incomplete map would be invalid.

## Claims currently allowed

- real ROI-wise fMRI tokenization for eight verified ROIs;
- learnable multi-layer routing improves the Stage-1 alignment objective;
- learned routing is an L24-biased multi-layer fusion;
- conservative contrastive training improves selected B2I retrieval metrics
  with an alignment trade-off.

## Claims not currently allowed

- a clear neuroanatomical ROI-to-CLIP-depth hierarchy;
- universal retrieval superiority of soft routing;
- MLLM/caption improvement before complete captions and real ablations;
- strict Shikra reproduction with the generic prefix bridge.
