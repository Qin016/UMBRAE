# NeuroRoute Stage-1 routing results

## Scope

This report summarizes the subject-specific Stage-1 representation
alignment experiments for NeuroRoute. The results establish whether a
learnable multi-layer CLIP router improves the current alignment
objective. They do not establish a neuroanatomical hierarchy between
brain ROIs and CLIP layers.

## Experimental setup

Experiments were run independently for `subj01`, `subj02`, `subj05`,
and `subj07`. Each subject used:

- subject-specific, voxel-order-verified `nsdgeneral.npy` ROI indices;
- the same eight-ROI NeuroRoute-v1 set;
- frozen CLIP ViT-L/14 features from layers 4, 8, 12, 16, 20, and 24;
- an MLP `BrainToCLIPProjector`;
- MSE plus cosine-distance representation alignment;
- identical optimizer, batch-size, selected-layer, and training-length
  settings across routing methods.

Models were trained independently per subject. This is a cross-subject
validation of the experimental finding, not a joint multi-subject model.

## ROI set

The eight required ROIs were:

`V1`, `V2`, `V3`, `hV4`, `FFA`, `EBA`, `PPA`, and `OPA`.

RSC was not used because the official `floc-places:RSC` voxels have zero
overlap with the UMBRAE `nsdgeneral` input space for all four subjects.
No RSC indices were approximated or fabricated.

For grouped expected-depth analysis:

- early visual ROIs: `V1`, `V2`, `V3`, `hV4`;
- high-level ROIs: `FFA`, `EBA`, `PPA`, `OPA`.

## Compared methods

1. **Soft + projector**: sample- and ROI-dependent learned soft routing
   over all six selected CLIP layers.
2. **Uniform + projector**: fixed weight `1/6` for every selected layer.
3. **Single-L24 + projector**: fixed weight 1 for CLIP layer 24 and 0
   for all other layers.

The brain-to-CLIP projector was enabled for every method, so differences
reflect the routing target rather than the presence or absence of the
projector.

## Per-subject results

Best validation alignment loss:

| Subject | Soft | Uniform | Single-L24 | Soft vs uniform | Soft vs Single-L24 |
|---|---:|---:|---:|---:|---:|
| subj01 | **0.103179** | 0.129467 | 0.141285 | 20.30% | 26.97% |
| subj02 | **0.102983** | 0.128507 | 0.140701 | 19.86% | 26.81% |
| subj05 | **0.101401** | 0.125884 | 0.139145 | 19.45% | 27.13% |
| subj07 | **0.103666** | 0.130385 | 0.141405 | 20.49% | 26.69% |

Soft routing achieved the lowest validation alignment loss for every
subject. The advantage was consistent rather than being driven by one
subject.

## Average results

| Method | Mean best validation loss | SD | Mean final validation loss |
|---|---:|---:|---:|
| Soft + projector | **0.102807** | 0.000849 | 0.103067 |
| Uniform + projector | 0.128561 | 0.001682 | 0.129324 |
| Single-L24 + projector | 0.140634 | 0.000900 | 0.140634 |

Across subjects, soft routing improved best validation loss by:

- 19.45%–20.49% relative to uniform routing;
- 26.69%–27.13% relative to single-L24 routing.

This supports the claim that the learned multi-layer mixture provides
value for the current Stage-1 alignment objective beyond both an
unstructured uniform mixture and the strongest tested single-layer
target.

## Routing interpretation

Expected CLIP-layer depth for ROI `r` was calculated as:

```text
expected_depth(r) = sum_l routing_weight(r,l) * selected_layer_index(l)
```

Cross-subject mean results:

| Method | Mean L24 usage | Early ROI mean depth | High-level ROI mean depth | High − early |
|---|---:|---:|---:|---:|
| Soft + projector | 0.3460 | 17.504 | 17.533 | +0.029 |
| Uniform + projector | 0.1667 | 14.000 | 14.000 | 0.000 |
| Single-L24 + projector | 1.0000 | 24.000 | 24.000 | 0.000 |

Soft routing learned an L24-biased multi-layer mixture rather than a
single-layer solution. Its advantage over single-L24 indicates that
non-L24 layers still contribute useful information to the alignment
target.

## Interpretation

The supported Stage-1 conclusion is:

> With the current frozen CLIP feature bank and brain-to-CLIP projector,
> learned soft multi-layer routing provides a stable representation
> alignment advantage across S1, S2, S5, and S7.

The result is reproducible across the four tested subjects and survives
comparison against uniform and single-L24 projector-enabled baselines.

This is evidence for the utility of adaptive multi-layer feature mixing
under the current Stage-1 loss. It is not evidence that individual ROIs
have learned distinct CLIP-layer correspondences.

## Limitation: no clear ROI-specific layer hierarchy

The mean expected depth difference between high-level and early visual
ROIs was only `+0.029` CLIP layers. This is negligible relative to the
selected depth range from layer 4 to layer 24.

Therefore, the current results do **not** support claims such as:

- early visual cortex preferentially aligns to intermediate CLIP layers;
- high-level category-selective cortex preferentially aligns to deeper
  CLIP layers;
- the learned router recovers a neuroanatomically meaningful hierarchy.

All ROI groups currently occupy a similar expected-depth regime. The
soft router's performance gain should be described as an adaptive
multi-layer representation advantage, not ROI-specific hierarchical
specialization.

## Artifacts

Machine-readable and Markdown tables are stored under:

`stage1_outputs/cross_subject_summary/`

Key files:

- `cross_subject_comparison.csv`
- `expected_depth_per_roi.csv`
- `per_subject_results.md`
- `average_results.md`
- `relative_improvements.md`
- `routing_<subject>_<method>.csv`

