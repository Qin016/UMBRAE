# NeuroRoute srFGW implementation audit

Audit date: 2026-08-20

## Scope and no-rerun guarantee

This is a repository audit and implementation map only. No FGW/srFGW code was
implemented, no UMBRAE or Shikra source was modified, no training/evaluation
experiment was launched, and no existing result artifact was overwritten.

The protected existing output roots are:

- `umbrae/stage1_outputs/`
- `umbrae/stage2_outputs/`
- `umbrae/umbrae_neuroroute_outputs/`
- `umbrae/train_logs/`
- the legacy workspace-level `stage1_outputs/`

Any later offline correspondence job must use a new output root and fail on an
already populated output directory by default. It must not call either Stage-1
or Stage-2 training entry point.

## Executive findings

The assumed eight-ROI, six-CLIP-layer pipeline exists and is usable as a frozen
offline representation source. For the completed subj01 alignment-only soft
checkpoint, the central contracts are:

```text
stored nsdgeneral sample       [3, 15724] float16
Stage-1 loader output          [B, 15724] float32
raw ROI tokens                 [B, 8, 1024]
projected ROI tokens           [B, 8, 1024]
CLIP hidden state per layer    [B, 257, 1024]
CLIP non-CLS patch bank        [B, 6, 256, 1024]
CLIP mean-patch layer features [B, 6, 1024]
routing weights                [B, 8, 6]
routed targets                 [B, 8, 1024]
original UMBRAE tokens         [B, 256, 1024]
strict Shikra bridge output    [B, 256, 4096]
```

The exact visual representation routed in Stage-1 is the mean of the 256
non-CLS patch tokens for each selected CLIP block. It is not the CLS token.
The six selected blocks are one-based layers `[4, 8, 12, 16, 20, 24]`.

The actual Stage-1 `BrainToCLIPProjector` is shared across all eight ROIs. The
actual tokenizer is also configured as `shared_mlp`, with a learned ROI identity
embedding added afterward. It is not configured with one projector per ROI.

The raw NSD shards retain repeat rows, `num_uniques.npy`, and `trial.npy`, so
repeat-derived reliability/noise summaries are technically possible in a new
offline reader. The current NeuroRoute loaders do not expose that information:
Stage-1 selects/averages a repeat before returning a sample, and Stage-2 always
averages it. Neither returns `trial`. A repeat-aware analysis therefore needs a
new read-only cache path and cannot simply consume batches from the existing
loaders.

No FGW, srFGW, optimal-transport solver, representation cache, geometry
analyzer, or stability analyzer currently exists in the repository. POT or an
equivalent srFGW dependency is also not declared in `umbrae/environment.yaml`.

## Files inspected

Core ROI and representation implementation:

- `umbrae/models/roi_mapping.py`
- `umbrae/models/roi_tokenizer.py`
- `umbrae/models/brain_clip_projector.py`
- `umbrae/models/clip_layer_bank.py`
- `umbrae/models/roi_layer_router.py`
- `umbrae/scripts/prepare_roi_indices.py`
- `umbrae/scripts/prepare_roi_nsdgeneral_indices.py`
- `umbrae/scripts/prepare_neuroroute_roi_collections.py`
- `umbrae/scripts/build_neuroroute_roi_mapping.py`
- `umbrae/scripts/build_nsd_roi_indices.py`
- `umbrae/scripts/verify_nsd_roi_files.py`
- `umbrae/scripts/train_stage1_routing.py`

Stage-2, UMBRAE, Perceiver, and Shikra integration:

- `umbrae/model.py`
- `umbrae/perceiver.py`
- `umbrae/models/neuroroute_mllm_adapter.py`
- `umbrae/models/umbrae_neuroroute_adapter.py`
- `umbrae/scripts/train_stage2_mllm_neuroroute.py`
- `umbrae/scripts/train_umbrae_neuroroute.py`
- `umbrae/utils.py`
- `umbrae/train.py`
- `umbrae/train_brainx.py`

Diagnostics, tests, configs, and result records:

- `umbrae/scripts/inspect_stage1_outputs.py`
- `umbrae/scripts/plot_routing_dynamics.py`
- `umbrae/scripts/analyze_routing_structure.py`
- `umbrae/scripts/summarize_cross_subject_stage1.py`
- `umbrae/scripts/summarize_stage1_retrieval.py`
- `umbrae/scripts/diagnose_stage1_checkpoint.py`
- `umbrae/scripts/coco73k_caption_utils.py`
- `umbrae/tests/test_roi_tokenizer_real.py`
- `umbrae/tests/test_stage1_routing_pipeline.py`
- `umbrae/configs/neuroroute_roi_set_v1_8roi.json`
- `roi_indices/subj{01,02,05,07}_neuroroute_v1.json`
- per-collection artifacts under `roi_indices/subj{01,02,05,07}/`
- checkpoint configs/metrics under
  `umbrae/stage1_outputs/cross_subject_projector/subj01/{soft,uniform,single_L24}/`
- adapter configs and summaries under `umbrae/stage2_outputs/` and
  `umbrae/umbrae_neuroroute_outputs/`
- `umbrae/docs/NEUROROUTE_ALL_EXPERIMENTS_SUMMARY.md`
- `umbrae/docs/STAGE1_ROUTING_RESULTS.md`
- `umbrae/docs/STAGE1_RETRIEVAL_RESULTS.md`
- `umbrae/docs/UMBRAE_NEUROROUTE_INTEGRATION.md`
- `umbrae/docs/UMBRAE_NEUROROUTE_RESULTS_S1.md`
- `umbrae/docs/NEUROROUTE_FULL_PIPELINE_STATUS.md`

Read-only metadata/content checks were also made against subj01 train/validation/
test tar shards and all four 300-sample validation shards under
`umbrae/nsd/webdataset_avg_split/`. The selected checkpoint was opened read-only
with memory mapping to confirm its epoch, config, state-key shapes, and frozen
identity CLIP projection heads.

## Actual component map

Paths below are relative to the workspace root
`/opt/data/private/BA/UMBRAE`.

| Concern | Actual path | Class/function | Audit finding |
|---|---|---|---|
| ROI-set definition | `umbrae/configs/neuroroute_roi_set_v1_8roi.json` | JSON config | Requires `V1,V2,V3,hV4,FFA,EBA,PPA,OPA`; excludes RSC because it has no `nsdgeneral` overlap. |
| ROI volume-to-vector preparation | `umbrae/scripts/prepare_roi_indices.py` | script functions and `main` | Builds whole-brain ROI index collections. |
| ROI-to-`nsdgeneral` conversion | `umbrae/scripts/prepare_roi_nsdgeneral_indices.py` | `build_flat_to_nsdgeneral_position`, `convert_roi_indices_to_nsdgeneral`, `validate_against_samples` | Converts volume-space ROI voxels to exact flattened `nsdgeneral.npy` positions and validates against paired whole-brain arrays. |
| ROI collection orchestration | `umbrae/scripts/prepare_neuroroute_roi_collections.py` | `iter_sample_pairs`, `validate_collection`, `aggregate_validation`, `main` | Runs conversion for `prf-visualrois`, `floc-bodies`, `floc-faces`, and `floc-places`; checks multiple samples/repeats. |
| Final eight-ROI merge | `umbrae/scripts/build_neuroroute_roi_mapping.py` | `load_collection`, `resolve_roi`, `merge_preserving_order`, `build_overlap_report`, `main` | Creates the real, voxel-order-verified merged mapping consumed by training. |
| Generic ROI mapping helper | `umbrae/models/roi_mapping.py` | `load_roi_indices`, `validate_roi_indices` | Supports a different `roi_indices` dictionary schema; it is not the loader used by current Stage-1/Stage-2 merged mappings. |
| Actual Stage-1 ROI loader | `umbrae/scripts/train_stage1_routing.py` | `load_roi_mapping` | Reads `payload["rois"][name]["indices"]` and requires both `roi_mapping_is_real` and `voxel_order_verified`. |
| Actual Stage-2 ROI loader | `umbrae/scripts/train_stage2_mllm_neuroroute.py` | `load_roi_mapping` | Duplicates the real/verified merged-mapping checks; strict UMBRAE integration imports this function. |
| ROI tokenization | `umbrae/models/roi_tokenizer.py` | `ROITokenizer` | Accepts `[B,V]` or `[B,repeats,V]`; its own 3-D path averages all rows before ROI extraction. Returns `roi_tokens` and stable ROI-name order. |
| Brain-to-CLIP projection | `umbrae/models/brain_clip_projector.py` | `BrainToCLIPProjector` | One shared linear/MLP/residual-MLP module applied to every ROI along the last dimension. |
| CLIP multi-layer bank | `umbrae/models/clip_layer_bank.py` | `CLIPLayerBank.forward`, `aggregate_layer_tokens` | Extracts hidden states for L4/L8/L12/L16/L20/L24, removes CLS, applies a per-layer linear head, and mean-pools patches. |
| ROI-to-layer routing | `umbrae/models/roi_layer_router.py` | `ROILayerRouter` | Produces `[B,R,L]` routing weights and `[B,R,D]` routed targets; queries come from raw ROI tokens. |
| Fixed routing baselines | `umbrae/scripts/train_stage1_routing.py` | `FixedLayerRouter` | Implements uniform, random, hard, and single-layer routing with the same output API. |
| Stage-1 model/training | `umbrae/scripts/train_stage1_routing.py` | `NSDTarDataset`, `Stage1RoutingModel`, `compute_losses`, `run_epoch`, `save_checkpoint`, `run_training` | Owns the actual end-to-end ROI/CLIP alignment implementation and output format. |
| Stage-1 checkpoint restoration | `umbrae/scripts/train_stage2_mllm_neuroroute.py` | `FrozenStage1FMRIEncoder` | Restores only `roi_tokenizer.*` and `brain_clip_projector.*`; it intentionally does not instantiate CLIP and does not restore the learned router. |
| Generic NeuroRoute Stage-2 adapter | `umbrae/models/neuroroute_mllm_adapter.py` | `NeuroRouteMLLMAdapter` | Produces a generic MLLM prefix; includes concat/gated/cross-attention modes. This is not the strict 256-token Shikra bridge. |
| Generic Stage-2 training | `umbrae/scripts/train_stage2_mllm_neuroroute.py` | `Stage2TarDataset`, `NeuroRouteStage2Model`, `run_training` | Earlier/generic `inputs_embeds` Stage-2 path and shared checkpoint loader. |
| Strict UMBRAE-NeuroRoute adapter | `umbrae/models/umbrae_neuroroute_adapter.py` | `UMBRAENeuroRouteAdapter`, `_PerceiverBlock` | Fuses original UMBRAE and projected ROI tokens, resamples to exactly 256 tokens, and projects 1024 to Shikra's 4096-dimensional embedding space. |
| Original UMBRAE Perceiver | `umbrae/model.py` and `umbrae/perceiver.py` | `Perceiver`, `BrainX`, `BrainXS`, `PerceiverAttention`, `PerceiverResampler` | Original UMBRAE brain encoder and its six-layer Perceiver resampler. |
| Strict Stage-2 training/bridge | `umbrae/scripts/train_umbrae_neuroroute.py` | `FrozenUMBRAEEncoder`, `UMBRAENeuroRouteModel`, `shikra_image_prompt`, `replace_shikra_patch_embeddings`, `run_training` | Current strict integration. Requires 256 visual tokens in `shikra_patch` mode and replaces the positions between one `<im_start>` and the expected `<im_end>`. |

## ROI mappings and subject-specific files

The final mappings used by the completed runs are:

- `roi_indices/subj01_neuroroute_v1.json`
- `roi_indices/subj02_neuroroute_v1.json`
- `roi_indices/subj05_neuroroute_v1.json`
- `roi_indices/subj07_neuroroute_v1.json`

All four declare `roi_mapping_is_real=true`,
`voxel_order_verified=true`, the same stable ROI order, no unresolved required
ROI, and validation over three samples/nine repeat arrays.

| Subject | V1 | V2 | V3 | hV4 | FFA | EBA | PPA | OPA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subj01 | 1350 | 1433 | 1187 | 687 | 687 | 2238 | 720 | 1532 |
| subj02 | 1102 | 1075 | 1097 | 483 | 868 | 2351 | 755 | 1322 |
| subj05 | 1113 | 1081 | 925 | 542 | 763 | 2869 | 975 | 1162 |
| subj07 | 1142 | 986 | 726 | 397 | 343 | 2294 | 652 | 1078 |

The per-collection source artifacts are under `roi_indices/subjXX/`, including
`*_indices.pkl`, `*_nsdgeneral_indices.pkl`, `*_nsdgeneral_indices.npz`,
`*_meta.json`, and `subjXX_validation.json` for the four ROI collections.

One important schema discrepancy exists: `models.roi_mapping.load_roi_indices`
expects either a top-level ROI dictionary or `payload["roi_indices"]`. The final
`subjXX_neuroroute_v1.json` files instead contain metadata plus
`payload["rois"][roi]["indices"]`. Current training works because both training
scripts define their own `load_roi_mapping`; the generic model helper should not
be used directly for these merged JSON files without an adapter.

## Exact tensor and representation contracts

### fMRI and ROI path

For subj01, a raw shard member is `nsdgeneral.npy: [3,15724]`. The Stage-1
`NSDTarDataset._select_repeats` uses the first `num_uniques` rows and returns
one `[15724]` vector according to `fmri_repeat_mode`. All completed cross-subject
projector runs use `fmri_repeat_mode="mean"`. Batching therefore supplies
`[B,15724]` to `Stage1RoutingModel`, not `[B,3,15724]`.

`ROITokenizer` is configured in `Stage1RoutingModel.__init__` as:

```text
tokenizer_type       shared_mlp
token_dim            1024
use_roi_embeddings   true
num ROIs             8
```

For subj01, the largest ROI is EBA with 2238 voxels. The shared tokenizer pads
each ROI vector to `[B,8,2238]`, applies one MLP
`LayerNorm(2238) -> Linear(2238,1024) -> GELU -> Linear(1024,1024)`, then adds
the `[8,1024]` ROI embedding table. Its output is:

```text
raw roi_tokens z: [B,8,1024]
```

`BrainToCLIPProjector` is one shared MLP, not a `ModuleList` indexed by ROI:

```text
LayerNorm(1024)
-> Linear(1024,1024)
-> GELU
-> Dropout(0.1)
-> Linear(1024,1024)
-> Dropout(0.1)
```

It is broadcast over the `[B,8]` leading axes and returns:

```text
projected_roi_tokens: [B,8,1024]
```

The learned soft router receives **raw** `roi_tokens`, while the alignment loss
compares **projected** ROI tokens with routed CLIP targets. This distinction must
be retained in the cache schema.

### CLIP path and pooling

`CLIPLayerBank` uses CLIP ViT-L/14 at 224 pixels. Its 16-by-16 patch grid gives
256 patch positions plus one CLS position. For each one-based selected block
`l`, the implementation reads:

```python
outputs.hidden_states[l][:, 1:, :]
```

Thus the exact per-layer source is the block hidden state with the CLS token
discarded. It is neither CLS pooling nor a pooled CLIP model output. A separate
linear head is defined for each selected layer; with CLIP hidden size and target
size both equal to 1024, these heads are initialized to identity. Stage-1 then
freezes the entire `CLIPLayerBank`, including these heads. Inspection of the
selected subj01 checkpoint confirmed exactly identity weights and zero bias for
all six heads.

The returned tensors are:

```text
layer_tokens:  [B,6,256,1024]
pooled_tokens: [B,6,1024] = mean(layer_tokens, dim=patch)
```

`Stage1RoutingModel.forward` passes `pooled_tokens` to the router. The relevant
visual node representation for initial ROI-to-layer structural analysis is
therefore `[B,6,1024]` mean-patch features. The `[B,6,256,1024]` bank is only
needed if a later, separately specified patch-level analysis is desired.

### Perceiver and LayerNorm placement

There are three attention implementations and they should not be conflated:

1. In the strict `UMBRAENeuroRouteAdapter`, UMBRAE source tokens first pass
   through `umbrae_input[0] = LayerNorm(umbrae_dim)` and ROI tokens through
   `roi_expander[0] = LayerNorm(neuroroute_dim)`. For the default
   `perceiver_resampler`, each `_PerceiverBlock.forward` then applies
   `query_norm` to the 256 learned queries and `source_norm` to keys and values
   immediately inside the cross-attention call. The FFN is also pre-normalized.
2. In original UMBRAE, `model.Perceiver.forward` applies `ln_vision` before
   calling `PerceiverResampler`. Each `PerceiverAttention.forward` additionally
   applies `norm_media` and `norm_latents`, and `PerceiverResampler` applies a
   final `norm` after all residual blocks.
3. `NeuroRouteMLLMAdapter` has a separate `nn.MultiheadAttention` option in the
   generic Stage-2 path. Its query/key/value inputs are not separately normalized
   before attention; its `prefix_projector` begins with LayerNorm only after
   fusion.

The completed strict subj01 experiments use item 1 with
`fusion_type="perceiver_resampler"`, not the generic adapter in item 3.

## NSD repeat and trial audit

The local data are tar shards under:

- `umbrae/nsd/webdataset_avg_split/train/`
- `umbrae/nsd/webdataset_avg_split/val/`
- `umbrae/nsd/webdataset_avg_split/test/`

There are no standalone `trial.npy` or `num_uniques.npy` files in this tree;
they are members of each sample group inside the tar files. Every inspected
validation sample contains:

```text
sampleXXXXXXXXXXXX.nsdgeneral.npy  [3,V] float16
sampleXXXXXXXXXXXX.num_uniques.npy [1]   int64
sampleXXXXXXXXXXXX.trial.npy       [1]   int64
sampleXXXXXXXXXXXX.coco73k.npy     [1]   int64
```

The 300-sample validation shard counts are:

| Subject | `num_uniques=1` | `=2` | `=3` | Missing `trial`/`num_uniques` |
|---|---:|---:|---:|---:|
| subj01 | 12 | 24 | 264 | 0 |
| subj02 | 13 | 30 | 257 | 0 |
| subj05 | 9 | 28 | 263 | 0 |
| subj07 | 18 | 30 | 252 | 0 |

For additional subj01 checks, `train_subj01_0.tar` has counts `{1:14, 2:57,
3:429}` and `test_subj01_0.tar` has `{1:14, 2:60, 3:426}`. A one-repeat sample
was observed with all three stored rows duplicated, confirming that the full
fixed-size array must not be treated as three independent measurements. The
current Stage-1 convention of using only `fmri[:num_uniques]` is the safe local
contract. No semantics should be assigned to rows beyond that count.

Current loader behavior is:

- `scripts.train_stage1_routing.NSDTarDataset` reads `num_uniques`, selects
  valid rows, reduces them by `mean`, `first`, or `random`, and does not include
  `trial.npy` in its sample index or return dictionary.
- `scripts.train_stage2_mllm_neuroroute.Stage2TarDataset` reads
  `num_uniques`, always averages valid rows, and does not read/return `trial`.
- `models.roi_tokenizer.ROITokenizer` can accept `[B,repeats,V]`, but blindly
  averages the full repeat axis and has no valid-repeat mask. Passing padded
  three-row tensors to it is therefore not a valid replacement for using
  `num_uniques`.
- `utils.get_dataloaders` knows the `trial` and `reps` member names, but actual
  original UMBRAE training calls override `to_tuple` to `['voxels','images']`.

Conclusion: noise-aware geometry is feasible from the raw shard contents for
samples with `num_uniques >= 2`, but is not exposed by the current NeuroRoute
data loaders. The initial main representation should reproduce the checkpoint's
mean-input protocol. A secondary reliability cache may encode each valid repeat
individually by flattening the valid repeat axis into the batch, then retain an
explicit repeat mask. Because the tokenizer is nonlinear, the token of the mean
fMRI vector and the mean of repeat-wise tokens must be stored as different
quantities, not treated as interchangeable.

`trial.npy` is only one scalar per stimulus sample, not one ID per repeat row.
It can be preserved as sample metadata, but the audited code does not establish
session/run identity for individual repeat rows. Session-aware or temporal-noise
protocols would require additional metadata and must not be inferred from this
scalar.

## Frozen representation probe selection

Use this checkpoint for the initial subj01 offline analysis:

```text
umbrae/stage1_outputs/cross_subject_projector/subj01/soft/checkpoint_best.pt
```

Reasons:

- It is the completed 10-epoch, non-debug, projector-enabled subj01 soft run.
- It uses the full six-layer bank and the desired mean-repeat protocol.
- Its best checkpoint is epoch 10 with validation alignment loss `0.1031794975`.
- It is the main alignment-only representation and outperforms the matched
  uniform (`0.129467`) and single-L24 (`0.141285`) Stage-1 objectives.
- The contrastive checkpoints deliberately trade worse alignment for retrieval,
  so they are not an unconditional replacement for a structural representation
  probe.
- Uniform remains a necessary frozen control and single-L24 a necessary depth
  control, but neither should be the primary probe.

The exact matched controls are:

```text
umbrae/stage1_outputs/cross_subject_projector/subj01/uniform/checkpoint_best.pt
umbrae/stage1_outputs/cross_subject_projector/subj01/single_L24/checkpoint_best.pt
```

`FrozenStage1FMRIEncoder` is reusable for the brain branch. It restores the
tokenizer and shared brain-to-CLIP projector only. For exact visual features, a
new cache loader must also reconstruct `CLIPLayerBank` from checkpoint config
and load the `clip_layer_bank.*` state, or equivalently verify the frozen CLIP
model plus identity heads. The offline analysis must not use `routed_targets`
as the six visual nodes because those targets are already mixed across layers
and would make an ROI-to-layer correspondence analysis circular.

## Existing routing diagnostics

Reusable diagnostics and artifacts include:

- `umbrae/models/roi_layer_router.py::ROILayerRouter.forward`: temperature,
  mean entropy, mean maximum weight, active layers per ROI, logits, and weights.
- `umbrae/scripts/train_stage1_routing.py::run_epoch`: loss components, raw and
  projected norms, routed-target norm, retrieval proxy, and validation routing
  mean/std matrices.
- `umbrae/scripts/train_stage1_routing.py::build_routing_dynamics_record`:
  per-epoch top layers, layer usage, entropy, and maximum weight.
- `umbrae/scripts/inspect_stage1_outputs.py::routing_diagnostics` and
  `inspect_output`: validates and compares final `[R,L]` matrices and can export
  CSV.
- `umbrae/scripts/plot_routing_dynamics.py`: plots per-epoch routing dynamics.
- `umbrae/scripts/summarize_cross_subject_stage1.py`: cross-subject loss,
  expected-depth, and routing summaries.
- `umbrae/scripts/train_stage2_mllm_neuroroute.py::routing_diagnostics`:
  L24 usage and early/high-level expected depth.
- `umbrae/scripts/analyze_routing_structure.py`: entropy, coverage, ROI
  diversity, expected-depth gap, CSV, JSON, and optional heatmaps.
- `umbrae/scripts/diagnose_stage1_checkpoint.py`: checkpoint completeness and
  retrieval-pooler diagnostics.

These summarize learned routing matrices, not representation geometry, transport
plans, srFGW objective components, or stability. They can supply labels and
comparison baselines but do not replace the proposed geometry analysis.

## Existing output inventory

Important completed output locations are:

- Stage-1 matched four-subject runs:
  `umbrae/stage1_outputs/cross_subject_projector/subjXX/{soft,uniform,single_L24}/`
- Stage-1 cross-subject summaries:
  `umbrae/stage1_outputs/cross_subject_summary/`
- Stage-1 retrieval runs/summaries:
  `umbrae/stage1_outputs/retrieval/` and
  `umbrae/stage1_outputs/retrieval_summary/`
- Generic subj01 Stage-2:
  `umbrae/stage2_outputs/subj01_real_3epochs/`
- Strict subj01 UMBRAE-NeuroRoute five-way ablation:
  `umbrae/umbrae_neuroroute_outputs/subj01_real_3epochs/{umbrae_only,neuroroute_only,umbrae_plus_soft,umbrae_plus_uniform,umbrae_plus_single_l24}/`
- Structured alpha/temperature runs:
  `umbrae/umbrae_neuroroute_outputs/subj01_structured/{alpha_0p1,alpha_0p3,alpha_0p5,temp_2p0}/`
- Structured routing diagnostics:
  `umbrae/umbrae_neuroroute_outputs/subj01_structured/routing_analysis/`
- Smoke/benchmark outputs:
  `umbrae/umbrae_neuroroute_outputs/subj01_smoke/`,
  `subj01_structured_smoke/`, and `subj01_benchmark/`
- A separate legacy debug output exists at
  `stage1_outputs/subj01_debug/` outside the `umbrae/` project directory.

The strict completed runs record `bridge_type="shikra_patch"`,
`num_visual_tokens=256`, frozen UMBRAE/NeuroRoute/Shikra, and
`fusion_type="perceiver_resampler"`. The ordinary `umbrae_plus_soft`,
`umbrae_plus_uniform`, and `umbrae_plus_single_l24` modes use ROI representations
learned under their corresponding Stage-1 checkpoints. Only the structured
modes use the saved `[R,L]` routing matrix directly in token expansion; ordinary
soft/uniform/single modes do not consume it inside the adapter.

## Reusable modules

The first offline implementation should reuse, without modifying:

- `ROITokenizer`
- `BrainToCLIPProjector`
- `CLIPLayerBank`
- `FrozenStage1FMRIEncoder` for exact brain checkpoint restoration
- Stage-1 `load_roi_mapping` semantics for the merged mapping schema
- `coco73k_caption_utils.expand_paths` and its safe tar/NPY helpers where useful
- existing ROI names, selected-layer JSON/config values, and routing matrices as
  labels/controls

The strict UMBRAE encoder, adapter, Shikra model, caption dataset, and patch
replacement code are not required for offline ROI-to-CLIP geometry and should
not be imported or modified.

## Missing modules and dependencies

The following are absent:

- a repeat-preserving, mask-aware offline NSD representation reader;
- a versioned cache schema containing sample IDs, COCO IDs, trial scalar,
  repeat count/mask, mean-input brain tokens, optional repeat-wise brain tokens,
  and per-layer CLIP features;
- within-domain ROI and CLIP-layer geometry construction;
- feature-cost construction with explicit normalization choices;
- an srFGW solver/wrapper with marginal validation and objective decomposition;
- null/permutation baselines, bootstrap confidence intervals, seed sweeps, and
  repeat/subsample stability analysis;
- cache provenance checks tying outputs to checkpoint/config/mapping/data split;
- a declared POT or equivalent optimal-transport dependency.

No file matching FGW, srFGW, or Gromov-Wasserstein implementation terminology
was found in the Python/Markdown/config source audit.

## Minimal proposed files

Do not add these during this audit. The next implementation step should remain
offline and minimal:

1. `umbrae/scripts/cache_roi_clip_representations.py`
   - Read existing shards and the selected frozen checkpoint only.
   - Reproduce the Stage-1 mean-repeat brain representation exactly.
   - Optionally cache valid repeats separately with a mask; never pass padded
     rows blindly through `ROITokenizer`.
   - Cache projected ROI nodes `[N,8,1024]` and mean-patch CLIP layer nodes
     `[N,6,1024]`; cache raw ROI nodes separately for sensitivity analysis.
   - Preserve `sample_key`, `coco73k_id`, scalar `trial`, `num_uniques`, ROI
     order, selected layers, checkpoint path/hash, mapping path/hash, dtype, and
     split provenance.
   - Default to a new cache root and refuse overwrite.
2. `umbrae/scripts/analyze_representation_geometry.py`
   - Build/validate ROI and layer relational matrices from the cache.
   - Make metric, centering, normalization, aggregation unit, and repeat-noise
     correction explicit.
   - Emit descriptive geometry and null diagnostics without solving srFGW.
3. `umbrae/models/fgw_correspondence.py`
   - Contain only solver-facing data validation, cost normalization, srFGW call,
     transport-plan checks, and objective-component reporting.
   - Keep PyTorch model/training code out of this module.
4. `umbrae/scripts/run_fgw_correspondence.py`
   - Consume immutable caches/geometries, run a declared hyperparameter grid,
     and save plans/metrics/config to a new output directory.
   - Never invoke Stage-1 or Stage-2 training.
5. `umbrae/scripts/analyze_fgw_stability.py`
   - Bootstrap samples, vary seeds/subsamples, compare mean-input and
     noise-aware variants, run label-permutation/null controls, and summarize
     plan stability with uncertainty.

Suggested execution order is cache validation, descriptive geometry, a small
deterministic solver validation on synthetic costs, the primary frozen soft
probe, the uniform/single-L24 controls, and finally repeat/bootstrap stability.
This is an implementation plan, not authorization to run any of those steps.

## Known engineering and scientific risks

- **Schema duplication:** three different ROI loaders exist, and the generic
  `models.roi_mapping` schema does not match the merged mapping JSON.
- **Script-level model ownership:** `Stage1RoutingModel` and
  `FrozenStage1FMRIEncoder` live in training scripts rather than a reusable model
  module. Importing them is possible but creates avoidable coupling.
- **Repeat masking:** `ROITokenizer` averages all rows of a 3-D input and cannot
  honor per-sample `num_uniques`; padded/duplicated rows would bias reliability.
- **Nonlinear repeat aggregation:** tokenizing the mean fMRI and averaging
  repeat-wise tokens are different estimands.
- **Trial semantics:** one scalar `trial` per sample is insufficient for
  session/run-aware repeat noise modeling.
- **Circular targets:** routed targets already combine CLIP layers using the
  learned router and must not serve as the layer-node bank for discovering
  correspondence.
- **Representation ambiguity:** raw ROI tokens drive the router, but projected
  ROI tokens are aligned to CLIP. The primary analysis should use projected
  tokens and report raw-token sensitivity rather than silently choosing one.
- **Pooling scope:** current Stage-1 evidence concerns mean patch features.
  Switching to CLS or patch-level geometry changes the experiment.
- **Small node sets:** the transport plan is only 8-by-6; apparent structure can
  be unstable even with many stimuli and needs null/stability analysis.
- **Scale and metric sensitivity:** within-domain distances and the fused feature
  term require normalization before varying the FGW trade-off parameter.
- **Semi-relaxed marginals:** the fixed and free marginals, mass normalization,
  orientation of the 8-by-6 plan, convergence tolerance, and initialization must
  be explicit and validated.
- **Dependency drift:** no OT package is pinned, while the environment declares
  Python 3.8/PyTorch 1.13-era dependencies. Solver/API compatibility must be
  checked before adding a dependency.
- **Cache size:** patch-token caches are about 256 times larger than pooled layer
  caches and are unnecessary for the initial 8-ROI/6-layer study.
- **CLIP exactness:** use the same hidden-state indexing, preprocessing, frozen
  identity heads, and local/model revision as Stage-1; do not substitute a
  generic CLIP pooled output.
- **Checkpoint leakage/provenance:** never fit preprocessing or choose srFGW
  hyperparameters on held-out results without recording that choice. Cache keys
  must include checkpoint, mapping, subject, and split.
- **Output safety:** several completed output trees already exist, including a
  duplicate legacy root. New scripts must require an explicit new destination
  and refuse overwrite by default.

## Discrepancies between the assumed and actual pipeline

1. Repeat data exist in the raw shards, but the current NeuroRoute loaders do
   not expose repeat tensors or trial metadata to models.
2. CLIP per-layer representations are mean non-CLS patch tokens, not CLS tokens.
3. The actual brain projector is shared across ROIs; only ROI identity embeddings
   and ROI voxel content distinguish ROI tokens.
4. The learned router consumes raw ROI tokens, while alignment and downstream
   Stage-2 use projected ROI tokens.
5. `FrozenStage1FMRIEncoder` does not restore the router or CLIP bank. Stage-2
   loads a saved validation-mean routing matrix from a neighboring NPY file when
   needed.
6. In the strict adapter, ordinary soft/uniform/single variants differ through
   their frozen Stage-1 ROI representation checkpoints. Direct routing-matrix
   conditioning is enabled only for the structured alpha/temperature modes.
7. There are two Stage-2 systems: a generic prefix implementation and the later
   strict UMBRAE/Shikra 256-position bridge. The existing strict experiments use
   the latter.
8. The general-purpose ROI loader is not compatible with the final merged ROI
   JSON schema used by training without adaptation.
9. Existing routing diagnostics analyze learned `[8,6]` weights and expected
   depth; no existing code analyzes representational geometry or transport
   stability.

## Audit stop point

The repository is ready for a separate, offline cache-and-analysis
implementation using the completed subj01 soft checkpoint as the initial frozen
probe and uniform/single-L24 as controls. That implementation has intentionally
not begun here. Existing UMBRAE, Shikra, Stage-1, Stage-2, and experiment outputs
remain unchanged and will not be rerun or overwritten by this audit.

## Post-hoc verification after Prompt 0–2

Verification date: 2026-08-20

```text
AUDIT_STATUS = VERIFIED_READY
```

This status was assigned only after checking each hard-gate component against
the current filesystem and opening the selected Stage-1 checkpoint read-only.
None of the eight required components below was inferred from the prompt.

### Repository identity

```text
current_working_directory = /opt/data/private/BA/UMBRAE
git_repository_present = true
git_root = /opt/data/private/BA/UMBRAE
branch = main
HEAD = ea5dab0042ae0c6d089379f39cdd46cb2973dd00
git_status = unavailable
git_status_reason = Git executable is absent; branch and HEAD were read from
                    .git/HEAD and .git/refs/heads/main
```

The repository identity is therefore recoverable, but dirty/clean worktree
provenance is unavailable. Lack of the Git executable did not block the audit
because the required source and artifacts are locally present and readable.

### Hard-gate component verification

Paths are relative to `/opt/data/private/BA/UMBRAE`.

| Required component | Exact path | Exact class/function or artifact | Code-backed finding | Result |
|---|---|---|---|---|
| Real NSD eight-ROI mapping | `roi_indices/subj01_neuroroute_v1.json` | JSON artifact, `rois[*].indices` | Read directly: `subject=subj01`, `roi_mapping_is_real=true`, `voxel_order_verified=true`, `roi_names=[V1,V2,V3,hV4,FFA,EBA,PPA,OPA]`, and `unresolved_rois=[]`. | FOUND, not assumed |
| ROI index loader | `umbrae/scripts/train_stage1_routing.py` | `load_roi_mapping` | Reads the merged `payload["rois"][name]["indices"]` schema and enforces the real/verified flags. `train_stage2_mllm_neuroroute.py::load_roi_mapping` implements the same validation for Stage-2. | FOUND, not assumed |
| Real-ROI voxel encoder | `umbrae/models/roi_tokenizer.py` | `ROITokenizer` | Extracts subject-specific voxel indices, pads ROI vectors for the configured shared MLP, adds ROI identity embeddings, and returns `roi_tokens`. | FOUND, not assumed |
| ROI-to-CLIP projection | `umbrae/models/brain_clip_projector.py` | `BrainToCLIPProjector` | Applies one shared projector along the last dimension of `[B,8,1024]`; it is not an ROI-specific `ModuleList`. | FOUND, not assumed |
| CLIP L4/L8/L12/L16/L20/L24 extraction | `umbrae/models/clip_layer_bank.py` | `CLIPLayerBank.forward`, `aggregate_layer_tokens` | Reads `outputs.hidden_states[layer][:,1:,:]`, stacks the six configured blocks, applies per-layer heads, then `mean(dim=2)` over non-CLS patches. | FOUND, not assumed |
| Stage-1 NeuroRoute checkpoint | `umbrae/stage1_outputs/cross_subject_projector/subj01/soft/checkpoint_best.pt` | epoch-10 soft checkpoint artifact | Read-only `torch.load(..., map_location="meta", mmap=True)` confirmed `router_type=soft`, layers `[4,8,12,16,20,24]`, and both `roi_tokenizer.*` and `brain_clip_projector.*` state keys. File size is 1,328,854,034 bytes. | FOUND, not assumed |
| Strict UMBRAE-NeuroRoute Stage-2 integration | `umbrae/models/umbrae_neuroroute_adapter.py`; `umbrae/scripts/train_umbrae_neuroroute.py` | `UMBRAENeuroRouteAdapter`, `FrozenUMBRAEEncoder`, `UMBRAENeuroRouteModel`, `replace_shikra_patch_embeddings`, `run_training` | The model obtains projected ROI tokens from the frozen Stage-1 encoder, combines them with UMBRAE tokens, and the strict bridge replaces exactly the Shikra image-patch positions. Completed adapter configs record `shikra_patch`, 256 tokens, and frozen upstream components. | FOUND, not assumed |
| Perceiver used by completed NeuroRoute experiments | `umbrae/models/umbrae_neuroroute_adapter.py` | `_PerceiverBlock`, `UMBRAENeuroRouteAdapter` | `_PerceiverBlock.forward` applies `query_norm` and `source_norm` immediately before cross-attention. Completed strict experiment configs record `fusion_type="perceiver_resampler"`. Original UMBRAE Perceiver code also exists at `umbrae/model.py::Perceiver` and `umbrae/perceiver.py::{PerceiverAttention,PerceiverResampler}` but is a distinct path. | FOUND, not assumed |

The post-hoc checks agree with the original audit's shape, pooling, shared
projector, Stage-2, and LayerNorm findings. The hard gate did not identify a
repository mismatch.

### Prompt-1/2 artifact verification

The Prompt-1 cache is present at:

```text
umbrae/fgw_cache/subj01/train_seed42/
```

Its three immutable feature groups are present and reusable:

| Existing group | Rows | Brain token shape | Projected shape | CLIP shape | Reuse status |
|---|---:|---|---|---|---|
| `geometry_fit` | 5,135 | `[5135,8,1024]` | `[5135,8,1024]` | `[5135,6,1024]` | reusable by ID/index |
| `feature_cost_fit` | 1,712 | `[1712,8,1024]` | `[1712,8,1024]` | `[1712,6,1024]` | reusable by ID/index |
| `heldout_eval` | 1,712 | `[1712,8,1024]` | `[1712,8,1024]` | `[1712,6,1024]` | reusable by ID/index |

All split-level `metadata.json` and `cache_config.json` files are present.
Across the three groups there are 8,559 rows and 8,559 unique stable
`coco73k:<id>` identifiers; every row also records original `sample_index`,
split-local index, scalar `trial_id`, subject, source dataset split, source tar,
and `number_of_repeats`. The cached fMRI features are repeat-mean tokens, not
repeat-resolved measurements.

The original Prompt-2 output and report remain present and are potentially
reusable only as legacy diagnostics:

```text
umbrae/fgw_geometry/subj01/projected_cosine_spearman_seed42_n1000_b100/
umbrae/docs/FGW_GEOMETRY_RELIABILITY_S1.md
```

All required matrices, JSON summaries, CSV diagnostics, and heatmaps were found.
They must not be overwritten. In particular, their disjoint-stimulus
`split-half` analysis measures stimulus-subset stability rather than neural
measurement reliability, their primary representation is projected tokens, and
their bootstrap-with-replacement is retained only as a legacy secondary
diagnostic after remediation.

No cache or geometry artifact was deleted or regenerated during this Prompt-0R
verification. Verification passed, so the provenance/split repair may proceed
without re-extracting CLIP or brain features.
