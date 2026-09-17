# FGW Cross-Subject Locked Validation V1

## Scope and hard stop

This report records Prompt 5A only: locked, offline validation replication on
NSD `subj02`, `subj05`, and `subj07`. The frozen `subj01` result was used only
as a protocol reference. No replication-subject `offline_test` index array or
feature array was loaded after manifest creation; no discovery+validation refit,
Prompt 5B, Stage-2, caption experiment, UMBRAE training, Shikra training, or
Stage-1 training was run.

The locked scientific configuration was:

```text
method = sr_fgw
beta = 0.5
lambda_cov = 0
entropy = 0
ROIs = [V1, V2, V3, hV4, FFA, EBA, PPA, OPA]
CLIP layers = [4, 8, 12, 16, 20, 24]
brain input = roi_tokens_before_projector
CLIP pooling = mean_non_cls_patch_tokens
validation composite = 0.5 * normalized feature loss + 0.5 * normalized GW loss
```

The global preregistration was written before any replication validation
feature access. Its SHA-256 is
`881ff0400521dbfccb93cef6adddc06e766236d141eeea1af457c7b30959f482`.

## Main result

| Subject | Geometry stable | Feature-only preferred layer | Selected seed | FGW val composite | Baseline val composite | ΔGW | Structural null pass | Identifiability | Validation status |
| ------- | --------------- | ---------------------------- | ------------- | ----------------: | ---------------------: | --: | -------------------- | --------------- | ----------------- |
| subj01 (reference only) | GO | L24 | random-1006 | 0.55443 | 0.72558 | -0.25837 | yes | pass; known FFA/PPA/OPA ambiguity | READY_FOR_OFFLINE_TEST (historical) |
| subj02 | GO | L24 | feature-informed | 0.54025 | 0.69405 | -0.22872 | yes | pass; FFA/PPA locally ambiguous | READY_FOR_OFFLINE_TEST |
| subj05 | GO | L24 | random-1013 | 0.50236 | 0.71911 | -0.35459 | yes | pass | READY_FOR_OFFLINE_TEST |
| subj07 | GO | L24 | feature-informed | 0.55121 | 0.72886 | -0.18124 | **no** | primary threshold passes, but matched-plan reproducibility fails; hV4/PPA ambiguous | **INCONCLUSIVE** |

`ΔGW` is the full-validation normalized GW loss of locked srFGW minus that of
the frozen primary non-structural baseline; lower and more negative is better.
The fixed primary baseline is
`source_constrained_feature_coverage(beta=0, lambda_cov=1, entropy=0)`.
`balanced_outer_ot` was evaluated only as the fixed secondary baseline.

Prompt-5A statuses are therefore:

```text
subj02 VALIDATION_STATUS = READY_FOR_OFFLINE_TEST
subj05 VALIDATION_STATUS = READY_FOR_OFFLINE_TEST
subj07 VALIDATION_STATUS = INCONCLUSIVE
```

Only `subj02` and `subj05` are eligible for a future, separately authorized
Prompt 5B. No such test has been run here.

## Preflight and representation caches

All three subjects passed the asset preflight. Each has a real, voxel-order
verified, subject-specific mapping with no missing ROI, the same ROI-tokenizer
architecture and 1,024-dimensional Stage-1 representation protocol, the exact
six CLIP layers, and non-CLS patch-token mean pooling.

| Subject | V1 | V2 | V3 | hV4 | FFA | EBA | PPA | OPA |
| ------- | --: | --: | --: | ---: | --: | --: | --: | --: |
| subj02 | 1102 | 1075 | 1097 | 483 | 868 | 2351 | 755 | 1322 |
| subj05 | 1113 | 1081 | 925 | 542 | 763 | 2869 | 975 | 1162 |
| subj07 | 1142 | 986 | 726 | 397 | 343 | 2294 | 652 | 1078 |

No reusable caches existed, so frozen inference caches were generated from the
existing subject-specific Stage-1 `soft/checkpoint_best.pt` files. Their exact
arrays are:

```text
brain_roi_features      [N, 8, 1024]  (before BrainToCLIPProjector)
projected_roi_features  [N, 8, 1024]
clip_layer_features     [N, 6, 1024]
```

The caches are offline-only, contain stable `coco73k` stimulus IDs, have no
NaN/Inf values, and record deterministic ordering and checkpoint/mapping hashes.
The extraction batch size was increased from 16 to 64 for throughput; models
remained in frozen eval/inference mode and representation semantics were
unchanged. The interrupted partial subj02 cache was moved to the system trash
and is recoverable; it is not referenced by any result.

## Leakage-safe manifests

The same seed-42 unique-stimulus algorithm used for subj01 was used. All
downstream validation/test IDs were protected and all overlap counts are zero.
No repeated stimulus group crosses an offline split.

| Subject | Eligible unique stimuli | Discovery | Validation | Offline test (sealed) | Manifest SHA-256 |
| ------- | ----------------------: | --------: | ---------: | --------------------: | ---------------- |
| subj02 | 8559 | 5135 | 1712 | 1712 | `a1c3279359137508282ed5dffe0982ff9f53688235e676688d72dc66edffc815` |
| subj05 | 8181 | 4909 | 1636 | 1636 | `6e8b1836eccdab8917617e0280ba6069ee80bb23b515cabead116fa50c93e591` |
| subj07 | 8189 | 4913 | 1638 | 1638 | `bc53abad6bc62ce32b2a228d7d52022fdd0783e0dfc162ca914c9f5bc9ce6ad8` |

The subj05/subj07 counts differ from subj01 because their verified eligible
training sets contain 8,181 and 8,189 unique stimuli. The 60/20/20 rule was
applied without forcing subj01's numerical counts.

## Discovery geometry and feature compatibility

Primary geometry used cosine stimulus RDMs and one minus Spearman correlation
between RDM upper triangles. It is an **ROI representational geometry relation
matrix**, not anatomical distance and not evidence of a biological hierarchy.

For 100 pairs of disjoint 600-stimulus subsets from a fixed 1,500-stimulus
discovery anchor set, mean upper-triangle Spearman stability was:

| Subject | Brain mean ρ | CLIP mean ρ | Independent-ROI geometry null p | Geometry status |
| ------- | -----------: | ----------: | ------------------------------: | --------------- |
| subj02 | 0.98997 | 0.99157 | 0.001996 | GO |
| subj05 | 0.98579 | 0.99561 | 0.001996 | GO |
| subj07 | 0.98820 | 0.99275 | 0.001996 | GO |

These are stimulus-subset stability estimates, not neural measurement
reliability. Repeat-based reliability and crossnobis remain unavailable because
the cache contains one ROI-token row after averaging valid fMRI repeats and
does not preserve independent repeat/session partitions. No disjoint-stimulus
proxy was relabeled as repeat reliability.

The discovery feature matrix `M [8,6]` used five unique-stimulus folds,
multi-output ridge probes with `alpha=100`, fold-local brain centering/scaling,
fold-local CLIP centering, and mean OOF cosine error. All three subjects again
favored L24 descriptively. This observation was not a PASS criterion.

## Subject-specific calibration and frozen fitting

The numerical subj01 scales were not reused. Each subject was calibrated once
on discovery matrices using the uniform feasible plan
`T_ref[r,l]=(1/8)*(1/6)`:

| Subject | s_feature | s_gw |
| ------- | --------: | ---: |
| subj02 | 0.18379350 | 0.19451308 |
| subj05 | 0.17698953 | 0.17499160 |
| subj07 | 0.19026171 | 0.20812575 |

All subjects used the exact subj01 initialization set: uniform,
feature-informed, and random seeds 1001–1018. Selection used the lowest
converged discovery objective only. The validation split never selected the
model, beta, coverage, entropy, baseline family, or initialization.

## Held-out validation, nulls, and sensitivity

| Subject | Paired ΔGW 95% CI | Paired Δcomposite 95% CI | Brain-ID p | CLIP-ID p | Independent-ROI p | Matched-plan p | Feature-pairing p |
| ------- | ----------------: | -----------------------: | ---------: | --------: | ----------------: | -------------: | ----------------: |
| subj02 | [-0.23771, -0.22038] | [-0.15811, -0.14948] | 0.00498 | 0.00498 | 0.00498 | 0.00200 | 1.00000 |
| subj05 | [-0.36336, -0.34111] | [-0.22136, -0.21013] | 0.00498 | 0.00995 | 0.00498 | 0.00200 | 1.00000 |
| subj07 | [-0.19438, -0.16941] | [-0.18475, -0.17264] | 0.00498 | **0.09453** | 0.00498 | 0.00200 | 1.00000 |

Thus subj07 shows a strong held-out advantage over the frozen baseline, stable
validation subsampling, and two successful structural null families. It remains
`INCONCLUSIVE` because the preregistered CLIP-identity mismatch null does not
separate at `p<=0.05`, and its near-optimal-plan reproducibility does not exceed
the matched-plan null threshold. The model was not changed to rescue it.

Local-minimum diagnostics show subj02 ambiguity concentrated in FFA and PPA,
no flagged ROI for subj05 under the pairwise median-row-correlation diagnostic,
and subj07 ambiguity in hV4 and PPA. These are optimization/identifiability
observations, not biological differences or unique ROI-to-layer assignments.

The registered feature-variance sensitivity returned identical preferred
layers and zero Frobenius distance from the primary plan for all three subjects.
It was not used for selection. The feature-pairing null gave `p=1.0` for every
subject and, as preregistered, did not enter the PASS decision.

## Implementation and provenance

New preparation entry point:

- `scripts/prepare_locked_cross_subject_validation.py`: validates subject assets,
  records cache provenance, computes the rule-based uniform-plan calibration,
  and writes the locked per-subject validation config before validation access.

The existing `scripts/validate_fgw_correspondence.py` was generalized to accept
the three-config locked replication set, enforce the frozen non-structural
baseline, and include matched-plan specialization/reproducibility in the
decision. Its numerical routines, solver, null fitting, OOF probes, and
variance-standardization logic were reused unchanged.

Per-subject roots:

- `fgw_outputs/subj02/cross_subject_validation_v1/`
- `fgw_outputs/subj05/cross_subject_validation_v1/`
- `fgw_outputs/subj07/cross_subject_validation_v1/`

Each contains the manifest, cache provenance, preflight audit, geometry and
stability/null assets, OOF `M`, loss calibration, per-initialization plans,
selected discovery initialization, validation matrices, structural and feature
nulls, matched-plan analysis, variance sensitivity, fixed baseline metrics, and
`validation/validation_summary.json`.

Direct synthetic/helper tests passed for FGW correspondence, validation
helpers, representation caching, and stimulus-manifest construction. The
optional POT reference check was skipped because the `ot` package is not
installed; the repository's own solver tests passed.

The frozen subj01 offline-test directory tree hash was
`19af838b0b2c1e90c7e751f94622db65a18fb5092221f16066878e73c8493370`
before and after Prompt 5A. Its frozen plan SHA-256 remains
`51e65527f3bbc1a3951ec376dd8be3662cab9e6b5f682d2a0da26fb347e20760`.

## Scientific conclusion at the Prompt-5A boundary

The locked subj01 formulation replicated through validation in subj02 and
subj05 without hyperparameter rescue. Subj07 has a real-looking held-out
performance advantage but fails two preregistered specificity/reproducibility
checks and must remain inconclusive. This is cross-subject validation evidence,
not yet one-shot cross-subject offline-test evidence, not a biological hierarchy,
not a unique ROI-to-layer assignment, and not evidence of downstream caption
utility.
