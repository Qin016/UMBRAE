# P11-B pRF-Grounded Fine-Grained Structural Representation Report

## 1. Setup
Subj01, locked R²>=10.1 XY K64 mapping, 8,559 train and 300 validation samples; test sealed. Real and within-ROI×hemisphere Random use identical 64-unit size vectors, architecture, 1,292,800 trainable parameters, initialization, batch order, teacher W, optimizer, and 30-epoch exposure. BrainX, LoRA, Fusion, adapters, mm_projector, Shikra, caption, and grounding are absent.

## 2. Objective
`L = local cosine + 0.1 local MSE + 0.1 within-sample unit-relation MSE + 0.1 per-unit cross-sample InfoNCE`, temperature 0.07.

## 3. Capacity Equivalence
REAL_RANDOM_CAPACITY_MATCH = PASS

OPTIMIZATION_PROTOCOL_MATCH = PASS

Both initialization hashes are `ae909b87d627cefc0db7b03a723f0b9d4b77a32c89a12b075fa6e03e12e097d2`; both best checkpoints are epoch 30 and both ran 30 epochs.

## 4. Training Curves
All requested curves are in `figures/`. Both runs converged without nonfinite loss or gradients.

## 5. Local Visual Recovery
Both models recover substantial teacher cosine (~0.625), but Real−Random is negligible and its paired CI crosses zero. Random has lower MSE and higher local-sample retrieval.

## 6. Spatial Specificity
Real unit-localization MRR is 0.120497; Random is 0.121559. The Real advantage criterion is not met.

## 7. Local Structure
Real relation Spearman (0.493128) exceeds Random (0.470574), but both brain unit sets are much more mutually similar than the teacher (Real 0.852634, Random 0.848794, teacher 0.617562). Both locality gaps are negative, documenting semantic leakage/global collapse.

## 8. ROI Breakdown
| ROI | Real Local Cos | Random Local Cos | Δ | Real Loc MRR | Random Loc MRR | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 0.607563 | 0.607494 | +0.000068 | 0.098168 | 0.099083 | -0.000915 |
| V2 | 0.624675 | 0.623904 | +0.000771 | 0.113689 | 0.114513 | -0.000824 |
| V3 | 0.640821 | 0.641826 | -0.001005 | 0.138091 | 0.140353 | -0.002262 |
| hV4 | 0.636114 | 0.635597 | +0.000517 | 0.147705 | 0.147579 | +0.000126 |

No ROI shows a consistent joint Real advantage in local cosine, localization, and sample retrieval.

## 9. Unit-Level Breakdown
`per_unit_real_random.csv` contains all 64 units, anatomical metadata, Real/Random local recovery, localization, sample retrieval, and deltas. Unit-bootstrap intervals test whether effects are broadly distributed.

## 10. Real vs Random
| Metric | Real | Random | Delta | Bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Local Cosine | 0.625419 | 0.625340 | +0.000079 | [-0.000712, +0.000872] |
| Unit Localization MRR | 0.120497 | 0.121559 | -0.001062 | [-0.002845, +0.000744] |
| Local Sample MRR | 0.034504 | 0.037274 | -0.002770 | [-0.004366, -0.001290] |
| Unit Relation Spearman | 0.493128 | 0.470574 | +0.022555 | [+0.016414, +0.029001] |
| Locality Gap | -0.107221 | -0.107032 | -0.000190 | [-0.000967, +0.000570] |

Bootstraps use 10,000 paired validation-sample resamples; separate 10,000 unit resamples are also saved.

## 11. Spatial Visualization
The visual-field delta scatter and ROI/curve figures show no coherent widespread Real advantage.

## 12. Mechanism Interpretation
The encoder can learn a local-teacher-correlated representation, but correct pRF-grounded voxel membership does not outperform its capacity- and teacher-matched Random control on the decisive spatial-specificity and stimulus-retrieval metrics. Real's relation-Spearman improvement alone is insufficient to validate retinotopic correspondence. Secondary control: Teacher shuffle local cosine=0.622971, localization MRR=0.113897, sample MRR=0.029854, relation Spearman=0.437911, locality gap=-0.116789.

## 13. Final status
LOCAL_VISUAL_RECOVERY = WEAK

SPATIAL_SPECIFICITY = NEGATIVE

PRF_GROUNDED_STRUCTURE_SIGNAL = NEGATIVE

REAL_RANDOM_CAPACITY_MATCH = PASS

FINE_GRAINED_REPRESENTATION = NOT_VALIDATED

P11B_STATUS = NO_RELIABLE_PRF_STRUCTURE_ADVANTAGE

NEXT_STAGE = REASSESS_LOCAL_ENCODING_OR_FMRI_INFORMATION_LIMIT

No downstream or next-stage experiment was started.
