# P11-A.5 Continuous Retinotopy Asset Recovery Report

## 1. Why P11-A Was Blocked

The original local release had categorical visual/eccentricity ROIs but no continuous maps. P11-A correctly refused to fabricate pRF centers or sigma.

## 2. Official Asset Search

Official `nsddatapaper/main/analysis_prf.m` confirms released func1pt8mm volumes named `prf_angle.nii.gz`, `prf_eccentricity.nii.gz`, `prf_size.nii.gz`, `prf_R2.nii.gz`, and `prf_exponent.nii.gz`. Official S3 listings also confirm native-surface `lh/rh.prf*.mgz`, but these were not needed because precomputed target-space volumes exist.

## 3. Asset Semantics

The model is CSS: stimulus is dotted with a unit-normalized isotropic Gaussian, raised to an exponent, scaled, and HRF-convolved. Angle is 0–360 degrees, with 0=right horizontal and 90=upper vertical. Eccentricity is degrees visual angle. Released size is **effective size `sigma/sqrt(exponent)`**, not Gaussian sigma. Gaussian sigma is separately derived as `size*sqrt(exponent)`. Quality is training R² in percent and may be negative.

## 4. Data Recovery

All five subj01 files downloaded successfully from official public S3 without authentication. Raw files remain under `/opt/data/private/BA/NSD/nsd_prf_recovery/subj01/source/func1pt8mm`; hashes and byte counts are locked in `download_manifest.json`. `MANUAL_DOWNLOAD_REQUIRED=false`.

## 5. Native-Space Provenance

All maps are official subj01 native func1pt8mm volumes with shape `[81,104,83]`, 1.8-mm sampling, and exactly the same affine as `roi/nsdgeneral.nii.gz`. Surface vertex geometry is not involved in the selected route.

## 6. Mapping Pipeline

`official native func1pt8mm parameter volume -> identical func1pt8mm grid -> exact nsdgeneral mask selection in NumPy C-order`. There is no interpolation, nearest-vertex assignment, resampling, or anatomical-xyz-to-image substitution.

## 7. Mapping Validation

- nsdgeneral output length: 15,724; duplicates: 0.
- V1/V2/V3/hV4 index sets remain exact against NeuroRoute.
- Continuous-vs-categorical eccentricity exact band agreement: 69.89%; same-or-adjacent band: 94.68%.
- With R²>=10.1, left cortex in right visual field: 95.61%; right cortex in left visual field: 91.79%.
- `MAPPING_VALIDATION=PASS`. Exact-bin mismatches are reported, not modified; their predominantly adjacent-band form is consistent with the categorical ROIs having been drawn/mapped through a surface representation.

## 8. Coverage

| ROI | Total | Full pRF | Coverage | Full & R2>0 | Full & R2>=10.1 | Median R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 1350 | 1350 | 1.0000 | 1313 | 1184 | 48.96 |
| V2 | 1433 | 1433 | 1.0000 | 1362 | 1193 | 47.12 |
| V3 | 1187 | 1187 | 1.0000 | 1163 | 1036 | 54.67 |
| hV4 | 687 | 686 | 0.9985 | 635 | 543 | 33.43 |

Raw global and per-ROI quality distributions are in `p11a5_summary.json`. R²>0 is used only as a readiness audit; R²>=10.1 is retained as a stricter source-derived candidate. No training/K-dependent threshold was selected.

## 9. Available pRF Tier

- FULL_PRF: 4656 / 4657
- CENTER_ONLY: 0
- CATEGORICAL_ONLY: 1
- FULL_PRF with R²>0: 4473
- FULL_PRF with R²>=10.1: 3956

## 10. Remaining Blockers

The continuous-pRF asset and mapping blocker is resolved. Before training, resumed P11-A/P11-B-prep must still lock a quality threshold, decide CSS raw-sigma versus effective-size supervision explicitly, and construct/select K. P11-A.5 does not perform clustering.

## 11. Categorical Fallback Feasibility

ROI × hemisphere × categorical-eccentricity units are constructible and saved as a non-spatial-affinity baseline. They contain no polar-angle localization and must not be called pRF-grounded Gaussian tokens.

## 12. Final Decision

```text
CONTINUOUS_POLAR_ANGLE = AVAILABLE
CONTINUOUS_ECCENTRICITY = AVAILABLE
PRF_SIZE = AVAILABLE
PRF_QUALITY = AVAILABLE
PRF_XY_MAPPING = READY
FULL_GAUSSIAN_AFFINITY = READY
MAPPING_VALIDATION = PASS
PRF_DATA_AVAILABILITY = SUFFICIENT
P11B_READY = true
MANUAL_DOWNLOAD_REQUIRED = false
REQUIRED_FILES = []
P11A5_STATUS = BLOCKER_RESOLVED
```

No optimizer, backward, model update, clustering, or P11-B training was started.
