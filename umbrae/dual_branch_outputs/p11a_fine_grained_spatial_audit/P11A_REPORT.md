# P11-A Fine-Grained Spatial Brain Representation Audit

## 1. Motivation

P8–P10 showed that marginal correction, token-wise adaptation, and token mixing cannot recover the Exact CLIP Token Oracle's downstream utility. P11-A therefore audited whether a spatially traceable brain representation can be constructed. No training was started.

## 2. Available pRF / Retinotopy Assets

The local NSD copy contains categorical `prf-visualrois` and `prf-eccrois` volumes and hemisphere-specific variants. It does **not** contain continuous voxel-wise pRF angle/x/y, eccentricity, size/sigma, or R²/variance-explained files. Full inventory: `docs/P11_PRF_ASSET_AUDIT.md` and `p11_prf_asset_inventory.json`.

Official code evidence: `cvnlab/nsdcode/examples/examples_nsdmapdata.py` uses `freesurfer/subjXX/label/lh.prfangle.mgz` for polar-angle mapping; the expected file is absent from this local NSD copy.

## 3. nsdgeneral ↔ pRF Mapping

The reliable part is `nsdgeneral vector index -> C-order volume index -> func1pt8mm ijk/world xyz -> aligned categorical visual ROI/eccentricity ROI/hemisphere`. The constructed `subj01_voxel_prf_table.npz` contains all 15724 voxels. Continuous pRF columns are NaN and `valid_prf=false`, explicitly complying with the no-imputation rule.

There is no reliable continuation from those volume voxels to continuous pRF parameters because the source parameter maps are absent.

## 4. pRF Coverage

| ROI | Total | Categorical ecc label | Valid continuous pRF | Quality passed | Coverage | Median ecc | Median sigma |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| V1 | 1350 | 1346 | 0 | 0 | 0.000 | unavailable | unavailable |
| V2 | 1433 | 1429 | 0 | 0 | 0.000 | unavailable | unavailable |
| V3 | 1187 | 1182 | 0 | 0 | 0.000 | unavailable | unavailable |
| hV4 | 687 | 684 | 0 | 0 | 0.000 | unavailable | unavailable |

Quality distribution and a defensible threshold cannot be reported because no fit-quality metric exists locally. No threshold was invented.

## 5. Visual-Field Sanity

Hemisphere provenance is available from aligned left/right masks, but contralateral visual-field sanity cannot be tested without polar angle/x/y. The eccentricity figure reports categorical label counts only. Scatter, pRF-size, laterality-affinity, and visual-field coverage panels are explicitly marked unavailable rather than populated with fabricated coordinates.

## 6. CLIP Patch Geometry

The repository's `models/clip_patch_teacher.py` applies `patch_grid.flatten(2).transpose(1,2)`, proving row-major 16×16 ordering (`index=row*16+column`) for the 256 patch tokens. `clip_patch_index_map.json` records normalized image centers. `NORMALIZED_COORDINATE_ONLY=true`; absolute visual-angle centers are null because no validated pRF-to-stimulus convention can be joined locally.

## 7. Voxel-to-Patch Affinity

The intended kernel is `exp(-||c_p-mu_v||²/(2 sigma_v²))`, normalized over patches. It was **not evaluated** because neither `mu_v` nor a defined positive Gaussian sigma is available. No categorical eccentricity midpoint, ROI mean, nearest neighbor, or zero fill was substituted.

## 8. Candidate Fine-Grained Units

XY-only and XY+size ROI-conditioned clustering both require missing continuous pRF data. Rejected manifests and empty `[0,256]` affinity sentinels were emitted for K={32,48,64,96,128} so downstream code cannot mistake absence for a valid mapping.

## 9. Candidate Comparison

| K_retino | K_total nominal | Min vox/unit | Median vox/unit | Spatial compactness | Affinity redundancy | Patch coverage | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 36 | unavailable | unavailable | unavailable | unavailable | unavailable | REJECTED_MISSING_CONTINUOUS_PRF |
| 48 | 52 | unavailable | unavailable | unavailable | unavailable | unavailable | REJECTED_MISSING_CONTINUOUS_PRF |
| 64 | 68 | unavailable | unavailable | unavailable | unavailable | unavailable | REJECTED_MISSING_CONTINUOUS_PRF |
| 96 | 100 | unavailable | unavailable | unavailable | unavailable | unavailable | REJECTED_MISSING_CONTINUOUS_PRF |
| 128 | 132 | unavailable | unavailable | unavailable | unavailable | unavailable | REJECTED_MISSING_CONTINUOUS_PRF |

No PRIMARY or SECONDARY K is selected. Selecting one would use only a requested token count, contrary to the protocol.

## 10. Selected Mapping

`selected_mapping.json` is a blocked manifest with zero units, not a usable tokenizer mapping. `PRIMARY_K_RETINO=null`, `PRIMARY_K_TOTAL=null`, and `SECONDARY_K=null`.

## 11. High-Level ROI Handling

FFA/EBA/PPA/OPA remain conceptually one non-retinotopic semantic unit per ROI. They were not assigned fake spatial pRF centers or face/body/place visual teachers. Existing atlas overlaps remain documented in the NeuroRoute mapping and were not silently deduplicated.

## 12. Random Spatial Control

The future protocol is defined as within-parent-ROI permutation followed by repartition using the real unit-size sequence. It preserves token count, ROI provenance, unit sizes, voxel coverage, and (when present) quality distribution while destroying spatial organization. It is not executable until a valid real mapping exists, hence `RANDOM_CONTROL_READY=false`.

## 13. Future Spatial Teacher

Once real affinities exist, the retinotopic teacher is `V_teacher_r = sum_p W[r,p] V_clip[p]`. This differs from P3 by conditioning each unit on a traceable visual-field kernel instead of a global semantic target. No teacher tensor was generated and no training was started.

## 14. Risks

- Continuous pRF assets may need a separate NSD data release and a verified surface-to-func1pt8mm mapping.
- Angle convention, eccentricity units, pRF-size definition, and fit-quality semantics must be sourced from the matching release.
- hV4 coverage may differ from V1–V3 after quality filtering.
- Excessive K could fragment low-coverage voxels; it cannot be evaluated yet.
- High-level ROIs do not have an automatic retinotopic teacher.
- Subject-agnostic code is present, but cross-subject consistency cannot be evaluated before subj01 construction is valid.

## 15. Final Decision

```text
PRF_DATA_AVAILABILITY = INSUFFICIENT
FINE_GRAINED_RETINOTOPIC_TOKENIZATION = NOT_FEASIBLE
PRIMARY_K_RETINO = null
PRIMARY_K_TOTAL = null
SECONDARY_K = null
RANDOM_CONTROL_READY = false
P11B_READY = false
P11A_STATUS = BLOCKED
```

Blocker: The local NSD copy has categorical prf-visualrois/prf-eccrois labels but no voxel-wise polar angle or x/y center, continuous eccentricity, pRF size/sigma, or fit-quality data. Gaussian voxel-to-patch affinity and pRF-coordinate clustering therefore cannot be constructed without fabricating values.
