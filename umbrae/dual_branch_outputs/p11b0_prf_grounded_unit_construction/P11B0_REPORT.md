# P11-B0 pRF-Grounded Fine Unit Construction Report

## 1. Motivation
P8–P10 left a fine spatial-information deficit that coarse eight-ROI tokens cannot resolve. P11-B0 therefore locks a stimulus-independent, pRF-grounded interface before any neural encoder is designed.

## 2. Input pRF Assets
The exact P11-A.5 15,724-voxel nsdgeneral table is reused (SHA256 `92e48761028b81cf14252c5d550cba946762dac5ed5d95c01b86b6afeb3a50db`). V1/V2/V3/hV4 contain 4,656 full-pRF voxels. Angle is 0° right and 90° upper; eccentricity and x/y are dva; Gaussian sigma is official CSS effective size × sqrt(exponent); R² is percent.

## 3. NSD Stimulus Visual Geometry
PASS. The official NSD paper states 8.4° × 8.4° total extent for pRF, fLoc, and NSD; official pRF release code independently uses `8.4/200`. The square natural-image extent is fixation-centered [-4.2,+4.2]° on both axes. No circular mask is applied to NSD natural-scene images (the pRF mapping stimulus itself uses a circular region, which is not substituted for the natural-image geometry).

## 4. CLIP Patch Geometry
The local square 256×256 RGB sample becomes 224×224 under bicubic resize; center crop removes no field. ViT-L/14 gives 16×16 row-major tokens. Top row maps to positive visual-field y. `PATCH_FOV_MAPPING=DIRECT`.

## 5. Voxel→Patch Gaussian Affinity
Each voxel uses `softmax_p(-||c_p-mu_v||²/(2 sigma_v²))`. Rows sum to one after max-subtracted exponentiation; no nonpositive/nonfinite sigma and no clamp were required. Patch centers, not rectangle integrals, are the locked first approximation.

## 6. Quality Policy Comparison
R²>0 retains 4,473 voxels; R²>=10.1 retains 3,956. The stricter official candidate becomes primary because it removes low-quality spatial outliers that form singleton clusters for the >0 policy at K>=48 while retaining robust unit counts. R²>0 remains the secondary quality policy.

## 7. Candidate Fine Units
All 20 combinations (2 quality × 2 feature sets × K32/48/64/96/128) were built offline. Exact values and rejection codes are in `candidate_mapping_comparison.csv`. KMeans is conditioned on ROI×hemisphere, seed42/n_init10, with within-group z-scoring and seeds 0–4 ARI audit.

## 8. Spatial Compactness
For the selected policy/XY features, compactness improves from K32 through K64 while minimum unit size remains 12 at K64. K96 reaches a 3-voxel minimum and K128 a singleton, crossing the robustness boundary despite finer centers.

## 9. Patch Affinity Diversity
Entropy, effective patch count, top-10 mass, pairwise W cosine, effective rank, singular spectrum, and top-5 patch coverage are recorded for every candidate. Cross-hierarchy overlaps were retained as scientifically legitimate.

## 10. Selected Mapping
`PRIMARY_MAPPING=quality_r2_ge10p1_xy_k64`; `PRIMARY_K_RETINO=64`; `PRIMARY_K_TOTAL=68`; quality R²>=10.1; feature XY; quality-weighted W. It is the finest candidate with every unit >=10 voxels and avoids using CLIP or downstream evidence.

## 11. Secondary Mapping
`SECONDARY_MAPPING=quality_r2_ge10p1_xy_k48` (K48, same policy/features) is the coarser granularity ablation with a 15-voxel minimum and higher stability.

## 12. High-Level ROI Tokens
FFA/EBA/PPA/OPA retain their verified NeuroRoute voxel sets as one functional token each. Their `patch_affinity` is null, never uniform.

## 13. Random Spatial Control
The primary random control permutes voxel membership without replacement within ROI×hemisphere, preserving K, unit sizes, voxel pool, labels, order, capacity, and the exact real teacher W. The teacher-shuffle control separately preserves real memberships and permutes W within the same group.

## 14. Spatial Teacher Operator
The parameter-free operator computes `T_r(x)=sum_p W_rp V_p(x)`. Raw CLIP scale is primary; normalized-patch aggregation is optional diagnostic. Validated output shape is `[2, 64, 1024]`; no learnable parameters exist.

## 15. Teacher Diversity Diagnostic
After mapping files were locked, a fixed validation subset was read from the frozen CLIP cache. Mean between-unit cosine is `0.6319937035441399` and mean local-to-global cosine is `0.7917952165007591`. These values did not influence mapping selection.

## 16. Risks
pRF fits remain noisy and threshold-sensitive; CSS size semantics must remain locked; center sampling approximates patch-area integration; the 16×16 grid is finite; peripheral kernels are truncated and renormalized; high K creates small units; high-level ROIs lack pRF teachers; and this subj01 mapping is not portable to another subject without subject-specific reconstruction.

## 17. Final Decision
NSD_VISUAL_ANGLE_MAPPING = PASS

VOXEL_PATCH_AFFINITY = VALID

QUALITY_POLICY_LOCKED = r2_ge10p1

PRIMARY_MAPPING_ID = quality_r2_ge10p1_xy_k64

PRIMARY_K_RETINO = 64

PRIMARY_K_TOTAL = 68

SECONDARY_MAPPING_ID = quality_r2_ge10p1_xy_k48

REAL_MAPPING_LOCKED = true

RANDOM_SPATIAL_CONTROL_READY = true

SPATIAL_TEACHER_READY = true

P11B_TRAINING_READY = true

P11B0_STATUS = COMPLETE

NO TRAINING STARTED.
