# P8 Downstream-Compatible Structural Calibration Diagnosis

## 1. Setup
Locked test, 982 images and 2,419 grounding queries; frozen P7 checkpoints, mm_projector and Shikra. Alpha grid: 0, .25, .5, .75, 1.

## 2. Statistical Validation
10,000 paired caption, cluster-grounding and stimulus-RSA bootstraps. Real-Random CIs cross zero for all primary downstream metrics; anatomical evidence is directional only.

## 3. Geometry–Utility Interpolation
Full Real alpha=.25 is the CIDEr sweet spot: CIDEr 0.610868, BLEU-4 0.191003, grounding accuracy 0.198429, RSA 0.465636. It improves alpha=1 and preserves/improves key baseline utilities. Diagnosis: over-calibration.

## 4. Projector Interface Diagnostics
Amplification ratios are {'lora': 1.0158549254664335, 'full_real': 1.0332460377596766, 'full_random': 1.0337001392240983}. RSA gains persist after projection; no strong decoder-interface amplification evidence.

## 5. Hard Retrieval
K=99 hard negatives remove R@1 saturation. Full models improve median rank from 8 to 7; Real and Random remain close.

## 6. Real vs Random Evidence
Architectural capacity and group sizes match, but checkpoint epochs differ and only one random seed exists. REAL_RANDOM_CAPACITY_MATCH = FAIL.

## 7. Sample-Level Analysis
Caption and grounding paired wins/ties plus baseline-difficulty and target-size groups are in p8_sample_level_analysis.json.

## 8. Failure / Success Cases
Fixed P7 sample IDs 0-9 are retained without reselection. Alpha endpoints have zero max absolute error. Failures remain in metric denominators.

## 9. Scientific Interpretation
H1 is supported: a weaker residual strength improves utility while retaining geometry gains. H2 is not strongly supported. Anatomy remains directional only.

## 10. Decision for P9
P8_DIAGNOSIS = OVER_CALIBRATION
NEXT_STAGE = P9_PRESERVATION_CONSTRAINED_CALIBRATION
P9 should prioritize stronger/projector-aware preservation and weaker fusion; do not proceed to parcel/pRF yet.

P8_STATUS = COMPLETE
NEW_TRAINING_STARTED = false
PARCEL_STAGE_STARTED = false
PRF_STAGE_STARTED = false
RETINOTOPY_STAGE_STARTED = false
P9_RECOMMENDATION = A_STRONGER_PRESERVATION_PLUS_C_WEAKER_FUSION_AND_D_PROJECTOR_AWARE_PRESERVATION
