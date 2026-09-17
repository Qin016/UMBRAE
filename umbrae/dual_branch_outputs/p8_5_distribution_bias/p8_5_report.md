# P8.5 Distribution Bias Diagnosis Report

## 1. Setup
Train-only fit: 8,559 samples / 2,191,104 tokens. Test is post-hoc reused test. No training.

## 2. Mean Shift
Full Real–CLIP mean L2=0.209359, relative=0.023498, cosine=0.999957.

## 3. Channel Scale
Full Real mean std ratio=0.466117; std-channel correlation=0.998513. A large scale mismatch exists.

## 4. Covariance / Correlation
Normalized covariance distance=0.132196; correlation Frobenius distance=99.464805.

## 5. PCA Geometry
Effective rank: Full Real=4.862, CLIP=57.296. Full Real is far more anisotropic.

## 6. Distribution Distance
Train MMD and train-fitted test-application distances are recorded in distribution_metrics.json. Analytic corrections repair marginal moments but do not imply semantic alignment.

## 7. Analytic Corrections
Mean-only, channel affine, and WCT are fitted solely from train statistics. WCT epsilon=1e-4; Full Real source covariance clamps 242 dimensions.

## 8. Representation Results
Affine/WCT improve hard retrieval, but WCT lowers RSA and projected RSA. See representation_metrics.json and hard_retrieval.json.

## 9. Caption Results
| Variant | CIDEr | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|
| UMBRAE | 0.606413 | 0.190289 | 0.436173 |
| Full raw | 0.590972 | 0.181831 | 0.434995 |
| alpha=.25 | 0.610868 | 0.191003 | 0.435860 |
| Mean | 0.587069 | 0.181631 | 0.433790 |
| Affine | 0.540766 | 0.086127 | 0.396560 |
| WCT | 0.514437 | 0.113162 | 0.404274 |

## 10. Grounding Results
Mean correction is mixed and tiny. Affine/WCT strongly reduce IoU/accuracy and increase parse failures.

## 11. Projector-Space Diagnostics
Corrected projected norms/RSA/cosine are in representation_metrics.json. Statistical matching does not preserve the decoder interface.

## 12. Mechanism Comparison
Alpha=.25 clearly outperforms mean/affine/WCT downstream. The P8 sweet spot is primarily magnitude control, not marginal distribution-bias removal.

## 13. Final Interpretation
MEAN_BIAS_SIGNAL = WEAK
CHANNEL_SCALE_BIAS_SIGNAL = NEGATIVE
COVARIANCE_BIAS_SIGNAL = NEGATIVE
DISTRIBUTION_BIAS_HYPOTHESIS = NOT_SUPPORTED
OVER_CALIBRATION_HYPOTHESIS = SUPPORTED
NEXT_STAGE = P9_PRESERVATION_CONSTRAINED_STRENGTH_CONTROL
P8_5_STATUS = COMPLETE
