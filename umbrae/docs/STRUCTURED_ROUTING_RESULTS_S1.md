# Structured Routing Results — Subj01

## Motivation

The strict-Shikra subj01 benchmark found that soft NeuroRoute augmentation
improved caption metrics over UMBRAE-only, but uniform multi-layer
augmentation was stronger. The soft Stage-1 matrix has substantial L24
usage and relatively low layer coverage, while uniform routing enforces
complete coverage.

This experiment tests whether preserving uniform coverage while adding a
limited soft-routing deviation can improve the result.

## Protocol

Existing baseline outputs were reused without retraining:

- UMBRAE-only
- NeuroRoute-only
- UMBRAE + soft
- UMBRAE + uniform
- UMBRAE + single-L24

Only four new runs were trained:

- uniform residual soft, alpha 0.1
- uniform residual soft, alpha 0.3
- uniform residual soft, alpha 0.5
- temperature-smoothed soft, tau 2.0

All new runs used the complete 8,559-sample subj01 train split, the
300-sample validation split, three epochs, batch size two, frozen UMBRAE,
frozen NeuroRoute, frozen Shikra, adapter-only optimization, and the
strict 256-token Shikra patch bridge.

For structured modes, each fMRI-derived ROI representation is expanded
into six CLIP-layer-conditioned subtokens and weighted by the effective
routing matrix before Perceiver resampling. Stage-1 is not retrained.
This is an explicit structured-fusion extension; its output should not be
interpreted as reconstructing unavailable image CLIP features.

## Caption results

Metrics are from the final third epoch. Best validation loss is read from
the best checkpoint.

| Method | Best val ↓ | Final val ↓ | BLEU-4 ↑ | ROUGE-L ↑ | CIDEr ↑ |
|---|---:|---:|---:|---:|---:|
| UMBRAE-only | **1.884107** | 1.916646 | 0.071218 | 0.294891 | 0.594173 |
| Soft | 1.909896 | 1.909896 | 0.084187 | 0.304803 | 0.713035 |
| Uniform | 1.908940 | 1.908940 | **0.091184** | **0.307187** | **0.758230** |
| Single-L24 | 1.922040 | 1.922040 | 0.072228 | 0.284027 | 0.537156 |
| Alpha 0.1 | 1.910761 | 1.910761 | 0.077578 | 0.292739 | 0.604175 |
| Alpha 0.3 | 1.908892 | **1.908892** | 0.077615 | 0.295642 | 0.615365 |
| Alpha 0.5 | 1.915305 | 1.915305 | 0.074344 | 0.287580 | 0.578109 |
| Temperature 2.0 | 1.918370 | 1.918370 | 0.074060 | 0.290924 | 0.594332 |

Alpha 0.3 has the lowest final validation loss by a negligible 0.0025%
relative to uniform, but its caption metrics are substantially lower:

- BLEU-4: 14.88% below uniform
- CIDEr: 18.84% below uniform
- ROUGE-L: 3.76% below uniform

No structured method beats uniform on both CIDEr and BLEU-4. Structured
routing therefore does not satisfy the success criterion.

## Routing diagnostics

`layer_coverage_score` is the effective number of aggregate layers divided
by six. `roi_diversity_score` is mean pairwise total-variation distance
between ROI routing distributions.

| Routing | Entropy | Coverage | ROI diversity | Early depth | High-level depth | Gap |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 1.7918 | 1.0000 | 0.0000 | 14.0000 | 14.0000 | 0.0000 |
| Alpha 0.1 | 1.7883 | 0.9966 | 0.0017 | 14.3427 | 14.3417 | -0.0011 |
| Alpha 0.3 | 1.7611 | 0.9698 | 0.0050 | 15.0282 | 15.0250 | -0.0032 |
| Alpha 0.5 | 1.7056 | 0.9176 | 0.0083 | 15.7136 | 15.7083 | -0.0053 |
| Temperature 2.0 | 1.6584 | 0.8752 | 0.0090 | 16.1161 | 16.1140 | -0.0021 |
| Soft | 1.4124 | 0.6845 | 0.0165 | 17.4273 | 17.4167 | -0.0106 |
| Single-L24 | ~0 | 0.1667 | 0.0000 | 24.0000 | 24.0000 | 0.0000 |

Increasing alpha moves expected depth toward deeper layers and reduces
coverage. Caption performance also trends away from the uniform result:
alpha 0.1 and 0.3 are similar, while alpha 0.5 is worse. This is
consistent with stronger soft contribution adding high-level redundancy
instead of useful specialization.

Temperature 2.0 increases entropy and coverage relative to original soft
routing, but remains below both soft and uniform caption performance.
Smoothing alone is insufficient.

The early/high-level expected-depth gap remains effectively zero for every
variant. The routing diagnostics do not support an ROI-specific CLIP-layer
hierarchy.

## Conclusion

Pure uniform multi-layer coverage remains the strongest subj01 caption
method. Structured residual and temperature-soft routing do not improve
over uniform, and they do not improve over the existing soft baseline on
caption metrics.

The supported interpretation is:

- ROI multi-layer augmentation remains beneficial relative to
  UMBRAE-only under several settings.
- broad, stable layer coverage is more important than the current learned
  layer specialization for subj01 captioning.
- the Stage-1 soft matrix has little ROI diversity and no meaningful
  early-versus-high-level depth separation.

No cross-subject, ROI-hierarchy, or SOTA claim follows from this
experiment.
