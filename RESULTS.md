# Experiment results and repository map

This fork keeps the original UMBRAE history and adds the NeuroRoute, dual-branch,
and FGW experiments developed on top of it. This page is the entry point for
the published code and result artifacts.

For a version-by-version Chinese development history, including each method,
improvement, outcome, limitation, and recommended next step, see
[VERSIONS.md](VERSIONS.md).

## Headline result

Across S1, S2, S5, and S7, real ROI tokenization and soft multi-layer routing
improve Stage-1 alignment over uniform and final-layer-only targets. The learned
routing does not establish a clear ROI-specific CLIP-layer hierarchy. In the
strict S1 Shikra caption experiment, ROI augmentation improves over UMBRAE-only,
while uniform ROI coverage outperforms learned soft routing. The experiments
therefore support NeuroRoute as an augmentation to UMBRAE rather than a drop-in
replacement.

The detailed claims, limitations, and subject-level tables are in the
[full experiment summary](umbrae/docs/NEUROROUTE_ALL_EXPERIMENTS_SUMMARY.md).

## Repository map

| Path | Contents |
| --- | --- |
| `umbrae/models/` | ROI tokenization, routing, FGW, dual-branch, projector, and adapter modules |
| `umbrae/losses/` | Alignment, relational, calibration, distillation, and UOT losses |
| `umbrae/scripts/` | Training, evaluation, cache construction, audits, and result summarizers |
| `umbrae/configs/` | Locked experiment and ablation configurations |
| `umbrae/tests/` | Unit, shape, protocol, leakage, and synthetic pipeline tests |
| `roi_indices/` | Subject-level ROI metadata, mappings, and QC visualizations |
| `umbrae/docs/` | Experiment reports, audits, protocols, and interpretation |
| `umbrae/*_outputs/` | Lightweight aggregate metrics, comparisons, provenance, and plots |

## Result guides

- [Stage-1 routing](umbrae/docs/STAGE1_ROUTING_RESULTS.md)
- [Stage-1 retrieval](umbrae/docs/STAGE1_RETRIEVAL_RESULTS.md)
- [S1 Stage-2 caption results](umbrae/docs/STAGE2_CAPTION_RESULTS_S1.md)
- [UMBRAE + NeuroRoute integration](umbrae/docs/UMBRAE_NEUROROUTE_RESULTS_S1.md)
- [Structured routing](umbrae/docs/STRUCTURED_ROUTING_RESULTS_S1.md)
- [FGW correspondence validation](umbrae/docs/FGW_CORRESPONDENCE_VALIDATION_S1_V1.md)
- [FGW locked offline test](umbrae/docs/FGW_CORRESPONDENCE_OFFLINE_TEST_S1_V1.md)
- [FGW cross-subject validation](umbrae/docs/FGW_CROSS_SUBJECT_VALIDATION_V1.md)
- [FGW cross-subject offline replication](umbrae/docs/FGW_CROSS_SUBJECT_OFFLINE_REPLICATION_V1.md)
- [FGW Stage-2 development](umbrae/docs/FGW_STAGE2_S1_DEVELOPMENT_V1.md)

## What is and is not versioned

The Git repository includes source code, tests, configurations, reports,
aggregate JSON/CSV metrics, small diagnostic figures, and provenance metadata.
It excludes restricted NSD data, downloaded third-party models, feature caches,
training checkpoints, raw COCO annotations, logs, and sample-level predictions.

This separation is intentional: a checkout remains reviewable and below GitHub
file limits, while the reports retain the exact experimental conclusions. See
the configuration files and report-specific reproduction sections for commands
and expected local paths. Paths captured in provenance files document the
original run environment and may need to be adjusted on another machine.

## Reproduction

Start with [REPRODUCTION.md](REPRODUCTION.md) for the verified UMBRAE smoke test.
The individual reports under `umbrae/docs/` describe the inputs, locked splits,
commands, and stopping boundaries for each NeuroRoute and FGW experiment.
