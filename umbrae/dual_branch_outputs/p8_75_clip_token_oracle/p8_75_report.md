# P8.75 Exact CLIP-Token Oracle Report

## 1. Oracle Definition

`EXACT_CLIP_TOKEN_ORACLE` is GT image → frozen CLIP `hidden_states[-2][:,1:,:]` → the current frozen mm_projector → the current frozen Shikra. It is not Shikra's native image route and is not a deployable brain-decoding model.

## 2. Teacher Equivalence

Teacher equivalence is `PASS`. The online extraction uses the same RGB/224 bicubic/antialias/center-crop/CLIP-normalization definition as the locked training cache. Full details are in `teacher_equivalence.json`.

## 3. Caption Oracle

| Input Representation | Brain-derived? | RSA | Projected Oracle Gap | CIDEr | BLEU-4 | ROUGE-L | Mean IoU | Acc@0.5 | Parse Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UMBRAE | Yes | 0.459076 | 0.462916 | 0.606413 | 0.190289 | 0.436173 | 0.220826 | 0.197602 | 7 |
| LoRA | Yes | 0.468177 | 0.462295 | 0.581678 | 0.183341 | 0.434125 | 0.216320 | 0.195122 | 7 |
| Full Real | Yes | 0.480000 | 0.462612 | 0.590972 | 0.181831 | 0.434995 | 0.217941 | 0.198016 | 4 |
| Full Real α=.25 | Yes / diagnostic | 0.465642 | 0.462273 | 0.610868 | 0.191003 | 0.435860 | 0.220622 | 0.198429 | 6 |
| Exact CLIP Token Oracle | No, GT image oracle | 1.000000 | 0.000000 | 1.635679 | 0.506905 | 0.663752 | 0.479991 | 0.523357 | 1 |

All caption and grounding failures remain in their locked denominators. Oracle generation success, empty counts, and mean/median lengths are recorded in the task metric files.

## 4. Grounding Oracle

The same table reports all 2,419 locked queries, including parse failures.

## 5. Oracle Headroom

Oracle − UMBRAE CIDEr = 1.029266; Oracle − UMBRAE grounding accuracy = 0.325754. Oracle − Full Real CIDEr = 1.044707; Oracle − Full Real grounding accuracy = 0.325341.

## 6. Paired Token Reconstruction

`paired_token_diagnostics.json` reports same-sample token cosine/MSE and per-sample errors. These are paired conditional diagnostics, unlike P8.5 marginal-distribution matching. BrainX uses element-wise `[B,256,1024]` supervision, but token index is not claimed as anatomical patch identity. Optional set OT was skipped because it is not required for the oracle decision.

## 7. Projector-Space Oracle Gap

`projector_oracle_gap.json` reports paired projected token cosine/MSE, pooled cosine, norm ratio, and variance for `[982,256,4096]` decoder inputs.

## 8. Geometry vs Exact Visual-Token Recovery

Compare each row's RSA with its projected oracle gap. RSA is global sample geometry; projected paired gap measures recovery of the actual decoder input. Oracle RSA=1 and gap=0 are identity sanity endpoints only.

## 9. Decoder Capacity Diagnosis

DECODER_CAPACITY_HEADROOM = LARGE

PROJECTOR_DECODER_CEILING = NOT_SUPPORTED

BRAIN_TO_VISUAL_INTERFACE_GAP = LARGE

## 10. Mechanism Decision

NEXT_STAGE_PRIORITY = BRAIN_TOKEN_COMPATIBILITY

P8_75_STATUS = COMPLETE

No P9, decoder retraining, or any model training was started.
