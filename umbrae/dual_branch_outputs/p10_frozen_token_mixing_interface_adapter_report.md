# P10 Frozen Representation Token-Mixing Interface Adapter Report

## 1. Motivation

P9 showed a generic benefit from token-wise residual adaptation but only weak P6 information advantage. P10 is the final lightweight frozen-representation capacity diagnostic: it tests whether self-attention can recover decoder-relevant information whose organization spans multiple Brain tokens.

Only the P10 adapter and two residual gates were trained. BrainX/P6, StructuralBranch, Fusion, LoRA, CLIP, Shikra `mm_projector`, Shikra, prompts, generation, parser, and evaluators remained frozen. Training and selection used train/validation only; no P10 post-hoc test was run.

## 2. Architecture

One block: `LN(1024) → 8-head MHSA → gated residual → LN(1024) → FFN(1024→256→1024, GELU) → gated residual`. Token count and width remain `[B,256,1024]`; dropout is 0 and no positional embedding or learned output query is added.

Exact trainable parameters: 4,728,066 (3.227335% of BrainX), including two scalar gates.

## 3. Initialization Integrity

| Representation | Max abs error | Mean abs error | Token cosine | Pooled cosine | Relative delta | g_attn | g_ffn | Initial state hash |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UMBRAE | 0.00000000 | 0.00000000 | 1.00000000 | 1.00000000 | 0.00000000 | 0.05002085 | 0.05002085 | `e092687bc91acf2b8f5c343ebcd14a2695ea57e46235edc4a366a57c25e73775` |
| P6 Full Real | 0.00000000 | 0.00000000 | 1.00000000 | 1.00000000 | 0.00000000 | 0.05002085 | 0.05002085 | `e092687bc91acf2b8f5c343ebcd14a2695ea57e46235edc4a366a57c25e73775` |

The two initial state hashes are identical; initialization is an exact identity mapping.

## 4. UMBRAE TokenMix Training

| Epoch | Gap | CIDEr | Grounding | Attention entropy | Off-diagonal mass | g_attn | g_ffn | Correction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.390220 | — | — | 4.864808 | 0.992830 | 0.051555 | 0.053392 | 0.040068 |
| 2 | 0.389397 | 0.861140 | 0.227381 | 4.454316 | 0.993145 | 0.054557 | 0.056216 | 0.050249 |
| 3 | 0.388997 | — | — | 4.299081 | 0.993057 | 0.057931 | 0.058737 | 0.054387 |
| 4 | 0.388853 | 0.872433 | 0.232143 | 4.230378 | 0.993037 | 0.061359 | 0.061281 | 0.060815 |
| 5 | 0.388583 | — | — | 4.219133 | 0.993004 | 0.064819 | 0.064070 | 0.068680 |
| 6 | 0.388504 | 0.867145 | 0.227381 | 4.187525 | 0.993102 | 0.068500 | 0.066893 | 0.076036 |
| 7 | 0.388392 | — | — | 4.173984 | 0.993164 | 0.072194 | 0.069768 | 0.082720 |
| 8 | 0.388405 | 0.865492 | 0.229762 | 4.118051 | 0.993272 | 0.075766 | 0.072404 | 0.088003 |
| 9 | 0.388245 | — | — | 4.126107 | 0.993260 | 0.079329 | 0.074965 | 0.095434 |
| 10 | 0.388218 | 0.877025 | 0.232143 | 4.131534 | 0.993265 | 0.082653 | 0.077127 | 0.101406 |
| 11 | 0.388297 | — | — | 4.091313 | 0.993426 | 0.085700 | 0.079203 | 0.102324 |
| 12 | 0.388146 | 0.856587 | 0.233333 | 4.119841 | 0.993412 | 0.088838 | 0.081098 | 0.105715 |
| 13 | 0.388131 | — | — | 4.115579 | 0.993479 | 0.092049 | 0.083066 | 0.107525 |
| 14 | 0.388112 | 0.879376 | 0.234524 | 4.146048 | 0.993452 | 0.095127 | 0.084908 | 0.110161 |
| 15 | 0.388084 | — | — | 4.150082 | 0.993494 | 0.098251 | 0.086742 | 0.111075 |
| 16 | 0.388059 | 0.873036 | 0.233333 | 4.141452 | 0.993566 | 0.101522 | 0.088515 | 0.112428 |
| 17 | 0.388034 | — | — | 4.158964 | 0.993574 | 0.104996 | 0.090517 | 0.114506 |
| 18 | 0.388024 | 0.861014 | 0.229762 | 4.159888 | 0.993668 | 0.108500 | 0.092494 | 0.114091 |
| 19 | 0.387996 | — | — | 4.161345 | 0.993604 | 0.112171 | 0.094365 | 0.118177 |
| 20 | 0.387979 | 0.858981 | 0.228571 | 4.158594 | 0.993692 | 0.115791 | 0.096302 | 0.117902 |

## 5. P6 Full Real TokenMix Training

| Epoch | Gap | CIDEr | Grounding | Attention entropy | Off-diagonal mass | g_attn | g_ffn | Correction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.389057 | — | — | 5.214647 | 0.992307 | 0.051424 | 0.053554 | 0.039158 |
| 2 | 0.388462 | 0.854810 | 0.230952 | 4.862912 | 0.992313 | 0.053566 | 0.057809 | 0.048484 |
| 3 | 0.388029 | — | — | 4.588706 | 0.992810 | 0.056593 | 0.061681 | 0.051489 |
| 4 | 0.387872 | 0.859395 | 0.230952 | 4.560988 | 0.992731 | 0.059750 | 0.064941 | 0.057109 |
| 5 | 0.387673 | — | — | 4.582552 | 0.992829 | 0.062959 | 0.068127 | 0.062940 |
| 6 | 0.387590 | 0.837050 | 0.225000 | 4.598035 | 0.992923 | 0.066578 | 0.071236 | 0.068387 |
| 7 | 0.387471 | — | — | 4.601933 | 0.993027 | 0.070374 | 0.074386 | 0.073779 |
| 8 | 0.387429 | 0.853685 | 0.230952 | 4.576461 | 0.993170 | 0.074260 | 0.077128 | 0.078475 |
| 9 | 0.387318 | — | — | 4.600475 | 0.993106 | 0.078162 | 0.079650 | 0.083450 |
| 10 | 0.387281 | 0.833227 | 0.230952 | 4.580017 | 0.993183 | 0.081796 | 0.081835 | 0.088215 |
| 11 | 0.387342 | — | — | 4.577142 | 0.993240 | 0.085329 | 0.084075 | 0.090721 |
| 12 | 0.387207 | 0.838006 | 0.230952 | 4.611365 | 0.993289 | 0.088988 | 0.086238 | 0.093139 |
| 13 | 0.387246 | — | — | 4.617756 | 0.993304 | 0.092544 | 0.088668 | 0.095514 |
| 14 | 0.387187 | 0.835551 | 0.235714 | 4.632936 | 0.993230 | 0.096220 | 0.090691 | 0.098328 |
| 15 | 0.387193 | — | — | 4.623838 | 0.993265 | 0.099742 | 0.092990 | 0.098347 |
| 16 | 0.387162 | 0.832171 | 0.236905 | 4.621299 | 0.993383 | 0.103458 | 0.095386 | 0.099569 |
| 17 | 0.387157 | — | — | 4.635153 | 0.993330 | 0.107417 | 0.097738 | 0.101605 |
| 18 | 0.387165 | 0.851237 | 0.235714 | 4.656929 | 0.993378 | 0.111398 | 0.100208 | 0.101547 |
| 19 | 0.387131 | — | — | 4.673882 | 0.993269 | 0.115529 | 0.102660 | 0.105975 |
| 20 | 0.387089 | 0.840163 | 0.232143 | 4.651395 | 0.993332 | 0.119633 | 0.105176 | 0.107976 |

## 6. P9 vs P10

| Representation | Adapter | Projected gap | CIDEr | BLEU-4 | Mean IoU | Grounding acc |
|---|---|---:|---:|---:|---:|---:|
| UMBRAE | None | 0.393471 | 0.877624 | 0.248003 | 0.241070 | 0.236905 |
| UMBRAE | P9 Token-Wise | 0.389247 | 0.892492 | 0.251195 | 0.241806 | 0.240476 |
| UMBRAE | P10 TokenMix | no compatible checkpoint | — | — | — | — |
| P6 Full Real | None | 0.391750 | 0.857438 | 0.237390 | 0.237709 | 0.226190 |
| P6 Full Real | P9 Token-Wise | 0.387438 | 0.869061 | 0.245800 | 0.237924 | 0.234524 |
| P6 Full Real | P10 TokenMix | 0.387872 | 0.859395 | 0.241686 | 0.240316 | 0.230952 |
| Exact CLIP Oracle | — | 0 | not run on validation | — | — | — |

UMBRAE has no downstream-compatible P10 checkpoint. Its required `best.pth` is explicitly tagged `diagnostic_interface_best_not_selected` and must not be treated as a selected model; `final_val_metrics.selected` is null.

| Representation | P9 gap − P10 gap | P10 CIDEr − P9 CIDEr | P10 grounding − P9 grounding |
|---|---:|---:|---:|
| UMBRAE | — | — | — |
| P6 Full Real | -0.000434 | -0.009666 | -0.003571 |

## 7. Interface Recovery

| Representation | Adapter | Gap recovery ratio | Projected cosine | Projected norm ratio |
|---|---|---:|---:|---:|
| UMBRAE | None | 0.000000 | 0.606529 | 0.681914 |
| UMBRAE | P10 interface-best (downstream rejected) | 0.013958 | 0.612021 | 0.677569 |
| P6 Full Real | None | 0.000000 | 0.608250 | 0.666529 |
| P6 Full Real | P10 TokenMix selected | 0.009898 | 0.612128 | 0.678387 |

## 8. Token Mixing Diagnostics

| Representation | Epoch | Entropy | Normalized entropy | Max weight | Off-diagonal mass | Collapse |
|---|---:|---:|---:|---:|---:|---|
| UMBRAE | 20 (interface-best rejected) | 4.158594 | 0.749948 | 0.151165 | 0.993692 | False |
| P6 Full Real | 4 | 4.560988 | 0.822514 | 0.106365 | 0.992731 | False |

Per-head token usage is preserved in each run's `attention_diagnostics.jsonl`. Collapse was recorded without changing the architecture.

## 9. Downstream

The complete caption metrics (CIDEr, BLEU-4, ROUGE-L, empty count, output lengths) and grounding metrics (mean IoU, Accuracy@0.5, parse failures) are contained in the main P9/P10 table above and each run's `final_val_metrics.json`.

## 10. Does P6 Benefit More From Token Mixing?

`P6_TOKEN_MIXING_ADVANTAGE = WEAK`

The decision requires simultaneous gap, CIDEr, and grounding advantages; mixed directions are not labeled positive.

## 11. Hypothesis Decision

`TOKEN_ORGANIZATION_MISMATCH = NOT_SUPPORTED`

`P6_INFORMATION_GAIN = WEAK`

`FROZEN_REPRESENTATION_READOUT_LIMIT = REACHED`

## 12. Next-Stage Decision

`P10_STATUS = FROZEN_REPRESENTATION_INFORMATION_LIMIT`

`NEXT_STAGE = FINE_GRAINED_BRAIN_REPRESENTATION`

`EVIDENCE_FAVORS = B_FINE_GRAINED_DECODER_RELEVANT_INFORMATION_IS_MISSING`

The adapter demonstrably used off-diagonal attention without collapse, yet did not improve P9's joint interface/downstream result. The evidence therefore favors genuinely missing fine-grained decoder-relevant information over information being present but merely organized across the wrong tokens.

Operationally, 'improvement' means a strict simultaneous improvement in projected gap, CIDEr, and grounding because the protocol supplied no separate significance threshold. Raw deltas are reported above. No next stage was started.
