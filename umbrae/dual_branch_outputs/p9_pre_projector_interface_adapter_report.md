# P9 Pre-Projector Interface Adapter Report

## 1. Setup

P9 uses train (8,559) and locked validation (300) only. The Brain representation, P6 Structural/Fusion/LoRA components, Shikra `mm_projector`, Shikra decoder, CLIP teacher, tokenizer, prompts, generation, parser, and evaluator remained frozen/unchanged. Only the pre-projector adapter and its scalar residual gate were optimized. No post-hoc test was run or used for selection.

Two matched runs used seed 42, identical sample order, architecture, initialization, teacher, loss, AdamW settings, batch size 64, and the same 20-epoch cap.

## 2. Adapter Architecture

`LayerNorm(1024) → Linear(1024,256) → GELU → Linear(256,1024)` with `Z_hat = Z + sigmoid(gate_logit) * delta`. The final linear was zero initialized and `gate_logit=-2.2`.

- Adapter parameters excluding gate: 527,616
- Gate parameters: 1
- Total P9-trainable parameters: 527,617
- Trainable / BrainX base: 0.360147%

## 3. Initialization Equivalence

| Representation | Max abs error | Mean abs error | Cosine | Initial gate |
|---|---:|---:|---:|---:|
| UMBRAE | 0.00000000 | 0.00000000 | 1.00000000 | 0.09975048 |
| P6 Full Real | 0.00000000 | 0.00000000 | 1.00000024 | 0.09975048 |

## 4. UMBRAE + Adapter Training

| Epoch | Projected gap | Projected cosine | CIDEr | Grounding acc | Gate | Correction ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.389979 | 0.610021 | — | — | 0.108450 | 0.041497 |
| 2 | 0.389247 | 0.610753 | 0.892492 | 0.240476 | 0.122184 | 0.051225 |
| 3 | 0.388950 | 0.611050 | — | — | 0.134425 | 0.056109 |
| 4 | 0.388811 | 0.611189 | 0.875103 | 0.230952 | 0.146789 | 0.064032 |
| 5 | 0.388583 | 0.611417 | — | — | 0.159858 | 0.073183 |
| 6 | 0.388495 | 0.611505 | 0.871501 | 0.230952 | 0.172767 | 0.081437 |
| 7 | 0.388380 | 0.611620 | — | — | 0.185048 | 0.087981 |
| 8 | 0.388347 | 0.611653 | 0.869075 | 0.235714 | 0.196585 | 0.093196 |
| 9 | 0.388215 | 0.611785 | — | — | 0.207961 | 0.099631 |
| 10 | 0.388172 | 0.611828 | 0.878814 | 0.232143 | 0.218609 | 0.104471 |
| 11 | 0.388212 | 0.611788 | — | — | 0.228618 | 0.105907 |
| 12 | 0.388105 | 0.611895 | 0.867748 | 0.229762 | 0.238311 | 0.108555 |
| 13 | 0.388088 | 0.611912 | — | — | 0.248580 | 0.111245 |
| 14 | 0.388034 | 0.611966 | 0.867743 | 0.232143 | 0.258106 | 0.113269 |
| 15 | 0.387974 | 0.612026 | — | — | 0.267402 | 0.114827 |
| 16 | 0.387994 | 0.612006 | 0.861005 | 0.233333 | 0.276995 | 0.116199 |
| 17 | 0.387970 | 0.612030 | — | — | 0.286763 | 0.118787 |
| 18 | 0.387920 | 0.612080 | 0.867258 | 0.230952 | 0.295879 | 0.118966 |
| 19 | 0.387895 | 0.612105 | — | — | 0.305756 | 0.121016 |
| 20 | 0.387829 | 0.612171 | 0.858989 | 0.229762 | 0.314989 | 0.121953 |

## 5. P6 Full Real + Adapter Training

| Epoch | Projected gap | Projected cosine | CIDEr | Grounding acc | Gate | Correction ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.388807 | 0.611193 | — | — | 0.108834 | 0.041686 |
| 2 | 0.388225 | 0.611775 | 0.846777 | 0.227381 | 0.122179 | 0.051735 |
| 3 | 0.387967 | 0.612033 | — | — | 0.134329 | 0.056492 |
| 4 | 0.387824 | 0.612176 | 0.864357 | 0.229762 | 0.148009 | 0.064588 |
| 5 | 0.387647 | 0.612353 | — | — | 0.161601 | 0.073689 |
| 6 | 0.387556 | 0.612444 | 0.858441 | 0.234524 | 0.174418 | 0.081489 |
| 7 | 0.387456 | 0.612544 | — | — | 0.186133 | 0.087751 |
| 8 | 0.387438 | 0.612562 | 0.869061 | 0.234524 | 0.197148 | 0.091552 |
| 9 | 0.387327 | 0.612673 | — | — | 0.208132 | 0.096700 |
| 10 | 0.387294 | 0.612706 | 0.848749 | 0.235714 | 0.218843 | 0.100979 |
| 11 | 0.387321 | 0.612679 | — | — | 0.228820 | 0.102197 |
| 12 | 0.387274 | 0.612726 | 0.856010 | 0.233333 | 0.238843 | 0.104065 |
| 13 | 0.387238 | 0.612762 | — | — | 0.249480 | 0.106925 |
| 14 | 0.387218 | 0.612782 | 0.849476 | 0.232143 | 0.259601 | 0.108014 |
| 15 | 0.387161 | 0.612839 | — | — | 0.269414 | 0.109664 |
| 16 | 0.387176 | 0.612824 | 0.846493 | 0.233333 | 0.279884 | 0.110972 |
| 17 | 0.387177 | 0.612823 | — | — | 0.290415 | 0.112671 |
| 18 | 0.387117 | 0.612883 | 0.847740 | 0.230952 | 0.300479 | 0.113160 |
| 19 | 0.387118 | 0.612882 | — | — | 0.311410 | 0.114856 |
| 20 | 0.387046 | 0.612954 | 0.834675 | 0.233333 | 0.322157 | 0.116566 |

## 6. Interface Recovery

| Brain representation | Adapter | Projected gap | Gap recovery | Projected cosine | Norm ratio |
|---|---|---:|---:|---:|---:|
| UMBRAE | None | 0.393471 | 0.000000 | 0.606529 | 0.681914 |
| UMBRAE | Interface Adapter (epoch 2) | 0.389247 | 0.010735 | 0.610753 | 0.684558 |
| P6 Full Real | None | 0.391750 | 0.000000 | 0.608250 | 0.666529 |
| P6 Full Real | Interface Adapter (epoch 8) | 0.387438 | 0.011007 | 0.612562 | 0.678694 |
| Exact CLIP Oracle | — | 0 | identity | 1 | 1 |

## 7. Representation Metrics

| Representation | Adapter | Spearman RSA | Pearson RSA | Hard R@1 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| UMBRAE | None | 0.853007 | 0.866147 | 0.993333 | 0.996667 | 0.996667 | 0.995256 |
| UMBRAE | Interface Adapter | 0.843185 | 0.855728 | 0.993333 | 0.996667 | 0.996667 | 0.995185 |
| P6 Full Real | None | 0.874542 | 0.888198 | 0.996667 | 1.000000 | 1.000000 | 0.997333 |
| P6 Full Real | Interface Adapter | 0.851298 | 0.865219 | 0.993333 | 0.996667 | 0.996667 | 0.995278 |

RSA and hard retrieval are diagnostics only and were not used for checkpoint selection.

## 8. Validation Downstream

| Brain representation | Adapter | CIDEr | BLEU-4 | ROUGE-L | Mean IoU | Grounding acc | Parse failures |
|---|---|---:|---:|---:|---:|---:|---:|
| UMBRAE | None | 0.877624 | 0.248003 | 0.496099 | 0.241070 | 0.236905 | 0 |
| UMBRAE | Interface Adapter | 0.892492 | 0.251195 | 0.495911 | 0.241806 | 0.240476 | 0 |
| P6 Full Real | None | 0.857438 | 0.237390 | 0.489669 | 0.237709 | 0.226190 | 0 |
| P6 Full Real | Interface Adapter | 0.869061 | 0.245800 | 0.494628 | 0.237924 | 0.234524 | 0 |
| Exact CLIP Oracle | — | not run on validation | — | — | — | — | — |

Checkpoint compatibility required both `CIDEr >= source CIDEr` and `grounding accuracy >= source grounding accuracy`; among compatible checkpoints, the lowest projected gap was selected. The earlier Exact CLIP values (CIDEr 1.635679, grounding 0.523357) came from reused test and are used only as explicitly labeled recovery-reference constants, not as P9 validation evidence or selection data.

## 9. Does P6 contain more usable information?

`P6_INFORMATION_GAIN = WEAK`

This label compares only downstream-compatible selected checkpoints. A missing compatible checkpoint is not replaced by an interface-only best checkpoint.

## 10. Interface Adapter Effect

`UMBRAE_INTERFACE_ADAPTER = POSITIVE`

`P6_INTERFACE_ADAPTER = POSITIVE`

## 11. Oracle Headroom Recovery

| Representation | Gap recovery ratio | CIDEr recovery fraction | Grounding recovery fraction |
|---|---:|---:|---:|
| UMBRAE | 0.010735 | 0.019614 | 0.012468 |
| P6 Full Real | 0.011007 | 0.014935 | 0.028043 |

## 12. Scientific Interpretation

`INTERFACE_MISMATCH_HYPOTHESIS = PARTIALLY_SUPPORTED`

`FROZEN_P6_INFORMATION_CONTENT = PARTIAL`

`SIMPLE_TOKENWISE_ADAPTER = SUFFICIENT`

`P9_STATUS = GENERIC_INTERFACE_ADAPTATION_WORKS`

The result is validation-only. No P9 post-hoc reused-test evaluation was run, no test metric influenced training or selection, and no next-stage token-mixing or joint fine-tuning was started.
