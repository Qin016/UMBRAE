# NeuroRoute Stage-2 caption benchmark: S1

## Scope

This is a real-data caption-first benchmark on subject 1 using 8,559
training samples and 300 validation samples. Every sample has a validated
NSD `nsdId -> cocoId -> COCO captions` mapping.

Stage-1 and the 7B language model are frozen. Only the Stage-2 global
projector/adapter path is trained. Validation and generation use fMRI-derived
tokens only.

The bridge is `generic_inputs_embeds_prefix`. This is not a strict Shikra
`<im_patch>` reproduction.

## Methods

- `l24_only`: one fMRI-derived global L24-style token.
- `routed_only`: eight fMRI-derived Stage-1 aligned soft-routing ROI tokens.
- `concat_soft`: global single-L24 path plus soft-routing ROI tokens.
- `concat_uniform`: same global path plus uniform-routing ROI tokens.
- `concat_single_l24`: same global path plus single-L24 ROI tokens.

All concat variants use the same single-L24 global checkpoint.

## Three-epoch results

| Method | Val loss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| L24 only | 2.2089 | 0.2334 | 0.0978 | 0.0465 | 0.0270 | 0.1828 | 0.2205 |
| Routed only | 2.1860 | 0.2282 | **0.1055** | **0.0577** | **0.0359** | **0.2123** | 0.2213 |
| Concat soft | **2.1543** | **0.2371** | 0.1054 | 0.0548 | 0.0320 | 0.2012 | **0.2264** |
| Concat uniform | 2.1831 | 0.2123 | 0.0929 | 0.0504 | 0.0316 | 0.1755 | 0.2121 |
| Concat single-L24 | 2.2038 | 0.2256 | 0.1033 | 0.0562 | 0.0351 | 0.1952 | 0.2231 |

METEOR and SPICE were unavailable because Java is not installed.

`concat_soft` best validation loss improves over:

- `l24_only` by 2.47%;
- `concat_uniform` by 1.32%;
- `concat_single_l24` by 2.24%.

## Interpretation

The results support a downstream benefit from Stage-1 ROI representations:
both `routed_only` and `concat_soft` outperform the L24-only baseline on
validation loss and several caption metrics.

Learnable soft routing has a modest but consistent aggregate advantage over
uniform routing: lower validation loss and higher BLEU-1/2/3/4, CIDEr, and
ROUGE-L.

The comparison with duplicated single-L24 ROI tokens is mixed:
`concat_soft` has lower loss, higher BLEU-1/2, CIDEr, and ROUGE-L, while
`concat_single_l24` has slightly higher BLEU-3/4. This is partial evidence
for multi-layer ROI information, not a universal metric win.

`routed_only` is not poor: it obtains the best BLEU-4 and CIDEr. Therefore
the current S1 result does not show that routed tokens can only be
complementary. `concat_soft` is nevertheless the most balanced method by
validation loss, BLEU-1, and ROUGE-L.

These results do not establish an ROI-specific CLIP-depth hierarchy.

## Leakage and artifacts

For all five runs:

```text
uses_image_clip_tokens_at_eval = false
oracle_image_token_mode = false
mllm_bridge_type = generic_inputs_embeds_prefix
mllm_loader_type = llama_direct_for_shikra_weights
```

Artifacts and machine-readable summaries are under:

```text
stage2_outputs/subj01_real_3epochs/
stage2_outputs/subj01_real_3epochs/summary/
```
