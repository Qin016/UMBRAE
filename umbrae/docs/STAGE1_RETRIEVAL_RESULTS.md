# NeuroRoute Stage-1 retrieval results

## Protocol

Retrieval uses 300 validation fMRI/image pairs per subject. All four
validation sets contain 300 unique `coco73k` IDs, so the implemented
multi-positive protocol reduces to one positive per query for these
particular shards. The evaluator still matches by ID rather than assuming
the diagonal.

The common evaluation protocol is:

- brain query: fMRI-only `ROITokenizer -> BrainToCLIPProjector -> mean ROI`;
- image gallery: frozen CLIP L24 pooled feature;
- cosine similarity after L2 normalization;
- no image feature enters the brain branch.

## Failed attention+routed smoke checkpoint

The original `soft_contrastive_attention` checkpoint contains valid
attention-pooler weights, but was trained for only one epoch with ten
train/validation steps. Mean/attention and L24/routed evaluation all stayed
near the 300-way random baseline. This identifies the checkpoint as an
undertrained smoke run, not an attention-loading failure.

## Conservative S1 contrastive sweep

Soft routing was trained for 10 epochs with mean ROI pooling, L24 image
targets, temperature 0.07, and contrastive weights 0.01, 0.03, and 0.05.

| Method | B2I R@1 | R@5 | R@10 | MedR | MeanR |
|---|---:|---:|---:|---:|---:|
| Soft alignment-only | 1.00% | 3.33% | 5.33% | 120.5 | 127.53 |
| Uniform alignment-only | 0.33% | 3.00% | 5.33% | 124.5 | 132.79 |
| Single-L24 alignment-only | 0.67% | 4.00% | 6.00% | 108.0 | 119.46 |
| Soft contrastive 0.01 | 1.00% | 3.67% | 6.67% | 112.0 | 122.36 |
| Soft contrastive 0.03 | 1.00% | 5.00% | 7.33% | 98.5 | 115.14 |
| Soft contrastive 0.05 | 1.00% | 5.00% | 8.00% | 96.5 | 112.68 |

Weight 0.05 gave the strongest S1 B2I R@10 and ranks. Its best-checkpoint
alignment component was 0.108835 versus 0.103179 for alignment-only soft,
a 5.48% degradation. It is therefore a retrieval-oriented trade-off, not
an unconditional replacement for the alignment-only checkpoint.

## Cross-subject results

Brain-to-image averages across S1/S2/S5/S7:

| Method | R@1 | R@5 | R@10 | MedR | MeanR |
|---|---:|---:|---:|---:|---:|
| Soft alignment-only | 0.92% | 3.17% | 5.75% | 120.88 | 129.42 |
| Uniform alignment-only | 0.42% | 2.75% | 5.08% | 124.38 | 133.03 |
| Single-L24 alignment-only | **1.42%** | 4.67% | 7.58% | 109.75 | 121.74 |
| Soft contrastive 0.05 | 1.08% | **4.83%** | **9.92%** | **94.63** | **112.08** |

Image-to-brain averages:

| Method | R@1 | R@5 | R@10 | MedR | MeanR |
|---|---:|---:|---:|---:|---:|
| Soft alignment-only | 2.67% | 12.50% | 21.75% | 39.38 | 61.32 |
| Uniform alignment-only | 2.25% | 8.83% | 15.17% | 56.63 | 77.78 |
| Single-L24 alignment-only | **6.33%** | **18.42%** | **31.00%** | **28.00** | **45.82** |
| Soft contrastive 0.05 | 4.67% | 17.75% | 28.83% | 29.13 | 48.24 |

The contrastive checkpoint's alignment component degraded by 5.48%,
5.57%, 7.68%, and 5.66% for S1, S2, S5, and S7 respectively.

## Interpretation

Supported conclusions:

- Alignment-only soft routing is consistently better than uniform on the
  Stage-1 alignment objective and is modestly better than uniform retrieval
  on average.
- Alignment-only soft routing does not beat single-L24 retrieval.
- Conservative L24 contrastive training improves B2I R@5, R@10, and rank
  relative to alignment-only soft and gives the best average B2I R@10/rank.
- Single-L24 remains stronger for B2I R@1 and most I2B metrics.

Thus retrieval partially supports a learned soft representation, but does
not establish universal soft-routing superiority. It also does not support
a distinct ROI-to-depth hierarchy.

Machine-readable tables are stored in:

`stage1_outputs/retrieval_summary/cross_subject_l24/`
