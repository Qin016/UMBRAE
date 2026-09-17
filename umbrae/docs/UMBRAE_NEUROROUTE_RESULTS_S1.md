# UMBRAE–NeuroRoute S1 Caption Results

## Protocol

The real subj01 ablation used:

- all 8,559 UMBRAE subj01 training samples;
- all 300 subj01 validation samples;
- three epochs, batch size two, seed 42;
- frozen original UMBRAE BrainX;
- frozen Stage-1 NeuroRoute encoder;
- frozen Shikra/LLaMA;
- adapter-only optimization;
- the original 256-token Shikra
  `<im_start>/<im_patch>/<im_end>` replacement protocol;
- fMRI-derived visual tokens only during validation.

Caption mapping coverage was 100% for both splits. Every method used the
same UMBRAE checkpoint, Shikra weights, prompt, split, optimizer settings,
and decoding parameters. Soft, uniform, and single-L24 used their
corresponding Stage-1 checkpoints.

## Results

Caption metrics below are from the final third epoch. `Best val` is read
from `checkpoint_best.pt`.

| Method | Best val loss ↓ | Final val loss ↓ | BLEU-1 ↑ | BLEU-2 ↑ | BLEU-3 ↑ | BLEU-4 ↑ | ROUGE-L ↑ | CIDEr ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UMBRAE only | **1.884107** | 1.916646 | 0.286008 | 0.166102 | 0.106018 | 0.071218 | 0.294891 | 0.594173 |
| NeuroRoute only | 2.175133 | 2.175133 | 0.245429 | 0.115848 | 0.065491 | 0.041993 | 0.230652 | 0.246029 |
| UMBRAE + soft | 1.909896 | 1.909896 | 0.317801 | 0.187405 | 0.121390 | 0.084187 | 0.304803 | 0.713035 |
| UMBRAE + uniform | 1.908940 | **1.908940** | **0.325883** | **0.196788** | **0.130363** | **0.091184** | **0.307187** | **0.758230** |
| UMBRAE + single L24 | 1.922040 | 1.922040 | 0.281394 | 0.161991 | 0.104864 | 0.072228 | 0.284027 | 0.537156 |

METEOR and SPICE were unavailable because the environment does not have a
Java runtime. Each run generated 300 validation captions.

## Relative results

UMBRAE + soft compared with UMBRAE only:

- final validation loss: 0.35% lower;
- BLEU-4: 18.21% higher;
- ROUGE-L: 3.36% higher;
- CIDEr: 20.00% higher;
- best validation loss: 1.37% worse.

UMBRAE + soft compared with UMBRAE + uniform:

- final validation loss: 0.05% worse;
- BLEU-4: 7.67% worse;
- ROUGE-L: 0.78% worse;
- CIDEr: 5.96% worse.

UMBRAE + soft compared with UMBRAE + single L24:

- final validation loss: 0.63% lower;
- BLEU-4: 16.56% higher;
- ROUGE-L: 7.31% higher;
- CIDEr: 32.74% higher.

## Interpretation

The results support a limited augmentation claim: adding soft NeuroRoute
ROI tokens improves all final-epoch caption metrics over UMBRAE-only and
over the single-L24 ROI-token baseline. However, UMBRAE-only retains the
lowest best validation loss.

Learnable soft routing is not supported as the best routing strategy in
this experiment. Uniform multi-layer ROI tokens achieve the strongest
final validation loss and every reported caption metric. The result
therefore suggests that ROI-aware/multi-layer augmentation is useful under
this setup, but does not demonstrate that learned soft routing is better
than uniform fusion.

NeuroRoute-only is substantially weaker than the combined methods. This
supports treating NeuroRoute ROI tokens as complementary to the original
UMBRAE global path rather than as its replacement.

These results do not establish an ROI-specific CLIP hierarchy. They are
also subj01-only and use the local 300-sample validation split with one
reference caption per sample. They should not be presented as a
cross-subject or paper-level SOTA result.

## Leakage and bridge checks

All five runs recorded:

```text
bridge_type = shikra_patch
uses_image_clip_tokens_at_eval = false
oracle_image_token_mode = false
```

The MLLM received exactly 256 fMRI-derived visual embeddings replacing the
256 Shikra `<im_patch>` positions. No image pixels or image CLIP features
entered the validation brain branch.

## Outputs

Results are stored under:

```text
umbrae_neuroroute_outputs/subj01_real_3epochs/
```

Machine-readable comparison tables are in:

```text
umbrae_neuroroute_outputs/subj01_real_3epochs/summary/
  umbrae_neuroroute_caption_summary.csv
  umbrae_neuroroute_caption_summary.json
```
