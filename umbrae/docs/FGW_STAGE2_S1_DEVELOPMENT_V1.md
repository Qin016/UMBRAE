# FGW Stage-2 S1 Development V1

This is the subj01-only architecture-matched development experiment. Gamma was
selected from the preregistered grid using validation multi-reference CIDEr only.
No protected downstream test sample was decoded and no protected-test metric was
computed.

## Locked result

`LOCKED_GAMMA = 0.25`

`STAGE2_DEVELOPMENT_STATUS = READY_FOR_REPLICATION`

The locked file was written before the row-shuffled run began. The lock is final
for subsequent subj02/subj05 replication.

## Validation results

| Method | Gamma | 48-token architecture | Selected epoch | Val CIDEr | BLEU-4 | ROUGE-L | Val LM loss |
|---|---:|:---:|---:|---:|---:|---:|---:|
| Uniform prior | 0.00 | yes | 2 | 0.400370 | 0.170904 | 0.408784 | 1.915642 |
| FGW | 0.25 | yes | 2 | 0.422199 | 0.163608 | 0.411014 | 1.919949 |
| FGW | 0.50 | yes | 1 | 0.415072 | 0.164804 | 0.402783 | 1.944667 |
| FGW | 1.00 | yes | 1 | 0.393640 | 0.164329 | 0.397054 | 1.938374 |
| Row-shuffled FGW | 0.25 | yes | 3 | 0.442869 | 0.181781 | 0.415112 | 1.911359 |

## Full run diagnostics

| Run | BLEU-1 | BLEU-2 | BLEU-3 | Checkpoint SHA256 | Time (s) | Peak GPU (GiB) | Trainable | Frozen | Leakage |
|---|---:|---:|---:|---|---:|---:|---:|---:|:---:|
| uniform_prior | 0.523488 | 0.349191 | 0.241464 | `1984f72f69937971dfd2079e062e43816b4e191ab7e3f115c4b81f6c1da8af87` | 4549.9 | 18.324 | 40,167,424 | 6,890,397,052 | pass |
| fgw_gamma025 | 0.527030 | 0.350629 | 0.237280 | `e2bcdff6560e87e13f8fec857477b7fff9804386edbb657624db665f8f072151` | 4572.9 | 18.324 | 40,167,424 | 6,890,397,052 | pass |
| fgw_gamma050 | 0.512859 | 0.343822 | 0.235666 | `900b90c231396c934f96fd23e758d037bb0abce0e80344ba875a0ea374f36171` | 4588.6 | 18.324 | 40,167,424 | 6,890,397,052 | pass |
| fgw_gamma100 | 0.514769 | 0.341827 | 0.234094 | `edab9eb76642abddaaeee00c23ecbfa86fdffe354e1fd18269a4503837852f29` | 4913.5 | 18.324 | 40,167,424 | 6,890,397,052 | pass |
| row_shuffled_fgw | 0.529925 | 0.360780 | 0.252071 | `82ec4e71ebe874a5c1bb9ef141aee52aaa6e4c422415110d9d237c2d8f5259b9` | 5603.2 | 18.324 | 40,167,424 | 6,890,397,052 | pass |

## Locked comparisons

- `FGW_locked_minus_Uniform`: CIDEr +0.021829; BLEU-4 -0.007296; ROUGE-L +0.002230.
- `FGW_locked_minus_RowShuffled`: CIDEr -0.020669; BLEU-4 -0.018172; ROUGE-L -0.004098.


Paired bootstrap intervals in the JSON summary are descriptive only (10,000
resamples, seed 62002) and are available for the evaluator's per-stimulus CIDEr
and ROUGE-L outputs. BLEU-4 is a corpus statistic in this evaluator, so only its
point difference is reported. These diagnostics did not alter locked gamma.

## Architecture and parameter audit

All five runs used 48 ROI-layer tokens, the same Perceiver and Shikra patch bridge,
seed 42, AdamW (learning rate 1e-4, weight decay 1e-2), batch size 2, and three
epochs. Every run had 40,167,424 trainable and 6,890,397,052 frozen parameters;
the trainable parameter names were identical and correspondence contributed zero
trainable parameters. Each epoch was evaluated, and every reported metric comes
from that run's CIDEr-selected checkpoint.

## Data and caption audit

- Train: 8,559 unique stimuli; 0 missing references; 2 duplicate-reference cases;
  reference-count histogram 4:14, 5:8,525, 6:20.
- Validation: 300 unique stimuli; exactly 5 references each; 0 missing and 0
  duplicate-reference cases.
- Protected final test: provenance/count audit only, 982 unique stimuli; 0 missing
  and 0 duplicate-reference cases; reference-count histogram 5:980, 6:2.
- Train, validation, and protected-test stimulus sets are disjoint.

All eight required leakage assertions passed for every run. In particular, image
CLIP tokens were not used at evaluation, oracle mode was off, the transport plan
was frozen with no gradient, and the protected test was used neither for training
nor checkpoint selection.
