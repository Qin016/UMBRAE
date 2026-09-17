# FGW Stage-2 Attention-Prior Implementation V1

Date: 2026-08-21

Status: Prompt 6A implemented and tested. Prompt 6B was not started. No real
Stage-2 training, caption evaluation, or gamma selection was run.

## Conditions and insertion

The strict adapter now supports:

- `uniform_prior`
- `fgw_prior`
- `row_shuffled_fgw_prior`

All three use the same 256 UMBRAE keys, the same unweighted 48 ROI-layer keys,
the same two-block Perceiver, and the same trainable adapter. Only the frozen
attention bias differs. The bias is shared across queries and heads and is
added only at the first block (`--fgw-bias-layers first`). Positions outside
the dynamically derived ROI-layer slice receive exact zero. The audited
strict fusion path has no latent/self keys.

For frozen transport plan `T`:

```text
A[r,l] = T[r,l] / sum_k T[r,k]
q_gamma[r,l] = (A[r,l] + 1e-8)^gamma
               / sum_k (A[r,k] + 1e-8)^gamma
bias[r,l] = log(6 * q_gamma[r,l])
```

Uniform `A` and `gamma=0` both produce exact zero bias. Every row satisfies
`sum_l exp(bias[r,l]) = 6` within numerical tolerance. There is no clipping,
learned transform, or correspondence parameter.

## Frozen plans and provenance

The allow-listed loader rejects subj07 and validates exact axis order, shape
`[8,6]`, nonnegativity, total mass 1, row mass 1/8, and SHA256 over the exact
`.npy` bytes.

| Subject | Frozen plan SHA256 |
|---|---|
| subj01 | `51e65527f3bbc1a3951ec376dd8be3662cab9e6b5f682d2a0da26fb347e20760` |
| subj02 | `a3d3ab31770ecc0fc62599597538a7b931df2721fdd22aabb13e8e7e8f7c8da0` |
| subj05 | `a3d3ab31770ecc0fc62599597538a7b931df2721fdd22aabb13e8e7e8f7c8da0` |

The loaded method provenance is frozen as `sr_fgw`, beta `0.5`, coverage
weight `0`, and entropy `0`. Each future Stage-2 run writes
`fgw_plan_provenance.json`, embeds the provenance in `adapter_config.json`,
and stores the frozen plan and transformed bias as non-gradient model buffers.

## Preregistered row null

`fgw_outputs/stage2_fgw_preregistered_null.json` records NumPy seed `62001`
and the first generated derangement:

```text
[3,4,6,2,1,7,5,0]
```

That is, the rows assigned to `V1,V2,V3,hV4,FFA,EBA,PPA,OPA` come from
`hV4,FFA,PPA,V3,V2,OPA,EBA,V1`, respectively. It has no fixed points and is
shared unchanged by S1/S2/S5. Tests verify preservation of the target
marginal, total mass, and row-entropy multiset.

## Bias diagnostics (no training)

All three frozen plans contain exact zeros and, in this repository, have the
same final transport values. Consequently S1/S2/S5 have the same diagnostics:

| gamma | min | max | mean absolute | population std |
|---:|---:|---:|---:|---:|
| 0.25 | -2.862201 | 1.742969 | 2.675662 | 1.716246 |
| 0.50 | -7.419081 | 1.791260 | 6.481111 | 3.432491 |
| 1.00 | -16.628921 | 1.791759 | 14.156061 | 6.864982 |

One frozen subj01 training sample (`sample000000300`) was passed through the
frozen BrainX and Stage-1 ROI encoder and the completed uniform adapter weights
in eval/no-grad mode. The first strict fusion block's raw zero-prior content
logits had:

```text
shape = [1,8,256,304]
min = -206.094330
max = 230.443161
mean = 0.719697
mean absolute = 46.651966
population std = 59.785221
```

Thus none of the three proposed bias ranges is numerically pathological
relative to the audited content logits. This is diagnostic only; no gamma was
selected and `delta` remains exactly `1e-8`.

## Trainable-parameter audit

The existing adapter-only protocol is preserved. For the real 1024-dimension,
two-block configuration:

| Component | Status | Parameter count |
|---|---|---:|
| Fusion adapter | trainable | 40,167,424 |
| Frozen UMBRAE BrainX | frozen | 146,500,608 |
| Frozen ROI tokenizer/projector | frozen | 5,456,252 |
| Frozen Shikra/Llama bridge model | frozen | 6,738,440,192 |
| Total frozen | frozen | 6,890,397,052 |
| FGW correspondence/bias parameters | trainable | 0 |

Every run prints and saves all trainable parameter names plus trainable and
frozen counts in `parameter_audit.json`. The existing protocol trains the
whole adapter module, including its pre-existing alternative-fusion members;
the FGW implementation adds buffers but no parameters.

## Leakage audit

The Stage-2 dataset returns only `fmri`, caption text, sample ID, and COCO ID.
The model path uses fMRI-derived UMBRAE tokens and fMRI-derived ROI tokens.
There is no image tensor, image CLIP hidden state, oracle token, FGW solver,
feature-cost recomputation, or geometry recomputation in the path. The
training entry point continues to reject oracle-image-token mode. Frozen T is
the only correspondence artifact consumed.

## Tests

`tests/test_fgw_stage2_attention_prior.py` covers all requested invariants:
plan/order/hash validation; A normalization; uniform and gamma-zero bias;
row prior mass; exact key placement; attention broadcasting; derangement
invariants; gamma-zero forward equivalence; legacy 48-token uniform control;
strict `[B,256,4096]` output; no image/CLIP dataset field; and no transport
gradient.

Executed with `/opt/conda/envs/brainx/bin/python`:

```text
test_fgw_stage2_attention_prior.py: 9 test groups passed
test_umbrae_neuroroute_adapter.py: passed
test_structured_routing_modes.py: passed
test_train_umbrae_neuroroute_synthetic.py: passed (synthetic smoke only)
py_compile for all modified Python files: passed
```

The environment does not have pytest installed, so the new test file follows
the repository's existing directly executable test convention. The synthetic
smoke test used generated tensors/tiny injected models only; it did not access
real Stage-2 outcomes or train a scientific model.
