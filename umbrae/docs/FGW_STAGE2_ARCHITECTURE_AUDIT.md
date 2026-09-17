# FGW Stage-2 Architecture Audit

Date: 2026-08-21

Scope: Prompt 6A only. No scientific Stage-2 training or caption-based model
selection was run.

## Authoritative strict Stage-2 path

The architecture-matched UMBRAE/Shikra path is:

```text
scripts/train_umbrae_neuroroute.py
  -> FrozenUMBRAEEncoder (model.py: BrainX/BrainXS)
  -> FrozenStage1FMRIEncoder (ROITokenizer + BrainToCLIPProjector)
  -> models/umbrae_neuroroute_adapter.py: UMBRAENeuroRouteAdapter
  -> replace_shikra_patch_embeddings
  -> frozen Shikra/Llama
```

`scripts/train_stage2_mllm_neuroroute.py` is also present, but its
`NeuroRouteMLLMAdapter` uses a generic short prefix and is not the completed
strict UMBRAE augmentation used by the S1 caption experiments.

## Audited token path

For the completed strict experiments, `roi_token_expansion=1`, six selected
CLIP layers are ordered `L4,L8,L12,L16,L20,L24`, and the named ROI order is
`V1,V2,V3,hV4,FFA,EBA,PPA,OPA`.

The original strict `umbrae_plus_uniform`, `umbrae_plus_soft`, and
`umbrae_plus_single_l24` modes create 8 ROI tokens, not 48. The later legacy
structured modes (`umbrae_plus_uniform_residual_soft` and
`umbrae_plus_temperature_soft`) create 48 tokens in this exact order:

```text
(V1,L4), (V1,L8), ..., (V1,L24),
(V2,L4), ..., (V2,L24),
...,
(OPA,L4), ..., (OPA,L24)
```

The 48-token construction is:

```text
projected fMRI ROI token
  -> token-wise LayerNorm
  -> Linear + GELU
  -> source-type embedding (ROI identity is already present in the frozen
     ROITokenizer representation)
  -> add learned CLIP-layer identity embedding
```

The legacy structured path then performs
`layer_token *= routing_weight * 6`. This happens before `_PerceiverBlock`'s
token-wise `source_norm = LayerNorm(1024)`. Therefore the old path does encode
routing mass as a pre-LayerNorm scalar. The new FGW prior modes reuse the same
48-token construction but omit that multiplication entirely.

## Strict fusion Perceiver

The adapter's `perceiver_resampler` is a two-block stack of the repository's
custom `_PerceiverBlock`; it is distinct from the six-layer lucidrains
`PerceiverResampler` inside the frozen UMBRAE BrainX encoder.

Each strict fusion block applies:

```text
Q = query_norm(queries)
K = source_norm(source)
V = source_norm(source)
attention = MultiheadAttention(Q,K,V), 8 heads
queries = queries + attention
queries = queries + FFN(queries)
```

PyTorch `MultiheadAttention` computes projected Q/K/V internally. The audited
content score is `QK^T / sqrt(128)` per head.

For the new three prior modes with batch size `B`:

| Quantity | Actual value |
|---|---:|
| Output queries | 256 |
| UMBRAE source keys | positions `[0,256)` |
| ROI-layer source keys | positions `[256,304)` |
| Latent/self source keys | none (count 0) |
| Total source/media keys | 304 |
| Attention-logit shape | `[B,8,256,304]` |
| Cross-attention blocks consuming these keys | 2 |
| Blocks receiving the FGW prior | first block only |

The strict fusion Perceiver does **not** concatenate latent/self keys to the
media source. The separate frozen BrainX Perceiver does concatenate its media
and latents internally, but that upstream module is frozen and is not the
Stage-2 prior insertion point.

The key start is derived at runtime from the actual UMBRAE tensor length; the
implementation does not assume position 256 when constructing the mask.

## Shikra bridge

The adapter always produces `[B,256,4096]`. In strict `shikra_patch` mode,
`replace_shikra_patch_embeddings` requires exactly one `<im_start>`, replaces
exactly the following 256 placeholder positions, and requires `<im_end>`
immediately afterward. The visual sequence is not appended as a generic
prefix in the primary path.

## Existing S1 result directories

Completed strict S1 directories are under
`umbrae_neuroroute_outputs/subj01_real_3epochs/`:

- `umbrae_plus_uniform`
- `umbrae_plus_soft`
- `umbrae_plus_single_l24`

Their configs record `fusion_type=perceiver_resampler`,
`bridge_type=shikra_patch`, 256 visual tokens, and frozen UMBRAE, NeuroRoute,
and MLLM components.

## Uniform architecture equivalence

`UNIFORM_ARCHITECTURE_EQUIVALENT = false` for the completed S1
`umbrae_plus_uniform` result versus the new `uniform_prior`: the former has 8
ROI keys and the latter has 48 ROI-layer keys. Existing S1 uniform caption
metrics therefore cannot be reused as the architecture-matched primary
baseline.

As an implementation control, the new zero-bias `uniform_prior` is
deterministically equivalent (within `1e-6`) to the repository's legacy
48-token structured path supplied with exactly uniform routing. No completed
primary S1 result directory used that 48-token architecture.
