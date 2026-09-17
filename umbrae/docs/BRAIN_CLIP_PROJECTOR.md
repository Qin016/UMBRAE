# Brain-to-CLIP projector

`BrainToCLIPProjector` maps ROI-token representations into the frozen
CLIP feature distribution before Stage-1 alignment:

```text
raw ROI tokens [B,R,D_in]
    -> BrainToCLIPProjector
projected ROI tokens [B,R,D_out]
    <-> routed CLIP targets [B,R,D_out]
```

Raw ROI tokens are produced from heterogeneous voxel groups and include
ROI identity information. Their scale and coordinate distribution need
not match frozen CLIP hidden states. Directly forcing raw ROI tokens to
match CLIP features makes the tokenizer simultaneously learn anatomical
aggregation and CLIP-space calibration, which can encourage shortcut
routing or scale mismatch. The projector isolates this representation
conversion while preserving raw ROI tokens for routing queries and later
brain-side processing.

Supported heads:

- `linear`: LayerNorm, linear projection, dropout.
- `mlp`: LayerNorm, two-layer GELU MLP, dropout.
- `residual_mlp`: residual two-layer MLP; requires `D_in == D_out`.

Stage-1 defaults to an MLP with `hidden_dim=D_in` and dropout `0.1`.
Disable it with `--no-use-brain-clip-projector` to reproduce the earlier
direct-alignment path.

`--detach-routed-targets` detaches frozen CLIP layer features before
routing. It does not detach the routing probabilities themselves, so the
soft router query/key projections remain trainable.
