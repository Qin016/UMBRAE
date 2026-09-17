# ROI-to-CLIP-layer soft router

`ROILayerRouter` maps ROI tokens to mixtures of selected CLIP layer
features without changing the existing UMBRAE training path.

## Shapes

- ROI tokens `z`: `[B, R, D]`
- CLIP layer features `v`: `[B, L, D]`
- Routing weights `A`: `[B, R, L]`
- Routed targets: `[B, R, D]`

For NeuroRoute v1, `R=8`. `L` is the number of layers selected by
`CLIPLayerBank`, and both inputs must use the same common feature
dimension `D`.

## Definition

The router projects ROI tokens into queries and CLIP layer features into
keys:

```text
A[r,l] = softmax((Q(z[r]) dot K(v[l])) / sqrt(hidden_dim) / tau)
routed_target[r] = sum_l A[r,l] * v[l]
```

The temperature `tau` is fixed by default and can optionally be
learnable. Optional top-k routing keeps the largest `k` probabilities,
sets the others to zero, and renormalizes the retained weights.

## Regularization

- Entropy loss is minimized with a small coefficient to encourage
  sharper routing without directly forcing one-hot assignments.
- Balance loss is the KL divergence between average layer usage and a
  uniform distribution. It discourages all ROIs from collapsing onto
  one CLIP layer.
- Smoothness loss penalizes squared differences between neighboring
  layer probabilities. It is appropriate when selected CLIP layers are
  ordered and nearby layers are expected to behave similarly.

L1 sparsity is not applied to softmax probabilities because every
routing vector already has L1 norm equal to one. An L1 penalty would
therefore be constant and provide no sparsity gradient.

The three weights default to zero. The module is not connected to the
training loop in Step 4.

## Example

```python
router = ROILayerRouter(
    feature_dim=64,
    hidden_dim=32,
    temperature=0.7,
    topk=None,
)
out = router(roi_tokens, clip_layer_features)
routed_targets = out["routed_targets"]
routing_weights = out["routing_weights"]
```
