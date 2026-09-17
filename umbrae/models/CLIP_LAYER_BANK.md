# CLIP Multi-Layer Bank

`CLIPLayerBank` extracts hidden states from selected one-based Transformer
blocks of a Hugging Face CLIP vision encoder.

For CLIP ViT-L/14 and the default layers `[4, 8, 12, 16, 20, 24]`:

- Input images: `[B, 3, H, W]`, floating point in `[0, 1]`.
- `layer_tokens`: `[B, 6, 256, target_dim]`.
- `pooled_tokens`: `[B, 6, target_dim]`.
- `selected_layers`: `[4, 8, 12, 16, 20, 24]`.

The CLS token is excluded. This preserves the 256 patch-token contract used by
the existing UMBRAE image feature extraction.

Each selected layer has an independent linear projection to `target_dim`.
When the CLIP hidden dimension already equals `target_dim`, the projection is
initialized as identity. CLIP is frozen by default; projection heads remain
trainable.

Training scripts retain the original `BrainEncoder` path by default. With
`--use_clip_layer_bank`, projected layer tokens are averaged across the layer
dimension to produce `[B, 256, target_dim]`, which keeps the current
reconstruction loss interface unchanged.

Example:

```bash
python tests/test_clip_layer_bank.py
```

The smoke test injects a small randomly initialized CLIP vision model with the
same 24-layer/256-patch interface, so it validates the module without downloading
the 1.71 GB ViT-L/14 checkpoint. Real training loads the configured pretrained
CLIP model normally.

Training configuration:

```bash
--use_clip_layer_bank \
--clip_selected_layers 4 8 12 16 20 24 \
--clip_layer_target_dim 1024 \
--freeze_clip
```
