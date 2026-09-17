# Dual-branch UMBRAE MVP foundation

P2 implements only Stage A and Stage B. It does not implement LoRA, Stage C,
caption training, spatial OT, pRF, multi-layer CLIP, routing, GW, or FGW.

The frozen semantic output is `Z_sem [B,256,1024]`. The trainable structural
output is `H_struct [B,8,1024]`. Gated cross-attention preserves the semantic
token interface and returns `Z_cal [B,256,1024]` for the existing 1024-to-4096
Shikra projector.

## Cache contract

Stage A does not require a semantic cache. Stage B may cache frozen
`Z_sem [256,1024]` and frozen CLIP `V_patch [256,1024]`. Cache metadata must
record sample ID, subject, BrainX checkpoint path/hash, CLIP model and layer
definition, repeat aggregation policy, dtype, and feature shapes.

Never cache `H_struct`, because the structural branch is trainable. Once Stage C
adds LoRA inside BrainX, cached Z_sem is forbidden: a cached tensor bypasses the
trainable Q/V path and therefore prevents LoRA from affecting the loss or
receiving gradients.

The Stage-B P2 formal objective currently contains semantic UOT only. The
training skeleton exposes a clean additional-loss hook. The optional
`smoke_fusion_connectivity` term is solely a short gradient-connectivity probe;
it is not a proposed scientific objective or loss weight.
