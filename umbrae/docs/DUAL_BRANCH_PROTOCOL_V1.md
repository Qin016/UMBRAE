# Dual-Branch Protocol V1

This file locks the subj01 MVP protocol before Stage A training. The machine-readable authority is `configs/dual_branch/protocol_v1.json`. No Stage A/B/C training, UOT tuning, LoRA implementation, or test evaluation was performed while creating this snapshot.

## Data and leakage gate

The manifests were produced by opening every tar member and reading its metadata and image bytes, rather than trusting shard names. Train uses `train_subj01_0.tar` through `train_subj01_17.tar` (8,559 samples/stimuli), validation uses `val_subj01_0.tar` (300), and test uses `test_subj01_0.tar` and `test_subj01_1.tar` (982). Every shard's embedded subject agrees with subj01.

Image SHA256 is the primary stimulus identity and `coco73k` is the secondary identity. Train/validation/test intersections are zero for sample ID, image SHA256, and coco73k ID. There are no duplicate sample IDs or image hashes within a split. The leakage gate therefore passes. The test manifest exists only for integrity auditing.

## Repeat aggregation

The locked train, validation, and test policy is `mean_over_num_uniques_valid_repeats`. One deterministic, preaggregated `[B,V]` fMRI tensor is supplied to both branches.

This intentionally differs from original UMBRAE code. Original training uses `repeat_index = train_i % 3` and indexes that slot without consulting `num_uniques`. Original validation and inference average every stored repeat slot with `torch.mean(voxel, axis=1)`, also without consulting `num_uniques`. The new policy avoids invalid padded slots, is deterministic, and prevents semantic/structural branch repeat mismatch.

## Frozen encoders

BrainX is loaded from the absolute checkpoint recorded in the JSON, whose SHA256 is `b38aa7ce9d11d56c36ede011ec757a3fda4194c5deaa5f7b0f5f8e42187bb4fb`. Strict reload verifies model kind BrainX, 146,500,608 parameters, and semantic output `[B,256,1024]`.

The only visual teacher is `openai/clip-vit-large-patch14`. Its locked output is `hidden_states[-2][:,1:,:]`, the block-23-equivalent 256 raw patch tokens with shape `[B,256,1024]`. Input is RGB; preprocessing is deterministic resize 224 (bicubic, antialias), center crop 224, then CLIP mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation `(0.26862954, 0.26130258, 0.27577711)`. Multi-layer CLIP, pooled layer banks, DINO, and other teachers are outside protocol v1.

## ROI and random control

ROI order is V1, V2, V3, hV4, FFA, EBA, PPA, OPA. Counts are 1,350; 1,433; 1,187; 687; 687; 2,238; 720; and 1,532. There are 9,834 assignments over 9,666 unique voxels. Nonzero off-diagonal intersections are V3–OPA 4, hV4–FFA 7, and EBA–OPA 157. The complete matrix and mapping hash are in `roi_audit.json`.

The matched random control applies one seed-42 random bijection over the exact real ROI voxel union to every ROI membership. Consequently it exactly preserves the voxel universe, every ROI size, every pairwise intersection, and higher-order intersections. It does not inspect image labels, CLIP features, or fMRI response magnitude. Tests establish identical seed reproducibility and different-seed change.

## Evaluation

`q_sem`, `q_visual`, and later `q_cal` are the arithmetic mean over their respective 256 tokens, producing `[B,1024]`, followed by L2 normalization for cosine evaluation. This is a new unified protocol evaluator; original downstream task code is not substituted as an alternate pooling rule.

Retrieval uses one square candidate pool containing all 300 validation stimuli for every method. Brain-to-image is primary; image-to-brain is secondary. Both report Recall@1/5/10, median rank, and mean rank.

RSA constructs cosine-distance RDMs (`1 - cosine`), extracts the strict upper triangle, and reports Spearman correlation as primary. Pearson is auxiliary. A future differentiable RDM training loss is not the evaluation RSA.

## Baseline and experiment identity

`BASELINE_UMBRAE_V1` is frozen BrainX versus frozen CLIP only. It does not use the structural branch, fusion, UOT, or LoRA. Its snapshot directory is `protocol_outputs/protocol_v1/subj01/umbrae_s1_baseline_seed42_protocolv1`. Each experiment uses `{method}_{sN}_{stage}_seed{seed}_{protocol}` and saves config, metrics, protocol version, checkpoint/ROI hashes, seed, and git commit. Git is not installed in this workspace, so this snapshot explicitly records `git_commit_hash: UNAVAILABLE` rather than fabricating an identifier.

## Test seal

Validation is the default evaluator split. Test additionally requires `--allow-test` and is forbidden for Stage A/B hyperparameters, UOT/fusion loss weights, LoRA rank, checkpoint selection, and early stopping. It may be opened only for a preregistered final model or predeclared frozen comparison, with the same candidate pool and protocol for every method. See `TEST_SET_POLICY.md`.

## UOT diagnostics

The retained legacy entropy is `-sum(T * log(T)) / (sum(T) + eps)` (the implementation uses the already available exact `log_transport`). The normalized diagnostic first sets `p=T/(sum(T)+eps)`, computes `H=-sum(p*log(p+eps))` independently per sample, then reports `H/log(R*P)` (R=8, P=256), `exp(H)` effective support, and `max(p)`. Row and column mass and their coefficients of variation are diagnostics only; protocol v1 does not constrain UOT marginals or change epsilon/tau.
