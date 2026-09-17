# UMBRAE reproduction notes

## Compatibility changes

1. Recreated the `brainx` Conda environment with Python 3.10 instead of the
   unconstrained Python 3.14 environment. The original project documents
   Python 3.10, and the Python 3.14 environment installed PyTorch 2.12.1 with
   CUDA 13.0, which cannot run on the host's NVIDIA 550.127.05 driver.
2. Used PyTorch 2.3.0 and torchvision 0.18.0 with CUDA 12.1. These versions are
   compatible with the host driver and RTX A6000.
3. Pinned Transformers to 4.31.0 and Accelerate to 0.20.3. The repository's
   unpinned Transformers dependency had resolved to an incompatible future
   release, while Transformers 4.31.0 requires Accelerate 0.20.3 or newer.
4. Pinned NumPy below 2 to avoid incompatibilities with packages and serialized
   artifacts produced before NumPy 2.
5. Added optional `--max_samples` and `--max_new_tokens` inference arguments.
   Their defaults preserve the original full evaluation (all samples and 512
   generated tokens); they allow a bounded end-to-end reproduction check.
6. Passed `--data_path nsd` because the downloaded dataset in this workspace is
   named `umbrae/nsd`, rather than the `nsd_data` placeholder in the README.

## End-to-end smoke reproduction

Run from the `umbrae` directory:

```bash
conda activate brainx
python inference.py \
  --data_path nsd \
  --fmri_encoder brainx \
  --subj 1 \
  --prompt "Describe this image <image> as simply as possible." \
  --brainx_path train_logs/brainx-v-1-4/last.pth \
  --save_path evaluation/reproduction_smoke \
  --max_samples 1 \
  --max_new_tokens 64
```

## Verified result

Verified on 2026-06-19 with an NVIDIA RTX A6000:

- BrainX checkpoint load: 0 missing keys, 0 unexpected keys.
- Brain encoder output: `[1, 256, 1024]`, all values finite.
- Projected Shikra input: `[1, 256, 4096]`.
- Generated caption: `A group of people cooking in a kitchen`.
- Machine-readable output:
  `umbrae/evaluation/reproduction_smoke/sub01_dim1024/fmricap.json`.
- Python syntax checks passed for `inference.py`, `model.py`, `perceiver.py`,
  and `utils.py`.

This verifies the complete single-sample inference path from an NSD fMRI sample
through BrainX, the multimodal projector, and Shikra-7B. It is a bounded
end-to-end reproduction check, not a rerun of the full 982-sample benchmark.
