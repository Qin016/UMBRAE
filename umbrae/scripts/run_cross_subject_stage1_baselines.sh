#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/brainx/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/stage1_outputs/cross_subject_projector}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

run_one() {
  local subject="$1"
  local method="$2"
  local subject_number="${subject#subj}"
  local output_dir="${OUTPUT_ROOT}/${subject}/${method}"
  local log_path="${OUTPUT_ROOT}/${subject}/${method}.log"

  if [[ -f "${output_dir}/checkpoint_best.pt" \
        && -f "${output_dir}/routing_dynamics.jsonl" ]]; then
    echo "SKIP complete: ${subject} ${method}"
    return
  fi

  mkdir -p "${OUTPUT_ROOT}/${subject}"
  local router_args=()
  case "${method}" in
    soft)
      router_args=(--router-type soft)
      ;;
    uniform)
      router_args=(--router-type uniform)
      ;;
    single_L24)
      router_args=(--router-type single --single-router-layer 24)
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 2
      ;;
  esac

  echo "START ${subject} ${method} -> ${output_dir}"
  "${PYTHON_BIN}" scripts/train_stage1_routing.py \
    --subject "${subject}" \
    --train-tar "nsd/webdataset_avg_split/train/train_subj${subject_number}_0.tar" \
    --val-tar "nsd/webdataset_avg_split/val/val_subj${subject_number}_0.tar" \
    --roi-mapping-json "/opt/data/private/BA/UMBRAE/roi_indices/${subject}_neuroroute_v1.json" \
    --selected-clip-layers 4 8 12 16 20 24 \
    --roi-token-dim 1024 \
    --clip-layer-target-dim 1024 \
    --use-brain-clip-projector \
    --projector-type mlp \
    --projector-hidden-dim 1024 \
    --projector-dropout 0.1 \
    --detach-routed-targets \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay 0.01 \
    --router-entropy-weight 0.0 \
    --router-balance-weight 0.0 \
    --router-smoothness-weight 0.0 \
    --debug-max-steps 0 \
    --seed 42 \
    --output-dir "${output_dir}" \
    "${router_args[@]}" \
    > >(tee "${log_path}")
  echo "DONE ${subject} ${method}"
}

for subject in subj01 subj02 subj05 subj07; do
  for method in soft uniform single_L24; do
    run_one "${subject}" "${method}"
  done
done

echo "ALL CROSS-SUBJECT STAGE-1 BASELINES COMPLETE"
