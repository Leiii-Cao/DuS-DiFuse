#!/usr/bin/env bash
set -euo pipefail

# ======================== User configuration ========================

# Use Accelerate from the currently active Python environment.
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

# All paths are resolved relative to the DuS-DiFuse repository root.
CONFIG="${CONFIG:-configs/train/generative_modulation.yaml}"
DATA_ROOT="${DATA_ROOT:-data/generation/train/images}"
LABEL_ROOT="${LABEL_ROOT:-data/generation/train/labels}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/generation}"

# Leave empty to initialize ControlNet from the frozen SD UNet. Set this to a
# previously saved ControlNet state dict to initialize from existing weights.
# Optimizer state and global step are not stored in these checkpoints.
CONTROLNET_RESUME="${CONTROLNET_RESUME:-}"

# Optional runtime overrides. Leave empty to use the YAML values.
TRAIN_STEPS="${TRAIN_STEPS:-}"
LOG_EVERY="${LOG_EVERY:-}"
CKPT_EVERY="${CKPT_EVERY:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"

# ====================== End user configuration ======================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_generative_modulation.py"

command=(
  env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
  "${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}"
  "${TRAIN_SCRIPT}"
  --config "${CONFIG}"
  --data-root "${DATA_ROOT}"
  --label-root "${LABEL_ROOT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${CONTROLNET_RESUME}" ]]; then
  command+=(--controlnet-resume "${CONTROLNET_RESUME}")
fi

if [[ -n "${TRAIN_STEPS}" ]]; then
  command+=(--train-steps "${TRAIN_STEPS}")
fi

if [[ -n "${LOG_EVERY}" ]]; then
  command+=(--log-every "${LOG_EVERY}")
fi

if [[ -n "${CKPT_EVERY}" ]]; then
  command+=(--ckpt-every "${CKPT_EVERY}")
fi

if [[ -n "${BATCH_SIZE}" ]]; then
  command+=(--batch-size "${BATCH_SIZE}")
fi

if [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  command+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")
fi

printf 'Project root: %s\n' "${PROJECT_ROOT}"
printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
"${command[@]}"
