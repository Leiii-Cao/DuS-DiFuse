#!/usr/bin/env bash
set -euo pipefail

# ======================== User configuration ========================

# Use Accelerate from the currently active Python environment.
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

# All paths are resolved relative to the DuS-DiFuse repository root.
CONFIG="${CONFIG:-configs/train/degradation_decoupling.yaml}"
DATA_ROOT="${DATA_ROOT:-data/GDFusion/train}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/sde}"

# Leave empty to initialize a new degradation-decoupling UNet. Set this to a
# previously saved UNet state dict to initialize from it. This restores model
# weights only; optimizer state and global step are not stored in SDE checkpoints.
UNET_RESUME="${UNET_RESUME:-}"

# Optional runtime overrides. Leave empty to use the YAML values.
TRAIN_STEPS="${TRAIN_STEPS:-}"
LOG_EVERY="${LOG_EVERY:-}"
CKPT_EVERY="${CKPT_EVERY:-}"

# ====================== End user configuration ======================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_degradation_decoupling.py"

command=(
  env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
  "${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}"
  "${TRAIN_SCRIPT}"
  --config "${CONFIG}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${UNET_RESUME}" ]]; then
  command+=(--unet-resume "${UNET_RESUME}")
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

printf 'Project root: %s\n' "${PROJECT_ROOT}"
printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
"${command[@]}"
