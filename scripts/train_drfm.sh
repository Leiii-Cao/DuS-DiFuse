#!/usr/bin/env bash
set -euo pipefail

# ======================== User configuration ========================

# Use Accelerate from the currently active Python environment.
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

# All paths are resolved relative to the DuS-DiFuse repository root.
CONFIG="${CONFIG:-configs/train/drfm.yaml}"
DATA_ROOT="${DATA_ROOT:-data/GDFusion/train}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/drfm}"

# DRFM pseudo-supervision requires trained SDE and GFCM weights.
UNET_RESUME="${UNET_RESUME:-weights/sde.pt}"
GFCM_RESUME="${GFCM_RESUME:-weights/GFCM.pt}"

# Leave empty to initialize DRFM randomly. Set this to a previously saved
# DRFM state dict to initialize from it. Optimizer state and global step are
# not stored in DRFM checkpoints.
DRFM_RESUME="${DRFM_RESUME:-}"

# Optional runtime overrides. Leave empty to use the YAML values.
# The YAML default is a global batch size of 4, which gives 1 sample per
# process when using the default four-GPU setup.
BATCH_SIZE="${BATCH_SIZE:-}"
TRAIN_STEPS="${TRAIN_STEPS:-}"
LOG_EVERY="${LOG_EVERY:-}"
CKPT_EVERY="${CKPT_EVERY:-}"
NUM_WORKERS="${NUM_WORKERS:-}"

# ====================== End user configuration ======================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_drfm.py"

command=(
  env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
  "${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}"
  "${TRAIN_SCRIPT}"
  --config "${CONFIG}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --unet-resume "${UNET_RESUME}"
  --gfcm-resume "${GFCM_RESUME}"
)

if [[ -n "${DRFM_RESUME}" ]]; then
  command+=(--drfm-resume "${DRFM_RESUME}")
fi

if [[ -n "${BATCH_SIZE}" ]]; then
  command+=(--batch-size "${BATCH_SIZE}")
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

if [[ -n "${NUM_WORKERS}" ]]; then
  command+=(--num-workers "${NUM_WORKERS}")
fi

printf 'Project root: %s\n' "${PROJECT_ROOT}"
printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
"${command[@]}"
