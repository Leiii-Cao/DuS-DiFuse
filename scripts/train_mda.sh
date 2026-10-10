#!/usr/bin/env bash
set -euo pipefail

# ======================== User configuration ========================

# The script uses the Python/Accelerate commands from the currently active
# environment. Override ACCELERATE_BIN only when the executable has another name.
ACCELERATE_BIN="accelerate"
CUDA_DEVICES="0"
NUM_PROCESSES=1

# All paths are resolved relative to the DuS-DiFuse repository root.
CONFIG="configs/train/mda.yaml"
DATA_ROOT="data/GDFusion/train"
OUTPUT_DIR="experiments/mda"

# Generic ViT-H-14 OpenCLIP weights used to initialize a new MDA run.
INIT_PATH="weights/open_clip_pytorch_model.bin"

# Leave empty for a new run. Set this to best.pth or latest.pth to restore the
# model, optimizer, completed epoch, and best loss. RESUME overrides INIT_PATH.
RESUME=""

# ====================== End user configuration ======================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_mda.py"

command=(
  env "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
  "${ACCELERATE_BIN}" launch --num_processes "${NUM_PROCESSES}"
  "${TRAIN_SCRIPT}"
  --config "${CONFIG}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${INIT_PATH}" ]]; then
  command+=(--init-path "${INIT_PATH}")
fi

if [[ -n "${RESUME}" ]]; then
  command+=(--resume "${RESUME}")
fi

printf 'Project root: %s\n' "${PROJECT_ROOT}"
printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'

cd "${PROJECT_ROOT}"
"${command[@]}"
