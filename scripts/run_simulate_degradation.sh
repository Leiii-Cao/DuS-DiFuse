#!/usr/bin/env bash
set -euo pipefail

# ======================== User configuration ========================

# Python executable from the currently active environment.
PYTHON_BIN="python"

# The two input folders must contain filename-aligned image pairs.
VISIBLE_DIR="data/GDFusion/test/MSRS/vis"
INFRARED_DIR="data/GDFusion/test/MSRS/ir"

# Output will be written under OUTPUT_ROOT using the two names below.
OUTPUT_ROOT="data/GDFusion/test/MSRS"
VISIBLE_OUTPUT_NAME="vis_lq"
INFRARED_OUTPUT_NAME="ir_lq"

# One degradation type is sampled for each image from the corresponding list.
# A one-item list, such as (rain), applies that type to every image.
# Visible choices: none noise blur rain snow haze
# Infrared choices: none noise lowcontrast stripe
VISIBLE_TYPES=(noise blur rain snow haze)
INFRARED_TYPES=(noise lowcontrast stripe)

# Haze requires a filename-aligned depth folder. Leave empty when haze is unused.
DEPTH_DIR="data/GDFusion/test/MSRS/depth"

# The same seed and filenames reproduce the same degradation results.
SEED=231

# ====================== End user configuration ======================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/simulate_degradation.py"

command=(
  "${PYTHON_BIN}" "${PYTHON_SCRIPT}"
  --visible-dir "${VISIBLE_DIR}"
  --infrared-dir "${INFRARED_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --visible-output-name "${VISIBLE_OUTPUT_NAME}"
  --infrared-output-name "${INFRARED_OUTPUT_NAME}"
  --visible-types "${VISIBLE_TYPES[@]}"
  --infrared-types "${INFRARED_TYPES[@]}"
  --seed "${SEED}"
  --overwrite
)

if [[ -n "${DEPTH_DIR}" ]]; then
  command+=(--depth-dir "${DEPTH_DIR}")
fi

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
