#!/usr/bin/env bash
set -euo pipefail

# Run from the project directory even when this script is launched elsewhere.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------
CONFIG="configs/inference.yaml"              # Model/checkpoint configuration.
INPUT_ROOT="data/example"                    # Used only when explicit folders are empty.
VISIBLE_DIR="" # Degraded visible images.
INFRARED_DIR="" # Degraded infrared images.
OUTPUT_DIR="results"                 # Fused images and optional masks.

# -----------------------------------------------------------------------------
# Runtime and diffusion
# -----------------------------------------------------------------------------
PYTHON_BIN="python"                          # Python from the active environment.
DEVICE="cuda:0"                              # Examples: cuda:0, cuda:1, cpu.
STEPS=25                                     # Reverse diffusion steps; must be >= 2.

# -----------------------------------------------------------------------------
# Generative Modulation
# -----------------------------------------------------------------------------
GENERATIVE_MODULATION=false                  # true: enable Generative Modulation.
TEXT_PROMPT=""                               # Example: "pedestrian, car"; empty means global.

# -----------------------------------------------------------------------------
# Tiled inference for large images / limited GPU memory
# -----------------------------------------------------------------------------
TILED=false                                  # true: enable tiled inference.
TILE_SIZE=512                                # Must be positive and divisible by 64.
TILE_STRIDE=384                              # Must be positive and divisible by 64.
DRY_RUN="${DRY_RUN:-false}"                  # true: print command without running it.

# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------
cmd=(
    "$PYTHON_BIN" inference.py
    --config "$CONFIG"
    --output "$OUTPUT_DIR"
    --device "$DEVICE"
    --steps "$STEPS"
    --text-prompt "$TEXT_PROMPT"
    --tile-size "$TILE_SIZE"
    --tile-stride "$TILE_STRIDE"
)

if [[ -n "$VISIBLE_DIR" || -n "$INFRARED_DIR" ]]; then
    if [[ -z "$VISIBLE_DIR" || -z "$INFRARED_DIR" ]]; then
        echo "VISIBLE_DIR and INFRARED_DIR must be set together." >&2
        exit 2
    fi
    cmd+=(--visible-dir "$VISIBLE_DIR")
    cmd+=(--infrared-dir "$INFRARED_DIR")
else
    cmd+=(--input-root "$INPUT_ROOT")
fi

if [[ "$GENERATIVE_MODULATION" == true ]]; then cmd+=(--generation); else cmd+=(--no-generation); fi
if [[ "$TILED" == true ]]; then cmd+=(--tiled); fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == true ]]; then exit 0; fi
"${cmd[@]}"
