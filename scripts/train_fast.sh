#!/usr/bin/env bash

export DEBUG_MODE=false
export WANDB_MODE=online
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_VISIBLE_DEVICES=0,1,2

CONFIG_NAME=${1}
EXP_NAME=${2}
shift 2

TRAIN_ARGS=()
RESUME=false
for arg in "$@"; do
    if [[ "${arg}" == "--resume=true" ]]; then
        RESUME=true
    elif [[ "${arg}" == "--overwrite" ]]; then
        continue
    else
        TRAIN_ARGS+=("${arg}")
    fi
done

if [[ "$RESUME" != "true" ]]; then
    TRAIN_ARGS+=("--overwrite")
fi

LOG_DIR=./logs
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${EXP_NAME}_fast_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

env | sort | tee "${LOG_FILE}"
uv run python scripts/train_fast.py "${CONFIG_NAME}" --exp-name="${EXP_NAME}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
