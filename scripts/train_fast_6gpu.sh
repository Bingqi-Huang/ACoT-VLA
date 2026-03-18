#!/usr/bin/env bash

export DEBUG_MODE=false
export WANDB_MODE=online
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

export ACOT_CHALLENGE_INIT_WEIGHTS=/home/bingqi/data/bingqi/Project/ACoT-VLA/checkpoints/baseline/30000/params
export R2A_CACHE=/home/bingqi/data/bingqi/ACoT-Data

CONFIG_NAME=${1}
EXP_NAME=${2}
shift 2

TRAIN_ARGS=()
RESUME=false
OVERWRITE=false
for arg in "$@"; do
    case "${arg}" in
        --resume|--resume=true)
            RESUME=true
            ;;
        --overwrite|--overwrite=true|--overwirte)
            OVERWRITE=true
            ;;
        *)
            TRAIN_ARGS+=("${arg}")
            ;;
    esac
done

if [[ "${RESUME}" == "true" && "${OVERWRITE}" == "true" ]]; then
    echo "Error: --resume and --overwrite cannot be used together." >&2
    exit 1
fi

if [[ "${RESUME}" != "true" && "${OVERWRITE}" != "true" ]]; then
    OVERWRITE=true
fi

[[ "${RESUME}" == "true" ]] && TRAIN_ARGS+=("--resume")
[[ "${OVERWRITE}" == "true" ]] && TRAIN_ARGS+=("--overwrite")

LOG_DIR=./logs
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${EXP_NAME}_fast_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

env | sort | tee "${LOG_FILE}"
uv run python scripts/train_fast.py "${CONFIG_NAME}" --exp-name="${EXP_NAME}" --r2a-cache-root="${R2A_CACHE}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
