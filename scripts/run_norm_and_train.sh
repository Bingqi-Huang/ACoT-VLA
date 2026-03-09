#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <config_name> <exp_name> [additional train.sh args...]"
    exit 1
fi

CONFIG_NAME=$1
EXP_NAME=$2
shift 2

echo "[run_norm_and_train] Computing norm stats for ${CONFIG_NAME}..."
uv run python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"

echo "[run_norm_and_train] Norm stats finished. Starting training for ${CONFIG_NAME}/${EXP_NAME}..."
bash scripts/train.sh "${CONFIG_NAME}" "${EXP_NAME}" --overwrite "$@"
