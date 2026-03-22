#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:-0}
port=${2:-8999}

export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-16}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${gpu_id}}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/root/.cache/openpi}
export PYTHONPATH="/submission:/submission/src:/submission/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export ACOT_ROUTED_CONFIG=${ACOT_ROUTED_CONFIG:-acot_challenge_lora_conservative}
export ACOT_ROUTED_BASE_CHECKPOINT=${ACOT_ROUTED_BASE_CHECKPOINT:-/submission/checkpoint/baseline/30000}
export ACOT_ROUTED_ADAPTER_DIR=${ACOT_ROUTED_ADAPTER_DIR:-/submission/adapters}
export ACOT_ROUTED_SPECIALIST_NORM_STATS_PATH=${ACOT_ROUTED_SPECIALIST_NORM_STATS_PATH:-/submission/assets/reasoning2action_sim_generalist/norm_stats.json}

extra_args=()
if [[ -n "${ACOT_ROUTED_SPECIALIST_NORM_STATS_PATH}" ]]; then
  extra_args+=(--policy.specialist-norm-stats-path "${ACOT_ROUTED_SPECIALIST_NORM_STATS_PATH}")
fi

cd /submission
exec uv run --no-sync python scripts/serve_policy.py \
  --port "${port}" \
  policy:adapter-routed \
  --policy.config "${ACOT_ROUTED_CONFIG}" \
  --policy.base-checkpoint "${ACOT_ROUTED_BASE_CHECKPOINT}" \
  --policy.adapter-dir "${ACOT_ROUTED_ADAPTER_DIR}" \
  "${extra_args[@]}"
