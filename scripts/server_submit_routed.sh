#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:-0}
port=${2:-8999}

export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-16}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${gpu_id}}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/root/.cache/openpi}
export PYTHONPATH="/submission:/submission/src:/submission/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
# Config must match the architecture of the base checkpoint (generalist_continued uses
# gemma_300m dual AEs + gemma_2b_lora backbone — same as the official baseline structure).
export ACOT_ROUTED_CONFIG=${ACOT_ROUTED_CONFIG:-acot_challenge_generalist_continued}
export ACOT_ROUTED_BASE_CHECKPOINT=${ACOT_ROUTED_BASE_CHECKPOINT:-/submission/checkpoint/generalist_continued}
export ACOT_ROUTED_ADAPTER_DIR=${ACOT_ROUTED_ADAPTER_DIR:-/submission/adapters}
# All tasks share norm stats embedded inside the generalist checkpoint dir
# (checkpoint/generalist_continued/assets/reasoning2action_sim_generalist/).
# No specialist norm stats override needed.

extra_args=()

cd /submission
exec uv run --no-sync python scripts/serve_policy.py \
  --port "${port}" \
  policy:adapter-routed \
  --policy.config "${ACOT_ROUTED_CONFIG}" \
  --policy.base-checkpoint "${ACOT_ROUTED_BASE_CHECKPOINT}" \
  --policy.adapter-dir "${ACOT_ROUTED_ADAPTER_DIR}" \
  "${extra_args[@]}"
