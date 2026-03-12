#!/usr/bin/env bash
set -euo pipefail

cart_num=${1:-0}
port=${2:-8999}

export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-16}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${cart_num}}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

export PYTHONPATH=/root/openpi/src:${PYTHONPATH:-/app:/app/src}
export ACOT_SERVE_CONFIG=${ACOT_SERVE_CONFIG:-acot_challenge_generalist_lora_generalist}
export ACOT_SERVE_CHECKPOINT=${ACOT_SERVE_CHECKPOINT:-/app/checkpoint/generalists-v1-10000}

GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py \
  --port "${port}" \
  policy:checkpoint \
  --policy.config "${ACOT_SERVE_CONFIG}" \
  --policy.dir "${ACOT_SERVE_CHECKPOINT}"
