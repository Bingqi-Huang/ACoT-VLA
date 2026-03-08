#!/usr/bin/env bash
set -euo pipefail

export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export PYTHONPATH="/root/openpi/src:${PYTHONPATH:-/app:/app/src}"

GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py \
  --env G2SIM \
  --port 8999 \
  --log-request-summaries \
  --log-request-limit "${SMOKE_LOG_REQUEST_LIMIT:-5}" \
  --prompt-preview-chars "${SMOKE_PROMPT_PREVIEW_CHARS:-120}"
