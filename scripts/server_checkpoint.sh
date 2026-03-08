cart_num=${1}
port=${2}

export TF_NUM_INTRAOP_THREADS=16
export CUDA_VISIBLE_DEVICES=${cart_num}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"

export PYTHONPATH=/root/openpi/src:${PYTHONPATH:-/app:/app/src}
GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py \
    --port ${port} \
    --policy checkpoint \
    --policy.config "${ACOT_SERVE_CONFIG:-acot_challenge_generalist_lora_clean_desktop}" \
    --policy.dir "${ACOT_SERVE_CHECKPOINT:?Set ACOT_SERVE_CHECKPOINT to the trained checkpoint directory}"
