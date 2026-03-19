#!/usr/bin/env bash
set -euo pipefail

export DEBUG_MODE=false
export WANDB_MODE=online
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

export ACOT_CHALLENGE_INIT_WEIGHTS=/home/bingqi/data/bingqi/Project/ACoT-VLA/checkpoints/baseline/30000/params
R2A_CACHE_ROOT=/home/bingqi/data/bingqi/ACoT-Data

LOG_DIR=./logs/specialists
mkdir -p "${LOG_DIR}"

echo "===== Starting 3 specialist trainings in parallel ====="
echo "Baseline checkpoint: ${ACOT_CHALLENGE_INIT_WEIGHTS}"
echo "Cache root:          ${R2A_CACHE_ROOT}"
echo "Logs:                ${LOG_DIR}"
echo ""

# GPU 0-1: clean_desktop (8000 steps, ~2.5h)
CUDA_VISIBLE_DEVICES=0,1 \
  uv run python scripts/train_fast.py acot_specialist_clean_desktop \
  --exp-name=spec_clean_v1 \
  --r2a-cache-root "${R2A_CACHE_ROOT}" \
  2>&1 | tee "${LOG_DIR}/clean_desktop.log" &
PID_CLEAN=$!
echo "[GPU 0-1] clean_desktop    PID=${PID_CLEAN}"

# GPU 2-3: stock_shelf (3000 steps, ~1h)
CUDA_VISIBLE_DEVICES=2,3 \
  uv run python scripts/train_fast.py acot_specialist_stock_shelf \
  --exp-name=spec_stock_v1 \
  --r2a-cache-root "${R2A_CACHE_ROOT}" \
  2>&1 | tee "${LOG_DIR}/stock_shelf.log" &
PID_STOCK=$!
echo "[GPU 2-3] stock_shelf      PID=${PID_STOCK}"

# GPU 4-5: sorting_packages (5000 steps, ~1.5h)
CUDA_VISIBLE_DEVICES=4,5 \
  uv run python scripts/train_fast.py acot_specialist_sorting \
  --exp-name=spec_sorting_v1 \
  --r2a-cache-root "${R2A_CACHE_ROOT}" \
  2>&1 | tee "${LOG_DIR}/sorting_packages.log" &
PID_SORT=$!
echo "[GPU 4-5] sorting_packages PID=${PID_SORT}"

echo ""
echo "All 3 started. Waiting for completion..."
echo "Monitor logs: tail -f ${LOG_DIR}/<task>.log"
echo ""

# Wait for each and report exit status
FAILED=0

wait ${PID_CLEAN} && echo "[DONE] clean_desktop" || { echo "[FAILED] clean_desktop"; FAILED=1; }
wait ${PID_STOCK}  && echo "[DONE] stock_shelf"   || { echo "[FAILED] stock_shelf";   FAILED=1; }
wait ${PID_SORT}   && echo "[DONE] sorting_packages" || { echo "[FAILED] sorting_packages"; FAILED=1; }

echo ""
if [[ ${FAILED} -eq 0 ]]; then
    echo "===== All specialists finished successfully ====="
    echo "Next: pick best checkpoints and run scripts/extract_and_route.sh"
else
    echo "===== Some specialists FAILED — check logs in ${LOG_DIR}/ ====="
    exit 1
fi
