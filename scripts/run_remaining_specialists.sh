#!/usr/bin/env bash
# scripts/run_remaining_specialists.sh
#
# Waits for the already-running sorting specialist to finish, then sequentially
# trains remaining specialists and extracts adapters.
#
# Generalist server eval scores (2026-03-23) that drive specialist decisions:
#   clean_desktop  0.40  → DEFINITE specialist  (unchanged from baseline 0.39)
#   hold_pot       0.875 → DEFINITE specialist  (regressed from baseline 1.0)
#   sorting        0.750 → DEFINITE specialist  (already running)
#   stock_shelf    0.775 → CONDITIONAL          (train; only route if specialist > 0.775)
#   place_block    0.708 → CONDITIONAL          (train; only route if specialist > 0.708)
#   pour_workpiece 0.813 → CONDITIONAL          (train; only route if specialist > 0.813)
#
# Usage:
#   export ACOT_CHALLENGE_GENERALIST_WEIGHTS=/abs/path/to/checkpoints/acot_challenge_generalist_continued/<exp>/<step>/params
#   bash scripts/run_remaining_specialists.sh --r2a-cache-root /path/to/r2a_cache
#
# Output adapters (all in adapters/):
#   sorting.npz        DEFINITE  — always route
#   clean_desktop.npz  DEFINITE  — always route
#   hold_pot.npz       DEFINITE  — always route
#   stock_shelf.npz    CONDITIONAL — check val_loss vs generalist before routing
#   place_block.npz    CONDITIONAL — check val_loss vs generalist before routing
#   pour_workpiece.npz CONDITIONAL — check val_loss vs generalist before routing

set -euo pipefail

# Keep an internal master log so the run can be diagnosed even if the caller's
# stdout/stderr pipe is interrupted.
RUN_START_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG_FILE="${RUN_REMAINING_MASTER_LOG:-logs/run_remaining_specialists_${RUN_START_TS}.log}"
mkdir -p "$(dirname "${MASTER_LOG_FILE}")"
touch "${MASTER_LOG_FILE}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    printf '%s\n' "${msg}" >> "${MASTER_LOG_FILE}"
    printf '%s\n' "${msg}" || true
}
die() { log "ERROR: $*"; exit 1; }

# Find the step with the lowest val_loss in a train_metrics.jsonl file.
# Prints just the step number.
best_val_step() {
    local jsonl="$1"
    uv run python - <<PYEOF
import json, pathlib, sys

p = pathlib.Path("${jsonl}")
if not p.exists():
    print("", end="")
    sys.exit(0)

best_step = None
best_loss = float("inf")
for line in p.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "val_loss" in d and d["val_loss"] is not None:
        step = d.get("step", d.get("train/step"))
        if step is None:
            continue
        try:
            step = int(step)
        except Exception:
            continue
        if d["val_loss"] < best_loss:
            best_loss = d["val_loss"]
            best_step = step

if best_step is None:
    print("", end="")
else:
    print(best_step, end="")
PYEOF
}

# Returns "yes" if train_metrics.jsonl contains any entry with
# step >= target_step, otherwise "no".
metrics_reached_step() {
    local metrics_file="$1"
    local target_step="$2"
    uv run python - <<PYEOF 2>/dev/null
import json, pathlib

p = pathlib.Path("${metrics_file}")
target = int("${target_step}")

if not p.exists():
    print("no")
    raise SystemExit(0)

for line in p.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    step = d.get("step", d.get("train/step"))
    if step is None:
        continue
    try:
        step = int(step)
    except Exception:
        continue
    if step >= target:
        print("yes")
        break
else:
    print("no")
PYEOF
}

# Print the highest step seen in metrics file (either "step" or "train/step").
latest_metrics_step() {
    local metrics_file="$1"
    uv run python - <<PYEOF 2>/dev/null
import json, pathlib

p = pathlib.Path("${metrics_file}")
if not p.exists():
    print("", end="")
    raise SystemExit(0)

max_step = None
for line in p.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    step = d.get("step", d.get("train/step"))
    if step is None:
        continue
    try:
        step = int(step)
    except Exception:
        continue
    if max_step is None or step > max_step:
        max_step = step

if max_step is None:
    print("", end="")
else:
    print(max_step, end="")
PYEOF
}

# Print the highest numeric checkpoint directory name under ckpt_base.
latest_checkpoint_step() {
    local ckpt_base="$1"
    local step
    step=$(ls -1 "${ckpt_base}" 2>/dev/null | rg '^[0-9]+$' | sort -n | tail -1 || true)
    printf '%s' "${step}"
}

# Train a specialist config and wait for it to finish, then return the best ckpt path.
# Usage: train_specialist <config_name> <exp_name> <num_train_steps>
# Sets global BEST_CKPT_DIR after completion.
train_specialist() {
    local config_name="$1"
    local exp_name="$2"
    local num_steps="$3"

    local ckpt_base="checkpoints/${config_name}/${exp_name}"
    local metrics_file="${ckpt_base}/train_metrics.jsonl"
    local final_step=$((num_steps - 1))

    log "========================================"
    log "Training: ${config_name}  exp=${exp_name}  steps=${num_steps}"
    log "========================================"

    # Avoid retraining completed stages when resuming the master script.
    local already_done="no"
    if [[ -f "${metrics_file}" ]]; then
        already_done=$(metrics_reached_step "${metrics_file}" "${final_step}")
    fi

    if [[ "${already_done}" == "yes" ]]; then
        log "Found existing completed run at step ${final_step}; skipping retrain."
    else
        local train_mode="overwrite"
        local train_flag="--overwrite"
        local last_ckpt_before
        last_ckpt_before=$(latest_checkpoint_step "${ckpt_base}")
        if [[ -n "${last_ckpt_before}" ]]; then
            train_mode="resume"
            train_flag="--resume"
            log "Found existing checkpoint step ${last_ckpt_before}; using --resume."
        fi

        # Keep training logs independent of caller stdout/stderr so a broken
        # parent pipe does not terminate the subprocess.
        local train_log="logs/${config_name}_${exp_name}_$(date +%Y%m%d_%H%M%S).log"
        nohup bash scripts/train_fast_6gpu.sh "${config_name}" "${exp_name}" "${train_flag}" \
            --r2a-cache-root "${R2A_CACHE_ROOT}" \
            >> "${train_log}" 2>&1 &
        local train_pid=$!
        log "Training process PID: ${train_pid} (mode=${train_mode}, log=${train_log})"

        # Wait for training to finish (process exit).
        local train_rc=0
        if wait "${train_pid}"; then
            train_rc=0
        else
            train_rc=$?
        fi
        log "Training process exited for ${config_name}/${exp_name} (exit_code=${train_rc})."

        local reached
        reached=$(metrics_reached_step "${metrics_file}" "${final_step}")
        if [[ "${reached}" != "yes" ]]; then
            local last_metrics
            local last_ckpt_after
            last_metrics=$(latest_metrics_step "${metrics_file}")
            last_ckpt_after=$(latest_checkpoint_step "${ckpt_base}")
            die "Training stopped before step ${final_step}. last_metrics_step=${last_metrics:-none}, last_checkpoint_step=${last_ckpt_after:-none}, exit_code=${train_rc}, train_log=${train_log}"
        fi
    fi

    # Find best val_loss checkpoint.
    local best_step
    best_step=$(best_val_step "${metrics_file}")
    if [[ -z "${best_step}" ]]; then
        log "WARNING: No val_loss entries found in ${metrics_file}. Using last checkpoint."
        # Fall back to the highest available step directory.
        best_step=$(latest_checkpoint_step "${ckpt_base}")
    fi

    if [[ -z "${best_step}" ]]; then
        die "Could not determine best checkpoint for ${config_name}/${exp_name}"
    fi

    BEST_CKPT_DIR="${ckpt_base}/${best_step}"
    log "Best checkpoint: ${BEST_CKPT_DIR} (val_loss step ${best_step})"
}

# Extract adapter from a checkpoint directory.
# Usage: extract_adapter <ckpt_dir> <adapter_name>
extract_adapter_npz() {
    local ckpt_dir="$1"
    local adapter_name="$2"
    local output="adapters/${adapter_name}.npz"

    log "Extracting adapter: ${ckpt_dir} → ${output}"
    mkdir -p adapters

    uv run python scripts/extract_adapter.py \
        --checkpoint "${ckpt_dir}" \
        --output "${output}" \
        --include-dual-ae

    log "Adapter saved: ${output}"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
R2A_CACHE_ROOT=""
GENERALIST_CKT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --r2a-cache-root)  R2A_CACHE_ROOT="$2";          shift 2 ;;
        --generalist-ckpt) GENERALIST_CKT_OVERRIDE="$2"; shift 2 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ -n "${R2A_CACHE_ROOT}" ]] || die "--r2a-cache-root is required (path to cached dataset)"

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
[[ -n "${ACOT_CHALLENGE_GENERALIST_WEIGHTS:-}" ]] || \
    die "ACOT_CHALLENGE_GENERALIST_WEIGHTS is not set (point to generalist params/ dir)"

log "Generalist weights: ${ACOT_CHALLENGE_GENERALIST_WEIGHTS}"
log "R2A cache root:     ${R2A_CACHE_ROOT}"

# ---------------------------------------------------------------------------
# Step 1: sorting — already done manually. adapter at adapters/sorting.npz.
# ---------------------------------------------------------------------------
log "Step 1/6: sorting skipped (done manually, adapter already extracted)"

# ---------------------------------------------------------------------------
# Step 2: train clean_desktop (7 500 steps — still at baseline level 0.40)
# ---------------------------------------------------------------------------
log "========================================"
log "Step 2/6: clean_desktop specialist"
log "========================================"
BEST_CKPT_DIR=""
train_specialist "acot_specialist_clean_desktop" "specialist_v1" 7500
extract_adapter_npz "${BEST_CKPT_DIR}" "clean_desktop"

# ---------------------------------------------------------------------------
# Step 3: train hold_pot (3 000 steps — regressed 1.0→0.875 in generalist)
# ---------------------------------------------------------------------------
log "========================================"
log "Step 3/6: hold_pot specialist  [recover regression 1.0->0.875]"
log "========================================"
BEST_CKPT_DIR=""
train_specialist "acot_specialist_hold_pot" "specialist_v1" 3000
extract_adapter_npz "${BEST_CKPT_DIR}" "hold_pot"

# ---------------------------------------------------------------------------
# Step 4: train stock_shelf (3 000 steps) — CONDITIONAL routing
# Generalist already at 0.775. Extract adapter, but only add to TASK_ROUTING
# after verifying specialist server score > 0.775.
# ---------------------------------------------------------------------------
log "========================================"
log "Step 4/6: stock_shelf specialist  [CONDITIONAL — generalist=0.775]"
log "========================================"
BEST_CKPT_DIR=""
train_specialist "acot_specialist_stock_shelf" "specialist_v1" 3000
extract_adapter_npz "${BEST_CKPT_DIR}" "stock_shelf"
log "ROUTING DECISION NEEDED: only activate stock_shelf routing if server eval > 0.775"

# ---------------------------------------------------------------------------
# Step 5: train place_block (3 000 steps) — CONDITIONAL routing
# Generalist already at 0.708. Extract adapter, but only add to TASK_ROUTING
# after verifying specialist server score > 0.708.
# ---------------------------------------------------------------------------
log "========================================"
log "Step 5/6: place_block specialist  [CONDITIONAL — generalist=0.708]"
log "========================================"
BEST_CKPT_DIR=""
train_specialist "acot_specialist_place_block" "specialist_v1" 3000
extract_adapter_npz "${BEST_CKPT_DIR}" "place_block"
log "ROUTING DECISION NEEDED: only activate place_block routing if server eval > 0.708"

# ---------------------------------------------------------------------------
# Step 6: train pour_workpiece (7 500 steps) — CONDITIONAL routing
# Generalist already at 0.813. Highest bar — only route if specialist clearly
# beats generalist. Regression risk outweighs small gain.
# ---------------------------------------------------------------------------
log "========================================"
log "Step 6/6: pour_workpiece specialist  [CONDITIONAL — generalist=0.813]"
log "========================================"
BEST_CKPT_DIR=""
train_specialist "acot_specialist_pour_workpiece" "specialist_v1" 7500
extract_adapter_npz "${BEST_CKPT_DIR}" "pour_workpiece"
log "ROUTING DECISION NEEDED: only activate pour_workpiece routing if server eval > 0.813"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "========================================"
log "All specialist training and extraction complete."
log "Adapters in: $(pwd)/adapters/"
ls -lh adapters/*.npz 2>/dev/null || log "(no .npz files found — check for errors above)"
log ""
log "ROUTING DECISIONS before building Docker:"
log "  ALWAYS route (definite wins):"
log "    sorting.npz       — generalist 0.750, specialist expected higher"
log "    clean_desktop.npz — generalist 0.400, specialist expected much higher"
log "    hold_pot.npz      — generalist 0.875, specialist should recover to ~1.0"
log ""
log "  CONDITIONAL (submit specialist solo first, compare to generalist 0.735 baseline):"
log "    stock_shelf.npz    — route only if server eval > 0.775"
log "    place_block.npz    — route only if server eval > 0.708"
log "    pour_workpiece.npz — route only if server eval > 0.813 (highest bar)"
log ""
log "  NEVER route (already at 1.0 with generalist):"
log "    open_door, scoop_popcorn"
log ""
log "  SKIP (specialist won't help — continuous action space):"
log "    sorting_packages_continuous"
log ""
log "Next steps:"
log "  1. Submit routed image with ONLY definite adapters first → get server score"
log "  2. Add conditional adapters one at a time and compare scores"
log "  3. git checkout -b submit/routed-v1"
log "  4. Edit TASK_ROUTING in src/openpi/policies/adapter_routed_policy.py"
log "     to comment out conditional entries until verified"
log "  5. Run local routed serve smoke test (scripts/server_routed.sh)"
log "  6. Build and push Docker image per AGENTS/runbook_submission.md"
log "========================================"
