# Training Runbook

This runbook captures the currently recommended training workflow and the known fast-path variants.

## Scope

- Environment setup
- Dataset extraction
- Norm stats generation
- Debug training
- Full training
- Checkpoint validation
- Fast-path acceleration workflow
- Hardware-specific notes

## Core Paths

- Legacy training path:
  - `scripts/train.py`
  - `scripts/train.sh`
- Additive fast-training path:
  - `scripts/train_fast.py`
  - `scripts/train_fast.sh`
  - `src/openpi/training/data_loader_fast.py`
  - helper cache scripts:
    - `scripts/precompute_subtask_index_cache.py`
    - `scripts/precompute_prompt_cache.py`

## Dataset Notes

- Current full-generalist training data is typically mounted at:
  - `/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim`
- The dataset is already in LeRobot format.
- It is highly fragmented:
  - thousands of per-episode parquet files
  - thousands of per-camera mp4 files
- This matters for throughput:
  - random episode sampling can amplify small-file open cost
  - video decode cost is significant
  - slower SATA / older memory systems are especially vulnerable to data stalls

## Standard Workflow

1. Verify environment variables.

Typical values:

```bash
export ACOT_CHALLENGE_DATA_ROOT=/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim
export UV_CACHE_DIR=/tmp/uv-cache
mkdir -p "${UV_CACHE_DIR}"
```

2. Verify extracted dataset roots exist.

3. Compute norm stats for the chosen config.

Example:

```bash
uv run python scripts/compute_norm_stats.py \
  --config-name acot_challenge_generalist_lora_generalist \
  --split train
```

4. Run a debug job first.

Legacy path example:

```bash
DEBUG_MODE=true uv run python scripts/train.py \
  acot_challenge_generalist_lora_generalist \
  --exp_name generalist_debug \
  --overwrite
```

5. Inspect checkpoint outputs and `train_metrics.jsonl`.

6. Run the full training job.

7. Run `scripts/eval_offline.py` on produced checkpoints if offline ranking is needed.

8. Record outcomes in `AGENTS/experiments.md`.

## Recommended Current Generalist Launch

The repo now includes a follow-up config for the next mainline generalist run:

- `acot_challenge_generalist_lora_generalist_tuned`

This config intentionally keeps the current state/action masking unchanged and only adjusts:

- `warmup_steps=2000`
- `decay_steps=24000`
- `num_train_steps=24000`
- `val_interval=1000`
- `val_num_batches=32`
- `save_interval=1000`
- `batch_size=120`

Suggested launch:

```bash
bash scripts/train.sh \
  acot_challenge_generalist_lora_generalist_tuned \
  generalist_tuned_v1
```

Use this config when the goal is to improve learning-rate timing and checkpoint selection without introducing a new mask ablation into the current mainline.

## Fast Path Workflow

The fast path is additive. It should not replace the legacy path until it is verified on the target hardware.

Current practical status:

- do not treat `scripts/train_fast.py` as the default launcher yet
- there is still no retained real cache-backed training run in the workspace
- `scripts/train_fast.py` still needs a checkpoint-save fix before it is safe to promote as the main entrypoint

### Intended Benefits

- reduce sampler startup cost via precomputed subtask indices
- avoid the heaviest legacy worker/thread explosion
- keep checkpoint format compatible with the rest of the codebase

### Current Fast-Path Commands

Precompute subtask cache:

```bash
uv run python scripts/precompute_subtask_index_cache.py \
  --config-name acot_challenge_generalist_lora_generalist \
  --split train
```

Optional validation cache:

```bash
uv run python scripts/precompute_subtask_index_cache.py \
  --config-name acot_challenge_generalist_lora_generalist \
  --split val
```

Debug fast run:

```bash
DEBUG_MODE=true uv run python scripts/train_fast.py \
  acot_challenge_generalist_lora_generalist \
  --exp_name generalist_fast_debug \
  --overwrite
```

### Generic Reasoning2Action frame-cache workflow

This is the current additive offline-cache direction for the competition training path.

Key property:

- build once for the `Reasoning2Action-Sim` family
- reuse across full generalist, 5-task, clean-desktop, and future specialists
- keep the cache below the config-specific transform boundary

Added entrypoints:

- `scripts/build_reasoning2action_frame_cache.py`
- `scripts/verify_reasoning2action_frame_cache.py`
- `scripts/compute_norm_stats_fast.py`

Build the cache on the preprocessing machine:

```bash
export ACOT_CHALLENGE_DATA_ROOT=/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim
export UV_CACHE_DIR=/tmp/uv-cache
mkdir -p "${UV_CACHE_DIR}"

UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/build_reasoning2action_frame_cache.py \
  --cache-root /path/to/r2a-frame-cache \
  --data-root "${ACOT_CHALLENGE_DATA_ROOT}" \
  --shard-size 2048 \
  --num-workers 16
```

Verify a target config against the built cache:

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/verify_reasoning2action_frame_cache.py \
  --cache-root /path/to/r2a-frame-cache \
  --config-name acot_challenge_generalist_lora_generalist \
  --split train
```

Optional cached norm-stats path:

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/compute_norm_stats_fast.py \
  --config-name acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --split train
```

Run cache-backed fast training:

```bash
DEBUG_MODE=true UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/train_fast.py \
  acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --exp_name generalist_fast_cache_debug \
  --overwrite
```

Compatibility target:

- cache is training-only
- checkpoint layout must remain unchanged
- existing offline eval and final Docker/websocket serving must continue to consume the produced checkpoints without adapters or cache-specific glue

Full fast run:

```bash
bash scripts/train_fast.sh \
  acot_challenge_generalist_lora_generalist \
  generalist_v1_bs96_fast \
  --val-interval=1000 \
  --val-num-batches=8
```

### Important Caveat

- For `acot_challenge_generalist_lora_generalist`, prompt-only token caching is not valid because tokenization depends on state (`discrete_state_input=True`).
- The currently useful cache is the subtask-index cache, not the prompt-token cache.
- The generic frame-cache direction is still the main optimization bet, but it remains a secondary path until a real cache-backed run and checkpoint-save path are validated end-to-end.

## Compatibility Notes

The fast path is intended to preserve:

- checkpoint directory layout under `checkpoints/<config>/<exp_name>/`
- `train_metrics.jsonl`
- compatibility with `scripts/eval_offline.py`
- compatibility with existing checkpoint-based inference / submission loading

This compatibility target is architectural intent and partial implementation status, not a blanket proof that every downstream path has been runtime-verified on every machine.

## Hardware Notes

### Older / I/O-Constrained Machines

- Symptoms seen:
  - GPU utilization and power sawtooth
  - low disk-util percentage despite apparent stalls
  - many training-time waits that are consistent with random file access and HEVC decode
- Main suspicion:
  - data-side video access / decode is a bigger issue than raw model compute

### Newer 2x RTX 5090 Machine

- Single-GPU behavior appears strong:
  - training enters the progress bar quickly
  - GPU can sustain much higher power draw
- Multi-GPU behavior is still under investigation:
  - both `scripts/train.py` and `scripts/train_fast.py` have shown very slow startup / first-step behavior on dual GPU
  - this currently looks more like JAX/XLA multi-GPU initialization / compilation cost than a fast-loader-only problem

## Batch Size Heuristic

For the 32G RTX 5090 machine, user-observed non-debug memory usage for `acot_challenge_generalist_lora_generalist` was roughly `25G` per GPU.

Practical heuristic:

- conservative single-GPU batch target: `96`
- likely larger safe single-GPU trial: `112`
- aggressive single-GPU trial: `120`

Treat these as starting points only; re-check peak memory with:

- first real training step
- validation
- checkpoint save / restore
- any larger host-to-device staging effects
