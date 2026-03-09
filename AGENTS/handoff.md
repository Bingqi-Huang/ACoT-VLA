# Handoff

Use this file for short session-to-session transfer notes.

## Template

Date:

Author:

What changed:

What was verified:

What is still broken or unknown:

Immediate next step:

## Current Note

Date: 2026-03-09

Author: Codex

What changed:

- Added `AGENTS/` shared workspace docs.
- Established the main submission constraint: final evaluation uses a Dockerized websocket server on port `8999`.
- Fixed two blocking LoRA issues for the planned challenge setup:
  - `gemma_300m_lora` is now part of the Gemma variant type alias.
  - `ACOTConfig.get_freeze_filter()` now keeps coarse-expert LoRA trainable.
- Added `grad_accum_steps` to `TrainConfig` and refactored `scripts/train.py` into compute/apply-grad phases so gradient accumulation works for both standard and ACoT training.
- Added targeted tests in `src/openpi/models/acot_vla_test.py`.
- Added `acot_challenge_generalist_lora_all` and generated `acot_specialist_*` configs in `src/openpi/training/config.py`.
- Added adapter extraction and routed serving:
  - `scripts/extract_adapter.py`
  - `src/openpi/policies/adapter_routed_policy.py`
  - `scripts/server_routed.sh`
  - `AdapterRouted` mode in `scripts/serve_policy.py`
- Adjusted the serving Dockerfile so `SERVER_SCRIPT` can override the launch script while preserving the current default.
- Fixed the checkpoint norm-stats path mismatch:
  - checkpoint saving now writes stats under `assets/<asset_id>/` when applicable
  - checkpoint loading falls back to root-level `assets/` stats for backward compatibility
- Refactored routed adapter serving to avoid rebuilding the model on adapter switches:
  - added `module_jit_with_state()` in `src/openpi/shared/nnx_utils.py`
  - `AdapterRoutedPolicy` now builds one base model, caches per-adapter `nnx.State` overlays, and reuses one stateful JIT sampler
- Changed the serving Docker default to `scripts/server_routed.sh` while keeping `SERVER_SCRIPT` as an override.
- Added `scripts/server_checkpoint.sh` for single-checkpoint test submissions such as `acot_challenge_generalist_lora_clean_desktop`.
- Rewrote `Training_Notes.md` around the clean-desktop single-model test-server workflow instead of the full adapter-routing workflow.
- Patched Python packaging to avoid the current `av==14.4.0` source-build failure on Ubuntu 22.04:
  - root `pyproject.toml` now overrides `av` to `14.0.1`
  - `packages/openpi-client/pyproject.toml` now uses `dependency-groups.dev` instead of deprecated `tool.uv.dev-dependencies`
- Added strict episode-level train/validation split support for LeRobot finetuning datasets:
  - `EpisodeSplitConfig` added to `DataConfig`
  - new split manifest utility in `src/openpi/training/episode_split.py`
  - split-aware dataset construction in `src/openpi/training/data_loader.py`
  - `src/openpi/training/sampler.py` now respects filtered episode IDs instead of assuming contiguous episode indices
- Added split-aware validation to `scripts/train.py` with `val_interval` and `val_num_batches`.
- Updated `scripts/compute_norm_stats.py` to support `--split`, defaulting the intended workflow to train-only stats.
- Added `scripts/generate_episode_split.py` and `docs/reasoning2action_episode_split.md`.
- Added focused tests in `src/openpi/training/episode_split_test.py`.
- Added offline checkpoint evaluation:
  - `scripts/eval_offline.py`
  - `src/openpi/training/offline_eval.py`
  - `src/openpi/training/offline_eval_test.py`
  - `docs/reasoning2action_offline_eval.md`
- Added deterministic teacher-forced offline hooks to `src/openpi/models/pi0.py` and `src/openpi/models/acot_vla.py` so validation can report per-example loss and action-reconstruction metrics without rollout.
- Improved train-time logging in `scripts/train.py`:
  - logs `train/loss`, `train/grad_norm`, `train/param_norm`, `train/learning_rate`, `train/steps_per_sec`, `train/samples_per_sec`, `train/wall_time_sec`
  - logs per-task train loss when raw batch `task` metadata is available
  - logs batch-level action-dimension MAE, joint MAE, and gripper metrics for pi0/pi0.5 model families
  - writes the same flat metrics stream to `<checkpoint_dir>/train_metrics.jsonl`
  - explicitly registers the `train/*`, `val/*`, and `checkpoint/*` metric families with W&B so the new series appear with consistent step axes
- Extended `src/openpi/training/data_loader.py` so the training loop can recover raw batch metadata (`task`, `episode_index`, `frame_index`) for logging without changing the model input path.
- Fixed offline eval for ACOT data configs in `src/openpi/training/offline_eval.py` by preserving dynamic attrs like `joint_action_shifts` when swapping in checkpoint norm stats; this addresses the clean-desktop eval crash where `create_torch_dataset()` received a plain `DataConfig`.
- Fixed a second offline eval issue in `scripts/eval_offline.py` by treating `train=False` as a static kwarg when JIT-compiling `compute_loss_per_example`; this addresses the `TracerBoolConversionError` triggered by `if train:` inside `preprocess_observation()`.
- Hardened `scripts/compute_norm_stats.py` against bad samples by wrapping the transformed dataset in `SafeDataset`, respecting the computed shuffle mode, and raising a clearer error only if no usable statistics were accumulated.
- Updated `src/openpi/training/data_loader.py` so fully empty collated batches are skipped instead of propagating `None` into JAX array conversion.
- Further updated `src/openpi/training/data_loader.py` so partially filtered batches are skipped too; this fixes training-time sharding crashes where `SafeDataset` dropped some bad samples and the surviving batch size no longer matched the configured per-process batch size.
- Added a tqdm progress bar to `scripts/eval_offline.py` so long offline-eval runs show per-checkpoint batch progress.
- Fixed the underlying split-vs-LeRobot incompatibility in `src/openpi/training/data_loader.py`: for selected episode subsets, openpi now patches each `LeRobotDataset` instance so `_get_query_indices()` maps the original global `episode_index` to the dataset-local episode position expected by `episode_data_index`. This preserves original episode ids for prompt logic but stops large numbers of legitimate split samples from being misread as out-of-bounds.

What was verified:

- Official docs confirm submission requirements and model-type declaration (`abs_joint` or `abs_pose`).
- Repo inference path loads checkpoint directories with `params` and `assets`.
- `python3 -m py_compile` passes on the modified files.
- `git diff --check` passes.
- `python3 -m py_compile` passes on the new episode-split and validation files.
- `python3 -m py_compile` passes on the offline-eval script/helper/model-hook changes.

What is still broken or unknown:

- No final submission image path has been implemented yet for a custom checkpoint.
- No verified local closed-loop evaluation run has been recorded yet in this folder.
- `uv run pytest` is currently blocked in this environment because dependency resolution tries to reach external package mirrors and fails.
- The new gradient accumulation path has not yet been exercised on actual hardware.
- No real adapter `.npz` files have been extracted and smoke-tested yet.
- The refactored routed adapter policy still needs runtime validation with real adapter files to confirm switch latency and no unexpected recompiles.
- `pytest` is not installed in the checked environment, so the new tests were added but not run through pytest here.
- Direct runtime smoke checks that import JAX are blocked because the environment currently has `ml_dtypes==0.4.1`, while JAX now requires `>=0.5`.
- The new val-loss path still needs a real clean-desktop training run to confirm the split manifest, train-only stats, and val-only loader behave correctly on actual data.
- The new offline evaluator still needs one real experiment directory run to confirm batch loading, per-task task-name recovery, and JSON/CSV outputs on actual Reasoning2Action checkpoints.
- The new train logging still needs one real clean-desktop run to confirm task metadata survives as expected and that the additional batch-metric JIT does not add unacceptable overhead.
- The fixed offline eval path still needs one real rerun on the previously failing clean-desktop checkpoint to confirm metric generation completes end-to-end.

Immediate next step:

- Run `scripts/generate_episode_split.py` for `acot_challenge_generalist_lora_clean_desktop`, recompute norm stats with `--split train`, then launch a short debug training run and confirm validation loss logs on the val split.
- After the debug run produces checkpoints, execute `scripts/eval_offline.py` on the experiment directory and inspect `eval/summary.csv` for checkpoint ranking.
- Inspect `<checkpoint_dir>/train_metrics.jsonl` from the same debug run and verify that the expected `train/task/*` and `train/action_mae/*` keys appear.

Git note:

- Commit substantial progress before handoff so the next agent can diff exact changes instead of reconstructing them from notes.
