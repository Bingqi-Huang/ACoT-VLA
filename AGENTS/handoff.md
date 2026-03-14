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

Date: 2026-03-14

Author: GitHub Copilot

What changed:

- Corrected routed-serving strategy to match challenge goal: base model stays on `checkpoint/baseline/30000`, and only routed tasks apply specialist adapters.
- `src/openpi/policies/adapter_routed_policy.py` now supports loading baseline checkpoints into LoRA config by zero-initializing missing LoRA tensors (instead of requiring a LoRA checkpoint as base).
- Kept task routing narrow: `clean_the_desktop -> clean_the_desktop_1500`; all other tasks fall back to `_default` / base behavior.
- Restored routed defaults in scripts/docker to baseline checkpoint:
  - `scripts/server_submit_routed.sh`
  - `scripts/server_routed.sh`
  - `scripts/docker/serve_policy.Dockerfile`
- Reverted build-context policy accordingly: `.dockerignore` excludes `checkpoint/specialists_clean_1500` again.
- Updated `AGENTS/docker_build.md` to baseline-as-base routed runbook.

What was verified:

- `docker build -f scripts/docker/serve_policy.Dockerfile -t routed-clean-desktop-1500:latest .` succeeds.
- Runtime smoke test loads checkpoint from `/submission/checkpoint/baseline/30000` and reaches websocket serving startup (`server listening on 0.0.0.0:8999`).

What is still broken or unknown:

- Smoke test ended with `address already in use` on port `8999`, indicating another process already bound the port during that run.

Immediate next step:

- Ensure port `8999` is free, rerun container, then send one routed request with `task_name=clean_the_desktop` and one non-routed task to validate adapter switching behavior end-to-end.

Date: 2026-03-14

Author: GitHub Copilot

What changed:

- Routed adapter mapping in `src/openpi/policies/adapter_routed_policy.py` now points `clean_the_desktop` to `clean_the_desktop_1500`.
- Removed routed mappings for specialist adapters not currently packaged, so non-matched tasks cleanly use `_default`/`_base`.
- `scripts/docker/serve_policy.Dockerfile` is now submission-ready and auto-serves routed policy from `/submission` on port `8999`.
- `AGENTS/docker_build.md` now documents the routed clean-desktop-1500 build and smoke-test commands.

What was verified:

- Workspace contains required artifacts:
  - `checkpoint/baseline/30000`
  - `adapters/_default.npz`
  - `adapters/clean_the_desktop_1500.npz`

What is still broken or unknown:

- Full end-to-end container runtime smoke test (actual server boot + websocket inference) has not been executed in this session.

Immediate next step:

- Build and run `scripts/docker/serve_policy.Dockerfile`, then verify service is listening on `8999` and test one routed `clean_the_desktop` request.

Date: 2026-03-12

Author: Codex

What changed:

- Fixed the blocking fast-path checkpoint save bug in `scripts/train_fast.py`.
- Fixed a fast/legacy behavior drift in `src/openpi/training/data_loader_fast.py` so `num_batches` loaders now keep iterating across dataset exhaustion like the legacy loader.
- Added basic stale-cache protection for cached subtask indices by rejecting cache metadata whose recorded dataset size no longer matches the active dataset.
- Added targeted fast-path regression coverage in `src/openpi/training/data_loader_fast_test.py`.

What was verified:

- `python3 -m py_compile` passes for:
  - `scripts/train_fast.py`
  - `src/openpi/training/data_loader_fast.py`
  - `src/openpi/training/data_loader_fast_test.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m pytest -q` passes for:
  - `src/openpi/training/data_loader_fast_test.py`
  - `scripts/verify_reasoning2action_frame_cache_test.py`
  - `src/openpi/training/r2a_frame_cache_test.py`

What is still broken or unknown:

- There is still no retained real cache-backed training run in the workspace, so fast-path wall-clock gain and long-run checkpoint compatibility remain unverified on target hardware.
- Cached subtask-index validation now checks dataset size, but not a stronger split/cache fingerprint; stale-cache cases with unchanged dataset length could still slip through.
- `scripts/server_routed.sh` still carries stale built-in defaults and must be overridden explicitly.

Immediate next step:

- Resume `acot_challenge_generalist_lora_generalist` from the retained `10000` checkpoint with `scripts/train_fast.py --r2a-cache-root ...` on the training machine and confirm:
  - next checkpoint save succeeds
  - validation still runs as expected
  - the resulting checkpoint remains readable by offline eval
  - measured startup/throughput is actually better than the legacy path

Date: 2026-03-12

Author: Codex

What changed:

- Pulled `AGENTS/status.md`, `AGENTS/PLAN.md`, `AGENTS/runbook_training.md`, and `Training_Notes.md` back toward the current codebase and retained artifacts.
- Replaced the removed generalist config name `acot_challenge_generalist_lora_all` in active training guidance with `acot_challenge_generalist_lora_generalist`.
- Updated docs to reflect the retained generalist artifact shape:
  - experiment name `generalist_v1_bs96`
  - one retained saved checkpoint at step `5000`
  - metrics recorded through step `8500`
- Documented that fast-path training is still additive and should not be treated as the default launcher yet.
- Added `acot_challenge_generalist_lora_generalist_tuned` as the recommended next-step generalist config.
- Recorded the accepted tuned-config changes:
  - `warmup_steps=2000`
  - `decay_steps=24000`
  - `num_train_steps=24000`
  - `val_interval=1000`
  - `val_num_batches=32`
  - `save_interval=1000`
  - `batch_size=120`
- Recorded the rejected ablation:
  - do not change the current state/action masks in the tuned config

What was verified:

- The updated docs now match the current active config names in `src/openpi/training/config.py`.
- The updated docs now match the retained experiment naming under `checkpoints/acot_challenge_generalist_lora_generalist/`.
- `src/openpi/training/config.py` contains `acot_challenge_generalist_lora_generalist_tuned`, and it keeps the original masking behavior.

What is still broken or unknown:

- `scripts/server_routed.sh` still carries stale built-in defaults and must be overridden explicitly.
- The tuned generalist config has not been run yet, so its wall-clock throughput and val behavior are still unverified.

Immediate next step:

- Choose between continuing `generalist_v1_bs96` to the next checkpoint boundary or launching `acot_challenge_generalist_lora_generalist_tuned` as the next mainline ablation, then use offline eval plus checkpoint serving before revisiting routed serving or fast-path promotion.

Date: 2026-03-11

Author: Codex

What changed:

- Replaced the placeholder submission runbook with a concrete agent-facing Docker packaging workflow in `AGENTS/runbook_submission.md`.
- The runbook now records the submission-specific lessons from the generalist checkpoint image:
  - use an isolated submission repo/worktree
  - stage the serving checkpoint under `checkpoint/`
  - remove `train_state/` before building
  - avoid `/app` because the official base image already contains a repo there
  - use `/submission` plus explicit `PYTHONPATH`
  - do not use `uv run` at container startup
  - prefetch tokenizer assets during `docker build`

What was verified:

- The documented pitfalls match the behavior observed during the successful generalist image submission workflow.
- The runbook now includes concrete launcher and Dockerfile templates instead of placeholders.

What is still broken or unknown:

- The runbook is currently written around the single-checkpoint generalist submission path; routed/adaptor submission should reuse the packaging rules but still needs its own final image template.

Immediate next step:

- Reuse `AGENTS/runbook_submission.md` as the starting point for the next submission image instead of rebuilding the packaging workflow from memory.

Date: 2026-03-11

Author: Codex

What changed:

- Fixed the local serve launch scripts to match the current `tyro` subcommand syntax:
  - `scripts/server_checkpoint.sh` now uses `policy:checkpoint`
  - `scripts/server_routed.sh` now uses `policy:adapter-routed`
- Recorded the fix in `AGENTS/status.md`.

What was verified:

- `uv run python scripts/serve_policy.py --help` now clearly shows the expected subcommands:
  - `policy:checkpoint`
  - `policy:default`
  - `policy:adapter-routed`
- The old `--policy checkpoint` / `--policy adapter-routed` format is confirmed incompatible with the current CLI.

What is still broken or unknown:

- Full runtime serving with the real generalist checkpoint still needs a real launch test after this script fix.

Immediate next step:

- Re-run `bash scripts/server_checkpoint.sh 0 8999` with:
  - `ACOT_SERVE_CONFIG=acot_challenge_generalist_lora_generalist`
  - `ACOT_SERVE_CHECKPOINT=<...>/generalist_v1_bs96/5000`
  and verify `/healthz` plus one real inference smoke test.

Date: 2026-03-11

Author: Codex

What changed:

- Added an additive fast-training path intended for `acot_challenge_generalist_lora_generalist` without modifying the legacy training path:
  - `scripts/train_fast.py`
  - `scripts/train_fast.sh`
  - `src/openpi/training/data_loader_fast.py`
  - `scripts/precompute_subtask_index_cache.py`
  - `scripts/precompute_prompt_cache.py`
- Documented fast-path usage in `Training_Notes.md`.
- Added several fast-path stability fixes:
  - strip metadata keys from device batches while keeping host metadata for logging
  - isolate one-shot preview loader from the long-lived training iterator
  - log preview images from host batches instead of device batches
- Recorded the fast-path and data-layout conclusions in `AGENTS/constraints.md`, `AGENTS/decisions.md`, `AGENTS/experiments.md`, `AGENTS/runbook_training.md`, and `AGENTS/status.md`.

What was verified:

- Synthetic transform-parity check for `acot_challenge_generalist_lora_generalist` showed the additive fast loader matches the legacy transform order and key tensor outputs.
- `python3 -m py_compile` passes for the current fast-path files.
- Dataset inspection on `/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim` confirms the training set is highly fragmented and video-heavy:
  - about `4179` parquet files
  - about `12778` mp4 files
  - per-episode parquet and per-camera mp4 layout
  - HEVC video, including high-resolution wrist cameras

What is still broken or unknown:

- Multi-GPU startup / first-step latency remains unresolved.
- Later user testing indicates that legacy `scripts/train.py` also stalls or compiles very slowly on dual GPU, so the remaining blocker is likely shared JAX/XLA multi-GPU behavior rather than a fast-loader-only bug.
- Fast-path compatibility with validation, offline eval, and downstream checkpoint-serving is a design goal and partial code-level target, but not all combinations have been runtime-verified on target hardware.
- For `acot_challenge_generalist_lora_generalist`, prompt-only token caching is intentionally unsupported because tokenization is state-dependent.

Immediate next step:

- Treat dual-GPU startup as a shared JAX/XLA issue and isolate it independently of the fast loader.
- In parallel, choose the next throughput path:
  - better online episode-locality / decoder-reuse loader
  - or an offline resized-video / training-cache path
  because the inspected dataset format strongly suggests that training throughput is bottlenecked by random video access and decode.

Date: 2026-03-11

Author: Codex

What changed:

- Reverted the abandoned per-config final-sample cache direction.
- Added a new additive generic `Reasoning2Action-Sim` frame-cache path:
  - `scripts/build_reasoning2action_frame_cache.py`
  - `scripts/verify_reasoning2action_frame_cache.py`
  - `scripts/compute_norm_stats_fast.py`
  - `src/openpi/training/r2a_frame_cache.py`
  - `src/openpi/training/data_loader_fast_r2a.py`
- Wired `scripts/train_fast.py` and `src/openpi/training/data_loader_fast.py` to accept an optional `--r2a-cache-root` flag.
  - without the flag, behavior stays on the existing raw-data fast path
  - with the flag, train/val loaders switch to the cache-backed dataset
- Updated `Training_Notes.md`, `AGENTS/status.md`, `AGENTS/decisions.md`, and `AGENTS/runbook_training.md` with the generic frame-cache workflow.

What was verified:

- `python -m py_compile` passes for the new generic frame-cache modules and the `train_fast.py` flag wiring.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m pytest -q src/openpi/training/r2a_frame_cache_test.py` passes.
- `uv run python scripts/train_fast.py --help` still works with the new optional cache flag parsing.

What is still broken or unknown:

- No real `Reasoning2Action-Sim` cache has been built yet on actual data.
- Raw-vs-cache parity has only synthetic/unit coverage so far, not a full real-data verification run.
- Cache-backed training throughput on the A100 machine is still unmeasured.
- The underlying dual-GPU JAX/XLA first-step issue remains separate and unresolved.

Immediate next step:

- Build one real frame cache on the preprocessing machine.
- Run `scripts/verify_reasoning2action_frame_cache.py` on `acot_challenge_generalist_lora_generalist`.
- Then benchmark `scripts/train_fast.py --r2a-cache-root ...` on the A100 machine against the raw fast path.

Date: 2026-03-10

Author: Codex

What changed:

- Updated routed-serving docs to reflect the reverse-engineered Genie Sim contract:
  - websocket `payload["task_name"]` is the ICRA `sub_task_name`
  - the outer benchmark scene name is not sent to the inference server
  - the current public routing surface is 10 route keys backed by 9 specialist adapters plus `_default`
- Updated `Training_Notes.md`, `AGENTS/PLAN.md`, `AGENTS/runbook_submission.md`, and `AGENTS/questions.md` to reflect the corrected routing model.
- Recorded the routing-contract decision in `AGENTS/decisions.md`.
- Recorded two further decisions in `AGENTS/decisions.md`:
  - undocumented aliases should not remain in the final ICRA router
  - `_part_*` dataset folders are same-task storage shards and must be merged into the same task family
- Did not change serving code; instead documented a focused code-adjustment plan for before final submission.

What was verified:

- The existing routed policy implementation already reads `obs["task_name"]`, so the corrected contract matches the current architecture.
- The existing post-process path already distinguishes sorting tasks by `task_name`, which remains compatible with the reverse-engineered evaluator behavior.

What is still broken or unknown:

- The router still contains a legacy defensive alias `grab_toy -> place_block_into_box`, even though the project decision is now to remove undocumented aliases from the final router.
- The current specialist configs still lag the shard-merging decision: they do not yet include all known shards for stock-shelf and sorting task families.
- There is still no automated smoke coverage over all 10 public route keys.
- No runtime check has yet been added to log raw incoming route keys and resolved adapter names in a structured way.

Immediate next step:

- Remove the legacy alias from routed serving, update specialist configs to include all known same-task shards, and then add one narrow routed-serving smoke test that exercises all 10 known keys, `sorting_packages_continuous`, and unknown fallback.

Date: 2026-03-10

Author: Codex

What changed:

- Hardened `scripts/train.py` JSONL metric logging for local checkpoint dirs:
  - switched the metric log path handling to `pathlib.Path`
  - create/touch `train_metrics.jsonl` during logger init
  - re-create the parent directory before each append
- Corrected `acot_challenge_generalist_lora_5_tasks` defaults so they match the smoke-tested initialization path unless explicitly overridden:
  - asset env var is now `ACOT_CHALLENGE_GENERALIST_5_TASKS_ASSET_ID`
  - init weights env var is now `ACOT_CHALLENGE_5_TASKS_INIT_WEIGHTS`, falling back to `ACOT_CHALLENGE_INIT_WEIGHTS`, then `gs://openpi-assets/checkpoints/pi05_base/params`
- Fixed the immediate high-loss culprit in the 5-task norm-stats path:
  - `scripts/compute_norm_stats.py` now computes stats from the entire sampled batch instead of only `batch[key][0]`
  - `scripts/compute_norm_stats.py` now always shuffles the sampled batches used for stats estimation
  - `DataConfigFactory._load_norm_stats()` now repairs near-zero std values using quantile span with a small floor to avoid catastrophic over-normalization from bad saved stats

What was verified:

- `python3 -m py_compile` still passes for `scripts/train.py` and `src/openpi/training/config.py`.
- Git inspection isolated the regression point for the 5-task config to commit `5f2a975`, where the config switched from the smoke-run `pi05_base` init to a local `baseline_checkpoint` and increased batch size from `60` to `96`.
- The training crash reported at step ~600 matches the post-`1f9b16f` JSONL logging changes in `scripts/train.py`.
- `python3 -m py_compile` passes for `scripts/compute_norm_stats.py`.
- A CPU-mode batch probe on `acot_challenge_generalist_lora_5_tasks` showed the problematic normalized action dimension drop from `std≈21.6` / `max≈100.9` to `std≈0.135` / `max≈6.28` after the norm-stat fixes.

What is still broken or unknown:

- The very large 5-task train loss appears before meaningful optimization progress and is likely dominated by config/init drift rather than the new logger crash, but this still needs a real rerun after restoring the intended init checkpoint path.
- The precise step-0 loss baseline for the older clean-desktop smoke run was not recoverable from the existing logs because those log files do not contain the later structured step summaries.

Immediate next step:

- Re-run `acot_challenge_generalist_lora_5_tasks` with the restored init-weight default or an explicit `ACOT_CHALLENGE_5_TASKS_INIT_WEIGHTS`, then compare the new step-0/step-100 loss against the previous `9e9-2e10` range before changing optimizer or batch-size settings.

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
- Fixed the underlying split-vs-LeRobot incompatibility in `src/openpi/training/data_loader.py`: selected-episode `LeRobotDataset` objects are now wrapped by a top-level compatibility dataset that remaps the original global `episode_index` to the dataset-local episode position expected by `episode_data_index`. This preserves original episode ids for prompt logic, avoids the large number of false out-of-bounds reads on split subsets, and stays pickle-safe for PyTorch `spawn` workers.
- Added `scripts/run_norm_and_train.py` as a convenience wrapper for unattended runs: it launches `compute_norm_stats.py` first and, only if that succeeds, launches `train.py` with `--overwrite=true` using the same Python environment.
- Added `scripts/run_norm_and_train.sh` for the same unattended workflow in pure shell. Also updated `scripts/train.sh` to accept and forward extra args to `train.py`, including translating the typo `--overwirte` into `--overwrite=true`.

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
- The new pickle-safe selected-episode compatibility wrapper still needs one real `scripts/compute_norm_stats.py` rerun with `num_workers > 0` to confirm the previous `patched_get_query_indices` multiprocessing failure is gone on the target environment.

Immediate next step:

- Re-run `scripts/compute_norm_stats.py` for the target split-enabled config with worker processes enabled and confirm it gets past the earlier `patched_get_query_indices` spawn error; then launch a short debug training run and confirm validation loss logs on the val split.
- After the debug run produces checkpoints, execute `scripts/eval_offline.py` on the experiment directory and inspect `eval/summary.csv` for checkpoint ranking.
- Inspect `<checkpoint_dir>/train_metrics.jsonl` from the same debug run and verify that the expected `train/task/*` and `train/action_mae/*` keys appear.

Git note:

- Commit substantial progress before handoff so the next agent can diff exact changes instead of reconstructing them from notes.

Date: 2026-03-12

Author: Codex

What changed:

- Fixed the current cache-verification blocker where `verify_reasoning2action_frame_cache.py` failed on `episode_index` / `frame_index` dtype-only differences (`int64` raw vs `int32` cache).
- Improved `verify_reasoning2action_frame_cache.py` mismatch output so scalar mismatches now include actual values.
- Updated `verify_reasoning2action_frame_cache.py` to compare nested batch outputs recursively, which fixes the current `np.array_equal` crash on dict-valued keys like `image` and `image_mask`.
- Updated `src/openpi/training/config.py` to auto-detect the local `Reasoning2Action-Sim` root when the machine uses `~/Datasets/huggingface/lerobot/...` instead of `~/Datasets/lerobot/...`.
- Updated `src/openpi/training/r2a_frame_cache.py` so future cache builds no longer downcast those metadata indices to `int32` during staging, assembly, or dataset reads.
- Added regression tests in `src/openpi/training/r2a_frame_cache_test.py` covering both the source-to-cache path and the final assembled-cache path.
- Added `scripts/verify_reasoning2action_frame_cache_test.py` for verifier helper coverage.

What was verified:

- `python -m py_compile` passes for:
  - `src/openpi/training/r2a_frame_cache.py`
  - `src/openpi/training/r2a_frame_cache_test.py`
  - `scripts/verify_reasoning2action_frame_cache.py`
- `python -m pytest -q src/openpi/training/r2a_frame_cache_test.py` passes with 3 tests.
- Direct reproduction against the real cache shows the previously reported failing sample index `169803` now passes the relaxed metadata-value check.
- `python -m pytest -q scripts/verify_reasoning2action_frame_cache_test.py src/openpi/training/r2a_frame_cache_test.py` passes with 5 tests.
- Direct reproduction against the real processed batch now passes the verifier comparison for all current batch keys.

Immediate next step:

- Re-run `scripts/verify_reasoning2action_frame_cache.py` on the real cache and see whether any further parity mismatch remains after the dtype-only metadata check is removed.
