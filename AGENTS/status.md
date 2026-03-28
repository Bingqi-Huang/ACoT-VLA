# Status

Last updated: 2026-03-28

## Session 2026-03-28: Fix lerobot import regression in v3 serving image

- Symptom:
  - `docker run ... generalist-v3:latest` failed at startup with `ModuleNotFoundError: No module named 'lerobot'`.
- Root cause:
  - `scripts/serve_policy.py` imported `adapter_routed_policy` at module import time.
  - `src/openpi/training/checkpoints.py` imported `openpi.training.data_loader` at runtime import time.
  - This created a startup import chain into training-only `lerobot` even for `policy:checkpoint` serving.
- Fix (aligned to prior `submitted-generalist-v2` behavior):
  - `scripts/serve_policy.py`: restore lazy import of `adapter_routed_policy` only inside `AdapterRouted` branch.
  - `src/openpi/training/checkpoints.py`: guard `data_loader` import behind `TYPE_CHECKING` and use string type annotations.
- Validation:
  - `python -m py_compile scripts/serve_policy.py src/openpi/training/checkpoints.py`
  - rebuilt image with `scripts/docker/serve_generalist_v3.Dockerfile`
  - container smoke test now reaches checkpoint restore and websocket startup:
    - `server listening on 0.0.0.0:8999`
  - no `lerobot` import error after fix.

## Session 2026-03-28: submitted-generalist-v3 branch setup from continued-training

- Created local branch:
  - `submitted-generalist-v3`
  - base: `origin/feature/generalist-continued-training`
- Preserved current `AGENTS/docker_build.md` content while switching base branch.
- Confirmed reweighted continued-training config is present in branch code:
  - `acot_challenge_generalist_continued_reweighted` in `src/openpi/training/config.py`
- Added submission-serving files missing from the feature branch:
  - `scripts/server_submit_generalist_v2.sh`
  - `scripts/docker/serve_generalist_v2.Dockerfile`
- Wired these serving files to reweighted defaults:
  - `ACOT_SERVE_CONFIG=acot_challenge_generalist_continued_reweighted`
  - `ACOT_SERVE_CHECKPOINT=/submission/checkpoint/generalist_continued_augumented`
- Build-time install strategy in the Dockerfile remains aligned with prior submission path:
  - `uv sync --frozen --no-dev --no-install-package lerobot`
- Basic validation completed:
  - `bash -n scripts/server_submit_generalist_v2.sh`
  - grep checks confirm reweighted config string in launcher + Dockerfile + config registry.

## Session 2026-03-21: Blackwell CUDA illegal-address fix for continued generalist training

- Target config:
  - `acot_challenge_generalist_continued`
- Failure mode (before fix):
  - multi-GPU run on 6x `RTX PRO 6000 Blackwell` failed with `CUDA_ERROR_ILLEGAL_ADDRESS`.
- Environment fix:
  - upgraded to CUDA13 JAX stack:
    - `jax[cuda13]==0.7.2`
    - `jaxlib==0.7.2`
    - `numpy>=2.0.0,<3.0.0`
    - `orbax-checkpoint==0.11.33`
    - `ml-dtypes==0.5.3`
  - runtime check now reports `cuda 13000`.
- Code compatibility fix:
  - patched `src/openpi/models/model.py::restore_params()` to handle Orbax `StepMetadata` (`item_metadata["params"]`) as well as legacy dict-style metadata.
- Validation result:
  - run `acot_challenge_generalist_continued_cuda13_fix1` successfully passed:
    - checkpoint restore
    - train-state initialization
    - real step advancement (`5/30000` before manual stop)
  - no `CUDA_ERROR_ILLEGAL_ADDRESS` in this fixed run log.
- Incident log:
  - `AGENTS/debug_cuda_illegal_address_2026-03-21.md`

## Session 2026-03-14: Routed clean-desktop-1500 serving readiness

- Updated routed task mapping so `clean_the_desktop` now activates adapter `clean_the_desktop_1500`.
- Scoped routed specialization to only currently packaged adapters; unmatched tasks now fall through to `_default` / `_base`.
- Converted `scripts/docker/serve_policy.Dockerfile` to submission-safe packaging:
  - isolated `/submission` layout
  - explicit copy of serving-only trees
  - tokenizer prefetch at build time
  - auto-launch via `scripts/server_submit_routed.sh` on port `8999`
- Refreshed `AGENTS/docker_build.md` to routed clean-desktop-1500 build/run/tag/push workflow.
- Documented that `checkpoint/specialists_clean_1500` is not required at inference after adapter extraction and is excluded from Docker build context via `.dockerignore`.
- Fixed submission Docker build failures on the `icra-admin/openpi_server` base image:
  - base image may not provide a usable plain `python` + deps for our submission tree
  - Dockerfile now runs `uv sync --frozen --no-dev` at build time
  - tokenizer prefetch now uses `uv run --no-sync ...`
  - routed startup script now uses `uv run --no-sync ...`
  - verified `docker build --no-cache -f scripts/docker/serve_policy.Dockerfile ...` succeeds
- Runtime smoke test found and fixed two additional issues:
  - `matplotlib` import in `src/openpi/shared/nnx_utils.py` was unnecessary for serving and caused startup failure on the base image; removed the hard import.
  - baseline-as-base routed serving is now explicitly supported for LoRA adapters by loading baseline params into the LoRA model with zero-initialized missing LoRA tensors in `adapter_routed_policy`.
- Build context policy for routed submission:
  - include `checkpoint/baseline/30000` (actual base model for non-routed tasks)
  - exclude `checkpoint/specialists_clean_1500` via `.dockerignore` after adapter extraction
- Fixed routed adapter state flatten/unflatten path handling for checkpoints that contain integer key path parts:
  - `src/openpi/policies/adapter_routed_policy.py` now stores flattened paths as tuple keys internally instead of `/`-joined strings.
- Verified runtime smoke test reaches websocket serving state with baseline base-checkpoint:
  - checkpoint loaded from `checkpoint/baseline/30000`
  - `server listening on 0.0.0.0:8999`

## Current State

- Repository has been examined for Reasoning2Action competition training and submission flow.
- Local dataset extraction helper has been added:
  - `scripts/prepare_reasoning2action_sim.sh`
- Local challenge config has been added:
  - `acot_icra_simulation_challenge_reasoning_to_action_local`
- Normalization stats script has been patched so stats are saved to the asset directory used by training.
- Blocking LoRA infrastructure fixes are now in place:
  - `gemma_300m_lora` added to the Gemma variant type alias.
  - ACoT freeze filtering now preserves LoRA params in the coarse action expert as well.
- Gradient accumulation support has been added to training:
  - `TrainConfig.grad_accum_steps`
  - split compute/apply grad path in `scripts/train.py`
  - accumulation now works for both standard and ACoT training loops
- Added targeted model tests for the LoRA variant and ACoT freeze-filter behavior:
  - `src/openpi/models/acot_vla_test.py`
- Added challenge-specific LoRA training configs:
  - `acot_challenge_generalist_lora_generalist`
  - `acot_challenge_generalist_lora_generalist_tuned`
  - `acot_challenge_generalist_lora_clean_desktop`
  - `acot_challenge_generalist_lora_5_tasks`
  - generated `acot_specialist_*` configs for the 9 challenge tasks
- Added a recommended follow-up generalist config that keeps the current masks unchanged and only tunes training cadence:
  - `acot_challenge_generalist_lora_generalist_tuned`
  - `warmup_steps=2000`
  - `decay_steps=24000`
  - `num_train_steps=24000`
  - `val_interval=1000`
  - `val_num_batches=32`
  - `save_interval=1000`
  - `batch_size=120`
- Added adapter-routing infrastructure for inference:
  - `scripts/extract_adapter.py`
  - `src/openpi/policies/adapter_routed_policy.py`
  - new `AdapterRouted` serve mode in `scripts/serve_policy.py`
  - `scripts/server_routed.sh`
- Fixed checkpoint norm-stats compatibility:
  - new checkpoints now save norm stats under `assets/<asset_id>/` for policy loading
  - loading also falls back to legacy root-level `assets/` stats for older checkpoints
- Adapter-routed serving now uses cached per-adapter `nnx.State` overlays with a single stateful JIT-compiled sampler instead of rebuilding the full model on task switches.
- Serving Docker now defaults to the routed launch script:
  - `scripts/docker/serve_policy.Dockerfile` starts `scripts/server_routed.sh` unless `SERVER_SCRIPT` overrides it
- Added a plain checkpoint-serving launch path for single-model test submissions:
  - `scripts/server_checkpoint.sh`
  - currently the most realistic single-model serve path for the retained generalist checkpoint
- Updated the serve launch scripts for the current `tyro` CLI subcommand format:
  - `scripts/server_checkpoint.sh` now invokes `serve_policy.py policy:checkpoint`
  - `scripts/server_routed.sh` now invokes `serve_policy.py policy:adapter-routed`
  - this fixes the previous `Unrecognized options: --policy` startup failure
- The routed serve script still has stale defaults:
  - `scripts/server_routed.sh` points to `acot_challenge_generalist_lora_all`
  - that config name does not exist in current `src/openpi/training/config.py`
  - routed serving should currently be treated as an opt-in path that requires explicit env overrides
- Adjusted Python packaging for the current Ubuntu 22.04 / Python 3.11 environment:
  - pinned transitive `av` resolution to `14.0.1` via `tool.uv.override-dependencies`
  - replaced deprecated `tool.uv.dev-dependencies` in `packages/openpi-client/pyproject.toml`
- Added deterministic episode-level train/validation splits for LeRobot finetuning datasets:
  - `EpisodeSplitConfig` on `DataConfig`
  - split manifests saved under `assets/<config>/episode_splits/*.json`
  - train loader now reads only train episodes
  - validation loader now reads only validation episodes
  - per-task episode/frame counts are printed for both splits
- Updated `scripts/compute_norm_stats.py` to support `--split train|val` so norm stats can be computed from train episodes only.
- Added validation support to `scripts/train.py`:
  - `TrainConfig.val_interval`
  - `TrainConfig.val_num_batches`
  - validation loss logging against the val episode split
- Added split tooling and docs:
  - `scripts/generate_episode_split.py`
  - `docs/reasoning2action_episode_split.md`
- Added standalone offline evaluation for finetuning checkpoints:
  - `scripts/eval_offline.py`
  - `src/openpi/training/offline_eval.py`
  - `docs/reasoning2action_offline_eval.md`
- Added deterministic teacher-forced offline model hooks for pi0/pi0.5 and ACoT:
  - `compute_loss_per_example(...)`
  - `teacher_force_actions(...)`
  - used for offline validation metrics without rollout

## Current Retained Artifacts

- The currently retained main experiment is:
  - `checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96`
- That experiment currently has:
  - one saved checkpoint at step `5000`
  - `train_metrics.jsonl` entries through step `8500`
  - best recorded validation loss `0.1684` at step `7000`
  - last recorded train loss `0.0787`
- Practical implication:
  - the most useful current training line is the retained generalist single-model path
  - the best visible val step has not been retained as a checkpoint yet
- Historical clean-desktop logs show a successful save at step `10000`, but that checkpoint tree is not retained in the current workspace.
- No `adapters/` directory or extracted adapter `.npz` files are present in the current workspace.
- Fast-path assets currently present in-worktree are limited to:
  - subtask-index cache files
  - a prompt-token cache artifact for the generalist config
  - there is not yet a retained real generic frame-cache root built from full data

## Dataset Understanding

- The downloaded `Reasoning2Action-Sim` data is already in LeRobot archive form.
- The user-downloaded folders contain split archive parts:
  - `meta.tar.gz.*`
  - `data.tar.gz.*`
  - `videos.tar.gz.*`
- Required preprocessing is archive extraction, not format conversion.

## Current Known Paths

- Raw downloaded data:
  `~/Datasets/hf_data/Reasoning2Action-Sim`
- Intended extracted LeRobot root:
  `~/Datasets/lerobot/Reasoning2Action-Sim`

## Submission Understanding

- Submission is done by pushing a Docker image and pasting the full image URL on the platform.
- The runtime contract is stricter than the training process.
- Architecture changes are allowed only if the final serving interface remains compliant.
- For meaningful finetuning validation, train/val separation now happens strictly at the episode level for LeRobot datasets when `episode_split` is configured.
- Offline checkpoint selection can now be done on the val episode split without simulation rollout, using per-checkpoint JSON metrics and a summary CSV.
- Finetuning runs now emit richer training diagnostics during `scripts/train.py`, including train loss, per-task train loss when task metadata is present, learning rate, grad norm, param norm, throughput, wall-clock time, checkpoint events, and batch-level action MAE metrics to both W&B and `train_metrics.jsonl` inside the experiment checkpoint directory; the W&B metric families are now explicitly defined so these series show up with stable step axes.
- Offline eval for ACOT checkpoints now preserves dynamic data-config attributes such as `joint_action_shifts` when injecting checkpoint norm stats, fixing the `AttributeError: 'DataConfig' object has no attribute 'joint_action_shifts'` crash seen on clean-desktop checkpoint evaluation.
- Offline eval now also marks the `train` kwarg as static when JIT-compiling `compute_loss_per_example`, fixing the `TracerBoolConversionError` raised inside `preprocess_observation()` during teacher-forced ACOT evaluation.
- Submission packaging guidance is now documented for coding agents in `AGENTS/runbook_submission.md`:
  - create and freeze a dedicated `submit-*` branch after a verified local serve test
  - perform Docker packaging in an isolated clone/worktree of that branch
  - prune non-serving files there before image build
  - use an isolated submission repo/worktree
  - stage the final checkpoint under `checkpoint/` and remove `train_state/`
  - do not overlay the official base image's `/app`; use `/submission`
  - do not use `uv run` at container startup
  - prefetch tokenizer assets during `docker build`
- `scripts/compute_norm_stats.py` now uses `SafeDataset` and respects its local shuffle flag, so isolated bad samples no longer crash worker batches or leave norm-stat accumulation empty; empty fully-skipped batches are also ignored by `TorchDataLoader`.
- `scripts/eval_offline.py` now shows a per-checkpoint tqdm progress bar during validation-batch evaluation.
- `TorchDataLoader` now skips incomplete post-filter batches as well, preventing multi-device `device_put` failures when `SafeDataset` drops bad samples and the remaining batch size is no longer divisible by the data-parallel mesh size.
- Root cause identified for the large number of post-split “bad samples”: the current LeRobot version uses global `episode_index` values to index a local `episode_data_index` array after subsetting episodes, which breaks on non-contiguous episode subsets. `src/openpi/training/data_loader.py` now wraps selected-episode LeRobot datasets in a top-level pickle-safe compatibility dataset so query/padding logic remaps global episode ids to local subset positions while preserving the original `episode_index` in returned samples.
- Added `scripts/run_norm_and_train.py` to sequentially run `compute_norm_stats.py` and then `train.py --overwrite=true` in one command, intended for unattended overnight finetuning starts after the split/data-loader fixes.
- Added a shell-first unattended runner `scripts/run_norm_and_train.sh`; `scripts/train.sh` now forwards extra CLI args to `train.py` and normalizes the common typo `--overwirte` to `--overwrite=true`.
- Diagnosed two regressions in the current 5-task finetuning path:
  - `scripts/train.py` JSONL metric logging used `etils.epath.Path.open("a")`, which can raise `FileNotFoundError` for a missing local file; logger now uses `pathlib.Path`, touches the file up front, and re-creates the parent directory before each append.
  - `acot_challenge_generalist_lora_5_tasks` had drifted from the smoke-tested init path by defaulting to a local `baseline_checkpoint` and by reusing the clean-desktop asset env var name; it now uses task-specific asset env naming and falls back to the same `ACOT_CHALLENGE_INIT_WEIGHTS` / `pi05_base` path used by the known-good smoke config unless explicitly overridden.
- Identified the immediate cause of the huge 5-task step-0 loss as biased / degenerate normalization stats rather than the optimizer:
  - `scripts/compute_norm_stats.py` was sampling only the first element of each batch and, for the common case without `max_frames`, doing so without shuffling, which biases stats toward the earliest contiguous frames/tasks.
  - For the current 5-task asset this produced a near-zero std on at least one action dimension; a CPU-mode probe showed normalized `actions[..., 14]` at `std≈21.6`, `max≈100.9` before the fix.
- `scripts/compute_norm_stats.py` now accumulates stats from the full batch and always shuffles its sampled batches.
- `DataConfigFactory._load_norm_stats()` now repairs near-zero std values using quantile span and a small floor, which reduced the same normalized action batch to `std≈0.135`, `max≈6.28` without requiring an immediate asset rewrite.
- Reverse-engineered Genie Sim routing contract has now been documented:
  - websocket `payload["task_name"]` is the evaluator `sub_task_name`
  - the outer benchmark `task_name` is not sent to the server
  - the current public ICRA surface is 10 route keys / 26 instances
  - `sorting_packages_continuous` should route by `task_name` and share the sorting adapter
  - routed-serving docs were updated; no code changes were applied yet
- Finalized two routing/data decisions from the reverse-engineering work:
  - undocumented aliases such as `grab_toy -> place_block_into_box` should not remain in the final ICRA router
  - same-task folders with `_part_*` suffixes are storage shards of one task and should be merged into the same task family during training
- Current code/doc mismatches are now explicit:
  - `acot_challenge_generalist_lora_all` is still referenced in older docs and routed-serving defaults, but the current codebase does not define that config
  - `src/openpi/policies/adapter_routed_policy.py` still carries the legacy alias `grab_toy -> place_block_into_box`
  - `acot_specialist_stock_shelf` and `acot_specialist_sorting` currently do not include all known shards of their task families, even though the full generalist config already includes `stock_and_straighten_shelf_part_2` and `sorting_packages_part_3`
- Added an additive fast-training path for the full generalist workflow:
  - `scripts/train_fast.py`
  - `scripts/train_fast.sh`
  - `src/openpi/training/data_loader_fast.py`
  - `scripts/precompute_subtask_index_cache.py`
  - `scripts/precompute_prompt_cache.py`
- Fast-path implementation details now in repo:
  - precomputed subtask-index cache support to reduce startup cost
  - host-side metadata retention while stripping metadata from device batches
  - preview-batch isolation so train-state initialization does not share the same long-lived data-prefetch iterator
  - host-side preview image logging to avoid forcing early device-side image materialization for W&B
- Fast-path transform parity for `acot_challenge_generalist_lora_generalist` was checked against the legacy transform chain on synthetic sample input:
  - repack -> data transforms -> normalize -> model transforms
  - `image`, `image_mask`, `state`, `actions`, `coarse_actions`, and tokenized prompt outputs matched
- For `acot_challenge_generalist_lora_generalist`, prompt-only token caching is intentionally disabled because `discrete_state_input=True` makes tokenization state-dependent.
- Generated fast-path cache outputs are intended to live under:
  - `assets/<config>/fast_cache/`
- Training docs now include fast-path usage instructions in:
  - `Training_Notes.md`
- Dataset layout has been inspected directly on the training root:
  - dataset root around `173G`
  - about `4179` parquet files
  - about `12778` mp4 files
  - videos stored per episode and per camera
  - sample metadata shows HEVC video with high-resolution wrist cameras (`1056x1280`)
- Current throughput diagnosis:
  - on older storage / memory hardware, steady-state bottleneck is likely dominated by random episode access plus video decode cost, not only by an overly large `num_workers`
- Current multi-GPU diagnosis:
  - single-GPU runs on newer hardware appear healthy and can drive GPU power much higher
  - dual-GPU startup / first-step latency is still unresolved
  - because both legacy `scripts/train.py` and additive `scripts/train_fast.py` show similar behavior, the main issue is currently believed to be shared JAX/XLA multi-GPU initialization / compilation rather than a fast-loader-only defect
- Implemented an additive `Reasoning2Action-Sim` generic frame-cache path:
  - `scripts/build_reasoning2action_frame_cache.py`
  - `scripts/verify_reasoning2action_frame_cache.py`
  - `scripts/compute_norm_stats_fast.py`
  - `src/openpi/training/r2a_frame_cache.py`
  - `src/openpi/training/data_loader_fast_r2a.py`
- `scripts/train_fast.py` now accepts optional `--r2a-cache-root`:
  - default behavior remains raw-data fast path
  - when provided, train/val loaders switch to the cache-backed dataset
- The generic frame cache stores an intermediate boundary only:
  - decoded/resized `224x224` images
  - raw state and raw action windows
  - prompt/task/frame metadata
  - it does not store normalized/tokenized/padded model-ready samples
- Fast path should still be treated as additive rather than primary:
  - there is no retained real cache-backed training run in the current workspace
  - no retained throughput benchmark shows a clear win on the target machine yet
  - `scripts/train_fast.py` checkpoint saving has been fixed for the main training loader path
  - fast loader `num_batches` iteration now mirrors the legacy loader across dataset exhaustion
  - cached fast-path subtask indices now reject obvious stale-cache cases when the recorded dataset size no longer matches the active dataset
  - targeted regression coverage now exists in `src/openpi/training/data_loader_fast_test.py`

## Session 2026-03-14: Competition Strategy Overhaul

Author: Claude Opus 4.6

### Analysis Completed

- Reviewed evaluation results: baseline=6.35, previous generalist (LoRA-everywhere step 20000)=5.08, net regression of -1.27.
- Categorized all 10 tasks by improvement potential:
  - PROTECT (near-perfect): open_door=1.0, scoop_popcorn=1.0, hold_pot=1.0, take_wrong_item=0.97 (3.97 pts)
  - IMPROVE (weak): clean_desktop=0.39, stock_shelf=0.24, place_block=0.42 (1.05 pts, ceiling 3.0)
  - MAINTAIN: pour_workpiece=0.61, sorting_packages=0.67
  - DEPRIORITIZE: sorting_continuous=0.05
- Traced the random-init issue end-to-end through code:
  - Both `lora_a` and `lora_b` use `nn.initializers.normal(stddev=0.01)` (not zero-init for `lora_b` like standard LoRA)
  - `ACOTCheckpointWeightLoader` uses `missing_regex=".*"` so all missing params are synthesized
  - Missing LoRA tensors get `random * 0.02` from weight loader, not the model's own init
  - Net perturbation per LoRA residual ~0.001 magnitude — small but nonzero
  - Baseline-compatible config has 0 missing tensors; LoRA config has 20 missing expert LoRA tensors
- Identified warmup bug in `acot_challenge_generalist_baseline_compatible`: warmup=10000 == total steps, LR never reaches peak

### Code Changes Made

1. **New config `acot_challenge_lora_conservative`** (`src/openpi/training/config.py`)
   - warmup=200, peak_lr=1e-5, 8000 steps, save/val every 500, batch_size=120, no EMA
   - Conservative schedule designed to limit drift from baseline while allowing weak-task improvement

2. **Fixed baseline-compatible warmup bug** (`src/openpi/training/config.py:2381`)
   - Changed warmup from 10,000 to 500 (was equal to total run length)

3. **LoRA-only adapter extraction** (`scripts/extract_adapter.py`)
   - Added `--lora-only` flag: extracts only LoRA tensors (~220MB per adapter in bfloat16)
   - Makes adapter routing viable on 24GB inference server

4. **Specialist configs support baseline init** (`src/openpi/training/config.py`)
   - `_make_reasoning2action_specialist_configs()` now falls back to `ACOT_CHALLENGE_INIT_WEIGHTS` when `ACOT_CHALLENGE_GENERALIST_WEIGHTS` is unset
   - Uses `ACOTCheckpointWeightLoader` for proper dual-expert weight remapping from baseline
   - Updated specialist hyperparams: lr=1e-5, 5000 steps, save_interval=500, no EMA

5. **Task routing updated for weak-task-only specialists** (`src/openpi/policies/adapter_routed_policy.py`)
   - `TASK_ROUTING` now only routes clean_desktop, stock_shelf, place_block to specialists
   - All other tasks fall through to `_default` (baseline/generalist), minimizing regression risk

### Strategy Established

Two-track approach with baseline as hard fallback:

- **Track A**: Conservative LoRA generalist — low LR, short run, dense checkpoints. Primary path.
- **Track B**: Baseline-compatible generalist — 0 random init, but 1.3B trainable params. Higher risk.
- **Track C** (later): Lightweight LoRA-only specialist routing for weak tasks only, using best generalist as base.
- **Hard rule**: Never submit below 6.35. Baseline is always the fallback.

### Verification

- All new/modified configs load correctly via `get_config()` (verified with `JAX_PLATFORMS=cpu`)
- `extract_adapter.py` `--lora-only` flag wired up correctly (verified via inspection)
- Unit tests could not run in this environment (JAX aborts without GPU), but are expected to pass on training machine

## Known Open Work

- Continue the retained `acot_challenge_generalist_lora_generalist` line to at least the next saved checkpoint boundary and keep the best-val step as a real checkpoint, not only in `train_metrics.jsonl`.
- Decide whether the next mainline run should continue `generalist_v1_bs96` or switch to `acot_challenge_generalist_lora_generalist_tuned`, which keeps the original masks and only changes schedule / validation / checkpoint cadence.
- Run `scripts/eval_offline.py` on the retained generalist checkpoint tree and use it for checkpoint selection on the current mainline.
- Finish or verify full extraction of all dataset parts.
- Validate a small debug training run on the actual target machine with `grad_accum_steps > 1`.
- Validate the new episode-split training path end-to-end on the clean-desktop smoke config and confirm validation loss logs as expected.
- Run the new offline evaluator on a real clean-desktop experiment directory and confirm the JSON/CSV outputs match expectations.
- Confirm the new `train_metrics.jsonl` contents and W&B curves on a real clean-desktop run, especially per-task task-name recovery and batch-level action/gripper metrics.
- Extract adapters from a real generalist/specialist checkpoint set and verify routed loading against those files.
- Validate the `AdapterRouted` CLI path with a real checkpoint + adapter directory.
- Measure task-switch latency once real adapter files are available.
- Re-run `scripts/eval_offline.py` on the previously failing clean-desktop checkpoint and confirm it now gets past dataset construction into actual metric computation.
- Re-run `scripts/compute_norm_stats.py` with worker processes enabled and confirm the new pickle-safe selected-episode wrapper gets past the earlier `patched_get_query_indices` spawn/pickling failure.
- Add routed-serving smoke coverage for all 10 public route keys plus unknown fallback.
- Remove the legacy `grab_toy -> place_block_into_box` alias from the routed-serving code so implementation matches the documented public ICRA contract.
- Update specialist configs so each task family consumes all known shards of that task:
  - `acot_specialist_stock_shelf` should include `stock_and_straighten_shelf` and `stock_and_straighten_shelf_part_2`
  - `acot_specialist_sorting` should include `sorting_packages_part_1`, `sorting_packages_part_2`, and `sorting_packages_part_3`
- Audit other task families for future `_part_*` shards and fold them into the same-task configs rather than creating new route keys.
- Record one real end-to-end routed-serving validation run with extracted adapters so the routing path is validated beyond static code inspection.
- Re-validate additive fast-path runtime behavior on the intended hardware, especially:
  - startup time improvement from subtask-index caching
  - compatibility with validation loader and `scripts/eval_offline.py`
  - any mismatch between legacy and fast checkpoint loading paths
- Build one real generic Reasoning2Action frame cache and measure:
  - wall-clock build time
  - resulting disk footprint
  - raw-vs-cache sample parity on actual data
  - cache-backed `train_fast.py` throughput on the A100 machine
- Determine whether the dual-GPU first-step stall is a JAX/XLA startup issue that also affects legacy `train.py`, or whether there is still a remaining fast-path-specific factor.
- Decide whether the next throughput push should target:
  - a better online dataloader with episode locality / decoder reuse
  - or continued optimization of the new generic frame-cache path
- Keep all frame-cache work additive and preserve legacy checkpoint / inference compatibility.

## Session 2026-03-14: Specialist Adapter Training + Routing Setup

Author: Claude Sonnet 4.6

### Code Changes Made

1. **New script `scripts/create_zero_lora_adapter.py`**
   - Creates a `_default.npz` adapter file with all LoRA tensors set to zero
   - Loads baseline checkpoint with `ACOTCheckpointWeightLoader(missing_init="zeros")`
   - Uses `jax.eval_shape` to avoid allocating full model on GPU (shape-only pass)
   - Filters to LoRA-only keys, saves in identical format to `extract_adapter.py`
   - Sanity-checks that all LoRA values are actually zero before writing
   - This zero adapter ensures strong tasks route to `_default` and behave exactly as baseline

### Purpose

Phase 1 of the specialist routing plan:
- Weak tasks (clean_desktop, stock_shelf, place_block) → specialist LoRA adapters
- Strong tasks → `_default.npz` (zero LoRA = pure baseline frozen weights)
- Zero LoRA contribution: `x @ 0 @ 0 * scaling = 0`, output = frozen base weights only

### Verification

- `python3 -m py_compile scripts/create_zero_lora_adapter.py` passes

### Next Steps

Phase 1 (run immediately):
```bash
python scripts/create_zero_lora_adapter.py \
  --checkpoint <baseline>/params \
  --output adapters/_default.npz
```

Phase 2 (train 3 specialists from baseline):
```bash
ACOT_CHALLENGE_INIT_WEIGHTS=<baseline>/params \
  bash scripts/train_fast.sh acot_specialist_clean_desktop exp_specialist_clean --r2a-cache-root=<path>

ACOT_CHALLENGE_INIT_WEIGHTS=<baseline>/params \
  bash scripts/train_fast.sh acot_specialist_stock_shelf exp_specialist_stock --r2a-cache-root=<path>

ACOT_CHALLENGE_INIT_WEIGHTS=<baseline>/params \
  bash scripts/train_fast.sh acot_specialist_place_block exp_specialist_place --r2a-cache-root=<path>
```

Phase 3 (extract best adapters):
```bash
python scripts/extract_adapter.py \
  --checkpoint checkpoints/acot_specialist_clean_desktop/exp_specialist_clean/<step> \
  --output adapters/acot_specialist_clean_desktop.npz --lora-only
```

Phase 4 (serve with routing):
```bash
python scripts/serve_policy.py policy:adapter-routed \
  --policy.config acot_challenge_lora_conservative \
  --policy.base-checkpoint <baseline_checkpoint> \
  --policy.adapter-dir adapters/
```

## Verification Notes

- `python3 -m py_compile` passes for the modified model/config/training files.
- `python3 -m py_compile` passes for the modified serving and adapter files.
- `python3 -m py_compile` passes for the new episode-split and validation files.
- `python3 -m py_compile` passes for the new offline evaluation files and the added model hooks.
- `python3 -m py_compile` passes for the new training logging changes in `scripts/train.py` and `src/openpi/training/data_loader.py`.
- `python3 -m py_compile` passes for the additive fast-training files after the latest host-preview / train-init isolation changes.
- `python -m py_compile` passes for the new generic frame-cache files and `train_fast.py` cache flag wiring.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m pytest -q src/openpi/training/r2a_frame_cache_test.py` passes.
- `git diff --check` passes.
- `uv run pytest ...` could not be completed in this environment because dependency resolution attempted network access and failed on DNS/package download.
- Direct Python smoke execution of the new split helpers is currently blocked in this environment because the installed `ml_dtypes` version is too old for JAX import (`0.4.1`, JAX requires `>=0.5`).

Date: 2026-03-12

Author: Codex

What changed:

- Fixed a false-negative in `scripts/verify_reasoning2action_frame_cache.py`:
  - `episode_index` and `frame_index` are now compared by value even when raw samples use `int64` and older cache files store `int32`.
- Improved the failure message in `scripts/verify_reasoning2action_frame_cache.py` so small mismatches now print actual scalar values, not only shape/dtype.
- `scripts/verify_reasoning2action_frame_cache.py` now compares nested batch structures recursively, so keys like `image` and `image_mask` no longer crash on dict-valued batches during parity checks.
- `src/openpi/training/config.py` now auto-detects the common local `Reasoning2Action-Sim` roots:
  - `~/Datasets/lerobot/Reasoning2Action-Sim`
  - `~/Datasets/huggingface/lerobot/Reasoning2Action-Sim`
- Hardened the cache build/read path in `src/openpi/training/r2a_frame_cache.py` so newly built caches preserve the original integer dtype for:
  - `episode_index`
  - `frame_index`
- Added dtype regression coverage in `src/openpi/training/r2a_frame_cache_test.py` for:
  - source-sample metadata conversion
  - final cache assembly
  - dataset reads from the assembled cache
- Added direct verifier helper coverage in `scripts/verify_reasoning2action_frame_cache_test.py` for:
  - nested dict batch comparison
  - dtype-relaxed metadata comparison

What was verified:

- Direct local reproduction against the real cache shows the previously failing sample index `169803` now passes `episode_index` / `frame_index` verification when `check_dtype=False`.
- Direct local reproduction against the real processed batch now passes for every key in:
  - `image`
  - `image_mask`
  - `state`
  - `coarse_actions`
  - `actions`
  - `task`
  - `episode_index`
  - `frame_index`
  - `tokenized_prompt`
  - `tokenized_prompt_mask`
