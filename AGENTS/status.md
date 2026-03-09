# Status

Last updated: 2026-03-09

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
  - `acot_challenge_generalist_lora_all`
  - generated `acot_specialist_*` configs for the 9 challenge tasks
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
  - intended for configs like `acot_challenge_generalist_lora_clean_desktop`
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
- `scripts/compute_norm_stats.py` now uses `SafeDataset` and respects its local shuffle flag, so isolated bad samples no longer crash worker batches or leave norm-stat accumulation empty; empty fully-skipped batches are also ignored by `TorchDataLoader`.
- `scripts/eval_offline.py` now shows a per-checkpoint tqdm progress bar during validation-batch evaluation.
- `TorchDataLoader` now skips incomplete post-filter batches as well, preventing multi-device `device_put` failures when `SafeDataset` drops bad samples and the remaining batch size is no longer divisible by the data-parallel mesh size.
- Root cause identified for the large number of post-split “bad samples”: the current LeRobot version uses global `episode_index` values to index a local `episode_data_index` array after subsetting episodes, which breaks on non-contiguous episode subsets. `src/openpi/training/data_loader.py` now patches selected-episode LeRobot datasets so query/padding logic remaps global episode ids to local subset positions while preserving the original `episode_index` in returned samples.
- Added `scripts/run_norm_and_train.py` to sequentially run `compute_norm_stats.py` and then `train.py --overwrite=true` in one command, intended for unattended overnight finetuning starts after the split/data-loader fixes.
- Added a shell-first unattended runner `scripts/run_norm_and_train.sh`; `scripts/train.sh` now forwards extra CLI args to `train.py` and normalizes the common typo `--overwirte` to `--overwrite=true`.

## Known Open Work

- Finish or verify full extraction of all dataset parts.
- Validate a small debug training run on the actual target machine with `grad_accum_steps > 1`.
- Validate the new episode-split training path end-to-end on the clean-desktop smoke config and confirm validation loss logs as expected.
- Run the new offline evaluator on a real clean-desktop experiment directory and confirm the JSON/CSV outputs match expectations.
- Confirm the new `train_metrics.jsonl` contents and W&B curves on a real clean-desktop run, especially per-task task-name recovery and batch-level action/gripper metrics.
- Extract adapters from a real generalist/specialist checkpoint set and verify routed loading against those files.
- Validate the `AdapterRouted` CLI path with a real checkpoint + adapter directory.
- Measure task-switch latency once real adapter files are available.
- Re-run `scripts/eval_offline.py` on the previously failing clean-desktop checkpoint and confirm it now gets past dataset construction into actual metric computation.

## Verification Notes

- `python3 -m py_compile` passes for the modified model/config/training files.
- `python3 -m py_compile` passes for the modified serving and adapter files.
- `python3 -m py_compile` passes for the new episode-split and validation files.
- `python3 -m py_compile` passes for the new offline evaluation files and the added model hooks.
- `python3 -m py_compile` passes for the new training logging changes in `scripts/train.py` and `src/openpi/training/data_loader.py`.
- `git diff --check` passes.
- `uv run pytest ...` could not be completed in this environment because dependency resolution attempted network access and failed on DNS/package download.
- Direct Python smoke execution of the new split helpers is currently blocked in this environment because the installed `ml_dtypes` version is too old for JAX import (`0.4.1`, JAX requires `>=0.5`).
