# Status

Last updated: 2026-03-08

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

## Known Open Work

- Finish or verify full extraction of all dataset parts.
- Validate a small debug training run on the actual target machine with `grad_accum_steps > 1`.
- Extract adapters from a real generalist/specialist checkpoint set and verify routed loading against those files.
- Validate the `AdapterRouted` CLI path with a real checkpoint + adapter directory.
- Measure task-switch latency once real adapter files are available.

## Verification Notes

- `python3 -m py_compile` passes for the modified model/config/training files.
- `python3 -m py_compile` passes for the modified serving and adapter files.
- `git diff --check` passes.
- `uv run pytest ...` could not be completed in this environment because dependency resolution attempted network access and failed on DNS/package download.
