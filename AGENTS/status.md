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
- Add the generalist + specialist challenge configs described in `AGENTS/PLAN.md`.
- Implement adapter extraction and adapter-routed serving.
- Create a dedicated submission Docker path for the final chosen model.

## Verification Notes

- `python3 -m py_compile` passes for the modified model/config/training files.
- `git diff --check` passes.
- `uv run pytest ...` could not be completed in this environment because dependency resolution attempted network access and failed on DNS/package download.
