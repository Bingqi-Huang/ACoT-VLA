# Status

Last updated: 2026-03-08

## Current State

- Repository has been examined for Reasoning2Action competition training and submission flow.
- Local dataset extraction helper has been added:
  - `scripts/prepare_reasoning2action_sim.sh`
- Local challenge config has been added:
  - `acot_icra_simulation_challenge_reasoning_to_action_local`
- Normalization stats script has been patched so stats are saved to the asset directory used by training.
- A dedicated smoke-submission path has been added:
  - `scripts/docker/smoke_submission.Dockerfile`
  - `scripts/server_smoke_submission.sh`
- The serving path now includes safe smoke instrumentation:
  - startup timing
  - summarized websocket payload keys
  - `task_name` presence/value
  - truncated prompt preview
  - inference and total request timing

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
- The current smoke submission intentionally stays on the existing G2SIM ACoT serving path.
- The smoke submission should be declared as `abs_joint`.
- The smoke image is based on the official baseline parent image documented by Genie Sim.

## Known Open Work

- Finish or verify full extraction of all dataset parts.
- Validate a small debug training run on the actual target machine.
- Decide the training strategy for limited GPU hardware:
  - baseline-style
  - reduced batch full finetune
  - more aggressive LoRA
- Verify locally that the official baseline parent image is accessible with the registry token on the target machine.
- Run the first platform smoke submission and inspect logs for real evaluator payload/task naming behavior.
