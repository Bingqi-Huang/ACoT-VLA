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

Date: 2026-03-08

Author: Codex

What changed:

- Added `AGENTS/` shared workspace docs.
- Established the main submission constraint: final evaluation uses a Dockerized websocket server on port `8999`.
- Fixed two blocking LoRA issues for the planned challenge setup:
  - `gemma_300m_lora` is now part of the Gemma variant type alias.
  - `ACOTConfig.get_freeze_filter()` now keeps coarse-expert LoRA trainable.
- Added `grad_accum_steps` to `TrainConfig` and refactored `scripts/train.py` into compute/apply-grad phases so gradient accumulation works for both standard and ACoT training.
- Added targeted tests in `src/openpi/models/acot_vla_test.py`.

What was verified:

- Official docs confirm submission requirements and model-type declaration (`abs_joint` or `abs_pose`).
- Repo inference path loads checkpoint directories with `params` and `assets`.
- `python3 -m py_compile` passes on the modified files.
- `git diff --check` passes.

What is still broken or unknown:

- No final submission image path has been implemented yet for a custom checkpoint.
- No verified local closed-loop evaluation run has been recorded yet in this folder.
- `uv run pytest` is currently blocked in this environment because dependency resolution tries to reach external package mirrors and fails.
- The new gradient accumulation path has not yet been exercised on actual hardware.

Immediate next step:

- Add the generalist + specialist LoRA configs from `AGENTS/PLAN.md`, then run a short debug training job with accumulation enabled.

Git note:

- Commit substantial progress before handoff so the next agent can diff exact changes instead of reconstructing them from notes.
