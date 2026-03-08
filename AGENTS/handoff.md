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
- No real adapter `.npz` files have been extracted and smoke-tested yet.
- The refactored routed adapter policy still needs runtime validation with real adapter files to confirm switch latency and no unexpected recompiles.

Immediate next step:

- Run a short debug training job for `acot_challenge_generalist_lora_all`, then extract one adapter and smoke-test `scripts/server_routed.sh` against it.

Git note:

- Commit substantial progress before handoff so the next agent can diff exact changes instead of reconstructing them from notes.
