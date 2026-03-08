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
- Added a dedicated smoke submission image path:
  - `scripts/docker/smoke_submission.Dockerfile`
  - `scripts/server_smoke_submission.sh`
- Added safe smoke instrumentation to the websocket serving path so platform logs show:
  - top-level payload keys
  - `task_name` present/absent and value
  - prompt preview
  - inference timing and startup timing

What was verified:

- Official docs confirm submission requirements and model-type declaration (`abs_joint` or `abs_pose`).
- Repo inference path loads checkpoint directories with `params` and `assets`.
- The repo’s G2SIM serving path remains the basis of the smoke submission; no routing or specialist logic was added.

What is still broken or unknown:

- The official baseline parent image has not yet been validated locally in this environment.
- No verified local closed-loop Genie Sim evaluation run has been recorded in this folder.
- The real evaluator payload still needs confirmation from the first submission logs.

Immediate next step:

- Build the smoke image, run the local health check, push it to the test-server registry, and inspect the first submission logs.

Git note:

- Commit substantial progress before handoff so the next agent can diff exact changes instead of reconstructing them from notes.
