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

What was verified:

- Official docs confirm submission requirements and model-type declaration (`abs_joint` or `abs_pose`).
- Repo inference path loads checkpoint directories with `params` and `assets`.

What is still broken or unknown:

- No final submission image path has been implemented yet for a custom checkpoint.
- No verified local closed-loop evaluation run has been recorded yet in this folder.

Immediate next step:

- Choose the training strategy for available hardware, then wire the matching checkpoint into a submission Docker image.

Git note:

- Commit substantial progress before handoff so the next agent can diff exact changes instead of reconstructing them from notes.
