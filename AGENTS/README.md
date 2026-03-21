# AGENTS Workspace Handbook

This directory is the shared operating memory for all coding agents working in this repository.

Every agent must read this folder before making substantial changes. At minimum, read:

1. `AGENTS/README.md`
2. `AGENTS/constraints.md`
3. `AGENTS/status.md`
4. `AGENTS/handoff.md`

## Git

- Use Git for version control and code management.
- Treat this repository as the source of truth for code, docs, and operational state.
- Commit meaningful progress in small, reviewable units.
- Do not rewrite history unless explicitly requested.
- Do not revert user changes unless explicitly requested.
- Record major workflow decisions in `AGENTS/decisions.md`.

## Purpose

- Preserve project constraints that must not be violated.
- Record current status, active assumptions, and open problems.
- Give future agents a stable place to continue work without re-discovering context.
- Separate long-lived operational knowledge from ad hoc chat history.

## Required Agent Behavior

- Read the required files above before editing training, inference, Docker, or submission code.
- Update `AGENTS/status.md` after any meaningful change in setup, training, inference, or submission workflow.
- Update `AGENTS/handoff.md` before ending a substantial work session.
- Record irreversible or high-impact decisions in `AGENTS/decisions.md`.
- If you discover a new hard constraint, add it to `AGENTS/constraints.md`.

## File Map

- `constraints.md`
  Hard constraints for training, inference, packaging, and competition submission.
- `status.md`
  Current project status, known working paths, and active blockers.
- `handoff.md`
  Short operational notes for the next agent.
- `decisions.md`
  Decision log with rationale and consequences.
- `runbook_training.md`
  Training workflow and operational checklist.
- `debug_cuda_illegal_address_2026-03-21.md`
  Blackwell CUDA illegal-address incident log and the confirmed fix path.
- `runbook_submission.md`
  Submission packaging and registry push checklist.
- `experiments.md`
  Placeholder tracker for experiment outcomes.
- `questions.md`
  Open questions that still require validation.

## Editing Rules For This Folder

- Keep entries factual and concise.
- Prefer dated entries when appending status or handoff notes.
- Do not delete historical notes unless they are clearly wrong; mark them outdated instead.
- Use checklists where operational steps matter.
- When unsure, append rather than rewrite.
