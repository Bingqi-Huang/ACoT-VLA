# Training Runbook

This is a placeholder runbook for future agents.

## Scope

- Environment setup
- Dataset extraction
- Norm stats generation
- Debug training
- Full training
- Checkpoint validation

## To Fill In

- Exact environment bootstrap commands
- Machine-specific overrides
- Recommended batch size / FSDP settings by hardware tier
- How to interpret early failures
- Where checkpoints are stored
- Which checkpoint step should be exported for submission

## Minimal Skeleton

1. Verify environment.
2. Verify extracted LeRobot dataset roots exist.
3. Verify norm stats exist for the chosen config.
4. Run a debug job.
5. Inspect checkpoint outputs.
6. Run the full training job.
7. Record results in `AGENTS/experiments.md`.
