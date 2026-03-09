# Decisions

Append important decisions here with rationale and consequences.

## Template

Date:

Decision:

Why:

Consequence:

## Seed Entries

Date: 2026-03-08

Decision:

- Treat the competition as submission-contract-constrained, not training-method-constrained.

Why:

- Official docs only constrain the final Dockerized websocket inference interface and output command type.

Consequence:

- Training and architecture may change, but inference packaging must remain compliant.

Date: 2026-03-08

Decision:

- Use `AGENTS/` as the persistent operating memory for future agents.

Why:

- Chat history is not a reliable long-term state store for multi-session agentic work.

Consequence:

- Future agents should update this folder after significant progress.

Date: 2026-03-09

Decision:

- Treat LeRobot finetuning validation as episode-split-constrained rather than sample-split-constrained.

Why:

- Frame-level or shuffled sample-level validation leaks temporal context across train and validation and is not meaningful for the Reasoning2Action smoke finetuning tasks.

Consequence:

- Train/validation splits are now generated per dataset by `episode_index`, saved to JSON manifests, reused across training/stat computation, and validation should only be interpreted when norm stats are computed from the train split.

Date: 2026-03-09

Decision:

- Use teacher-forced offline checkpoint evaluation as the first-line checkpoint selection signal for the Reasoning2Action finetuning workflow.

Why:

- Closed-loop simulation rollout is slower, more operationally fragile, and unnecessary for every checkpoint when the immediate goal is to compare finetuning progress on held-out episodes without leakage.

Consequence:

- Checkpoints can now be ranked from the val episode split using JSON metrics and a summary CSV before investing in heavier rollout-based validation or submission packaging.
