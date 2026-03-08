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

Date: 2026-03-08

Decision:

- Use a dedicated smoke-submission Docker path instead of mutating the repo's generic serving Dockerfile.
- Keep the smoke submission on the existing G2SIM ACoT websocket serving path.
- Use the official baseline image as the smoke submission parent image and declare the submission as `abs_joint`.

Why:

- The first submission goal is interface and evaluator validation, not model iteration.
- A separate Dockerfile and startup script reduce the risk of breaking existing local serving and training workflows.
- The official docs already describe the baseline image and the submission contract for websocket serving on port `8999`.

Consequence:

- Smoke-submission work should stay limited to packaging, startup, and observability.
- Any later custom-checkpoint submission can reuse the same smoke path or branch from it without reworking the websocket interface.
