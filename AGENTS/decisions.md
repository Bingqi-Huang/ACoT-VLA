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
