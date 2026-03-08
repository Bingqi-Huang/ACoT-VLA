# Constraints

This document records what must not be broken while iterating on training or model design.

## Hard Submission Contract

These are the constraints that matter at competition submission time.

- The final submission artifact is a Docker image.
- The Docker image must contain all runtime dependencies, inference code, and model artifacts needed to run.
- The Docker image must start directly via `ENTRYPOINT` or `CMD`; no manual post-start commands are allowed.
- The inference service must listen on port `8999`.
- The inference service must be exposed as a websocket policy server.
- The policy must accept the observation payload expected by Genie Sim / the Reasoning2Action evaluator.
- The policy must return one of the supported command types:
  - `abs_joint`
  - `abs_pose`
- The submission page requires selecting the model type consistent with the actual output type.
- The pushed image must be tagged under the registry endpoint and namespace provided by the test server.
- The image must be pullable and runnable by the evaluation system without private local dependencies.

## Git And Change Management

- Use Git to track all meaningful code and doc changes.
- Keep commits scoped and readable.
- Avoid destructive history edits unless explicitly requested.
- Before ending a work session, update `AGENTS/status.md` or `AGENTS/handoff.md` if context changed.

## What Can Change

These are generally safe to change if the hard submission contract above is preserved.

- Training method:
  - full finetuning
  - LoRA finetuning
  - mixed strategies
  - distillation
  - curriculum changes
  - different loss functions
- Dataset composition and sampling strategy.
- Prompt engineering and prompt replacement strategy during training.
- Checkpointing strategy.
- Model architecture, including:
  - backbone replacement
  - extra heads
  - action reasoning modules
  - tokenization scheme
  - latent interfaces

## What Cannot Change Without Rebuilding The Submission Layer

These changes are allowed only if the serving layer is updated to remain compliant.

- Observation schema expected by the model.
- Action representation emitted by the model.
- Policy transport protocol.
- Port number.
- Checkpoint loading method.
- Required normalization assets.
- GPU / memory assumptions for runtime inference.

## Practical Implication

Training is flexible.

Submission is not flexible.

If architecture changes are made, the serving layer must still:

- transform evaluator observations into model inputs
- load the trained checkpoint correctly
- produce valid output commands in the declared format
- serve over websocket on port `8999`

## Current Repo-Specific Notes

- The baseline serving entrypoint is `scripts/serve_policy.py`.
- The default Docker serving file is `scripts/docker/serve_policy.Dockerfile`.
- Default server launch currently goes through `scripts/server.sh`.
- Inference in this repo loads a trained checkpoint directory containing:
  - `params`
  - `assets`
- If training config changes, submission startup must point to the matching config and checkpoint.

## Official External References

- Competition quick start:
  `https://agibot-world.com/challenge2026/reasoning2action/quick-start`
- Genie Sim guide, integrate policy:
  `https://agibot-world.com/sim-evaluation/docs/#/v3?id=_353-integrate-your-own-policy`
- Genie Sim guide, submit policy:
  `https://agibot-world.com/sim-evaluation/docs/#/v3?id=_354-submit-your-policy`
