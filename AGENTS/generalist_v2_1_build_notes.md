# Generalist-v2.1 Build Notes

## Scope

This note records the fixes required to produce a working Docker image for the generalist checkpoint trained with `acot_challenge_generalist_continued` and served through `policy:checkpoint`.

Relevant files:

- `scripts/docker/serve_generalist_v2.Dockerfile`
- `scripts/serve_policy.py`
- `src/openpi/training/checkpoints.py`
- `AGENTS/docker_build.md`

## Final verified outcome

The following flow was verified locally:

1. `docker build --no-cache -f scripts/docker/serve_generalist_v2.Dockerfile -t generalist-v2:latest .`
2. `docker run --rm --network=host --gpus "device=1" -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 generalist-v2:latest`

Observed successful runtime milestones:

- checkpoint restored from `/submission/checkpoint/generalist_continued/params`
- norm stats loaded from `/submission/checkpoint/generalist_continued/assets/reasoning2action_sim_generalist`
- websocket server listening on `0.0.0.0:8999`

## Problems encountered and fixes

### 1) `pip: not found` during Docker build

Symptom:

- `RUN pip install --upgrade "jax[cuda12]" -q` failed because the base image does not expose a standalone `pip` in the Docker build environment.

Fix:

- standardize on `uv` for package management inside the project environment
- set `UV_PROJECT_ENVIRONMENT=/submission/.venv` so the environment path is deterministic inside the image

### 2) `.venv/bin/python` not found after `uv sync`

Symptom:

- `uv pip install --python /submission/.venv/bin/python ...` failed because `uv sync` created the environment at `.venv` relative to the working directory unless explicitly configured.

Fix:

- set `ENV UV_PROJECT_ENVIRONMENT=/submission/.venv`
- derive `PATH` from `${UV_PROJECT_ENVIRONMENT}/bin`

### 3) `lerobot` Git fetch blocked Docker builds

Symptom:

- `uv sync --frozen --no-dev` attempted to resolve the Git dependency `lerobot`
- initial failure came from Git LFS smudging test artifacts
- later failure came from unstable GitHub HTTP/2 fetches during Docker build

Root cause:

- `lerobot` is a training-time dependency in this repository but is not required for checkpoint-only policy serving

Fix:

- add `GIT_LFS_SKIP_SMUDGE=1` to the `uv sync` step
- add `--no-install-package lerobot` to the same `uv sync` step so the serving image does not depend on cloning that Git repository

### 4) JAX CUDA plugin mismatch at runtime

Symptom:

- runtime logs showed both CUDA 13 and CUDA 12 plugin initialization attempts
- errors included PJRT API mismatch and duplicate plugin registration

Root cause:

- `uv.lock` pins `jax[cuda13]==0.7.2` for local development
- installing `jax[cuda12]` on top without removing the CUDA 13 packages left mixed PJRT plugins in the same environment

Fix:

- explicitly uninstall `jax-cuda13-plugin`, `jax-cuda13-pjrt`, `jax`, and `jaxlib`
- reinstall `jax[cuda12]==0.7.2`
- pinning to `0.7.2` keeps the JAX package set aligned with the locked project version

### 5) `ModuleNotFoundError: lerobot` at container startup

Symptom:

- after skipping `lerobot` installation, the container still failed on startup because importing the serving code pulled in training-only modules that import `lerobot`

Root cause:

- `scripts/serve_policy.py` imported `adapter_routed_policy` at module import time, even when serving checkpoint-only policy mode
- `src/openpi/training/checkpoints.py` imported `openpi.training.data_loader` at module import time for type annotations
- `openpi.training.data_loader` imports `lerobot`

Fix:

- make `adapter_routed_policy` a lazy import inside the `AdapterRouted` match arm in `scripts/serve_policy.py`
- gate `openpi.training.data_loader` behind `TYPE_CHECKING` in `src/openpi/training/checkpoints.py`
- keep the runtime annotations as string annotations so serving code does not import dataset code during startup

## Files changed for the final solution

### `scripts/docker/serve_generalist_v2.Dockerfile`

Key changes:

- set `UV_PROJECT_ENVIRONMENT=/submission/.venv`
- run `GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --no-install-package lerobot`
- replace the locked CUDA 13 JAX stack with `jax[cuda12]==0.7.2`

### `scripts/serve_policy.py`

Key changes:

- lazy import `adapter_routed_policy` only when `policy:adapter-routed` is selected

### `src/openpi/training/checkpoints.py`

Key changes:

- move `openpi.training.data_loader` to a `TYPE_CHECKING` import
- keep `DataLoader` references as string annotations

### `AGENTS/docker_build.md`

Key changes:

- switch the documented build target to the non-routing generalist image
- update image naming to `generalist-v2.1`
- record the JAX / lerobot decisions at a high level

## Operational guidance

- keep the serving image focused on checkpoint inference; do not reintroduce training-only dependencies unless they are actually imported by the serving path
- if the locked JAX version changes in `pyproject.toml` / `uv.lock`, update the Dockerfile override so CUDA 12 uses the same base version
- if routed serving is revived later, verify separately whether the routed path still needs the lazy import and `TYPE_CHECKING` changes to stay independent of training dataset packages
