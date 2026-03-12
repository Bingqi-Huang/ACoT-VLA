# Submission Runbook

## Purpose

This document is for coding agents preparing the next Docker submission for the
ICRA Reasoning2Action challenge. It captures the workflow that actually worked
and the structural mistakes to avoid.

The recommended baseline submission path is a single-checkpoint generalist
image. Adapter-routed submission can reuse most of this flow, but the image
packaging rules below still apply.

## Non-Negotiable Constraints

- Final image must auto-start the inference server via `CMD` or `ENTRYPOINT`.
- Final server must listen on port `8999`.
- Submission form model type for the current generalist policy is `abs_joint`.
- The image must contain everything needed for inference:
  - Python runtime already present in the base image
  - project code
  - checkpoint `params`
  - checkpoint `assets`
  - tokenizer files and other model assets needed at runtime
- Do not rely on runtime dependency installation.
- Do not rely on runtime downloads from GitHub, GCS, or other remote endpoints.

## Critical Packaging Lessons

- Do not overlay the submission repo into `/app`.
  - The official base image already contains a full repo checkout at `/app`.
  - `COPY . /app` mixes the submission tree with base-image files and makes the
    runtime ambiguous.
  - Use an isolated directory such as `/submission` instead.
- Do not use `uv run` in the container start command.
  - In practice this triggered runtime package sync/build work inside the
    container.
  - Start the server with plain `python` and an explicit `PYTHONPATH` that
    points at the submission tree.
- Do not leave tokenizer download to first boot.
  - Pre-fetch tokenizer assets during `docker build`.
- Do not ship training-only checkpoint state.
  - `train_state/` is not needed for serving and only increases image size.

## Submission Branch Workflow

Prepare submission work on a dedicated branch. Do not prune files directly on
the active training branch.

Recommended sequence:

```bash
git switch -c submit-<submission_name>
git add <serve_fixes_and_required_files>
git commit -m "Prepare <submission_name> submission snapshot"
git push -u origin submit-<submission_name>
```

Use this branch to freeze the exact code revision that was already verified with
local checkpoint serving.

Rules:

- Only branch after a real local serve smoke test has passed.
- Keep the training branch intact.
- Treat the submission branch as disposable packaging state.
- Record the final commit SHA used for the image.

## Isolated Submission Checkout

Do Docker packaging in a separate clone or worktree built from the submission
branch, not in the training checkout.

Example:

```bash
git clone <repo_url> /path/to/submit-repo
cd /path/to/submit-repo
git fetch origin
git switch --track origin/submit-<submission_name>
```

Alternative:

```bash
git worktree add /path/to/submit-repo submit-<submission_name>
```

Goal:

- training checkout remains untouched
- submission checkout can be aggressively pruned
- Docker build context only sees the packaging tree

## Submission Tree Pruning

After entering the isolated submission checkout, remove non-submission material
that is not needed for serving.

Preview first:

```bash
git rm -r -n AGENTS docs examples .github
git rm -n README.md Training_Notes.md log
git clean -fdn
```

Then apply the pruning:

```bash
git rm -r AGENTS docs examples .github
git rm README.md Training_Notes.md log
git clean -fd
git commit -m "Prune submission tree to serving-only contents"
```

Notes:

- Only do this in the isolated submission checkout or dedicated submission
  branch.
- Do not do this on `main` or the active training branch.
- `git clean -fd` removes untracked files and directories, so always preview
  with `git clean -fdn` first.
- The goal is not aesthetic cleanup; it is to keep the build context minimal and
  deterministic.

## Recommended Workspace Layout

Prepare submission in an isolated repo clone or worktree, not in the active
training checkout.

Expected top-level contents in the submission repo:

- `packages/`
- `scripts/`
- `src/`
- `third_party/`
- `pyproject.toml`
- `uv.lock`
- `.python-version` if present in repo
- `checkpoint/<submission_checkpoint>/`

Keep the checkpoint under `checkpoint/` (singular), not `checkpoints/`.
Current `.dockerignore` excludes `checkpoints/`, so a copied checkpoint there
will not enter the image.

## Checkpoint Staging

Copy the chosen final checkpoint into the submission repo under a dedicated
directory, for example:

```bash
mkdir -p checkpoint/generalists_v1_bs96_step5000
rsync -aH /path/to/original/checkpoint/5000/ checkpoint/generalists_v1_bs96_step5000/
rm -rf checkpoint/generalists_v1_bs96_step5000/train_state
```

Required contents after staging:

- `checkpoint/generalists_v1_bs96_step5000/params`
- `checkpoint/generalists_v1_bs96_step5000/assets`

Do not remove checkpoint `assets`; policy loading needs normalization stats from
that tree.

## Build Context Hygiene

Use a strict `.dockerignore`. Keep `checkpoint/` included, but exclude large or
irrelevant local material:

```text
.venv
checkpoints
data
.git
__pycache__
*.pyc
.pytest_cache
.cache
.vscode
wandb
logs
```

Do not rely on `COPY . /...` if the repo contains extra local material you do
not intend to ship. Prefer explicit `COPY` lines in the Dockerfile.

## Submission Start Script

Create a submission-specific launcher, for example
`scripts/server_submit_generalist.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:-0}
port=${2:-8999}

export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-16}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${gpu_id}}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/root/.cache/openpi}
export PYTHONPATH="/submission:/submission/src:/submission/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export ACOT_SERVE_CONFIG=${ACOT_SERVE_CONFIG:-acot_challenge_generalist_lora_generalist}
export ACOT_SERVE_CHECKPOINT=${ACOT_SERVE_CHECKPOINT:-/submission/checkpoint/generalists_v1_bs96_step5000}

cd /submission
exec python scripts/serve_policy.py \
  --port "${port}" \
  policy:checkpoint \
  --policy.config "${ACOT_SERVE_CONFIG}" \
  --policy.dir "${ACOT_SERVE_CHECKPOINT}"
```

Important properties of this launcher:

- Starts from `/submission`, not `/app`
- Uses plain `python`, not `uv run`
- Uses `policy:checkpoint` subcommand syntax
- Keeps XLA memory fraction externally overridable

## Submission Dockerfile

Create a dedicated submission Dockerfile, for example
`scripts/docker/serve_policy.generalists_v1.Dockerfile`:

```dockerfile
FROM sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest

WORKDIR /submission

COPY pyproject.toml uv.lock .python-version LICENSE /submission/
COPY packages /submission/packages
COPY scripts /submission/scripts
COPY src /submission/src
COPY third_party /submission/third_party
COPY checkpoint /submission/checkpoint

RUN chmod +x /submission/scripts/server_submit_generalist.sh

ENV OPENPI_DATA_HOME=/root/.cache/openpi
ENV ACOT_SERVE_CONFIG=acot_challenge_generalist_lora_generalist
ENV ACOT_SERVE_CHECKPOINT=/submission/checkpoint/generalists_v1_bs96_step5000
ENV PYTHONPATH=/submission:/submission/src:/submission/packages/openpi-client/src

RUN python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"

EXPOSE 8999

CMD ["/bin/bash", "-lc", "/submission/scripts/server_submit_generalist.sh 0 8999"]
```

Rationale:

- Reuses the official base image for CUDA/JAX/runtime compatibility
- Keeps the submission tree isolated from the base-image repo
- Copies only the files needed for serving
- Downloads tokenizer assets at build time so first boot is offline

## Build Procedure

Run from the submission repo root:

```bash
docker build \
  -f scripts/docker/serve_policy.generalists_v1.Dockerfile \
  -t acot-generalists-v1-step5000:local \
  .
```

Do not assume the base image is empty. It is not.

## Required Smoke Tests

### 1. Offline boot test

Purpose: catch accidental runtime downloads or dependency sync.

```bash
docker run --rm --gpus all --network=none acot-generalists-v1-step5000:local
```

Watch container logs. The server should boot without messages of this form:

- `Updated https://github.com/...`
- `Built openpi ...`
- `Installed ... packages`
- `Downloading gs://...`

Terminate after the server reaches the listening state.

### 2. Host-network health check

```bash
docker run -it --rm --network=host --gpus all \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
  acot-generalists-v1-step5000:local
```

In another terminal:

```bash
curl http://127.0.0.1:8999/healthz
```

Expected result:

- container log includes `server listening on 0.0.0.0:8999`
- `curl` returns `OK`

### 3. Optional local simulation smoke test

If Genie Sim is available locally, run one short task pass against the local
server before pushing the image.

## Submission Steps

Once the image passes the smoke tests:

```bash
docker login sim-icra-registry.cn-beijing.cr.aliyuncs.com
docker tag acot-generalists-v1-step5000:local sim-icra-registry.cn-beijing.cr.aliyuncs.com/<namespace>/acot-generalists-v1-step5000:v1
docker push sim-icra-registry.cn-beijing.cr.aliyuncs.com/<namespace>/acot-generalists-v1-step5000:v1
```

Paste the full image URL into the submission page and choose model type
`abs_joint`.

## Final Pre-Push Checklist

- Submission repo is isolated from the training checkout
- Checkpoint lives under `checkpoint/`, not `checkpoints/`
- `train_state/` removed from the shipped checkpoint
- Dockerfile uses `/submission`, not `/app`
- Launcher uses `python`, not `uv run`
- Tokenizer is fetched during build
- Image starts automatically via `CMD`
- Service listens on port `8999`
- `curl /healthz` returns `OK`
- No runtime dependency install or remote download appears in logs
