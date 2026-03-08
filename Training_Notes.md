# Clean Desktop Test-Server Plan

This note is for a **single-model test submission** using `acot_challenge_generalist_lora_clean_desktop`.

It is **not** the full routed-adapter plan. For this test, the shortest correct path is:

1. compute norm stats for the clean-desktop config
2. run a short debug train
3. run the real clean-desktop train
4. serve that checkpoint directly as a websocket policy
5. package it into a Docker image
6. push the image and submit the full image URL on the official test server

## Why the previous plan was wrong

The earlier version of this note assumed:

- full 9-task generalist training
- specialist finetuning
- adapter extraction
- adapter-routed serving

That is the long-term competition plan, but it is **not** the right plan for a quick test with `acot_challenge_generalist_lora_clean_desktop`.

For the clean-desktop test, use a **plain checkpoint-serving path**, not the routed-adapter path.

## 1. Prepare the environment

Make sure the extracted dataset root exists, or set:

```bash
export ACOT_CHALLENGE_DATA_ROOT=~/Datasets/lerobot/Reasoning2Action-Sim
```

If you want to override the base initialization checkpoint, set:

```bash
export ACOT_CHALLENGE_INIT_WEIGHTS=/path/to/base/params
```

Optional local `uv` cache:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
mkdir -p "${UV_CACHE_DIR}"
```

## 2. Compute norm stats for the clean-desktop config

This must be run before training:

```bash
uv run python scripts/compute_norm_stats.py \
    --config-name acot_challenge_generalist_lora_clean_desktop
```

The config writes stats under the asset id for the clean-desktop config.

## 3. Run a short debug training job

Use debug mode first:

```bash
DEBUG_MODE=true uv run python scripts/train.py \
    --config-name acot_challenge_generalist_lora_clean_desktop \
    --exp_name clean_desktop_debug \
    --overwrite
```

Check:

- training starts
- loss logs appear
- no immediate OOM
- a debug checkpoint is written

## 4. Run the real clean-desktop training job

When the debug run is clean:

```bash
bash scripts/train.sh acot_challenge_generalist_lora_clean_desktop clean_desktop_v1
```

Or directly:

```bash
uv run python scripts/train.py \
    --config-name acot_challenge_generalist_lora_clean_desktop \
    --exp_name clean_desktop_v1
```

Expected checkpoint root:

```bash
./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_v1
```

Expected final-step checkpoint example:

```bash
./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_v1/50000
```

If you choose an earlier step, use the real step directory everywhere below.

## 5. Serve the trained checkpoint locally

Set the checkpoint to serve:

```bash
export ACOT_SERVE_CONFIG=acot_challenge_generalist_lora_clean_desktop
export ACOT_SERVE_CHECKPOINT=./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_v1/50000
```

Start the websocket server:

```bash
bash scripts/server_checkpoint.sh 0 8999
```

This uses:

- config: `acot_challenge_generalist_lora_clean_desktop`
- policy mode: checkpoint
- port: `8999`

## 6. Smoke test locally before Docker

Health check:

```bash
curl -L http://127.0.0.1:8999/healthz
```

What to verify:

- the server auto-starts
- `/healthz` returns `200`
- the checkpoint loads successfully
- websocket inference works with a clean-desktop observation payload

For this test submission, the model is still a **single checkpoint model**. No adapters are needed.

## 7. Build a Docker image for the official test server

The official docs require a Dockerized policy server that starts automatically and serves on port `8999`.

Build:

```bash
docker build -f scripts/docker/serve_policy.Dockerfile -t acot-clean-desktop-test .
```

Run locally:

```bash
docker run --rm -p 8999:8999 \
    -e SERVER_SCRIPT=./scripts/server_checkpoint.sh \
    -e ACOT_SERVE_CONFIG=acot_challenge_generalist_lora_clean_desktop \
    -e ACOT_SERVE_CHECKPOINT=./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_v1/50000 \
    acot-clean-desktop-test
```

Re-check:

```bash
curl -L http://127.0.0.1:8999/healthz
```

## 8. Submission notes from the official docs

For the official test server / submission flow:

- submit a **Docker image**
- the image must auto-start the policy server
- the service must listen on port `8999`
- the policy interface is websocket-based
- the submission form expects the **full image URL**
- choose the output type consistent with the model output

For this repo and this clean-desktop checkpoint path, the expected output type is:

```text
abs_joint
```

because the policy returns joint-space action commands rather than pose commands.

Official references:

- `https://agibot-world.com/challenge2026/reasoning2action/quick-start`
- `https://agibot-world.com/sim-evaluation/docs/#/v3?id=_354-submit-your-policy`

## 9. What must be inside the image

Before pushing, verify the image contains:

- the trained checkpoint directory
- checkpoint `params`
- checkpoint `assets`
- repo code and runtime dependencies
- startup path that launches the websocket server on `8999`

For this single-model test, adapters are **not required**.

## 10. Recommended immediate commands

```bash
export ACOT_CHALLENGE_DATA_ROOT=~/Datasets/lerobot/Reasoning2Action-Sim
export UV_CACHE_DIR=/tmp/uv-cache
mkdir -p "${UV_CACHE_DIR}"
uv run python scripts/compute_norm_stats.py --config-name acot_challenge_generalist_lora_clean_desktop
DEBUG_MODE=true uv run python scripts/train.py --config-name acot_challenge_generalist_lora_clean_desktop --exp_name clean_desktop_debug --overwrite
```
