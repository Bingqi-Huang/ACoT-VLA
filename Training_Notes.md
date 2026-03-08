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

---

# Full Routed-Adapter Plan

This is the **full competition plan**. Keep this as the long-term path after the clean-desktop test-server workflow above.

## 1. Prepare the environment

Make sure your extracted dataset root exists under `~/Datasets/lerobot/Reasoning2Action-Sim` or set:

```bash
export ACOT_CHALLENGE_DATA_ROOT=/your/Reasoning2Action-Sim
```

If you want a different base init checkpoint, set:

```bash
export ACOT_CHALLENGE_INIT_WEIGHTS=/path/to/base/params
```

## 2. Compute generalist norm stats

This must run before training the full generalist config:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/compute_norm_stats.py \
    --config-name acot_challenge_generalist_lora_all
```

This should write stats into the assets dir used by the config.

## 3. Run a short generalist debug train

First smoke test the accumulation path:

```bash
DEBUG_MODE=true UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py \
    --config-name acot_challenge_generalist_lora_all \
    --exp_name generalist_debug \
    --overwrite
```

Check:

- training starts
- no OOM
- loss logs appear
- checkpoint is written

## 4. Run a real generalist train

If debug is clean:

```bash
bash scripts/train.sh acot_challenge_generalist_lora_all generalist_v1
```

Or directly:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py \
    --config-name acot_challenge_generalist_lora_all \
    --exp_name generalist_v1
```

Wait until you have a usable checkpoint, ideally the target final step.

## 5. Point specialists at the real generalist checkpoint

Export the checkpoint path:

```bash
export ACOT_CHALLENGE_GENERALIST_WEIGHTS=./checkpoints/acot_challenge_generalist_lora_all/generalist_v1/50000/params
```

If your best step is not `50000`, use the real step directory.

## 6. Run one specialist debug job first

Example:

```bash
DEBUG_MODE=true UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py \
    --config-name acot_specialist_pour_workpiece \
    --exp_name specialist_debug \
    --overwrite
```

Check the same things as the generalist debug run.

## 7. Train the specialists

One by one or in parallel:

```bash
bash scripts/train.sh acot_specialist_pour_workpiece specialist_v1
bash scripts/train.sh acot_specialist_open_door specialist_v1
bash scripts/train.sh acot_specialist_scoop_popcorn specialist_v1
bash scripts/train.sh acot_specialist_hold_pot specialist_v1
bash scripts/train.sh acot_specialist_place_block specialist_v1
bash scripts/train.sh acot_specialist_take_wrong_item specialist_v1
bash scripts/train.sh acot_specialist_stock_shelf specialist_v1
bash scripts/train.sh acot_specialist_sorting specialist_v1
bash scripts/train.sh acot_specialist_clean_desktop specialist_v1
```

## 8. Extract adapters

Create the adapter directory:

```bash
mkdir -p adapters
```

Extract the fallback generalist adapter:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/extract_adapter.py \
    --checkpoint ./checkpoints/acot_challenge_generalist_lora_all/generalist_v1/50000 \
    --output ./adapters/_default.npz
```

Extract each specialist adapter:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/extract_adapter.py \
    --checkpoint ./checkpoints/acot_specialist_pour_workpiece/specialist_v1/10000 \
    --output ./adapters/pour_workpiece.npz
```

Repeat for the others.

## 9. Smoke test routed serving locally

Set the runtime paths:

```bash
export ACOT_ROUTED_CONFIG=acot_challenge_generalist_lora_all
export ACOT_ROUTED_BASE_CHECKPOINT=./checkpoints/acot_challenge_generalist_lora_all/generalist_v1/50000
export ACOT_ROUTED_ADAPTER_DIR=./adapters
```

Start the server:

```bash
bash scripts/server_routed.sh 0 8999
```

Check health:

```bash
curl -L http://127.0.0.1:8999/healthz
```

## 10. Test real routing behavior

Verify at least:

- `task_name="pour_workpiece"`
- `task_name="sorting_packages"`
- unknown task name fallback

Confirm:

- server responds
- action shape is valid
- waist behavior is correct for sorting
- fallback uses `_default`

## 11. Validate Docker startup

Build with routed startup:

```bash
SERVER_SCRIPT=./scripts/server_routed.sh docker build -f scripts/docker/serve_policy.Dockerfile -t acot-routed .
```

Run it:

```bash
docker run --rm -p 8999:8999 \
    -e SERVER_SCRIPT=./scripts/server_routed.sh \
    -e ACOT_ROUTED_CONFIG=acot_challenge_generalist_lora_all \
    -e ACOT_ROUTED_BASE_CHECKPOINT=./checkpoints/acot_challenge_generalist_lora_all/generalist_v1/50000 \
    -e ACOT_ROUTED_ADAPTER_DIR=./adapters \
    acot-routed
```

Recheck `/healthz`.

## 12. Before submission

Confirm:

- checkpoint exists in image
- adapters exist in image
- server auto-starts
- server listens on `8999`
- websocket path works

Then update:

```bash
AGENTS/status.md
AGENTS/handoff.md
AGENTS/experiments.md
```

## 13. Recommended immediate next 3 commands for the full plan

```bash
export ACOT_CHALLENGE_DATA_ROOT=~/Datasets/lerobot/Reasoning2Action-Sim
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/compute_norm_stats.py --config-name acot_challenge_generalist_lora_all
DEBUG_MODE=true UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py --config-name acot_challenge_generalist_lora_all --exp_name generalist_debug --overwrite
```
