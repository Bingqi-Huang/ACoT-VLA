# Training Notes

Current mainline note:

- the actively retained single-model training line in this workspace is `acot_challenge_generalist_lora_generalist`
- the clean-desktop path below is still useful as a smoke path, but its old checkpoint examples should be treated as placeholders unless that experiment is re-run and retained
- routed serving and fast-path training both exist in code, but neither should be treated as the default entrypoint yet

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
export ACOT_CHALLENGE_DATA_ROOT=/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim
```

If you want to override the base initialization checkpoint, set:

```bash
export ACOT_CHALLENGE_INIT_WEIGHTS=/data/admins/bingqi/Projects/ACoT-VLA/checkpoints/baseline_checkpoint/params
```

If we got any unaligned data(default is 0.0001):
```bash
export ACOT_CHALLENGE_VIDEO_TOLERANCE_S=0.15
```

Clean up depth camera info in meta data of datasets:
```bash
uv run ./scripts/cleanup_depth_metadata.py "$ACOT_CHALLENGE_DATA_ROOT"
```

Optional local `uv` cache:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
mkdir -p "${UV_CACHE_DIR}"
```

## Fast path for `acot_challenge_generalist_lora_generalist`

The current recommended experimental fast route for this config is:

1. build the generic `Reasoning2Action-Sim` frame cache
2. verify the cache against raw data
3. compute norm stats from the cache
4. train with `scripts/train_fast.py --r2a-cache-root ...`

This path is additive:

- old path: `scripts/train.py` and `scripts/train.sh`
- fast path: `scripts/train_fast.py` and `scripts/train_fast.sh`
- cache path: generic frame cache plus `train_fast.py --r2a-cache-root ...`

It does **not** change checkpoint format, offline eval compatibility, or final inference / Docker serving.

Current caveat:

- do not make `scripts/train_fast.py` the default launcher yet
- there is still no retained real cache-backed training run in this workspace
- `scripts/train_fast.py` still needs a checkpoint-save fix before it is safe as the primary path

### Scope and caveat

This section is specifically for:

```text
acot_challenge_generalist_lora_generalist
```

Important caveat:

- this config uses `discrete_state_input=True`
- prompt tokenization depends on both `prompt` and `state`
- so prompt-only token cache is not the right optimization here
- the main optimization is the generic frame cache

### 1. Prepare the environment

```bash
export ACOT_CHALLENGE_DATA_ROOT=/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim
export UV_CACHE_DIR=/tmp/uv-cache
mkdir -p "${UV_CACHE_DIR}"
```

Optional:

```bash
export ACOT_CHALLENGE_INIT_WEIGHTS=/data/admins/bingqi/Projects/ACoT-VLA/checkpoints/baseline_checkpoint/params
export ACOT_CHALLENGE_VIDEO_TOLERANCE_S=0.15
```

If metadata cleanup has not been done yet:

```bash
uv run ./scripts/cleanup_depth_metadata.py "$ACOT_CHALLENGE_DATA_ROOT"
```

### 2. Build the generic frame cache

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/build_reasoning2action_frame_cache.py \
  --cache-root /path/to/r2a-frame-cache \
  --data-root "${ACOT_CHALLENGE_DATA_ROOT}" \
  --shard-size 2048 \
  --num-workers 16
```

This builds one reusable cache for the whole `Reasoning2Action-Sim` family, not only this config.

Behavior of the builder:

- it shows a per-sub-dataset progress bar while staging samples
- it also shows a final `assemble` progress bar when merging staged repos into the final cache
- it now supports resume at the sub-dataset level
- if a run is interrupted, rerun the same command and already completed sub-datasets will be reused instead of rebuilt from the beginning

### 3. Verify the cache

Before training, verify raw-vs-cache equivalence for this config:

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/verify_reasoning2action_frame_cache.py \
  --cache-root /path/to/r2a-frame-cache \
  --config-name acot_challenge_generalist_lora_generalist \
  --split train
```

### 4. Compute norm stats from the cache

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/compute_norm_stats_fast.py \
  --config-name acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --split train
```

This writes norm stats to the usual assets location, so downstream training and checkpoint loading still use the normal asset layout.

### 5. Run a short fast debug job

```bash
DEBUG_MODE=true UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/train_fast.py \
  acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --exp_name generalist_v1_bs96_fast_cache_debug \
  --overwrite
```

Check:

- first batch initializes correctly
- loss logs appear
- `train_metrics.jsonl` is written
- checkpoint writing works

### 6. Run the real fast training job

Direct Python entry:

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/train_fast.py \
  acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --exp_name generalist_v1_bs96_fast_cache \
  --val-interval=1000 \
  --val-num-batches=8
```

Shell wrapper:

```bash
bash scripts/train_fast.sh \
  acot_challenge_generalist_lora_generalist \
  generalist_v1_bs96_fast_cache \
  --r2a-cache-root=/path/to/r2a-frame-cache \
  --val-interval=1000 \
  --val-num-batches=8
```

Resume:

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/train_fast.py \
  acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --exp_name generalist_v1_bs96_fast_cache \
  --resume=true
```

Restart from scratch:

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/train_fast.py \
  acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --exp_name generalist_v1_bs96_fast_cache \
  --overwrite
```

### 7. Compatibility target

This route should preserve:

- checkpoint layout under `./checkpoints/<config>/<exp_name>/`
- `train_metrics.jsonl`
- existing offline eval entrypoints
- final checkpoint-based inference
- final Docker / websocket serving

The cache is training-only infrastructure and should not enter the final runtime package.

### 8. Fallback rule

If anything looks wrong:

- stop the cache-backed fast run
- keep the logs
- fall back to `scripts/train.py` or raw `scripts/train_fast.py` without `--r2a-cache-root`

Because the old path is untouched, fallback is trivial:

```bash
bash scripts/train.sh \
  acot_challenge_generalist_lora_generalist \
  generalist_v1_bs96 \
  --val-interval=1000 \
  --val-num-batches=8
```


## 2. Compute norm stats for the clean-desktop config

This must be run before training:

```bash
uv run python scripts/compute_norm_stats.py \
    --config-name acot_challenge_generalist_lora_clean_desktop \
    --split train
```

The config writes stats under the asset id for the clean-desktop config.

## 3. Run a short debug training job

Use debug mode first:

```bash
DEBUG_MODE=true uv run python scripts/train.py \
    acot_challenge_generalist_lora_clean_desktop \
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
bash scripts/train.sh \
  acot_challenge_generalist_lora_clean_desktop \
  clean_desktop_v1 \
  --val-interval=1000 \
  --val-num-batches=8 \
#   --resume=true Choose between two
#   --overwrite

```

Or directly:

```bash
uv run python scripts/train.py \
    acot_challenge_generalist_lora_clean_desktop \
    --exp_name clean_desktop_v1
```

Expected checkpoint root:

```bash
./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_v1
```

Expected final-step checkpoint example:

```bash
./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_v1/<selected_step>
```

Use the real saved step directory everywhere below. The current workspace does not retain a clean-desktop checkpoint tree, so treat this section as a smoke-workflow template rather than a live retained artifact.

## 5. Serve the trained checkpoint locally

Set the checkpoint to serve:

```bash
export ACOT_SERVE_CONFIG=acot_challenge_generalist_lora_clean_desktop
export ACOT_SERVE_CHECKPOINT=./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_v1/<selected_step>
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
    -e ACOT_SERVE_CHECKPOINT=./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_v1/<selected_step> \
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
DEBUG_MODE=true uv run python scripts/train.py acot_challenge_generalist_lora_clean_desktop --exp_name clean_desktop_debug --overwrite
```

---

# Full Routed-Adapter Plan

This is the **full competition plan**. Keep this as the long-term path after the clean-desktop test-server workflow above.

## Routing contract from Genie Sim

The routed-serving plan should be driven by the **actual websocket payload**, not by the benchmark YAML name.

- The simulator sends msgpack+numpy payloads over websocket, not JSON.
- The inference payload key to route on is `payload["task_name"]`.
- In the current ICRA evaluation chain, `payload["task_name"]` is the benchmark `sub_task_name`, not the outer background `benchmark.task_name`.
- The payload does **not** include `instance_id` or the outer background task name. Finer-grained routing beyond sub-task must rely on prompt or image content.
- `sorting_packages_continuous` must be routed by `task_name`; its prompt alone is too generic.
- The current G2 consumer path is effectively `abs_joint`. Keeping the server on the existing joint-action path is the safest submission contract.

For this repo, that means the routed server should treat the following as the public route-key surface:

| `payload["task_name"]` | Adapter target | Training data backing it | Notes |
|---|---|---|---|
| `hold_pot` | `hold_pot` | `hold_pot` | Rule-evaluated |
| `clean_the_desktop` | `clean_the_desktop` | all `clean_the_desktop*` shards, currently `clean_the_desktop_part_1` + `clean_the_desktop_part_2` | VLM-evaluated |
| `open_door` | `open_door` | `open_door` | Rule-evaluated |
| `place_block_into_box` | `place_block_into_box` | `place_block_into_box` | Rule-evaluated |
| `pour_workpiece` | `pour_workpiece` | `pour_workpiece` | Rule-evaluated |
| `scoop_popcorn` | `scoop_popcorn` | `scoop_popcorn` | VLM-evaluated |
| `sorting_packages` | `sorting_packages` | all `sorting_packages*` shards, currently `sorting_packages_part_1` + `sorting_packages_part_2` + `sorting_packages_part_3` | Rule-evaluated |
| `sorting_packages_continuous` | `sorting_packages` | same sorting shard set as above | Same adapter as `sorting_packages` |
| `stock_and_straighten_shelf` | `stock_and_straighten_shelf` | all `stock_and_straighten_shelf*` shards, currently `stock_and_straighten_shelf` + `stock_and_straighten_shelf_part_2` | Rule-evaluated |
| `take_wrong_item_shelf` | `take_wrong_item_shelf` | `take_wrong_item_shelf` | Rule-evaluated |

Operationally, this is **10 route keys backed by 9 specialist adapters**, plus the `_default` generalist fallback.

## Current code/doc mismatches to keep in mind

The docs above describe the intended ICRA submission contract. The current codebase is close, but not fully aligned yet.

- The current routed implementation still contains a legacy alias `grab_toy -> place_block_into_box` in `src/openpi/policies/adapter_routed_policy.py`. That alias is not part of the public ICRA route-key contract and should be removed from the final serving path.
- The current full generalist config already includes additional same-task storage shards such as `stock_and_straighten_shelf_part_2` and `sorting_packages_part_3`. This is correct in spirit because `_part_*` directories are not new tasks; they are additional data for the same routed task family.
- The current specialist configs are still incomplete relative to that shard-merging rule:
  - `acot_specialist_stock_shelf` should use both `stock_and_straighten_shelf` and `stock_and_straighten_shelf_part_2`
  - `acot_specialist_sorting` should use `sorting_packages_part_1`, `sorting_packages_part_2`, and `sorting_packages_part_3`
- Routed serving exists in code, but it still lacks the validation depth needed for a final submission image: no documented all-10-key smoke coverage, no structured raw-key-to-adapter logging, and no recorded end-to-end run with real extracted adapters.

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
    --config-name acot_challenge_generalist_lora_generalist
```

This should write stats into the assets dir used by the config.

## 3. Run a short generalist debug train

First smoke test the accumulation path:

```bash
DEBUG_MODE=true UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py \
    acot_challenge_generalist_lora_generalist \
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
bash scripts/train.sh acot_challenge_generalist_lora_generalist generalist_v1_bs96
```

Or directly:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py \
    acot_challenge_generalist_lora_generalist \
    --exp_name generalist_v1_bs96
```

Wait until you have a usable checkpoint, ideally the target final step.

## 5. Point specialists at the real generalist checkpoint

Export the checkpoint path:

```bash
export ACOT_CHALLENGE_GENERALIST_WEIGHTS=./checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96/<selected_step>/params
```

Use the real saved step directory. In the current workspace, the retained generalist example is `./checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96/5000`.

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

Notes:

- `acot_specialist_sorting` is the shared specialist for both `sorting_packages` and `sorting_packages_continuous`, and it should consume all `sorting_packages*` shards.
- `acot_specialist_clean_desktop` is trained from all clean-desktop shards and serves the single inference route key `clean_the_desktop`.
- `acot_specialist_stock_shelf` should consume all `stock_and_straighten_shelf*` shards.

## 8. Extract adapters

Create the adapter directory:

```bash
mkdir -p adapters
```

Extract the fallback generalist adapter:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/extract_adapter.py \
    --checkpoint ./checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96/<selected_step> \
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
export ACOT_ROUTED_CONFIG=acot_challenge_generalist_lora_generalist
export ACOT_ROUTED_BASE_CHECKPOINT=./checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96/<selected_step>
export ACOT_ROUTED_ADAPTER_DIR=./adapters
```

Important:

- `scripts/server_routed.sh` still has stale built-in defaults pointing at `acot_challenge_generalist_lora_all`
- always set the routed env vars explicitly until that script is cleaned up

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
- `task_name="sorting_packages_continuous"`
- `task_name="clean_the_desktop"`
- unknown task name fallback

Confirm:

- server responds
- action shape is valid
- waist behavior is correct for both sorting route keys
- fallback uses `_default`
- logged raw route key matches the adapter you expect

## 11. Validate Docker startup

Build with routed startup:

```bash
SERVER_SCRIPT=./scripts/server_routed.sh docker build -f scripts/docker/serve_policy.Dockerfile -t acot-routed .
```

Run it:

```bash
docker run --rm -p 8999:8999 \
    -e SERVER_SCRIPT=./scripts/server_routed.sh \
    -e ACOT_ROUTED_CONFIG=acot_challenge_generalist_lora_generalist \
    -e ACOT_ROUTED_BASE_CHECKPOINT=./checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96/<selected_step> \
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
- routed behavior is keyed only by `payload["task_name"]`
- unknown route keys fall back to `_default`
- `sorting_packages_continuous` reuses the `sorting_packages` adapter

Then update:

```bash
AGENTS/status.md
AGENTS/handoff.md
AGENTS/experiments.md
```

## 13. Planned code adjustments before final submission

No serving-code change is required immediately to adopt the corrected routing contract; the current routed path already reads `obs["task_name"]`.

Before final submission, the code plan should be:

1. Make the routing table configurable or at least centralize the documented 10-key mapping in one place, instead of relying on an implicit hard-coded table.
2. Add a focused routed-serving smoke test that covers all 10 public route keys plus unknown fallback.
3. Log both the raw incoming `task_name` and the resolved adapter name on route switches so evaluation-time misroutes are visible in container logs.
4. Remove the legacy `grab_toy -> place_block_into_box` alias from the routed-serving implementation so code matches the public ICRA contract.
5. Update specialist configs so each task family consumes all known same-task shards, especially stock-shelf and sorting.
6. Keep the existing joint-action serving path; do not switch to pose output unless the serving and post-process path is re-validated end-to-end.

## 14. Recommended immediate next 3 commands for the full plan

```bash
export ACOT_CHALLENGE_DATA_ROOT=~/Datasets/lerobot/Reasoning2Action-Sim
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/compute_norm_stats.py --config-name acot_challenge_generalist_lora_generalist
DEBUG_MODE=true UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py acot_challenge_generalist_lora_generalist --exp_name generalist_debug --overwrite
```

## 15. Reasoning2Action generic frame-cache fast path

This is the current additive cache direction for the competition training line.

It does **not** replace:

- `scripts/train.py`
- `scripts/train_fast.py`
- raw LeRobot / mp4 datasets

It adds a reusable intermediate cache for the whole `Reasoning2Action-Sim` family.

### What this cache contains

The cache is built once for the dataset family and is intended to be reused by:

- `acot_challenge_generalist_lora_generalist`
- `acot_challenge_generalist_lora_clean_desktop`
- `acot_challenge_generalist_lora_5_tasks`
- future `acot_specialist_*` configs

It precomputes only the expensive shared part:

- decode video frames from the original mp4 files
- resize the 3 training cameras to `224x224`
- store raw-ish frame/state/action/prompt/task metadata in shard files

It deliberately does **not** precompute:

- normalization
- prompt tokenization
- config-specific action slicing
- final pad / final model batch layout

That work is still done at training time through the existing fast transform path.

### Files added for this workflow

- `scripts/build_reasoning2action_frame_cache.py`
- `scripts/verify_reasoning2action_frame_cache.py`
- `scripts/compute_norm_stats_fast.py`
- `src/openpi/training/r2a_frame_cache.py`
- `src/openpi/training/data_loader_fast_r2a.py`

### End-to-end workflow

Recommended machine split:

- preprocessing machine: U9
- training machine: A100 box

This split is supported.

Recommended principle:

- do the expensive raw-video decode and cache build on U9
- do raw-vs-cache verification on U9 as well, because U9 definitely has the raw dataset
- copy the built cache plus required metadata assets to the training machine
- run `train_fast.py --r2a-cache-root ...` on the training machine

1. On U9, build the cache once.

```bash
export ACOT_CHALLENGE_DATA_ROOT=/path/to/Reasoning2Action-Sim
export UV_CACHE_DIR=/tmp/uv-cache
mkdir -p "${UV_CACHE_DIR}"

UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/build_reasoning2action_frame_cache.py \
  --cache-root /path/to/r2a-frame-cache \
  --data-root "${ACOT_CHALLENGE_DATA_ROOT}" \
  --shard-size 2048 \
  --num-workers 16
```

2. On U9, verify raw-vs-cache equivalence for a target config before copying.

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/verify_reasoning2action_frame_cache.py \
  --cache-root /path/to/r2a-frame-cache \
  --config-name acot_challenge_generalist_lora_generalist \
  --split train
```

3. On U9, if you want fully cache-based training on the training machine, also prepare the split/norm assets there.

Episode split manifests can be created by running either the raw train path once up to split generation, or any command that touches the split manifest for the target config. The important files end up under:

```bash
assets/<config>/episode_splits/
```

If you want cache-only norm-stats generation, you can also compute stats now:

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/compute_norm_stats_fast.py \
  --config-name acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --split train
```

4. Copy the built cache directory to the training machine.

Recommended copy targets:

- `/path/to/r2a-frame-cache`
- `assets/acot_challenge_generalist_lora_generalist/episode_splits/`
- any cache-based norm-stats output directory you computed on U9

Example:

```bash
rsync -avP /path/to/r2a-frame-cache/ user@trainbox:/path/to/r2a-frame-cache/
rsync -avP assets/acot_challenge_generalist_lora_generalist/episode_splits/ \
  user@trainbox:/data/admins/bingqi/Projects/ACoT-VLA/assets/acot_challenge_generalist_lora_generalist/episode_splits/
```

5. On the training machine, set the same config family and point training at the copied cache.

```bash
export UV_CACHE_DIR=/tmp/uv-cache
mkdir -p "${UV_CACHE_DIR}"
```

Important:

- if the training machine does **not** have the raw `Reasoning2Action-Sim` dataset mounted, do **not** rely on `verify_reasoning2action_frame_cache.py` there
- in that case, make sure the needed `assets/<config>/episode_splits/*.json` files already exist on the training machine
- the cache-backed fast loader only needs the split manifests and the config repo basenames; it does not need to decode raw mp4 again during training

6. If needed, compute norm stats from the copied cache on the training machine.

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/compute_norm_stats_fast.py \
  --config-name acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --split train
```

7. Run fast training against the cache by adding `--r2a-cache-root`.

Debug run:

```bash
DEBUG_MODE=true UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/train_fast.py \
  acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --exp_name generalist_fast_cache_debug \
  --overwrite
```

Full run:

```bash
UV_CACHE_DIR=${UV_CACHE_DIR} uv run python scripts/train_fast.py \
  acot_challenge_generalist_lora_generalist \
  --r2a-cache-root /path/to/r2a-frame-cache \
  --exp_name generalist_fast_cache_v1 \
  --val-interval=1000 \
  --val-num-batches=8
```

### Compatibility target

The frame cache is training-only infrastructure.

It should **not** change:

- checkpoint directory layout
- `train_metrics.jsonl`
- existing offline eval entrypoints
- final checkpoint-based inference
- final Docker / websocket serving contract

So the cache stays outside the final runtime package; only the produced checkpoint matters to serving.
