# Steps for building and testing the generalist-v2.1 submit-ready image

## Why this setup

This image serves a single checkpoint policy with no routing and no adapters.

Model source:

- Training config: `acot_challenge_generalist_continued`
- Defined in: `src/openpi/training/config.py`
- Serving mode: `policy:checkpoint`
- Launch script: `scripts/server_submit_generalist_v2.sh`
- Dockerfile: `scripts/docker/serve_generalist_v2.Dockerfile`

Checkpoint contract:

- checkpoint dir: `checkpoint/generalist_continued`
- required contents:
  - `params/`
  - `assets/reasoning2action_sim_generalist/`
  - `_CHECKPOINT_METADATA`

Build/runtime notes:

- The image intentionally skips installing `lerobot` during `uv sync`; it is not required for checkpoint serving and caused unstable Git fetches during Docker builds.
- `uv.lock` pins a CUDA 13 JAX stack for local RTX 5090 development. The Dockerfile replaces it with `jax[cuda12]==0.7.2` for the submission server.
- `scripts/serve_policy.py` and `src/openpi/training/checkpoints.py` were adjusted so checkpoint serving does not import training-only dataset dependencies at process startup.
- Full troubleshooting details are recorded in `AGENTS/generalist_v2_1_build_notes.md`.

## 1) Stage checkpoint

Copy the trained generalist-v2.1 checkpoint from the training machine:

```bash
rsync -avP -e "ssh -p 2222" /home/bingqi/data/bingqi/Project/ACoT-VLA/checkpoints/acot_challenge_generalist_continued_reweighted/generalist_augumented/10000/params bingqi@101.6.33.98:/home/bingqi/SharedData/Research/submit-ACoT-VLA-generalists-v2/checkpoint/generalist_continued_augumented
rsync -avP -e "ssh -p 2222" /home/bingqi/data/bingqi/Project/ACoT-VLA/checkpoints/acot_challenge_generalist_continued_reweighted/generalist_augumented/10000/assets bingqi@101.6.33.98:/home/bingqi/SharedData/Research/submit-ACoT-VLA-generalists-v2/checkpoint/generalist_continued_augumented
rsync -avP -e "ssh -p 2222" /home/bingqi/data/bingqi/Project/ACoT-VLA/checkpoints/acot_challenge_generalist_continued_reweighted/generalist_augumented/10000/_CHECKPOINT_METADATA bingqi@101.6.33.98:/home/bingqi/SharedData/Research/submit-ACoT-VLA-generalists-v2/checkpoint/generalist_continued_augumented
```

Required paths:

```bash
ls checkpoint/generalist_continued/params
ls checkpoint/generalist_continued/assets/reasoning2action_sim_generalist
ls checkpoint/generalist_continued/_CHECKPOINT_METADATA
```

## 2) Build image

```bash
docker build --no-cache \
    -f scripts/docker/serve_generalist_v3.Dockerfile \
    -t generalist-v3:latest \
    .
```

## 3) Run local serve smoke test

```bash
docker run -it --rm --network=host --gpus all \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    generalist-v3:latest
```

Expected behavior:

- container auto-starts server from `CMD`
- websocket policy server listens on `8999`
- serving path is `policy:checkpoint`
- config is `acot_challenge_generalist_continued`
- checkpoint path is `/submission/checkpoint/generalist_continued`
- no runtime dependency sync/install (`uv run --no-sync` is expected)
- no adapter loading or route selection

## 4) Optional explicit env override test

```bash
docker run -it --rm --network=host --gpus all \
    -e ACOT_SERVE_CONFIG=acot_challenge_generalist_continued \
    -e ACOT_SERVE_CHECKPOINT=/submission/checkpoint/generalist_continued \
    generalist-v2.1:latest
```

## 5) Offline ICRA benchmark

```bash
cd /mnt/SharedData/Research/genie_sim
./scripts/start_gui.sh
./scripts/into.sh
./scripts/run_icra_tasks.sh
```

## 6) Tag and push

```bash
docker tag generalist-v2.1:latest \
    sim-icra-registry.cn-beijing.cr.aliyuncs.com/onecable/generalist-v2.1:latest
docker push sim-icra-registry.cn-beijing.cr.aliyuncs.com/onecable/generalist-v2.1:latest
```
