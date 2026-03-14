# Steps for building and testing routed submit-ready image

## Why this setup

Goal for challenge routing:

- Keep baseline capability for non-target tasks.
- Use task-specific adapter only when the route key matches.

This image now follows that contract:

- base checkpoint: baseline (`checkpoint/baseline/30000`)
- default adapter: `_default.npz` (empty / zero-LoRA behavior)
- task adapter: `clean_the_desktop_1500.npz`

Implementation note:
LoRA config params missing from the baseline checkpoint are zero-initialized
at load time, so `_default` path preserves baseline behavior.

Current stable defaults:

- Base checkpoint: `checkpoint/baseline/30000`
- Routed adapter dir: `adapters/`
- Clean-desktop specialist adapter: `adapters/clean_the_desktop_1500.npz`
- Fallback adapter: `adapters/_default.npz` (can be empty)
- Routing launch script: `scripts/server_submit_routed.sh`
- Dockerfile: `scripts/docker/serve_policy.Dockerfile`

## 1) Stage checkpoint and adapters

Copy trained specialist checkpoint from training machine:

```bash
rsync -avP -e "ssh -p 2222" /data/admins/bingqi/Projects/ACoT-VLA/checkpoints/acot_specialist_clean_desktop/exp_specialist_clean/1500/params bingqi@101.6.33.98:/mnt/SharedData/Research/submit-ACoT-VLA-specialists-clean-1500/checkpoint/specialists_clean_1500/
rsync -avP -e "ssh -p 2222" /data/admins/bingqi/Projects/ACoT-VLA/checkpoints/acot_specialist_clean_desktop/exp_specialist_clean/1500/assets bingqi@101.6.33.98:/mnt/SharedData/Research/submit-ACoT-VLA-specialists-clean-1500/checkpoint/specialists_clean_1500/
rsync -avP -e "ssh -p 2222" /data/admins/bingqi/Projects/ACoT-VLA/checkpoints/acot_specialist_clean_desktop/exp_specialist_clean/1500/_CHECKPOINT_METADATA bingqi@101.6.33.98:/mnt/SharedData/Research/submit-ACoT-VLA-specialists-clean-1500/checkpoint/specialists_clean_1500/
```

Extract adapter (if not yet extracted):

```bash
uv run ./scripts/extract_adapter.py \
    --checkpoint ./checkpoint/specialists_clean_1500/ \
    --output ./adapters/clean_the_desktop_1500
```

Required paths:

```bash
ls checkpoint/baseline/30000/params
ls checkpoint/baseline/30000/assets
ls adapters/_default.npz
ls adapters/clean_the_desktop_1500.npz
```

After adapter extraction, `checkpoint/specialists_clean_1500` is not required for this routed image.
You can remove it to reduce build context and local disk usage:

```bash
rm -rf checkpoint/specialists_clean_1500
```

## 1.5) Build-context guard

This repo excludes the specialist full checkpoint from build context:

- `.dockerignore` includes: `checkpoint/specialists_clean_1500`

Quick check:

```bash
grep -n "checkpoint/specialists_clean_1500" .dockerignore
```

## 2) Build image

```bash
docker build --no-cache \
    -f scripts/docker/serve_policy.Dockerfile \
    -t routed-clean-desktop-1500:latest \
    .
```

## 3) Run local serve smoke test

```bash
docker run -it --rm --network=host --gpus all \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    routed-clean-desktop-1500:latest
```

Expected behavior:

- container auto-starts server from `CMD`
- websocket policy server listens on `8999`
- no runtime dependency sync/install (`uv run --no-sync` is expected)
- no runtime checkpoint download

## 4) Optional explicit env override test

```bash
docker run -it --rm --network=host --gpus all \
    -e ACOT_ROUTED_CONFIG=acot_challenge_lora_conservative \
    -e ACOT_ROUTED_BASE_CHECKPOINT=/submission/checkpoint/baseline/30000 \
    -e ACOT_ROUTED_ADAPTER_DIR=/submission/adapters \
    routed-clean-desktop-1500:latest
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
docker tag routed-clean-desktop-1500:latest \
    sim-icra-registry.cn-beijing.cr.aliyuncs.com/onecable/routed-clean-desktop-1500:latest
docker push sim-icra-registry.cn-beijing.cr.aliyuncs.com/onecable/routed-clean-desktop-1500:latest
```
