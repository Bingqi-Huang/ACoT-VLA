# Docker Build/Run Guide (Routed clean5000)

This document aligns our workflow with the official ICRA Reasoning2Action submission contract.

## Official Contract Mapping

Official requires:

- Inference service auto-starts from Docker `CMD` or `ENTRYPOINT`.
- Service listens on port `8999`.
- Image contains dependencies, code, checkpoints, and serving logic.
- Submission model type matches policy output (`abs_joint` or `abs_pose`).

Current repo status:

- Auto-start: yes (`scripts/server_submit_routed.sh` via Docker `CMD`).
- Port: yes (default `8999`).
- Artifacts in image: yes (`checkpoint/` + `adapters/` + code tree).
- Output type: `abs_joint` (submit page should select `abs_joint`).

## Runtime Design

Routed strategy for this image:

- Base checkpoint: `checkpoint/baseline/30000`
- Routed adapter dir: `adapters/`
- Specialist adapter: `adapters/clean_the_desktop_5000.npz`
- Fallback adapter: `adapters/_default.npz`

`adapter_routed_policy` loads missing LoRA tensors from baseline with zero init,
so non-routed tasks preserve baseline behavior.

## 1) Required Local Files

Verify these paths exist before build:

```bash
ls checkpoint/baseline/30000/params
ls checkpoint/baseline/30000/assets
ls adapters/_default.npz
ls adapters/clean_the_desktop_5000.npz
```

If adapter has not been extracted yet:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' uv run ./scripts/extract_adapter.py \
    --checkpoint ./checkpoint/specialists-open/ \
    --output ./adapters/acot_specialist_open_door
```

## 1.1) Personal rsync Staging Commands (Keep)

These are preserved for manual checkpoint sync from your training machine:

```bash
rsync -avP -e "ssh -p 2222" /home/bingqi/data/admins/bingqi/Projects/ACoT-VLA/checkpoints/acot_specialist_open_door/acot_specialist_open_door/4500/params bingqi@101.6.33.98:/mnt/SharedData/Research/submit-ACoT-VLA-specialists-box-pour-scoop-stock-pot-open/checkpoint/specialists-open/
rsync -avP -e "ssh -p 2222" /home/bingqi/data/admins/bingqi/Projects/ACoT-VLA/checkpoints/acot_specialist_open_door/acot_specialist_open_door/4500/assets bingqi@101.6.33.98:/mnt/SharedData/Research/submit-ACoT-VLA-specialists-box-pour-scoop-stock-pot-open/checkpoint/specialists-open/
rsync -avP -e "ssh -p 2222" /home/bingqi/data/admins/bingqi/Projects/ACoT-VLA/checkpoints/acot_specialist_open_door/acot_specialist_open_door/4500/_CHECKPOINT_METADATA bingqi@101.6.33.98:/mnt/SharedData/Research/submit-ACoT-VLA-specialists-box-pour-scoop-stock-pot-open/checkpoint/specialists-open/
```

After extraction, the full specialist checkpoint is not required in the image build context:

```bash
rm -rf checkpoint/specialists_clean_5000
```

## 2) Build Context Guard

Ensure `.dockerignore` excludes temporary heavy checkpoints:

```bash
grep -n "checkpoint/specialists_clean_5000" .dockerignore
```

## 3) Build Submission Image

Preferred Dockerfile for routed submission:

- `scripts/docker/serve_policy.Dockerfile`

Build command:

```bash
docker build --no-cache \
    -f scripts/docker/serve_policy.Dockerfile \
    -t routed-box-pour-scoop-stock-pot-open:latest  \
    .
```

## 4) Run Inference Service (Official-Compatible)

Use the same run style as official docs (no extra custom env args):

```bash
docker run -it --rm --network=host --gpus "device=1" \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    routed-box-pour-scoop-stock-pot-open:latest
```

Success signal in logs:

- `INFO:websockets.server:server listening on 0.0.0.0:8999`

## 5) Run ICRA Tasks In Genie Sim

```bash
cd /mnt/SharedData/Research/genie_sim
./scripts/start_gui.sh
./scripts/into.sh
./scripts/run_icra_tasks.sh
```

If inference runs on another host:

```bash
./scripts/run_icra_tasks.sh --infer-host <host_ip>:8999
```

Optional score aggregation:

```bash
python3 scripts/stat_average.py
```

## 6) Tag and Push

```bash
docker login sim-icra-registry.cn-beijing.cr.aliyuncs.com
docker tag routed-clean-box:latest \
    sim-icra-registry.cn-beijing.cr.aliyuncs.com/onecable/routed-clean-box:latest
docker push sim-icra-registry.cn-beijing.cr.aliyuncs.com/onecable/routed-clean-box:latest
```

Submit the full image URL on the platform and select model type `abs_joint`.

## 7) Final Checklist

- Container starts serving automatically without extra manual command inside container.
- Service listens on `8999`.
- `checkpoint/baseline/30000` is present in image.
- `adapters/_default.npz` and `adapters/clean_the_desktop_5000.npz` are present in image.
- No runtime remote dependency install/download is required for serving.
