# Submission Runbook

This runbook covers the minimal smoke-submission path for the AgiBot World Challenge Reasoning2Action track.

## Current Smoke Path

- Dockerfile:
  `scripts/docker/smoke_submission.Dockerfile`
- Startup command inside image:
  `./scripts/server_smoke_submission.sh`
- Serving entrypoint:
  `scripts/serve_policy.py --env G2SIM --port 8999`
- Submission model type:
  `abs_joint`
- Logging behavior:
  startup timing plus summarized websocket payload keys, task name presence/value, truncated prompt preview, and inference timing

## Hard Checks Before Submission

- Server auto-starts via `CMD` or `ENTRYPOINT`
- Server listens on port `8999`
- Model output type matches submission declaration: `abs_joint`
- Image starts from a single Docker image with no post-start commands
- Local health check passes
- Platform logs stay readable and do not dump raw observations

## Registry Login

Use the **Get Registry Token** button on the test server before pulling the official baseline image or pushing the final smoke image.

If the token exposes a login command, run it first. Otherwise use the endpoint/username/password shown on the submission page:

```bash
docker login <REGISTRY_ENDPOINT>
```

## Build

Build from repo root:

```bash
docker build -f scripts/docker/smoke_submission.Dockerfile -t acot-r2a-smoke:$(git rev-parse --short HEAD) .
```

## Local Smoke Validation

Start the container:

```bash
docker run --rm --network=host --gpus all acot-r2a-smoke:$(git rev-parse --short HEAD)
```

Expected startup signal in logs:

- policy load timing
- websocket bind on `0.0.0.0:8999`

Health check from another terminal:

```bash
curl http://127.0.0.1:8999/healthz
```

Expected response:

```text
OK
```

## Tag And Push

Tag the image with the registry endpoint and namespace provided by the test server:

```bash
docker tag acot-r2a-smoke:$(git rev-parse --short HEAD) <REGISTRY_ENDPOINT>/<NAMESPACE>/acot-r2a-smoke:smoke-$(git rev-parse --short HEAD)
docker push <REGISTRY_ENDPOINT>/<NAMESPACE>/acot-r2a-smoke:smoke-$(git rev-parse --short HEAD)
```

## What To Submit

Submit the full pushed image URL:

```text
<REGISTRY_ENDPOINT>/<NAMESPACE>/acot-r2a-smoke:smoke-<GIT_SHA>
```

Submission-page choice:

- Model type: `abs_joint`

## What The First Smoke Submission Should Teach Us

- whether the real evaluator includes `task_name`
- the exact top-level payload keys used by the evaluator
- whether prompt text and task naming align with the training/task conventions in this repo
- startup time and whether the evaluator waits long enough for model load
- whether the platform log download is sufficient to debug runtime failures
