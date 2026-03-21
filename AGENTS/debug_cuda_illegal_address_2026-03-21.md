# CUDA Illegal Address Fix Log (2026-03-21)

## Incident Summary

- Symptom: multi-GPU training for `acot_challenge_generalist_continued` failed with:
  - `CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered`
- Environment at failure time:
  - GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (6x, 96GB)
  - Old stack: CUDA12-era JAX (`jax[cuda12]==0.5.3`) and older Orbax restore assumptions.

## Root Cause Chain

1. Hardware/runtime mismatch:
   - Blackwell GPUs require a newer CUDA runtime stack.
   - Old CUDA12 JAX stack was unstable for this workload and triggered illegal-address failures during early replicated step execution.
2. Post-upgrade secondary break:
   - Upgrading JAX/Orbax exposed an API change in Orbax metadata.
   - `restore_params()` assumed dict-style metadata (`metadata["params"]`) and crashed on Orbax `StepMetadata`.

## Fixes Applied

## 1) Environment upgrade for Blackwell

- `pyproject.toml`
  - `jax[cuda12]==0.5.3` -> `jax[cuda13]==0.7.2`
  - `numpy>=1.22.4,<2.0.0` -> `numpy>=2.0.0,<3.0.0`
  - `orbax-checkpoint==0.11.13` -> `orbax-checkpoint==0.11.33`
  - `ml-dtypes` override -> `0.5.3`
- `packages/openpi-client/pyproject.toml`
  - aligned NumPy range to `>=2.0.0,<3.0.0`
- `uv.lock` regenerated.

Verified runtime:

- `jax==0.7.2`
- `jaxlib==0.7.2`
- CUDA backend reported by JAX: `cuda 13000`

## 2) Orbax restore compatibility patch

File:

- `src/openpi/models/model.py`

Patch behavior:

- `restore_params()` now supports both:
  - old dict-style metadata (`metadata["params"]`)
  - new Orbax `StepMetadata` (`metadata.item_metadata["params"]`)
- Includes fallback for simpler/legacy metadata shapes.

## Validation Evidence

Command used:

```bash
export ACOT_CHALLENGE_INIT_WEIGHTS=./checkpoints/baseline/30000/params/
bash scripts/train_fast_6gpu.sh \
  acot_challenge_generalist_continued \
  acot_challenge_generalist_continued_cuda13_fix1 \
  --overwrite \
  --r2a-cache-root /home/bingqi/data/bingqi/ACoT-Data
```

Observed in log:

- checkpoint restore completed successfully
- train state initialized
- entered real training loop and advanced to step 5 before manual stop
- no `CUDA_ERROR_ILLEGAL_ADDRESS` in this fixed run

Reference log:

- `logs/acot_challenge_generalist_continued_acot_challenge_generalist_continued_cuda13_fix1_fast_6gpu_20260321_203956.log`

## Operational Notes For Future Agents

- For Blackwell (`RTX PRO 6000`) training in this repo, keep CUDA13 JAX stack.
- If upgrading Orbax again, re-check `restore_params()` metadata assumptions first.
- A warning about donated buffers not usable is non-fatal here; do not treat it as the root cause.

## Quick Triage Checklist

1. Confirm JAX backend reports CUDA13:
   - `uv run python -c "import jax; print(jax.devices()[0].client.platform_version)"`
2. Confirm checkpoint restore path does not throw `StepMetadata` type errors.
3. Confirm first training steps advance (`1+` iterations), not just init.
4. Only after that, investigate model-level kernels if illegal-address reappears.
