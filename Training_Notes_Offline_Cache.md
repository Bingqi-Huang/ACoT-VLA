# Offline Cache Training Notes

## Scope

This note documents the additive offline shard-cache training path for:

- `acot_challenge_generalist_lora_generalist`

The legacy raw-data training path remains unchanged.

## New Entry Points

- Build cache:

```bash
uv run python scripts/build_offline_cache.py \
  --base-config-name acot_challenge_generalist_lora_generalist \
  --cache-root /path/to/offline-cache \
  --split all \
  --shard-size 2048 \
  --num-workers 8
```

- Verify cache:

```bash
uv run python scripts/verify_offline_cache.py \
  --base-config-name acot_challenge_generalist_lora_generalist \
  --cache-root /path/to/offline-cache \
  --split all
```

- Train from cache:

```bash
uv run python scripts/train_offline_cache.py \
  --base-config-name acot_challenge_generalist_lora_generalist \
  --cache-root /path/to/offline-cache \
  --exp-name generalist_offline_cache_v1 \
  --overwrite
```

- Offline eval from cache:

```bash
uv run python scripts/eval_offline_cache.py \
  --base-config-name acot_challenge_generalist_lora_generalist \
  --cache-root /path/to/offline-cache \
  --checkpoint checkpoints/acot_challenge_generalist_lora_generalist/generalist_offline_cache_v1
```

## Compatibility Contract

- Cache training uses the same config name, checkpoint layout, `train_metrics.jsonl`, W&B metric names, and checkpoint asset saving path as legacy `scripts/train.py`.
- Existing `scripts/eval_offline.py` should still work on checkpoints produced by `scripts/train_offline_cache.py`.
- No original training script, config registration, or legacy data loader behavior is changed.

## Cache Contents

Each cached sample stores the final legacy transformed sample payload:

- `image/*`
- `image_mask/*`
- `state`
- `actions`
- `coarse_actions`
- `tokenized_prompt`
- `tokenized_prompt_mask`
- `task_id`
- `episode_index`
- `frame_index`
- `source_global_index`

Images are stored as `uint8` so the model path still relies on `Observation.from_dict(...)` to convert them to `[-1, 1]` float32 at runtime, matching legacy behavior.

## Important Caveat

The current raw ACOT data transform path contains prompt injection randomness. The offline cache path makes this deterministic per sample index during cache generation so that:

- cache generation is reproducible
- verification can compare raw-vs-cached samples exactly

If strict stochastic prompt re-sampling per epoch is required later, that should be added as a follow-up cache format revision.
