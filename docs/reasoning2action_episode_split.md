# Episode-Level Train/Validation Split

Reasoning2Action finetuning now supports a deterministic episode-level train/validation split for LeRobot datasets.

## What it does

- Splits each task by `episode_index`, never by frame or timestep.
- Saves the exact split manifest as JSON under `assets/<config>/episode_splits/`.
- Reuses the same manifest across training and normalization-stat computation.
- Prints per-task episode/frame counts for both `train` and `val`.

## Generate or inspect the split

```bash
uv run python scripts/generate_episode_split.py \
  --config-name acot_challenge_generalist_lora_clean_desktop
```

This writes and prints a JSON manifest similar to:

```json
{
  "version": 1,
  "seed": 42,
  "train_ratio": 0.8,
  "datasets": [
    {
      "repo_id": ".../clean_the_desktop_part_1",
      "task_name": "clean_the_desktop_part_1",
      "train_episode_indices": [0, 1, 2],
      "val_episode_indices": [3]
    }
  ]
}
```

## Compute normalization stats without leakage

Use the training split when computing stats:

```bash
uv run python scripts/compute_norm_stats.py \
  --config-name acot_challenge_generalist_lora_clean_desktop \
  --split train
```

## Train with validation

`acot_challenge_generalist_lora_clean_desktop` now enables validation by default. You can also override it from CLI:

```bash
DEBUG_MODE=true uv run python scripts/train.py \
  --config-name acot_challenge_generalist_lora_clean_desktop \
  --exp_name clean_desktop_split_smoke \
  --val_interval 50 \
  --val_num_batches 2
```

Training reads only train episodes. Validation reads only val episodes.
