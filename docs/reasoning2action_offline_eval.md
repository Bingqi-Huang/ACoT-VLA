# Offline Evaluation For LoRA Finetuning

This repository now includes an offline validation pipeline for pi0.5 / ACoT finetuning checkpoints.

## What it evaluates

- Validation split only
- No rollout
- Batched teacher-forced offline reconstruction metrics
- Checkpoint-by-checkpoint JSON outputs plus a summary CSV

## Commands

Evaluate one checkpoint:

```bash
uv run python scripts/eval_offline.py \
  --config-name acot_challenge_generalist_lora_clean_desktop \
  --checkpoint ./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_split_smoke/2500
```

Evaluate every checkpoint under one experiment directory:

```bash
uv run python scripts/eval_offline.py \
  --config-name acot_challenge_generalist_lora_clean_desktop \
  --checkpoint ./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_split_smoke
```

Faster smoke evaluation on a subset of validation batches:

```bash
uv run python scripts/eval_offline.py \
  --config-name acot_challenge_generalist_lora_clean_desktop \
  --checkpoint ./checkpoints/acot_challenge_generalist_lora_clean_desktop/clean_desktop_split_smoke \
  --max-batches 8 \
  --batch-size 4
```

## Output layout

```text
checkpoints/<config>/<exp>/eval/
  checkpoint_2500.json
  checkpoint_5000.json
  ...
  summary.csv
```

Each checkpoint JSON includes:

- `overall_val_loss`
- `per_task_val_loss`
- `per_action_dim_mae`
- `horizon_step_mae`
- `horizon_segment_mae`
- `overall_joint_mae` and `joint_mae` when action dim is 8
- `gripper.accuracy` and `gripper.bce` when action dim is 8 and the target gripper is binary

## Workflow

Recommended order for the clean-desktop smoke pipeline:

1. Generate the deterministic episode split.
2. Compute normalization stats from `--split train`.
3. Train with validation enabled.
4. Run `scripts/eval_offline.py` on the produced checkpoints.
5. Use `summary.csv` to pick the best checkpoint for serving or submission tests.
