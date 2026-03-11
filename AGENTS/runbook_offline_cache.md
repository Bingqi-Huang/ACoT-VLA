# Offline Cache Runbook

## Purpose

This runbook covers the additive offline shard-cache training path for `acot_challenge_generalist_lora_generalist`.

Use this path when the bottleneck is raw LeRobot episode parquet + HEVC mp4 decode on the A100 machine.

## Files

- `scripts/build_offline_cache.py`
- `scripts/verify_offline_cache.py`
- `scripts/train_offline_cache.py`
- `scripts/eval_offline_cache.py`
- `src/openpi/training/offline_cache.py`
- `src/openpi/training/data_loader_offline.py`

## Workflow

1. Build the cache on the faster U9 machine.
2. Copy the cache directory to the A100 machine.
3. Run `scripts/verify_offline_cache.py` on the A100 machine before training.
4. Train with `scripts/train_offline_cache.py`.
5. Evaluate with either:
   - existing `scripts/eval_offline.py`
   - new `scripts/eval_offline_cache.py`

## Design Constraints

- Additive only. Do not replace or modify legacy `train.py`, `eval_offline.py`, `data_loader.py`, or config registration.
- Checkpoint format must remain compatible with existing serving and offline evaluation flows.
- Cache manifest validation must fail closed on config/model/norm-stats/split mismatch.

## Known Tradeoff

Prompt injection in the raw ACOT transform path is stochastic. The offline cache path currently freezes it deterministically per sample index for reproducibility and exact cache verification.
