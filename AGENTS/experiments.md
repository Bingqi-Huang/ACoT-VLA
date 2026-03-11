# Experiments

This is a placeholder tracker for training and evaluation experiments.

## Suggested Columns

- Date
- Experiment name
- Config
- Hardware
- Batch size
- FSDP devices
- Precision / finetune mode
- Dataset subset
- Result
- Notes

## Entries

- Date: 2026-03-11
  Experiment name: `fast_train_path_additive_impl`
  Config: `acot_challenge_generalist_lora_generalist`
  Hardware: code-only implementation session, mixed references from old 3x A100 box and newer 2x RTX 5090 box
  Batch size: n/a
  FSDP devices: n/a
  Precision / finetune mode: ACoT LoRA generalist
  Dataset subset: full generalist task family
  Result: additive fast path implemented
  Notes:
  - added `scripts/train_fast.py`, `scripts/train_fast.sh`
  - added `src/openpi/training/data_loader_fast.py`
  - added `scripts/precompute_subtask_index_cache.py`
  - added `scripts/precompute_prompt_cache.py`
  - fast path preserves legacy checkpoint layout and is intended to remain compatible with offline eval and checkpoint-based serving

- Date: 2026-03-11
  Experiment name: `generalist_data_layout_inspection`
  Config: `acot_challenge_generalist_lora_generalist`
  Hardware: dataset inspection only
  Batch size: n/a
  FSDP devices: n/a
  Precision / finetune mode: n/a
  Dataset subset: `/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim`
  Result: data format strongly suggests random I/O + HEVC decode pressure
  Notes:
  - dataset root size about `173G`
  - about `4179` parquet files
  - about `12778` mp4 files
  - each episode stored as its own parquet plus per-camera mp4 files
  - sample videos are HEVC; wrist images are high resolution (`1056x1280`)

- Date: 2026-03-11
  Experiment name: `dual_gpu_train_fast_startup_debug`
  Config: `acot_challenge_generalist_lora_generalist`
  Hardware: 2x RTX 5090, 32G each, NVMe, DDR5 256G
  Batch size: debug batch size from config
  FSDP devices: 2
  Precision / finetune mode: ACoT LoRA generalist
  Dataset subset: full generalist task family
  Result: unresolved
  Notes:
  - single-GPU training enters progress quickly and can drive GPU power much higher
  - multi-GPU `train_fast.py` initially appeared to stall near first-batch / preview stage; several additive fixes were applied
  - later observation suggests original `scripts/train.py` also stalls or spends a very long time around the same startup / first-step area
  - current hypothesis is that the main blocker is not fast-loader-specific but related to dual-GPU JAX/XLA first-step initialization or compilation
