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

- Date: 2026-03-13
  Experiment name: `generalist_lora_fast_step5000_eval`
  Config: `acot_challenge_generalist_lora_generalist`
  Hardware: 3x A100 40G (training), competition test server (eval)
  Batch size: 96
  FSDP devices: 3
  Precision / finetune mode: ACoT LoRA generalist (LoRA-everywhere, vision+LLM+experts frozen except LoRA)
  Dataset subset: full 15-part generalist task family
  Result: **5.08 total (vs baseline 6.35, regression of -1.27)**
  Notes:
  - checkpoint at step 20000 evaluated via `evaluation_result_generalist_lora_fast_step20000.xlsx`
  - per-task breakdown shows catastrophic forgetting on strong tasks:
    - open_door: 1.0 → 0.4 (-0.6, worst regression)
    - scoop_popcorn: 1.0 → 0.8 (-0.2)
    - hold_pot: 1.0 → 0.85 (-0.15)
    - clean_desktop: 0.39 → 0.19 (-0.2)
  - only stock_shelf improved: 0.24 → 0.40 (+0.16)
  - diagnosis: aggressive LR (4e-5), long schedule (40k steps), 20 randomly-initialized expert LoRA tensors
  - val loss trajectory: 5k=0.1765, 10k=0.2019, 15k=0.1721, 20k=0.1698 — non-monotonic, best unsaved region near 17k
  - conclusion: LoRA-everywhere with aggressive schedule causes catastrophic forgetting, val loss does not predict benchmark score

- Date: 2026-03-14
  Experiment name: `competition_strategy_overhaul`
  Config: new `acot_challenge_lora_conservative` (+ fixes to baseline-compatible and specialist configs)
  Hardware: code-only session
  Batch size: n/a
  FSDP devices: n/a
  Precision / finetune mode: n/a
  Dataset subset: n/a
  Result: configs created, strategy established
  Notes:
  - added `acot_challenge_lora_conservative`: warmup=200, lr=1e-5, 8k steps, save/val every 500
  - fixed baseline-compatible warmup bug (was 10000==total steps)
  - added `--lora-only` to `extract_adapter.py` for lightweight routing
  - updated specialist configs to support baseline init via `ACOT_CHALLENGE_INIT_WEIGHTS`
  - updated `TASK_ROUTING` to only route weak tasks (clean_desktop, stock_shelf, place_block)
  - all configs verified loading correctly
