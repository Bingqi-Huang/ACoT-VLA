# Specialist Adapter Training & Routing

Strategy: train per-task LoRA adapters for the 3 weak tasks from the baseline checkpoint, then
route those tasks to specialists at inference. Strong tasks fall through to `_default` (zero LoRA
= pure baseline behavior).

## LoRA Hyperparameters

### Architecture

| Branch | Variant | Depth | Width | LoRA rank | LoRA alpha | Scaling (`alpha/rank`) |
|---|---|---|---|---|---|---|
| PaliGemma backbone | `gemma_2b_lora` | 18 | 2048 | **16** | 16.0 | 1.0 |
| Coarse action expert | `gemma_300m_lora` | 18 | 1024 | **32** | 32.0 | 1.0 |
| Fine action expert | `gemma_300m_lora` | 18 | 1024 | **32** | 32.0 | 1.0 |

LoRA applied to: **attn** (q, k, v, out projections) and **ffn** (gate, linear projections) in
every layer of all 3 Gemma branches.
Init: `normal(stddev=0.01)` for both `lora_a` and `lora_b` (non-standard — `lora_b` is not
zero-initialized as in the original LoRA paper).

### Freeze filter (shared by generalist and all specialists)

```
Frozen:   vision encoder, LLM dense weights, LLM embedder, both action experts (dense)
Trainable: all LoRA params (3 branches), ACoT modules (reasoners, projections, time MLPs)
```

`get_freeze_filter(freeze_vision=True, freeze_llm=True, freeze_llm_embedder=True, freeze_dual_ae=[True, True])`

### Training schedule (specialists)

Steps are calibrated to ~1–2 epochs per task based on measured frame counts.

| Specialist | Datasets | Total frames | Steps | Approx. epochs |
|---|---|---|---|---|
| `acot_specialist_clean_desktop` | part_1 + part_2 + **addition** | 1.13M | **8 000** | 0.85 |
| `acot_specialist_stock_shelf` | shelf + shelf_part_2 | 200k | **3 000** | 1.8 |
| `acot_specialist_place_block` | place_block | 188k | **3 000** | 1.9 |

Note: `clean_the_desktop_addition` (211 ep, 568k frames) must be included — omitting it loses
half the clean-desktop training data.

| Param | Value |
|---|---|
| `batch_size` | 120 |
| `grad_accum_steps` | 1 |
| Effective batch size | **120** |
| `num_workers` | 24 |
| Optimizer | AdamW, `clip_gradient_norm=1.0` |
| LR schedule | Cosine decay over `num_train_steps` |
| `warmup_steps` | 200 |
| `peak_lr` | 1e-5 |
| `decay_lr` | 1e-6 |
| `ema_decay` | None |
| `save_interval` | 500 |
| `val_interval` | 500 |
| `val_num_batches` | 32 |
| Weight init | `ACOTCheckpointWeightLoader` from baseline (missing LoRA → zeros) |

## Task Routing

Only the 3 weakest tasks are routed to specialists. All others fall through to `_default`
(zero expert LoRA = pure baseline).

| Task name (websocket `task_name`) | Adapter |
|---|---|
| `clean_the_desktop` | `acot_specialist_clean_desktop` |
| `stock_and_straighten_shelf` | `acot_specialist_stock_shelf` |
| `place_block_into_box` | `acot_specialist_place_block` |
| `grab_toy` | `acot_specialist_place_block` |
| All others | `_default` (zero LoRA = baseline) |

## Step-by-Step Runbook

### Phase 0: Prerequisites

```bash
# Baseline checkpoint must be present at:
./checkpoints/baseline_checkpoint/params/

# Frame cache (optional, speeds up data loading):
<your_cache_root>/
```

### Phase 1: Create Zero-LoRA Default Adapter

```bash
uv run scripts/create_zero_lora_adapter.py \
  --checkpoint ./checkpoints/baseline_checkpoint/params \
  --output adapters/_default.npz
```

**What this produces:** `adapters/_default.npz` with 30 LoRA tensors:
- 10 non-zero: PaliGemma backbone LoRA loaded from baseline (trained weights, correct)
- 20 zero: expert branch LoRA (`_1`, `_2` suffixes) that don't exist in baseline

Strong tasks routed to `_default` will overlay these values onto the base model → exact
baseline inference behavior.

### Phase 2: Train 3 Specialists from Baseline

Run sequentially (same GPU) or in parallel (separate GPUs). Each takes ~5 000 steps.

```bash
# Clean desktop (weak task, baseline score ~0.39)
ACOT_CHALLENGE_INIT_WEIGHTS=./checkpoints/baseline_checkpoint/params \
  bash scripts/train_fast.sh acot_specialist_clean_desktop exp_specialist_clean \
  --r2a-cache-root=<your_cache_root>

# Stock shelf (weak task, baseline score ~0.24)
ACOT_CHALLENGE_INIT_WEIGHTS=./checkpoints/baseline_checkpoint/params \
  bash scripts/train_fast.sh acot_specialist_stock_shelf exp_specialist_stock \
  --r2a-cache-root=<your_cache_root>

# Place block (weak task, baseline score ~0.42)
ACOT_CHALLENGE_INIT_WEIGHTS=./checkpoints/baseline_checkpoint/params \
  bash scripts/train_fast.sh acot_specialist_place_block exp_specialist_place \
  --r2a-cache-root=<your_cache_root>
```

Abort a specialist if val loss at step 1000 is >50% higher than at step 500 (divergence signal).

### Phase 3: Select Best Checkpoint (Offline Eval)

```bash
# Offline eval on all 10 checkpoints for each specialist
for step in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000; do
  uv run scripts/eval_offline.py \
    --config-name acot_specialist_clean_desktop \
    --checkpoint checkpoints/acot_specialist_clean_desktop/exp_specialist_clean/${step} \
    --output-dir eval_results/specialists/clean_desktop/
done
```

Pick the checkpoint with lowest val loss per specialist (or run a rollout eval on the top 2-3).

### Phase 4: Extract LoRA-Only Adapters

```bash
python scripts/extract_adapter.py \
  --checkpoint checkpoints/acot_specialist_clean_desktop/exp_specialist_clean/<best_step> \
  --output adapters/acot_specialist_clean_desktop.npz --lora-only

python scripts/extract_adapter.py \
  --checkpoint checkpoints/acot_specialist_stock_shelf/exp_specialist_stock/<best_step> \
  --output adapters/acot_specialist_stock_shelf.npz --lora-only

python scripts/extract_adapter.py \
  --checkpoint checkpoints/acot_specialist_place_block/exp_specialist_place/<best_step> \
  --output adapters/acot_specialist_place_block.npz --lora-only
```

`--lora-only` extracts only the LoRA tensors (~220 MB per adapter in bfloat16), not the full
ACoT module weights. This keeps per-adapter memory low.

### Phase 5: Serve with Routing

Base checkpoint at inference: **baseline checkpoint** (not a trained specialist).

```bash
python scripts/serve_policy.py policy:adapter-routed \
  --policy.config acot_challenge_lora_conservative \
  --policy.base-checkpoint ./checkpoints/baseline_checkpoint \
  --policy.adapter-dir adapters/
```

The adapter dir must contain at minimum:
```
adapters/
  _default.npz                         # zero expert LoRA = baseline behavior
  acot_specialist_clean_desktop.npz    # LoRA only
  acot_specialist_stock_shelf.npz      # LoRA only
  acot_specialist_place_block.npz      # LoRA only
```

### Phase 6: Verify Before Submission

```
Hard rule: never submit below 6.35 (current baseline score).
```

1. Run offline eval on each specialist's task and confirm improvement over baseline.
2. Run a full 10-task rollout eval on the routed system.
3. Compare against 6.35 before any submission.
4. If a specialist underperforms its task vs baseline, remove its npz — it will fall through
   to `_default` (baseline behavior) automatically.

## Adapter File Sizes

| File | Contents | Approx. size |
|---|---|---|
| `_default.npz` | 30 LoRA tensors (backbone from baseline + zeroed expert) | ~220 MB |
| `acot_specialist_*.npz` | LoRA-only (30 tensors per specialist) | ~220 MB each |
| Total (4 files) | | ~880 MB |

## Key Design Decisions

**Why zero expert LoRA for `_default`?**
The LoRA model loaded from baseline has randomly-initialized expert LoRA (`normal * 0.02`) for
the two action expert branches, which slightly perturbs baseline behavior on strong tasks. The
`_default.npz` replaces those random values with zeros, restoring exact baseline outputs.

**Why LoRA-only adapters (not full ACoT module)?**
The ACoT modules (reasoners, projections, time MLPs) are already in the baseline checkpoint and
shared across all tasks. Specialist adapters need only the task-specific LoRA delta.

**Why init specialists from baseline (not a trained generalist)?**
Track A (conservative LoRA generalist) and Track B (baseline-compatible) could not beat baseline
(6.35). Direct specialist fine-tuning from baseline avoids compounding any generalist regression.
