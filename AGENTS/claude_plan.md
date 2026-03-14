# Plan: Specialist Adapter Training + Routing from Baseline

## Context

Track A (conservative LoRA generalist) and Track B (baseline-compatible) cannot beat baseline.
Goal: train per-task LoRA adapters for the 3 weak tasks from baseline, route those tasks to
specialists at inference, strong tasks use pure baseline behavior.

## The Random LoRA Init Problem at Inference

The LoRA model loaded from baseline has randomly-initialized LoRA params (`normal * 0.02`) for
both PaliGemma and both expert branches (none of these are in the baseline checkpoint).

At inference with adapter routing:
- Weak tasks: specialist LoRA **replaces** the random LoRA → correct specialist behavior ✓
- Strong tasks (no adapter): random LoRA stays → small perturbation from baseline ✗

**Fix**: create a `_default.npz` adapter with all LoRA tensors set to **zero**.
When strong tasks fall through to `_default`, their LoRA is zeroed out = exact baseline behavior.
Zero LoRA contribution = `x @ 0 @ 0 * scaling = 0`, so output = frozen base weights only = baseline.

**New script needed**: `scripts/create_zero_lora_adapter.py`
Initializes the LoRA model from baseline checkpoint with `missing_init="zeros"`, then immediately
extracts all LoRA tensors (all zeros), saves as `_default.npz`. No training needed.

## Phase 1: Create Zero-LoRA `_default` Adapter

```bash
python scripts/create_zero_lora_adapter.py \
  --checkpoint <baseline_checkpoint>/params \
  --output adapters/_default.npz
```

This script:
1. Calls `ACOTCheckpointWeightLoader(params_path, missing_init="zeros").load(model_params)`
2. Flattens params, filters to `.*lora.*` paths
3. Saves as npz in the same format as `extract_adapter.py` output

**Critical files**:
- `src/openpi/training/weight_loaders.py` — `ACOTCheckpointWeightLoader.load()` already supports `missing_init="zeros"`
- `src/openpi/models/model.py` — `restore_params()` for loading
- `scripts/extract_adapter.py` — reuse the npz serialization format (`PATHS_KEY`, `VALUE_KEY_TEMPLATE`)

## Phase 2: Train 3 Specialists from Baseline

Existing configs in `_make_reasoning2action_specialist_configs()`:
- `acot_specialist_clean_desktop` — `clean_the_desktop_part_1` + `clean_the_desktop_part_2`
- `acot_specialist_stock_shelf` — `stock_and_straighten_shelf` + `stock_and_straighten_shelf_part_2`
- `acot_specialist_place_block` — `place_block_into_box`

Hyperparams already set: warmup=200, peak_lr=1e-5, 5000 steps, save every 500, batch=18, grad_accum=4.
Weight loader: `ACOTCheckpointWeightLoader` from baseline (since `ACOT_CHALLENGE_GENERALIST_WEIGHTS` not set).

Launch (one per available GPU, or sequentially):
```bash
ACOT_CHALLENGE_INIT_WEIGHTS=<baseline>/params \
  bash scripts/train_fast.sh acot_specialist_clean_desktop exp_specialist_clean --r2a-cache-root=<path>

ACOT_CHALLENGE_INIT_WEIGHTS=<baseline>/params \
  bash scripts/train_fast.sh acot_specialist_stock_shelf exp_specialist_stock --r2a-cache-root=<path>

ACOT_CHALLENGE_INIT_WEIGHTS=<baseline>/params \
  bash scripts/train_fast.sh acot_specialist_place_block exp_specialist_place --r2a-cache-root=<path>
```

## Phase 3: Evaluate and Extract Specialists

**Offline eval** on all 10 checkpoints per specialist:
```bash
uv run scripts/eval_offline.py \
  --config-name acot_specialist_clean_desktop \
  --checkpoint checkpoints/acot_specialist_clean_desktop/exp_specialist_clean/<step> \
  --output-dir eval_results/specialists/clean_desktop/
```

**Rollout eval** on top 2-3 candidates per task on test server. Pick best checkpoint per specialist.

**Extract LoRA-only adapters** from best checkpoints:
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

## Phase 4: Serve with Routing

Base checkpoint at inference: baseline checkpoint (NOT a trained checkpoint).
`_default.npz`: zero LoRA → strong tasks behave exactly as baseline.
Specialist adapters: replace LoRA for weak tasks.

```bash
python scripts/serve_policy.py policy:adapter-routed \
  --policy.config-name acot_challenge_lora_conservative \
  --policy.checkpoint <baseline_checkpoint> \
  --policy.adapter-dir adapters/
```

TASK_ROUTING (already set in `adapter_routed_policy.py`):
- `clean_the_desktop` → `acot_specialist_clean_desktop`
- `stock_and_straighten_shelf` → `acot_specialist_stock_shelf`
- `place_block_into_box` → `acot_specialist_place_block`
- `grab_toy` → `acot_specialist_place_block`
- All others → `_default` (zero LoRA = pure baseline)

## Code Changes Required

### 1. New script `scripts/create_zero_lora_adapter.py`

Roughly:
```python
import pathlib
import numpy as np
import tyro
import flax.traverse_util
from openpi.models import model as _model
from openpi.training.weight_loaders import ACOTCheckpointWeightLoader
from openpi.training.config import get_config

def main(checkpoint: str, output: str, config_name: str = "acot_challenge_lora_conservative"):
    cfg = get_config(config_name)
    # Build model params structure
    model = cfg.model.create(jax.random.PRNGKey(0))
    params = nnx.state(model)
    # Load from baseline with zeros for missing LoRA
    loader = ACOTCheckpointWeightLoader(checkpoint, missing_init="zeros")
    loaded = loader.load(params)
    # Extract LoRA-only
    flat = flax.traverse_util.flatten_dict(loaded, sep="/")
    lora_params = {k: v for k, v in flat.items() if "lora" in k}
    # Save in same format as extract_adapter.py
    paths = list(lora_params.keys())
    payload = {
        "__paths__": np.asarray(paths, dtype=str),
        **{f"param_{i:04d}": v for i, v in enumerate(lora_params.values())}
    }
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    print(f"Wrote {len(paths)} zero LoRA tensors to {output}")

if __name__ == "__main__":
    tyro.cli(main)
```

### 2. No other code changes needed

- Specialist configs: already correctly configured (Phase 1 of previous session)
- `extract_adapter.py --lora-only`: already added
- `TASK_ROUTING`: already updated to route only 3 weak tasks
- `ACOTCheckpointWeightLoader(missing_init="zeros")`: already supported

## Verification

1. Run `create_zero_lora_adapter.py` and verify output npz contains only zero arrays
2. Load routed policy with baseline checkpoint + adapters dir (with only `_default.npz` present) and verify inference output matches baseline
3. After specialist training, run rollout eval on each specialist's task and confirm improvement over baseline for that task
4. Run full 10-task rollout eval on routed system and compare against 6.35

## Hard Rules
- Never submit below 6.35
- Abort specialist if val loss diverges by >50% from step-500 value
- Evaluate against baseline before any submission
