# Plan: Task-Routed LoRA Adapters for Reasoning2Action Challenge

## 1. Strategy Summary

Train one **generalist** ACoT-VLA model with LoRA on all three Gemma branches, then fine-tune **9 specialist adapters** from it. At inference, route on `payload["task_name"]`, which in the current Genie Sim ICRA chain is the **sub-task key** rather than the outer benchmark scene name. This yields **10 public route keys backed by 9 specialist adapters** because `sorting_packages_continuous` should reuse the `sorting_packages` adapter. This gives the specialization benefit of separate models at the memory cost of ~1.3 models.

Same-task dataset folders with names like `*_part_2` or `*_part_3` are storage shards of the same routed task family. They should be merged into the same training config and must not create new route keys.

## 2. Why This Wins

The evaluation protocol hands us the effective route key for free: websocket `payload["task_name"]` is the ICRA `sub_task_name`. This is the correct oracle router. Instead of one generalist compromising across the public ICRA sub-tasks, each sub-task family gets its own specialized adapter, while `sorting_packages_continuous` safely shares the sorting adapter because prompt-only routing is too ambiguous there.

| Approach | Specialization | Inference Memory | Engineering Cost |
|---|---|---|---|
| Single generalist | Low | 1x | Low |
| N separate full models | High | Nx | Medium |
| **Per-task LoRA adapters** | **High** | **~1.3x** | **Medium** |

The adapters (LoRA weights + ACoT modules) are ~140MB per task. 9 adapters total ~1.3GB on top of the ~6GB base model. JIT compilation is shared across all adapters since param shapes are identical.

## 3. Prerequisites and Bug Fixes

### 3.1 Fix `Variant` Type (BLOCKING)

**File:** `src/openpi/models/gemma.py:55`

`gemma_300m_lora` exists in `get_config()` (line 145) but is missing from the type alias. Any config using `coarse_action_expert_variant="gemma_300m_lora"` will fail.

**Before:**
```python
Variant = Literal["dummy", "gemma_300m", "gemma_2b", "gemma_2b_lora"]
```

**After:**
```python
Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]
```

### 3.2 Fix Freeze Filter LoRA Detection (BLOCKING)

**File:** `src/openpi/models/acot_vla.py:360`

The freeze filter does not check `coarse_action_expert_variant` for LoRA. If you freeze the coarse expert but use LoRA on it, the LoRA params get frozen too — training is silently broken.

**Before:**
```python
has_lora = "lora" in self.paligemma_variant or "lora" in self.action_expert_variant
```

**After:**
```python
has_lora = "lora" in self.paligemma_variant or "lora" in self.coarse_action_expert_variant or "lora" in self.action_expert_variant
```

### 3.3 Verification

After both fixes, run the existing tests to confirm nothing breaks:
```bash
uv run pytest src/openpi/models/ -x -q
```

Then verify the freeze filter works correctly for the LoRA-everywhere config by adding a quick smoke test (see Test Plan section).

## 4. Gradient Accumulation

The codebase has **no gradient accumulation** (`scripts/train.py`). The baseline batch_size=256 will not fit on 3xA100 40G without FSDP. We need gradient accumulation to achieve reasonable effective batch sizes with smaller per-device batches.

### 4.1 Add `grad_accum_steps` to TrainConfig

**File:** `src/openpi/training/config.py`, inside class `TrainConfig` (after line 1184)

Add a new field:
```python
# Number of gradient accumulation steps. Effective batch = batch_size * grad_accum_steps.
grad_accum_steps: int = 1
```

### 4.2 Implement Accumulation in train.py

**File:** `scripts/train.py`

The current pattern (lines 301-343) is:
```python
ptrain_step = jax.jit(functools.partial(acot_train_step, config), ...)
for step in pbar:
    train_state, info = ptrain_step(train_rng, train_state, batch)
    batch = next(data_iter)
```

**Refactor into two JIT-compiled functions:**

1. `acot_compute_grads(config, rng, state, batch)` → `(grads, loss, metrics)` — computes gradients for one micro-batch using the existing `nnx.DiffState` and `nnx.value_and_grad` pattern, but does NOT apply the optimizer. It should return the trainable gradients (filtered by `config.trainable_filter`), the loss scalar, and any auxiliary metrics.

2. `acot_apply_grads(config, rng, state, grads)` → `(new_state, metrics)` — applies averaged gradients via `state.tx.update`, does the `optax.apply_updates`, computes EMA update, computes grad_norm/param_norm, increments `state.step`.

**Modified training loop:**

```python
pcompute_grads = jax.jit(
    functools.partial(acot_compute_grads, config),
    in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
    out_shardings=(train_state_sharding, replicated_sharding, replicated_sharding),
)
papply_grads = jax.jit(
    functools.partial(acot_apply_grads, config),
    in_shardings=(replicated_sharding, train_state_sharding, train_state_sharding),
    out_shardings=(train_state_sharding, replicated_sharding),
    donate_argnums=(1,),
)

for step in pbar:
    accumulated_grads = None
    total_loss = 0.0

    for micro_step in range(config.grad_accum_steps):
        grads, loss, _ = pcompute_grads(train_rng, train_state, batch)
        total_loss = total_loss + loss
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = jax.tree.map(jnp.add, accumulated_grads, grads)
        batch = next(data_iter)

    avg_grads = jax.tree.map(lambda g: g / config.grad_accum_steps, accumulated_grads)
    train_state, info = papply_grads(train_rng, train_state, avg_grads)
    info["loss"] = total_loss / config.grad_accum_steps
    # ... logging, checkpointing (same as current code)
```

**Important implementation notes for the coding agent:**
- The `acot_compute_grads` function should follow the same pattern as the current `acot_train_step` lines 199-215 but stop after computing `grads` — do NOT call `state.tx.update`.
- Use `nnx.DiffState(0, config.trainable_filter)` for gradient filtering, same as the current code (line 214).
- The grads returned by `pcompute_grads` are nnx.State objects filtered to trainable params. Accumulation via `jax.tree.map(jnp.add, ...)` works on these.
- `state.params` is constant during the accumulation loop (no optimizer step between micro-batches). Only the data batch changes.
- `train_rng` can be folded in per micro-step using `jax.random.fold_in(rng, state.step * config.grad_accum_steps + micro_step)`.
- The standard (non-ACoT) `train_step` should be refactored similarly for consistency, but ACoT is the priority.
- When `grad_accum_steps=1`, the behavior must be equivalent to the original code.

### 4.3 Verification

```bash
DEBUG_MODE=true uv run python scripts/train.py \
    --config-name acot_icra_simulation_challenge_reasoning_to_action_local \
    --exp_name grad_accum_test \
    --batch_size 4 \
    --grad_accum_steps 2 \
    --num_train_steps 20
```

Acceptance:
- Training runs without OOM on a single GPU
- Loss decreases over 20 steps
- Checkpoint saves successfully

## 5. Training Configs

### 5.1 Generalist Config (LoRA-Everywhere)

**File:** `src/openpi/training/config.py` — add new `TrainConfig` entry before the closing `]` of `_CONFIGS`.

**Config name:** `acot_challenge_generalist_lora_all`

Key differences from the baseline `acot_icra_simulation_challenge_reasoning_to_action_local`:
- `coarse_action_expert_variant="gemma_300m_lora"` (was `"gemma_300m"`)
- `action_expert_variant="gemma_300m_lora"` (was `"gemma_300m"`)
- `freeze_vision=True` (was `False`)
- `freeze_dual_ae=[True, True]` (was `[False, False]`)
- `batch_size=18` (6 per GPU on 3xA100 40G)
- `grad_accum_steps=4` (effective batch=72)
- `num_train_steps=50_000`

This freezes ALL dense weights (vision, backbone, both experts) and trains ONLY:
- LoRA params on all 3 Gemma branches
- All ACoT-specific modules (reasoners, projections, time MLPs)

These trainable params ARE the "adapter" that gets specialized per task later.

**Freeze filter construction:**
```python
freeze_filter=acot_vla.ACOTConfig(
    paligemma_variant="gemma_2b_lora",
    coarse_action_expert_variant="gemma_300m_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter(
    freeze_vision=True,
    freeze_llm=True,
    freeze_llm_embedder=True,
    freeze_dual_ae=[True, True],
)
```

**Use the same data config** as the local baseline: same `repo_id` list (all 9 tasks), same `prompt_map_inject_to_training`, same transforms, same `subtask` sampler, same `joint_action_shifts=(2, 1)`, same `extra_delta_transform=(True, True)`.

**Use the same weight loader** as the local baseline (`ACOTCheckpointWeightLoader`). The loader handles missing LoRA params by initializing them randomly, which is correct.

### 5.2 Specialist Configs (Per-Task)

**File:** `src/openpi/training/config.py`

Create 9 specialist configs programmatically. Each specialist:
- Uses the **same model config** as the generalist (LoRA-everywhere, same architecture)
- Uses the **same freeze filter** (only LoRA + ACoT modules are trainable)
- Points to a **single task's dataset** in `repo_id`
- Uses `CheckpointWeightLoader` pointing to the **generalist checkpoint** (not the base pi05 weights)
- Uses **the generalist's norm stats** (set `asset_id` to the generalist's computed stats path)
- Uses a **shorter training schedule**: `num_train_steps=10_000`, smaller warmup
- Uses a **lower learning rate**: peak_lr=2e-5, cosine decay over 10K steps, warmup=500

**Task-to-config mapping:**

| Config Name | Task | repo_id |
|---|---|---|
| `acot_specialist_pour_workpiece` | Pour workpiece | `pour_workpiece` |
| `acot_specialist_open_door` | Open door | `open_door` |
| `acot_specialist_scoop_popcorn` | Scoop popcorn | `scoop_popcorn` |
| `acot_specialist_hold_pot` | Hold pot | `hold_pot` |
| `acot_specialist_place_block` | Place block into box | `place_block_into_box` |
| `acot_specialist_take_wrong_item` | Take wrong item from shelf | `take_wrong_item_shelf` |
| `acot_specialist_stock_shelf` | Stock and straighten shelf | all `stock_and_straighten_shelf*` shards, currently `stock_and_straighten_shelf`, `stock_and_straighten_shelf_part_2` |
| `acot_specialist_sorting` | Sorting packages | all `sorting_packages*` shards, currently `sorting_packages_part_1`, `sorting_packages_part_2`, `sorting_packages_part_3` |
| `acot_specialist_clean_desktop` | Clean the desktop | all `clean_the_desktop*` shards, currently `clean_the_desktop_part_1`, `clean_the_desktop_part_2` |

**Implementation note:** To reduce config boilerplate, define a helper function that generates a specialist `TrainConfig` given a task name, the full shard list for that task family, and the prompt map subset. Loop over the task table above and append to `_CONFIGS`.

Each specialist's `prompt_map_inject_to_training` should include ONLY the relevant task's entry from the generalist's full prompt map. The prompt map keys are the dataset's task labels (e.g., `"Unload workpiece_icra_SIM"` for pour_workpiece).

### 5.3 Norm Stats

All specialists must use the **generalist's norm stats**, not task-specific stats. This ensures the adapter swap doesn't need to swap normalization parameters.

For the generalist: compute norm stats before first training.
```bash
uv run scripts/compute_norm_stats.py --config-name acot_challenge_generalist_lora_all
```

For specialists: set `asset_id` in the `AssetsConfig` to point to the generalist's computed stats directory. The specialist configs should also set `assets_dir` to the same path as the generalist.

## 6. Training Order

### 6.1 Generalist (Phase 1)

```bash
# Compute norm stats
uv run scripts/compute_norm_stats.py --config-name acot_challenge_generalist_lora_all

# Train (3xA100 40G, no FSDP)
bash scripts/train.sh acot_challenge_generalist_lora_all generalist_v1
```

Acceptance:
- Loss converges over 50K steps
- Checkpoint at step 50K (or best by loss) is usable for specialist init

### 6.2 Specialists (Phase 2, parallelizable)

After the generalist finishes, train all 9 specialists. These can run in parallel on different GPU sets if available, or sequentially on the same machine. Each takes ~10K steps (much shorter than generalist).

```bash
# Example for one specialist
bash scripts/train.sh acot_specialist_pour_workpiece specialist_v1
```

Each specialist's `CheckpointWeightLoader` must point to the generalist's best checkpoint:
```
checkpoints/acot_challenge_generalist_lora_all/generalist_v1/<best_step>/params
```

## 7. Adapter Extraction

### 7.1 New Script: `scripts/extract_adapter.py`

**Purpose:** Extract only the adapter params (LoRA + ACoT modules) from a specialist checkpoint and save them as a compact `.npz` file.

**Adapter param identification:** A param is an adapter param if its flattened path matches ANY of these patterns:
```python
ADAPTER_PATTERNS = [
    "lora",                            # All LoRA params in all 3 Gemma branches
    "coarse_action_in_proj",           # Coarse action input projection
    "action_in_proj",                  # Fine action input projection
    "coarse_action_out_proj",          # Coarse action output projection
    "action_out_proj",                 # Fine action output projection
    "coarse_time_mlp_in",             # Coarse time embedding MLP
    "coarse_time_mlp_out",
    "time_mlp_in",                    # Fine time embedding MLP
    "time_mlp_out",
    "explicit_action_reasoner",       # EAR module
    "implicit_action_reasoner",       # IAR module (includes _interact)
    "action_reasoning_fusion",        # Fusion module
    "explicit_action_reason_proj",    # Reasoning projection
    "implicit_action_reason_proj",    # Reasoning projection
]
```

**Logic:**
1. Load the specialist checkpoint using `model.restore_params(checkpoint_path / "params", restore_type=np.ndarray)`
2. Flatten the params dict with `flax.traverse_util.flatten_dict(params, sep="/")`
3. Filter to only keys where ANY pattern in `ADAPTER_PATTERNS` appears as a substring
4. Save the filtered flat dict as a numpy `.npz` file

**CLI:**
```bash
uv run python scripts/extract_adapter.py \
    --checkpoint checkpoints/acot_specialist_pour_workpiece/specialist_v1/10000 \
    --output adapters/pour_workpiece.npz
```

### 7.2 Also Extract the Generalist Adapter

Extract the generalist's adapter params as the **fallback adapter** for unknown task names:
```bash
uv run python scripts/extract_adapter.py \
    --checkpoint checkpoints/acot_challenge_generalist_lora_all/generalist_v1/50000 \
    --output adapters/_default.npz
```

### 7.3 Adapter Directory Structure

```
adapters/
  _default.npz                  # Generalist fallback
  pour_workpiece.npz
  open_door.npz
  scoop_popcorn.npz
  hold_pot.npz
  place_block_into_box.npz
  take_wrong_item_shelf.npz
  stock_and_straighten_shelf.npz
  sorting_packages.npz
  clean_the_desktop.npz
```

Each `.npz` file is ~140MB. Total: ~1.4GB.

## 8. Adapter-Routed Policy

### 8.1 New File: `src/openpi/policies/adapter_routed_policy.py`

This is the key serving component. It loads one base model and 9+ adapter param sets, then swaps the correct adapter into the model's state based on websocket `payload["task_name"]` before each inference call.

**Architecture:**

```
┌──────────────────────────────────────────┐
│        AdapterRoutedPolicy               │
│                                          │
│  ┌─────────────┐   ┌─────────────────┐   │
│  │  Base Model  │   │  Adapter Store  │   │
│  │  (shared     │   │  ┌───────────┐  │   │
│  │   dense      │   │  │ _default  │  │   │
│  │   weights)   │   │  │ pour_wp   │  │   │
│  │             │   │  │ open_door │  │   │
│  │             │   │  │ ...       │  │   │
│  └─────────────┘   │  └───────────┘  │   │
│         │           └────────┬────────┘   │
│         │  graphdef          │ adapter    │
│         ▼                    ▼ diff       │
│  ┌──────────────────────────────────┐     │
│  │  _jitted_sample(state, rng, obs) │     │
│  │  (compiled once, reused for all) │     │
│  └──────────────────────────────────┘     │
└──────────────────────────────────────────┘
```

**Key implementation details:**

1. At init, split the base model into `(graphdef, base_state)` using `nnx.split(model)`.

2. Load each adapter `.npz` file. Store them as `{adapter_name: {flat_string_path: np.ndarray}}`.

3. Create ONE `jax.jit`-compiled sample function that takes `(state, rng, observation)`. Since `graphdef` is constant, capture it in the closure. The function does `nnx.merge(graphdef, state)` then calls `model.sample_actions(...)`. This follows the same pattern as `module_jit` in `src/openpi/shared/nnx_utils.py:35-41` but takes state as an explicit argument instead of capturing it.

4. On task switch (when `task_name` changes):
   - Get the adapter diff for the new task
   - Build a new `nnx.State` by taking the base state's pure dict, flattening, replacing adapter paths with adapter values, unflattening, and creating a new State
   - Reference pattern: `state.replace_by_pure_dict(merged)` as used in `scripts/train.py:99`
   - Cache this as `self._current_state`

5. On each `infer()` call: pass `self._current_state` to `self._jitted_sample`. The JIT cache is reused since all states have identical shapes.

**Routing table** (defined in the policy or loaded from a config):
```python
TASK_ROUTING = {
    "pour_workpiece": "pour_workpiece",
    "open_door": "open_door",
    "scoop_popcorn": "scoop_popcorn",
    "hold_pot": "hold_pot",
    "place_block_into_box": "place_block_into_box",
    "take_wrong_item_shelf": "take_wrong_item_shelf",
    "stock_and_straighten_shelf": "stock_and_straighten_shelf",
    "sorting_packages": "sorting_packages",
    "sorting_packages_continuous": "sorting_packages",
    "clean_the_desktop": "clean_the_desktop",
}
# Unknown task_name → "_default" (generalist adapter)
```

Treat the list above as the public ICRA contract.

For the current project decisions, the final ICRA router should not keep undocumented aliases at all. A legacy alias such as `grab_toy -> place_block_into_box` should be removed rather than preserved defensively.

**The `post_process` method** should be inherited from the existing `Policy` class (handles waist actions for `sorting_packages`).

### 8.2 Modify `scripts/serve_policy.py`

Add a new policy creation path for the adapter-routed mode:

```python
@dataclasses.dataclass
class AdapterRouted:
    """Load a routed policy with per-task adapters."""
    config: str           # Base model config name
    base_checkpoint: str  # Path to generalist checkpoint dir
    adapter_dir: str      # Path to directory containing adapter .npz files
```

Update the `Args.policy` union type to include `AdapterRouted` as an option.

When `AdapterRouted` is selected:
1. Load the base model from `base_checkpoint` (same as current `Checkpoint` path)
2. Load all `.npz` files from `adapter_dir`
3. Create an `AdapterRoutedPolicy` with the model, adapters, and transforms

**Keep the existing `Checkpoint` and `Default` paths working** — only add, don't break.

### 8.3 Docker and Submission

**File:** `scripts/docker/serve_policy.Dockerfile`

The Dockerfile just needs to include the adapters directory:
```dockerfile
FROM registry.agibot.com/genie-sim/openpi_server:latest
COPY . .
CMD /bin/bash -c "./scripts/server_routed.sh 0 8999"
```

**New file:** `scripts/server_routed.sh`
```bash
cart_num=${1}
port=${2}

export TF_NUM_INTRAOP_THREADS=16
export CUDA_VISIBLE_DEVICES=${cart_num}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"

export PYTHONPATH=/root/openpi/src:${PYTHONPATH:-/app:/app/src}
GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py \
    --port ${port} \
    --policy adapter-routed \
    --policy.config acot_challenge_generalist_lora_all \
    --policy.base-checkpoint ./checkpoints/generalist/params \
    --policy.adapter-dir ./adapters
```

The Docker image must contain:
- The generalist base checkpoint (in `checkpoints/generalist/`)
- All adapter `.npz` files (in `adapters/`)
- Norm stats (in the checkpoint's `assets/` directory)

## 9. Adapter Params Definition (Reference)

These are the exact trainable params when using the LoRA-everywhere freeze config. They constitute the "adapter" that is specialized per task.

**LoRA params** (inside `PaliGemma/llm/`):
- All paths containing `lora` — covers `lora_a`, `lora_b`, `gating_einsum_lora_a`, `gating_einsum_lora_b`, `linear_lora_a`, `linear_lora_b` across all 18 layers of all 3 Gemma branches.

**ACoT module params** (top-level model attributes, NOT inside `PaliGemma/`):
- `coarse_action_in_proj/` — Linear(action_dim → coarse_expert_width)
- `action_in_proj/` — Linear(action_dim → expert_width)
- `coarse_action_out_proj/` — Linear(coarse_expert_width → action_dim)
- `action_out_proj/` — Linear(expert_width → action_dim)
- `coarse_time_mlp_in/` — Linear(width → width)
- `coarse_time_mlp_out/` — Linear(width → width)
- `time_mlp_in/` — Linear(width → width)
- `time_mlp_out/` — Linear(width → width)
- `explicit_action_reasoner/` — UnifiedAttentionModule (4-head cross-attention)
- `implicit_action_reasoner/` — DownsampleExtractor (cross-attention over backbone layers)
- `implicit_action_reasoner_interact/` — UnifiedAttentionModule
- `explicit_action_reason_proj/` — Linear(2*width → width)
- `implicit_action_reason_proj/` — Linear(2*width → width)
- `action_reasoning_fusion/` — UnifiedAttentionModule

**Base params** (shared, frozen, NOT included in adapters):
- `PaliGemma/img/` — SigLIP vision encoder (~400M params)
- `PaliGemma/llm/` excluding `lora` paths — Dense Gemma weights for backbone + both experts (~2.6B params)

## 10. Risk Mitigation

### If LoRA-everywhere is too weak

**Signal:** Generalist with LoRA-everywhere significantly underperforms the baseline (which trains experts fully).

**Mitigation:** Unfreeze the 300m action experts (set `freeze_dual_ae=[False, False]`). This makes the "adapter" larger (~600MB per task for each expert), but still much cheaper than full model copies. The specialist training will update expert weights fully. At inference, swap expert weights + LoRA + ACoT modules.

### If a specialist underperforms its task

**Signal:** Specialist score < generalist score on the specialist's owned task.

**Mitigation:** Use the generalist adapter (`_default`) for that task. The routing table is a config, not code — easy to change before submission.

### If inference memory is too tight

**Signal:** OOM when loading base model + all adapters.

**Mitigation:** Keep adapters on CPU as numpy arrays. Transfer to GPU only on task switch (~140MB transfer, takes <100ms, happens once per episode).

### If adapter swapping breaks JIT cache

**Signal:** Each task switch causes a full recompilation (~60s delay).

**Mitigation:** Pre-warm the JIT cache at server startup by running one dummy inference with each adapter. The JIT should cache based on shapes (not values), so one compilation covers all adapters. If it doesn't, pre-build full `nnx.State` objects per task at startup (costs ~54GB RAM but eliminates re-merge overhead).

## 11. Test Plan

### Unit Tests

- [ ] `gemma_300m_lora` variant resolves without error in `get_config()`
- [ ] Freeze filter with LoRA on all 3 branches keeps all LoRA params trainable (verify by inspecting `nnx.state(model, config.trainable_filter)` and confirming lora params appear)
- [ ] `grad_accum_steps=1` produces identical training dynamics to the original code (compare loss at step 1 with fixed seed)
- [ ] `grad_accum_steps=4` produces reasonable loss trajectory (loss decreases over 50 steps)
- [ ] Adapter extraction script produces `.npz` files containing only adapter paths
- [ ] Adapter-routed policy routes each known task to the correct adapter
- [ ] Unknown task falls back to `_default` adapter
- [ ] Adapter swap does NOT trigger JIT recompilation (measure second call time < 100ms)

### Training Smoke Tests

- [ ] 200-step debug run with generalist LoRA-everywhere config (no OOM on target hardware)
- [ ] 200-step debug run with one specialist config initialized from generalist checkpoint
- [ ] Checkpoint save/load roundtrip (save at step 100, resume, verify step continues)

### Serving Smoke Tests

- [ ] Adapter-routed policy loads and starts websocket server on port 8999
- [ ] `/healthz` endpoint returns 200
- [ ] Inference with `task_name="pour_workpiece"` returns valid action shape
- [ ] Inference with `task_name="sorting_packages"` returns actions with waist dims
- [ ] Inference with `task_name="sorting_packages_continuous"` reuses the sorting adapter
- [ ] Inference with `task_name="clean_the_desktop"` resolves to the clean-desktop adapter
- [ ] Inference with unknown `task_name` uses fallback adapter
- [ ] Docker image builds and starts server automatically

## 12. Milestones

### M0: Infrastructure Ready
- Bug fixes merged
- Gradient accumulation working
- Generalist config parses and runs for 200 debug steps
- **Target:** 2 days

### M1: Generalist Trained
- Generalist trains to convergence on all current Reasoning2Action training datasets
- Loss stabilizes
- First baseline score from submission
- **Target:** 5-7 days after M0 (depends on GPU time)

### M2: Specialists Trained + Adapters Extracted
- All planned specialists trained from generalist checkpoint
- All adapter `.npz` files extracted
- **Target:** 3-5 days after M1

### M3: Routed Serving Works
- Adapter-routed policy loads and serves correctly
- Local smoke test passes
- **Target:** 2 days after M2

### M4: Competition Submission
- Docker image built with all adapters
- Submitted to competition server
- Score received and compared to generalist-only baseline
- **Target:** 1 day after M3

### M5: Iterate (if time permits)
- Identify worst-performing tasks from submission scores
- Re-train specialists with more steps, different LR, or unfrozen experts
- Re-submit

## 13. File Change Index

| File | Change Type | Description |
|---|---|---|
| `src/openpi/models/gemma.py:55` | Edit | Add `gemma_300m_lora` to `Variant` type |
| `src/openpi/models/acot_vla.py:360` | Edit | Fix `has_lora` to check `coarse_action_expert_variant` |
| `src/openpi/training/config.py` | Edit | Add `grad_accum_steps` field to `TrainConfig` |
| `src/openpi/training/config.py` | Edit | Add generalist + 9 specialist configs |
| `scripts/train.py` | Edit | Implement gradient accumulation loop |
| `scripts/extract_adapter.py` | New | Script to extract adapter params from checkpoint |
| `src/openpi/policies/adapter_routed_policy.py` | New | Adapter-routed policy class |
| `scripts/serve_policy.py` | Edit | Add `AdapterRouted` policy mode |
| `scripts/server_routed.sh` | New | Routed serving launch script |
| `scripts/docker/serve_policy.Dockerfile` | Edit | Use `server_routed.sh` for CMD |

## 14. Assumptions

- The evaluation server provides websocket `payload["task_name"]` in every observation, and this value is the ICRA `sub_task_name`.
- `3 x A100 40G` is the primary training machine; no FSDP.
- The competition inference GPU can hold ~8GB of model params (base + adapters).
- All public route keys use the same observation schema (3 cameras + proprioceptive state).
- The public route keys are the current reverse-engineered set of 10 sub-task names; unknown values must fall back to `_default`.
- v1 keeps the core ACoT loss, sampler, and architecture unchanged. Gains come from per-task specialization via LoRA adapter routing.
