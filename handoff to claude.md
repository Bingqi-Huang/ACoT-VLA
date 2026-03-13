# Handoff to Claude

## Scope of this handoff

This note captures the technical conclusions, evaluation context, code changes, constraints, and recommended next steps from the Codex session focused on:

- understanding the ICRA challenge setup in this repo
- understanding why the trained generalist underperformed the official baseline
- deciding between continued generalist training, routing, and baseline-first strategies
- fixing initialization compatibility issues in the training code
- preserving the competition-relevant context for the next agent

This handoff intentionally includes the evaluation details supplied by the user, because those details strongly affect what counts as a good training strategy and how benchmark results should be interpreted.

## Repo and environment context

- Repo root: `/data/admins/bingqi/Projects/ACoT-VLA`
- Main training launcher in active use: `scripts/train.sh`
- Fast path also exists: `scripts/train_fast.py` and `scripts/train_fast.sh`
- Current date of this session: 2026-03-13
- User has local retained checkpoints from the generalist line and also has a local baseline checkpoint(these are not avaliable on the machine that claude presents, those are on the training machine, but they do exist):
  - `checkpoints/baseline_checkpoint/params`
  - `checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96/...`

Important local workspace state observed during this session:

- There were unrelated local/staged edits in:
  - `src/openpi/training/r2a_frame_cache.py`
  - `src/openpi/training/r2a_frame_cache_test.py`
- Those edits were treated as user work and should not be reverted.

## What was read and established at the start

The current active mainline run was identified as:

- config: `acot_challenge_generalist_lora_generalist`
- launcher: `scripts/train.sh`
- model path: `src/openpi/training/config.py`

At the start of the session, the user stated they had already trained the `10000`, `15000`, and `20000` checkpoints of the generalist model.

The current generalist line in this repo was identified as a LoRA-heavy ACOT variant based on:

- `paligemma_variant="gemma_2b_lora"`
- `coarse_action_expert_variant="gemma_300m_lora"`
- `action_expert_variant="gemma_300m_lora"`
- explicit and implicit action reasoners enabled
- `downsample_based_implicit_extractor=True`
- freeze filter that freezes vision, the main LLM, the LLM embedder, and both action experts outside LoRA paths

That setup came from:

- `src/openpi/training/config.py`

The generalist run retained checkpoints at:

- `5000`
- `10000`
- `15000`
- `20000`

and the visible val-loss trajectory in the retained metrics showed:

- `5000`: `0.1765`
- `10000`: `0.2019`
- `15000`: `0.1721`
- `20000`: `0.1698`

There was also a better unsaved region around `17000` with a lower val loss, which immediately suggested checkpoint cadence was too sparse for model selection if rollout score is non-monotonic.

## Official evaluation semantics provided by the user

The user supplied a detailed breakdown of how the 10 ICRA tasks are actually scored after expansion from `instructions.json`, `problems.json`, and `TASK_STEPS`.

This is extremely important because several tasks are not scored the way a casual reading of the instruction suggests.

### Action composition semantics

The user highlighted three composition rules:

- `ActionList`: sequential execution
- `ActionSetWaitAny`: whichever branch finishes first wins, usually success checker vs `StepOut`
- `ActionSetWaitAll`: all inner conditions must be satisfied simultaneously

### Task-by-task scoring summary supplied by the user

1. `hold_pot`
- Natural language: grasp both pot handles and place the pot on the stove
- Actual stages:
  - `LiftUp`
  - `InBBox + Upright`
- Important scoring detail:
  - formal score only counts `LiftUp`
  - the placement stage affects episode completion but not `TASK_STEPS` summary

2. `clean_the_desktop`
- Pure VLM terminal-state scoring
- Success only when VLM score reaches `1.0`
- No explicit pick-place substeps in formal scoring

3. `open_door`
- Single `PushPull`
- Success is based on door joint state entering the target range
- It does not directly score contact with the handle

4. `place_block_into_box`
- Three-stage pick-place:
  - `Follow`
  - `PickUpOnGripper`
  - `Inside`
- All three are formally scored

5. `pour_workpiece`
- `ActionSetWaitAll` over four `Inside` conditions
- All four workpieces must be inside the target box

6. `scoop_popcorn`
- Pure VLM terminal-state task

7. `sorting_packages`
- Stages:
  - `Follow`
  - `PickUpOnGripper`
  - `Inside + Upright`
  - `PickUpOnGripper`
  - `Inside`
- Formal score includes the scan-table upright placement requirement

8. `sorting_packages_continuous`
- Four cycles of package sorting
- Formal aggregate only counts the four `Upright` outcomes on the scanner
- This makes the task score very skewed relative to full behavior

9. `stock_and_straighten_shelf`
- Stages:
  - `Follow`
  - `PickUpOnGripper`
  - `InBBox`
  - `Follow`
  - `Upright`
- The shelf placement uses a fixed spatial bounding box, not a container-style `Inside`

10. `take_wrong_item_shelf`
- Stages:
  - `Follow`
  - `Inside`
- No separate scored `PickUpOnGripper` stage even though grasping is usually necessary in practice

### Additional user-supplied scoring caveats

- Episode completion is not always the same as formal benchmark score.
- `StepOut` is a stage-level failure exit.
- `E2E=1` only if the last formal scored step is exactly `1.0`.
- `sorting_packages_continuous` has especially misleading aggregate scoring and should be inspected at progress-detail level.

## Evaluation results supplied by the user

The user compared:

- baseline results in `evaluation_result.xlsx`
- trained model results in `evaluation_result (1).xlsx`

The user’s summary, which is directionally important and should be preserved, was:

- baseline total: `6.35`
- trained model total: `5.08`
- overall drop: `-1.27`

### User-supplied per-task comparison

| Task | Baseline | New | Delta | Notes |
| --- | ---: | ---: | ---: | --- |
| Sorting Packages | 0.67 | 0.42 | -0.25 | strong regression in later chain |
| Sorting Packages Continuous | 0.05 | 0.02 | -0.03 | still bad, long-horizon not improved |
| Pour Workpiece | 0.61 | 0.61 | 0.00 | unchanged |
| Take Wrong Item Shelf | 0.97 | 0.95 | -0.02 | basically stable |
| Stock and Straighten Shelf | 0.24 | 0.40 | +0.16 | genuine improvement |
| Scoop Popcorn | 1.00 | 0.80 | -0.20 | VLM end-state regression |
| Open Door | 1.00 | 0.40 | -0.60 | strongest regression |
| Place Block into Box | 0.42 | 0.44 | +0.02 | nearly unchanged |
| Hold Pot | 1.00 | 0.85 | -0.15 | regression |
| Clean the Desktop | 0.39 | 0.19 | -0.20 | regression |

### Failure observations the user explicitly reported

`clean_the_desktop`
- the robot closes the laptop lid
- then appears to run out of time

`sorting_packages`
- the package may not be label-face-up on the scanner

`stock_and_straighten_shelf`
- the robot seems not to have learned to pick from the messy cart reliably
- then fails to place on the shelf with correct orientation

### Important interpretation preserved from the conversation

The trained model looked like:

- local improvement on some shelf-like manipulation primitives
- substantial loss of general benchmark stability

This looked much more like:

- overfitting
- catastrophic forgetting
- training/evaluation objective mismatch

than like a genuine leaderboard improvement.

## Initial strategic conclusions reached before code changes

### 1. Do not assume the trained generalist replaced the baseline

The trained generalist should not be treated as a clean replacement for the baseline because the benchmark result was worse overall.

### 2. Baseline-first is the safer competition strategy

The strongest near-term strategy identified in conversation was:

- keep the official baseline as the safe fallback
- only replace it if a new checkpoint actually beats it on benchmark score

### 3. Routing was discussed but deprioritized

A routed approach was considered, where:

- baseline would be used for already-strong tasks such as `open_door`, `scoop_popcorn`, `hold_pot`, probably `take_wrong_item_shelf`
- specialists or alternative checkpoints would be used on weak tasks

But routing was deprioritized for two reasons:

- the official inference server only has about 24 GB VRAM and also runs Isaac Sim
- the current routed-adapter implementation in this repo is not truly lightweight and is not a safe baseline+specialist runtime story yet

## Routing and serving constraints established in the conversation

The user explicitly stated:

- routing through checkpoints is not possible if this requires loading multiple full models at once
- official test server has only `24G` VRAM
- part of VRAM is needed by Isaac Sim

This constraint changed the plan materially.

The code inspection during the session found:

- `scripts/extract_adapter.py` currently extracts not only LoRA tensors but also large dense modules like:
  - action projections
  - time MLPs
  - explicit/implicit action reasoner modules
  - fusion and projection layers
- `src/openpi/policies/adapter_routed_policy.py` caches a merged `nnx.State` per adapter
- `BaseModelConfig.load()` requires architecture-compatible parameter trees

Conclusion:

- multi-checkpoint runtime routing is not a practical near-term path for the official server budget
- current “adapter routing” is still too heavy to treat as tiny task-specific overlays over the official baseline

## The initialization-compatibility diagnosis

### What was initially suspected

At first glance, logs suggested many missing trainable parameters when loading the local baseline checkpoint into the current LoRA-everywhere generalist config.

The initial grouping from warning logs suggested:

- missing implicit action reasoner parameters
- missing expert-branch LoRA parameters

### What was later discovered

This diagnosis was refined substantially.

The loader bug was:

- checkpoint restore produced numbered child keys as strings, e.g. `"0"`, `"1"`, ...
- the model state used integer keys, e.g. `0`, `1`, ...
- `_merge_params()` matched them directly without key normalization

This created a false appearance that the entire implicit reasoner subtree was missing.

### Corrected conclusion after fixing the loader

After normalizing checkpoint keys before merge:

- the baseline checkpoint **does** match the implicit reasoner tree
- the actual remaining mismatch in the current LoRA-everywhere generalist config is only the expert-branch LoRA tensors

The true remaining missing parameter count for the LoRA-everywhere config against the local baseline checkpoint was verified as:

- `20` real missing tensors

These are:

- `PaliGemma/llm/layers/attn/attn_vec_einsum_1/lora_a`
- `PaliGemma/llm/layers/attn/attn_vec_einsum_1/lora_b`
- `PaliGemma/llm/layers/attn/attn_vec_einsum_2/lora_a`
- `PaliGemma/llm/layers/attn/attn_vec_einsum_2/lora_b`
- `PaliGemma/llm/layers/attn/kv_einsum_1/lora_a`
- `PaliGemma/llm/layers/attn/kv_einsum_1/lora_b`
- `PaliGemma/llm/layers/attn/kv_einsum_2/lora_a`
- `PaliGemma/llm/layers/attn/kv_einsum_2/lora_b`
- `PaliGemma/llm/layers/attn/q_einsum_1/lora_a`
- `PaliGemma/llm/layers/attn/q_einsum_1/lora_b`
- `PaliGemma/llm/layers/attn/q_einsum_2/lora_a`
- `PaliGemma/llm/layers/attn/q_einsum_2/lora_b`
- `PaliGemma/llm/layers/mlp_1/gating_einsum_lora_a`
- `PaliGemma/llm/layers/mlp_1/gating_einsum_lora_b`
- `PaliGemma/llm/layers/mlp_1/linear_lora_a`
- `PaliGemma/llm/layers/mlp_1/linear_lora_b`
- `PaliGemma/llm/layers/mlp_2/gating_einsum_lora_a`
- `PaliGemma/llm/layers/mlp_2/gating_einsum_lora_b`
- `PaliGemma/llm/layers/mlp_2/linear_lora_a`
- `PaliGemma/llm/layers/mlp_2/linear_lora_b`

### Why this still matters

Even though the corrected missing count is much smaller than the initial impression, it still means:

- the current LoRA-everywhere generalist config is not a clean continuation of the official baseline checkpoint
- it introduces trainable expert-LoRA parameters not present in the baseline checkpoint
- those parameters would be synthesized rather than loaded from true baseline weights

This is enough to justify a dedicated baseline-compatible config for the next clean run.

## Code changes made in this session

### 1. Weight-loader fix

File:

- `src/openpi/training/weight_loaders.py`

Changes:

- added key normalization via `_model.convert_str_keys_to_int(...)` before parameter-tree flattening and merge
- added optional `strict` support to checkpoint loaders
- added optional `missing_init` argument plumbing
- added strict compatibility failure path for:
  - real missing params
  - shape mismatches

Purpose:

- prevent false missing-param reports from string-vs-int key mismatches
- allow fail-fast behavior for checkpoint/model incompatibility instead of silent synthesis

### 2. New baseline-compatible ACOT challenge config

File:

- `src/openpi/training/config.py`

Added:

- `_reasoning2action_baseline_compatible_model()`
- `_reasoning2action_baseline_compatible_freeze_filter()`
- new config:
  - `acot_challenge_generalist_baseline_compatible`

Design intent:

- reuse the merged Reasoning2Action challenge task family
- use a model tree that is compatible with the local baseline checkpoint
- use strict checkpoint loading so incompatibility fails fast

### 3. Tests added

File:

- `src/openpi/training/weight_loaders_test.py`

Added tests for:

- numbered checkpoint key normalization
- strict failure on missing params
- strict failure on shape mismatch

## Verification completed during the session

### Unit tests

Executed:

- `uv run pytest src/openpi/training/weight_loaders_test.py src/openpi/models/acot_vla_test.py`

Result:

- `5 passed`

### Metadata-based compatibility checks

The local baseline checkpoint metadata was compared against both model trees.

#### Baseline-compatible model

- model keys: `192`
- checkpoint keys: `192`
- missing: `0`
- extra: `0`

Conclusion:

- the new baseline-compatible model tree exactly matches the local baseline checkpoint tree at metadata level

#### Current LoRA-everywhere model

- model keys: `212`
- checkpoint keys: `192`
- missing: `20`
- extra: `0`

Conclusion:

- current LoRA-everywhere config still has exactly the 20 missing expert-LoRA tensors listed above

## Trainable-surface and VRAM conclusions

Two counts were computed during the session.

### Total parameter count

- `acot_challenge_generalist_lora_generalist`: about `3,889,560,112`
- `acot_challenge_generalist_baseline_compatible`: about `3,845,323,312`

These are similar total model sizes.

### Trainable parameter count

- `acot_challenge_generalist_lora_generalist`: about `110,359,360`
- `acot_challenge_generalist_baseline_compatible`: about `1,336,791,600`

This is the critical difference.

The baseline-compatible config is much heavier in training because gradients and Adam optimizer state scale with the trainable subset, not just the total model size.

### EMA conclusion

EMA was inspected in training code and found to be stored as a full second parameter tree:

- `ema_params=None if config.ema_decay is None else params`

So turning on EMA adds memory roughly proportional to another full model parameter tree, not just trainable params.

Competition-relevant implication:

- `compatible + EMA` is the heaviest option
- on `3 x 40G A100`, EMA should be treated cautiously

## Fast training path conclusions

The user stated that:

- norm stats for the fast path have already been computed
- the fast cache has already been built

This removes the main preparatory cost of using the fast path.

Fast-path conclusion from the conversation:

- yes, the fast path can be used
- it is additive and does not change checkpoint format
- but it should still be debugged briefly before the real run if there is any doubt

## Strategy conclusions reached by the end of the session

### Strong recommendation about the old run

The user asked whether to stop the previous LoRA-everywhere run.

Recommendation given:

- yes, stop it if the run is `acot_challenge_generalist_lora_generalist`

Reason:

- it is not baseline-compatible
- it is optimizing a structurally less trustworthy trainable surface for benchmark recovery
- its loss was not decreasing meaningfully

### Winning-the-challenge perspective

When asked to think only from the perspective of winning the challenge, the recommendation was:

1. keep the official baseline as the locked fallback
2. stop the old LoRA-everywhere generalist run
3. launch a new baseline-compatible run
4. use the fast path because cache and norm stats are already ready
5. select checkpoints by benchmark score, not train loss
6. do not invest in routing yet

### Conservative training recommendation from the conversation

The conversation converged on a conservative first rerun rather than an aggressive long run.

Reasoning:

- prior runs showed fast forgetting
- long warmup and high LR looked too aggressive
- benchmark score was not aligned with training loss

Recommended competition-minded schedule discussed in conversation:

- `num_train_steps`: `8000` to `10000`
- `warmup_steps`: `200` to `500`
- `peak_lr`: `1e-5` to `1.5e-5`
- `decay_lr`: about `1e-6`
- `save_interval`: `1000`
- `val_interval`: `1000`
- `val_num_batches`: `8` or `16`
- `ema_decay`: ideally `0.999` if memory allows, otherwise `None`

However, after later VRAM discussion, the safer advice was:

- do **not** enable EMA immediately on the first compatible run if VRAM is tight
- reduce batch size first if needed

### Important caution for the next agent

The code currently contains a new baseline-compatible config, but the best competition schedule and memory-safe batch size still require judgment against the actual `3 x 40G` training hardware.

The current code change in this session was focused on:

- init compatibility
- fail-fast correctness

not on fully finishing the competition-tuned schedule.

The next agent should review the exact committed values in `acot_challenge_generalist_baseline_compatible` before launch.

## Recommended immediate next steps for the next agent

1. Verify the exact committed hyperparameters of `acot_challenge_generalist_baseline_compatible`.
2. If needed, patch that config toward the conservative winning schedule:
   - shorter warmup
   - lower LR
   - shorter run
   - denser checkpointing
   - validation enabled every `1000` steps
3. Use `scripts/train_fast.py` or `scripts/train_fast.sh` because cache and norm stats already exist.
4. Keep the official baseline checkpoint or image as the fallback submission.
5. Evaluate real checkpoints early and often.
6. Only replace the baseline if a checkpoint beats it on the true benchmark.

## Commands and paths that matter

### Official baseline init path used during this work

```bash
export ACOT_CHALLENGE_INIT_WEIGHTS=/data/admins/bingqi/Projects/ACoT-VLA/checkpoints/baseline_checkpoint/params
```

### Fast-path launcher pattern

```bash
bash scripts/train_fast.sh <CONFIG_NAME> <EXP_NAME> --r2a-cache-root=/path/to/r2a-frame-cache
```

### Baseline-compatible config name added during this work

```text
acot_challenge_generalist_baseline_compatible
```

### Current generalist config that should not be treated as the main recovery path

```text
acot_challenge_generalist_lora_generalist
```

## Summary in one paragraph

The key outcome of this session is that the poor-performing generalist was not a clean continuation of the official baseline, but the problem was narrower than first feared: after fixing string-vs-int checkpoint key handling, the true remaining incompatibility is exactly 20 expert-LoRA tensors in the current LoRA-everywhere generalist config. A strict baseline-compatible training path was added, tests were added, and the competition strategy settled on a baseline-first approach: keep the official baseline as fallback, use the fast training path with the already-prepared cache and norm stats, run a conservative short baseline-compatible finetune, choose checkpoints by real benchmark score rather than train loss, and avoid routing for now because the server VRAM and current adapter implementation do not make it a reliable competition path yet.
