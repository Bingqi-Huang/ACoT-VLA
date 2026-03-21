# Decisions

Append important decisions here with rationale and consequences.

## Template

Date:

Decision:

Why:

Consequence:

## Seed Entries

Date: 2026-03-08

Decision:

- Treat the competition as submission-contract-constrained, not training-method-constrained.

Why:

- Official docs only constrain the final Dockerized websocket inference interface and output command type.

Consequence:

- Training and architecture may change, but inference packaging must remain compliant.

Date: 2026-03-08

Decision:

- Use `AGENTS/` as the persistent operating memory for future agents.

Why:

- Chat history is not a reliable long-term state store for multi-session agentic work.

Consequence:

- Future agents should update this folder after significant progress.

Date: 2026-03-09

Decision:

- Treat LeRobot finetuning validation as episode-split-constrained rather than sample-split-constrained.

Why:

- Frame-level or shuffled sample-level validation leaks temporal context across train and validation and is not meaningful for the Reasoning2Action smoke finetuning tasks.

Consequence:

- Train/validation splits are now generated per dataset by `episode_index`, saved to JSON manifests, reused across training/stat computation, and validation should only be interpreted when norm stats are computed from the train split.

Date: 2026-03-09

Decision:

- Use teacher-forced offline checkpoint evaluation as the first-line checkpoint selection signal for the Reasoning2Action finetuning workflow.

Why:

- Closed-loop simulation rollout is slower, more operationally fragile, and unnecessary for every checkpoint when the immediate goal is to compare finetuning progress on held-out episodes without leakage.

Consequence:

- Checkpoints can now be ranked from the val episode split using JSON metrics and a summary CSV before investing in heavier rollout-based validation or submission packaging.

Date: 2026-03-10

Decision:

- Treat websocket `payload["task_name"]` as the routing contract for ICRA submission, and interpret it as the evaluator `sub_task_name` rather than the outer benchmark scene name.

Why:

- Reverse-engineering of Genie Sim shows the policy payload writes `task_name=self.sub_task_name`, while the outer benchmark `task_name` is used only for config/scene selection and result directory layout.

Consequence:

- Routed serving should key on the documented sub-task names.
- The public routing surface is currently 10 route keys, with `sorting_packages_continuous` sharing the `sorting_packages` adapter.
- Prompt-only routing is insufficient for `sorting_packages_continuous`, and unknown values must fall back to `_default`.

Date: 2026-03-10

Decision:

- Do not keep undocumented task aliases in the final ICRA router.

Why:

- The reverse-engineered evaluator contract already gives the true public route keys via websocket `payload["task_name"] = sub_task_name`. Undocumented aliases risk masking route mismatches and confusing future maintenance.

Consequence:

- The final public routing table should contain only the documented ICRA sub-task keys.
- Legacy aliases such as `grab_toy -> place_block_into_box` should be removed from the final serving path unless a later reverse-engineering result proves they are actually used by evaluation.

Date: 2026-03-10

Decision:

- Treat same-task dataset folders with `_part_*` suffixes as storage shards of one routed task, not as separate tasks.

Why:

- For large tasks, the dataset is split across multiple folders only because one folder is too large; the split does not imply a different task semantics or a different route key.

Consequence:

- All known shards for a task should be merged into that task family during training.
- Specialist configs should use all shards of their task family, not a subset chosen only by folder naming.
- Routing remains keyed by the single public task name, not by shard name.

Date: 2026-03-14

Decision:

- Adopt a baseline-first competition strategy: never submit a model scoring below the official baseline (6.35).

Why:

- The previous LoRA-everywhere generalist regressed to 5.08 (-1.27 vs baseline), showing that training can easily make things worse.
- The baseline already scores near-perfect on 4 of 10 tasks (open_door, scoop_popcorn, hold_pot, take_wrong_item).
- Improvement ceiling is in the 3 weak tasks (clean_desktop, stock_shelf, place_block), worth at most +1.95 points.
- Optimizing for benchmark score rather than train loss is critical since they are not well correlated.

Consequence:

- Every checkpoint must be evaluated against 6.35 before being considered for submission.
- Training schedules should be conservative (low LR, short runs, dense checkpoints) to limit catastrophic forgetting.
- The baseline checkpoint must never be overwritten.

Date: 2026-03-14

Decision:

- Use conservative LoRA generalist as the primary training path, not the baseline-compatible config.

Why:

- LoRA constrains trainable surface to ~110M params (vs ~1.3B for baseline-compatible), limiting how far the model can drift.
- 3x40G A100s can comfortably handle LoRA training without EMA; baseline-compatible with EMA risks OOM.
- The 20 missing expert LoRA tensors initialize with ~0.001 magnitude noise, which is small enough that a conservative LR (1e-5) should not amplify it destructively.
- The baseline-compatible config's warmup bug (warmup==total_steps) would have prevented any meaningful training anyway.

Consequence:

- `acot_challenge_lora_conservative` is the first config to launch.
- `acot_challenge_generalist_baseline_compatible` is the second attempt if LoRA underperforms.
- Both tracks should be evaluated by benchmark score, not train loss.

Date: 2026-03-14

Decision:

- Route only weak tasks to specialist adapters; let strong tasks use the base model directly.

Why:

- The previous routing table mapped all 10 tasks to task-specific adapters. This is unnecessary and risky for tasks already at 1.0.
- Routing only weak tasks (clean_desktop, stock_shelf, place_block) minimizes the blast radius of specialist errors.
- A `_default.npz` adapter extracted from the best generalist preserves generalist behavior for non-routed tasks.

Consequence:

- `TASK_ROUTING` now contains only 4 entries (including grab_toy alias).
- Non-routed tasks fall through to `_default` or `_base`.
- The `_default.npz` adapter must be extracted from the best generalist checkpoint when using routing.

Date: 2026-03-14

Decision:

- Support LoRA-only adapter extraction for lightweight routing on the 24GB inference server.

Why:

- Full adapter extraction (15 patterns including dense modules) produces ~140MB adapters — too heavy for 9 adapters in 24GB VRAM alongside Isaac Sim.
- LoRA-only extraction produces ~220MB per adapter in bfloat16; 3 specialist adapters = ~660MB total.
- This leaves sufficient headroom for base model (~7.8GB) + JIT cache (~2-4GB) + Isaac Sim.

Consequence:

- `scripts/extract_adapter.py` now supports `--lora-only` flag.
- Specialist routing is viable on the competition server only with LoRA-only adapters, not full adapters.

Date: 2026-03-11

Decision:

- Keep the new fast-training workflow strictly additive instead of modifying the existing `train.py` / legacy dataloader path.

Why:

- The repository already has a known-good training path for `acot_challenge_generalist_lora_generalist`.
- User requirement for this round was explicit: do not break already working code.
- This reduces rollback cost when testing aggressive training-path optimizations.

Consequence:

- Fast-path work should live under separate files such as `scripts/train_fast.py` and `src/openpi/training/data_loader_fast.py`.
- Legacy checkpoint layout, eval scripts, and serving code remain the compatibility target.

Date: 2026-03-11

Decision:

- Treat the current full-generalist steady-state bottleneck as primarily data-side video access / decode cost, not just `num_workers` misconfiguration.

Why:

- The dataset layout is highly fragmented:
  - about 4179 parquet files
  - about 12778 mp4 files
  - each episode stored separately
  - three camera videos per episode
- Training-time reads require random episode-local parquet access plus random frame extraction from HEVC mp4 files.
- On slower hardware (SATA SSD, older memory subsystem), GPU under-utilization and power sawtooth are consistent with decode / random-I/O stalls.

Consequence:

- Merely tuning `num_workers` is not expected to fully solve throughput problems.
- High-value optimization directions are:
  - precomputed sampler caches
  - better episode-locality / decoder reuse
  - offline training caches or resized sidecar video assets

Date: 2026-03-11

Decision:

- For `acot_challenge_generalist_lora_generalist`, prompt-only token caching is not valid.

Why:

- This config uses `discrete_state_input=True`.
- Prompt tokenization therefore depends on both prompt text and normalized state, not only prompt text.

Consequence:

- `scripts/precompute_prompt_cache.py` should refuse this config unless a state-aware cache design is added later.
- The immediately useful fast-path cache for this config is the subtask-index cache, not prompt-token cache.

Date: 2026-03-11

Decision:

- Do not revive the abandoned per-config final-sample offline cache path; use a generic `Reasoning2Action-Sim` frame-shard cache instead.

Why:

- A per-config cache has poor amortization when a single cache build can take on the order of the same wall-clock time as a full training run.
- The competition roadmap needs reuse across:
  - full generalist
  - 5-task
  - clean-desktop
  - future specialist / adapter training
- The common expensive work is at the raw data boundary:
  - random mp4 access
  - HEVC decode
  - image resize
  not at the later config-specific transform boundary.

Consequence:

- Cache only the shared intermediate data boundary:
  - decoded+resized camera frames
  - raw state/action windows
  - prompt/task/frame metadata
- Keep normalization, tokenization, padding, and config-specific action slicing in the fast loader at runtime.
- The cache remains training-only infrastructure and must not affect checkpoint layout, offline eval inputs, or final Docker/websocket serving.

Date: 2026-03-21

Decision:

- Standardize Blackwell training on CUDA13 JAX stack and keep Orbax-restore compatibility guards in-tree.

Why:

- `RTX PRO 6000 Blackwell` training with older CUDA12-era JAX stack showed `CUDA_ERROR_ILLEGAL_ADDRESS` in early replicated steps.
- After upgrading JAX/Orbax, checkpoint restore broke due to metadata API drift (`StepMetadata` vs dict), so compatibility handling is required to keep older/newer checkpoints loadable.

Consequence:

- Keep these dependency baselines unless there is a fully re-validated replacement:
  - `jax[cuda13]==0.7.2`
  - `jaxlib==0.7.2`
  - `orbax-checkpoint==0.11.33`
  - NumPy 2.x
- Preserve `restore_params()` metadata compatibility logic in `src/openpi/models/model.py`.
- Future dependency upgrades must include a real multi-GPU smoke run and checkpoint-restore validation, not only import-level checks.
