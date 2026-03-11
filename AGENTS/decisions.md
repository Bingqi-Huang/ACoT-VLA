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
