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
