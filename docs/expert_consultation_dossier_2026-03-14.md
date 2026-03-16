# ACoT-VLA 专家咨询材料（自包含离线版，2026-03-14）

## 文档目标

本材料用于给无仓库访问权限的专家做技术咨询，因此本文将：

1. 用可验证的数据讲清当前训练与推理策略。
2. 直接嵌入关键代码原文与 AGENTS 关键原文。
3. 在文末给出可直接讨论的问题清单。

说明：

- 本文中的代码块与文本摘录均来自当前 workspace 的真实文件。
- 用于统计的数据聚合文件为 `docs/_expert_consult_data_2026-03-14.json`。

---

## 第一部分：当前情况（数据+结论）

## 1. 当前策略一句话

当前策略是“基线保分优先 + 保守 LoRA 迭代 + 弱任务定向路由”，并将 fast path 作为增量能力，不替代稳定主路径。

## 2. 已知关键结果

来自 AGENTS 文档的历史评测信号：

- baseline 分数：6.35
- LoRA-everywhere generalist（step 20000）分数：5.08
- 相对 baseline：-1.27

同一条记录里指出强任务出现退化（如 open_door 1.0 -> 0.4），这是当前策略转向的直接原因。

## 3. wandb 运行台账（本地扫描）

总计 13 次 run（8 online + 5 offline）：

- online: run-20260311_122703-cs5i3sx9
- online: run-20260311_123642-0bjait8z
- online: run-20260311_135138-pk1zvimy
- online: run-20260312_215641-0zas6bzj
- online: run-20260312_220012-vp83y2xf
- online: run-20260312_223905-kk5sklcs
- online: run-20260313_084029-kk5sklcs
- online: run-20260313_084250-kk5sklcs
- offline: offline-run-20260311_131353-ll6wbjhk
- offline: offline-run-20260311_131900-tjmiy55r
- offline: offline-run-20260311_132516-ic8drn9d
- offline: offline-run-20260311_133102-t9yvqcgc
- offline: offline-run-20260311_133455-vdxsyrsm

可用性：

- 完整 config+summary 的 run 只有 3 个。
- 其余 run 多数只有 metadata，且有明显网络上传失败痕迹（SSL EOF、timeout、502）。

有 summary 的 3 个 run：

1. run-20260311_122703-cs5i3sx9
   - runtime 27s（短启动/早停）
2. run-20260312_220012-vp83y2xf
   - step 200, train/loss 0.1048368, val/loss 0.1959288
3. run-20260313_084029-kk5sklcs
   - step 1900, train/loss 0.0794664, val/loss 0.1398673

## 4. 本地 train_metrics（最完整曲线）

文件：`checkpoints/acot_challenge_generalist_lora_generalist_tuned/generalists_fast_v1_bs64/train_metrics.jsonl`

统计结果：

- train 点数：23
- val 点数：3
- checkpoint 事件：1（step 1000）

loss 曲线：

- train/loss: 0.1591（step 0） -> 0.0787（末尾）
- train 最优: 0.07377（step 1100）
- val/loss: 0.19595（step 0） -> 0.13987（step 1000） -> 0.12886（step 2000）

任务级 step0 -> step2000（节选）：

- Clear_the_desktop: 0.1211 -> 0.0880（下降）
- Pop_the_popcorn: 0.0655 -> 0.0150（显著下降）
- Flip_workpiece_icra_SIM: 0.5110 -> 0.0963（显著下降）
- Turn_the_doorknob: 0.0651 -> 0.0766（上升）
- Carry_the_pot: 0.0607 -> 0.0901（上升）

结论：

- 训练损失整体向好，但与 benchmark 分数不必然一致（已有反例 5.08 < 6.35）。

---

## 第二部分：关键代码原文（完整嵌入）

## A. 配置核心原文（config）

来源：`src/openpi/training/config.py`

### A1. Prompt map、specialist specs、LoRA 模板与 freeze/data 函数

```python
_REASONING2ACTION_PROMPT_MAP = {
  "Turn the doorknob": ("Turn the doorknob and push the door", 0.5),
  "Make popcorn": ("Scoop the popcorn and pour it into the popcorn bucket", 0.5),
  "Carry the pot": ("Grasp the two handles of the pot and place it on the stove", 0.5),
  "Insert building block holes_2_SIM": (
    "Pick up the yellow circular block from the table, "
    "and place it into the round hole of the block box",
    0.2,
  ),
  "Remove misplaced beverages from shelves": (
    "Pick up the incorrectly placed item from the shelf, "
    "and place it into the shopping basket",
    0.2,
  ),
  "Stock supermarket shelves  \nStraighten products  \nAttend ICRA conference  \nOperate SIM card": (
    "Pick up the wei-chuan orange juice in the shopping basket, "
    "and place it on the shelf. "
    "Then, straighten the toppled wei-chuan grape juice",
    0.2,
  ),
  "Sort packages": (
    "Grab the <color> package on the table, "
    "turn the waist right to face the barcode scanner, "
    "place the package on the scanning table with the barcode facing up. "
    "Then, grab the package, "
    "rotate the waist and place the package in the blue bin. "
    "Finally, return the waist back to face the initial table",
    0.2,
  ),
  "Clear the desktop": (
    "Pick up the pen on the left side and place it into the pen holder, "
    "close the laptop, "
    "pick up the tissue on the table and place it into the trash bin on the right size. "
    "Then, pick up the mouse and place it on the right side of the laptop. "
    "Finally, straighten the colored pencil box",
    0.5,
  ),
}

_REASONING2ACTION_SPECIALIST_SPECS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], int], ...] = (
  ("acot_specialist_pour_workpiece", ("pour_workpiece",), ("Unload workpiece_icra_SIM",), 5000),
  ("acot_specialist_open_door", ("open_door",), ("Turn the doorknob",), 5000),
  ("acot_specialist_scoop_popcorn", ("scoop_popcorn",), ("Make popcorn",), 5000),
  ("acot_specialist_hold_pot", ("hold_pot",), ("Carry the pot",), 5000),
  ("acot_specialist_place_block", ("place_block_into_box",), ("Insert building block holes_2_SIM",), 3000),
  ("acot_specialist_take_wrong_item", ("take_wrong_item_shelf",), ("Remove misplaced beverages from shelves",), 5000),
  (
    "acot_specialist_stock_shelf",
    ("stock_and_straighten_shelf", "stock_and_straighten_shelf_part_2"),
    ("Stock supermarket shelves  \nStraighten products  \nAttend ICRA conference  \nOperate SIM card",),
    3000,
  ),
  ("acot_specialist_sorting", ("sorting_packages_part_1", "sorting_packages_part_2", "sorting_packages_part_3"), ("Sort packages",), 5000),
  (
    "acot_specialist_clean_desktop",
    ("clean_the_desktop_part_1", "clean_the_desktop_part_2", "clean_the_desktop_addition"),
    ("Clear the desktop",),
    8000,
  ),
)

def _reasoning2action_lora_model() -> acot_vla.ACOTConfig:
  return acot_vla.ACOTConfig(
    coarse_action_horizon=30,
    action_horizon=30,
    paligemma_variant="gemma_2b_lora",
    coarse_action_expert_variant="gemma_300m_lora",
    action_expert_variant="gemma_300m_lora",
    adopt_explicit_action_reasoner=True,
    adopt_implicit_action_reasoner=True,
    downsample_based_implicit_extractor=True,
    max_token_len=210,
  )

def _reasoning2action_lora_freeze_filter() -> nnx.filterlib.Filter:
  return _reasoning2action_lora_model().get_freeze_filter(
    freeze_vision=True,
    freeze_llm=True,
    freeze_llm_embedder=True,
    freeze_dual_ae=[True, True],
  )

def _reasoning2action_data_config(
  repo_ids: Sequence[str],
  prompt_keys: Sequence[str] | None = None,
  *,
  asset_id: str,
  split_name: str | None = None,
) -> LerobotACOTGo2DataConfig:
  selected_prompt_map = (
    _REASONING2ACTION_PROMPT_MAP
    if prompt_keys is None
    else {key: _REASONING2ACTION_PROMPT_MAP[key] for key in prompt_keys}
  )
  return LerobotACOTGo2DataConfig(
    default_prompt="This is the icra simulation challenge baseline config. Please refer to the README for details.",
    repo_id=list(repo_ids),
    assets=AssetsConfig(
      assets_dir=_REASONING2ACTION_ASSETS_DIR,
      asset_id=asset_id,
    ),
    prompt_map_inject_to_training=selected_prompt_map,
    repack_transforms=_reasoning2action_repack_transforms(),
    base_config=DataConfig(
      dataloader_sampler="subtask",
      prompt_from_hl_instruction=True,
      episode_split=EpisodeSplitConfig(split_name=split_name),
      video_tolerance_s=float(os.getenv("ACOT_CHALLENGE_VIDEO_TOLERANCE_S", "0.15")),
    ),
    joint_action_shifts=(2, 1),
    extra_delta_transform=(True, True),
    delta_action_mask=_transforms.make_bool_mask(14, -18),
  )
```

### A2. specialist 自动生成函数（完整）

```python
def _make_reasoning2action_specialist_configs() -> list[TrainConfig]:
  generalist_asset_id = os.getenv("ACOT_CHALLENGE_GENERALIST_ASSET_ID", "reasoning2action_sim_generalist")
  generalist_weights = os.getenv(
    "ACOT_CHALLENGE_GENERALIST_WEIGHTS",
    os.getenv(
      "ACOT_CHALLENGE_INIT_WEIGHTS",
      "gs://openpi-assets/checkpoints/pi05_base/params",
    ),
  )
  use_baseline_init = "ACOT_CHALLENGE_GENERALIST_WEIGHTS" not in os.environ

  specialist_configs = []
  for config_name, dataset_names, prompt_keys, num_train_steps in _REASONING2ACTION_SPECIALIST_SPECS:
    specialist_configs.append(
      TrainConfig(
        name=config_name,
        model=_reasoning2action_lora_model(),
        data=_reasoning2action_data_config(
          _reasoning2action_repo_ids(*dataset_names),
          prompt_keys,
          asset_id=generalist_asset_id,
          split_name=config_name,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
          warmup_steps=200,
          peak_lr=1e-5,
          decay_steps=num_train_steps,
          decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=(
          weight_loaders.ACOTCheckpointWeightLoader(generalist_weights)
          if use_baseline_init
          else weight_loaders.CheckpointWeightLoader(generalist_weights)
        ),
        num_train_steps=num_train_steps,
        save_interval=500 if not os.getenv("DEBUG_MODE", default=False) == "true" else 50,
        val_interval=500 if not os.getenv("DEBUG_MODE", default=False) == "true" else 50,
        val_num_batches=32 if not os.getenv("DEBUG_MODE", default=False) == "true" else 2,
        num_workers=24 if not os.getenv("DEBUG_MODE", default=False) == "true" else 1,
        batch_size=120 if not os.getenv("DEBUG_MODE", default=False) == "true" else 4,
        grad_accum_steps=1,
        freeze_filter=_reasoning2action_lora_freeze_filter(),
      )
    )
  return specialist_configs
```

### A3. 关键挑战配置（完整原文段）

```python
TrainConfig(
  name="acot_challenge_generalist_lora_generalist",
  model=_reasoning2action_lora_model(),
  data=_reasoning2action_data_config(...),
  lr_schedule=_optimizer.CosineDecaySchedule(
    warmup_steps=5_000,
    peak_lr=4e-5,
    decay_steps=40_000,
    decay_lr=4e-6,
  ),
  optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
  ema_decay=None,
  weight_loader=weight_loaders.ACOTCheckpointWeightLoader(...),
  num_train_steps=40_000,
  batch_size=96 if not os.getenv("DEBUG_MODE", default=False) == "true" else 4,
  grad_accum_steps=1,
  freeze_filter=_reasoning2action_lora_freeze_filter(),
),

TrainConfig(
  name="acot_challenge_generalist_baseline_compatible",
  model=_reasoning2action_baseline_compatible_model(),
  data=_reasoning2action_data_config(...),
  lr_schedule=_optimizer.CosineDecaySchedule(
    warmup_steps=500,
    peak_lr=2e-5,
    decay_steps=10_000,
    decay_lr=4e-6,
  ),
  optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
  ema_decay=None,
  weight_loader=weight_loaders.ACOTCheckpointWeightLoader(..., strict=True),
  num_train_steps=10_000,
  batch_size=96 if not os.getenv("DEBUG_MODE", default=False) == "true" else 4,
  grad_accum_steps=1,
  freeze_filter=_reasoning2action_baseline_compatible_freeze_filter(),
),

TrainConfig(
  name="acot_challenge_generalist_lora_generalist_tuned",
  model=_reasoning2action_lora_model(),
  data=dataclasses.replace(_reasoning2action_data_config(...)),
  lr_schedule=_optimizer.CosineDecaySchedule(
    warmup_steps=100,
    peak_lr=2.5e-5,
    decay_steps=5_000,
    decay_lr=2.0e-6,
  ),
  optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
  ema_decay=None,
  num_train_steps=5_000,
  save_interval=1000,
  val_interval=1000,
  val_num_batches=32,
  batch_size=120,
  grad_accum_steps=1,
  freeze_filter=_reasoning2action_lora_freeze_filter(),
),

TrainConfig(
  name="acot_challenge_lora_conservative",
  model=_reasoning2action_lora_model(),
  data=dataclasses.replace(_reasoning2action_data_config(...)),
  lr_schedule=_optimizer.CosineDecaySchedule(
    warmup_steps=200,
    peak_lr=1e-5,
    decay_steps=8_000,
    decay_lr=1e-6,
  ),
  optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
  ema_decay=None,
  num_train_steps=8_000,
  save_interval=500,
  val_interval=500,
  val_num_batches=32,
  batch_size=120,
  grad_accum_steps=1,
  freeze_filter=_reasoning2action_lora_freeze_filter(),
),
```

## B. Adapter Routed Policy 原文（完整）

来源：`src/openpi/policies/adapter_routed_policy.py`

```python
PATHS_KEY = "__paths__"
VALUE_KEY_TEMPLATE = "param_{index:04d}"
TASK_ROUTING = {
  "clean_the_desktop": "clean_the_desktop_1500",
}

def _path_to_tuple(path: str) -> tuple[str | int, ...]:
  out: list[str | int] = []
  for token in path.split("/"):
    if token.isdigit():
      out.append(int(token))
    else:
      out.append(token)
  return tuple(out)

def _load_adapter_file(adapter_path: pathlib.Path) -> dict[tuple[str | int, ...], np.ndarray]:
  with np.load(adapter_path, allow_pickle=False) as adapter_data:
    if PATHS_KEY in adapter_data.files:
      paths = adapter_data[PATHS_KEY].tolist()
      return {
        _path_to_tuple(path): np.asarray(adapter_data[VALUE_KEY_TEMPLATE.format(index=index)])
        for index, path in enumerate(paths)
      }
    return {_path_to_tuple(path): np.asarray(adapter_data[path]) for path in adapter_data.files}

def create_adapter_routed_policy(...):
  checkpoint_dir = _download.maybe_download(str(checkpoint_dir))
  adapter_dir = _download.maybe_download(str(adapter_dir))

  def _init_params(rng: jax.Array):
    model = train_config.model.create(rng)
    return nnx.state(model).to_pure_dict()

  params_shape = jax.eval_shape(_init_params, jax.random.PRNGKey(0))
  base_params = weight_loaders.ACOTCheckpointWeightLoader(
    str(checkpoint_dir / "params"),
    missing_init="zeros",
  ).load(params_shape)

class AdapterRoutedPolicy(_policy.Policy):
  def __init__(...):
    ...
    self._base_state_flat = flax.traverse_util.flatten_dict(self._base_state.to_pure_dict())
    self._adapters = self._load_adapters(self._adapter_dir)
    self._state_cache: dict[str, nnx.State] = {}
    self._sample_actions = nnx_utils.module_jit_with_state(base_model.sample_actions)
    ...

  def _resolve_adapter_name(self, task_name: str | None) -> str:
    if task_name is None:
      return "_default" if "_default" in self._adapters else "_base"
    return TASK_ROUTING.get(task_name, "_default" if "_default" in self._adapters else "_base")

  def _build_state(self, adapter_name: str) -> nnx.State:
    cached_state = self._state_cache.get(adapter_name)
    if cached_state is not None:
      return cached_state

    merged_params = dict(self._base_state_flat)
    merged_params.update(copy.deepcopy(self._adapters.get(adapter_name, {})))
    state = copy.deepcopy(self._base_state)
    state.replace_by_pure_dict(flax.traverse_util.unflatten_dict(merged_params))
    self._state_cache[adapter_name] = state
    return state

  def infer(self, obs: dict) -> dict:
    task_name = _normalize_task_name(jax.tree.map(lambda x: x, obs).get("task_name", None))
    self._activate_adapter(self._resolve_adapter_name(task_name))
    ...
    outputs["policy_timing"] = {"infer_ms": model_time * 1000}
    outputs["adapter_name"] = self._current_adapter_name
    return self.post_process(obs, outputs)
```

## C. Adapter 提取脚本原文（完整）

来源：`scripts/extract_adapter.py`

```python
ADAPTER_PATTERNS = (
  "lora",
  "coarse_action_in_proj",
  "action_in_proj",
  "coarse_action_out_proj",
  "action_out_proj",
  "coarse_time_mlp_in",
  "coarse_time_mlp_out",
  "time_mlp_in",
  "time_mlp_out",
  "explicit_action_reasoner",
  "implicit_action_reasoner",
  "implicit_action_reasoner_interact",
  "action_reasoning_fusion",
  "explicit_action_reason_proj",
  "implicit_action_reason_proj",
)
LORA_ONLY_PATTERNS = ("lora",)

def main(checkpoint: str, output: str, lora_only: bool = False) -> None:
  patterns = LORA_ONLY_PATTERNS if lora_only else ADAPTER_PATTERNS
  checkpoint_path = pathlib.Path(checkpoint)
  params = _model.restore_params(checkpoint_path / "params", restore_type=np.ndarray)
  flat_params = flax.traverse_util.flatten_dict(params, sep="/")

  filtered = {path: value for path, value in flat_params.items() if any(pattern in path for pattern in patterns)}
  if not filtered:
    raise ValueError(f"No adapter parameters matched under checkpoint: {checkpoint_path}")

  payload = {
    PATHS_KEY: np.asarray(list(filtered.keys()), dtype=str),
    **{VALUE_KEY_TEMPLATE.format(index=index): value for index, value in enumerate(filtered.values())},
  }
  np.savez(output_path, **payload)
```

## D. 服务脚本原文（checkpoint / routed）

来源：`scripts/server_checkpoint.sh`、`scripts/server_routed.sh`

```bash
# server_checkpoint.sh
export ACOT_SERVE_CONFIG=${ACOT_SERVE_CONFIG:-acot_challenge_generalist_lora_generalist}
export ACOT_SERVE_CHECKPOINT=${ACOT_SERVE_CHECKPOINT:-/app/checkpoint/generalists_v1_bs96_step5000}

GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py \
  --port "${port}" \
  policy:checkpoint \
  --policy.config "${ACOT_SERVE_CONFIG}" \
  --policy.dir "${ACOT_SERVE_CHECKPOINT}"
```

```bash
# server_routed.sh
export ACOT_ROUTED_CONFIG=${ACOT_ROUTED_CONFIG:-acot_challenge_lora_conservative}
export ACOT_ROUTED_BASE_CHECKPOINT=${ACOT_ROUTED_BASE_CHECKPOINT:-./checkpoint/baseline/30000}
export ACOT_ROUTED_ADAPTER_DIR=${ACOT_ROUTED_ADAPTER_DIR:-./adapters}

exec uv run --no-sync python scripts/serve_policy.py \
  --port "${port}" \
  policy:adapter-routed \
  --policy.config "${ACOT_ROUTED_CONFIG}" \
  --policy.base-checkpoint "${ACOT_ROUTED_BASE_CHECKPOINT}" \
  --policy.adapter-dir "${ACOT_ROUTED_ADAPTER_DIR}"
```

---

## 第三部分：AGENTS 关键原文（完整嵌入）

## A. AGENTS/status.md 关键段

```markdown
## Session 2026-03-14: Competition Strategy Overhaul

- Reviewed evaluation results: baseline=6.35, previous generalist (LoRA-everywhere step 20000)=5.08, net regression of -1.27.
- Categorized all 10 tasks by improvement potential:
  - PROTECT (near-perfect): open_door=1.0, scoop_popcorn=1.0, hold_pot=1.0, take_wrong_item=0.97 (3.97 pts)
  - IMPROVE (weak): clean_desktop=0.39, stock_shelf=0.24, place_block=0.42 (1.05 pts, ceiling 3.0)
- Traced random-init issue for missing LoRA tensors (~0.001 residual scale).

### Code Changes Made
1. New config `acot_challenge_lora_conservative`
2. Fixed warmup bug in `acot_challenge_generalist_baseline_compatible`
3. Added `--lora-only` in `scripts/extract_adapter.py`
4. Specialist configs support baseline init fallback
5. Routed policy shifted to weak-task-only strategy

### Strategy Established
- Hard rule: never submit below 6.35
- Track A: conservative LoRA
- Track B: baseline-compatible fallback
- Track C: weak-task lightweight routing
```

```markdown
## Current Retained Artifacts
- retained main experiment: checkpoints/acot_challenge_generalist_lora_generalist/generalist_v1_bs96
- one saved checkpoint at step 5000
- train_metrics entries through step 8500
- best val loss 0.1684 at step 7000
- last train loss 0.0787
```

## B. AGENTS/experiments.md 关键段

```markdown
- Date: 2026-03-13
  Experiment: generalist_lora_fast_step5000_eval
  Result: 5.08 total (vs baseline 6.35, regression -1.27)
  Notes:
  - checkpoint at step 20000 evaluated via evaluation_result_generalist_lora_fast_step20000.xlsx
  - open_door: 1.0 -> 0.4
  - scoop_popcorn: 1.0 -> 0.8
  - hold_pot: 1.0 -> 0.85
  - clean_desktop: 0.39 -> 0.19
  - stock_shelf improved: 0.24 -> 0.40
  - conclusion: val loss does not reliably predict benchmark score
```

## C. AGENTS/decisions.md 关键段

```markdown
Date: 2026-03-14
Decision: Adopt baseline-first strategy, never submit below 6.35.

Date: 2026-03-14
Decision: Use `acot_challenge_lora_conservative` as primary path.

Date: 2026-03-14
Decision: Route only weak tasks to specialists; strong tasks use base/default.

Date: 2026-03-14
Decision: Support `--lora-only` adapter extraction for lightweight routing.
```

## D. AGENTS/constraints.md 关键段

```markdown
Hard submission contract:
- final artifact is Docker image
- service must listen on port 8999
- websocket policy server
- valid output type: abs_joint or abs_pose
- image must be runnable by evaluator without private dependencies

Repo-specific constraints:
- fast path is additive only; must not break legacy train.py path
- inference checkpoint layout requires params + assets
- routed serving key is websocket payload["task_name"]
```

---

## 第四部分：仍待澄清的关键矛盾

1. 配置漂移：

- 代码中的 tuned config（warmup=100/steps=5000）与历史运行记录（warmup=2000/steps=20000）出现不一致。

2. 指标错配：

- train/val loss 下降并不保证 benchmark 提升（已有 5.08 反例）。

3. 路由落地：

- 当前代码路由表仅 clean_desktop 单键，仍未完全体现“3 弱任务路由”的最终目标。

4. 观测缺口：

- wandb 网络异常导致多数 run summary 缺失，需要更强本地事实源（train_metrics + checkpoint events）。

---

## 第五部分：请专家重点回答的问题

## A. 训练策略

1. 在当前任务分布下，应否坚持“保守 LoRA 单主线”，还是并行 baseline-compatible 对照更稳妥？
2. warmup/总步数比例如何设定能最大化“保强任务+提弱任务”？
3. 是否应引入任务重加权或 curriculum，减少强任务回退？
4. 除 loss 外，应该用什么更强代理指标做 checkpoint 选择？

## B. 路由与 adapter

5. `base + adapter overlay` 且 `missing_init=zeros` 的方案在泛化上是否合理？
6. 路由是先维持单键 A/B，还是直接扩到 3 个弱任务更有效？
7. 参数提取应继续字符串 pattern，还是改成结构化模块白名单导出？
8. LoRA-only vs full adapter 的性价比边界应如何定量评估？

## C. 评估与提交

9. 是否应把 challenge 小样本闭环评测前置到每个关键 checkpoint？
10. 对强任务退化，优先约束参数/数据/优化哪一层？
11. 是否建议建立本地 dashboard，弱化对 wandb 在线 summary 的依赖？
12. 从单 checkpoint 切到 routed 提交，门槛应该如何定义（总分、弱任务增益、强任务下限）？

---

## 附：本报告数据原件

- `docs/_expert_consult_data_2026-03-14.json`
- `checkpoints/acot_challenge_generalist_lora_generalist_tuned/generalists_fast_v1_bs64/train_metrics.jsonl`
- `wandb/*`
