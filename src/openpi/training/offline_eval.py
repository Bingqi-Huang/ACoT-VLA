from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def discover_checkpoints(path: pathlib.Path) -> list[pathlib.Path]:
    if (path / "params").exists():
        return [path]

    checkpoints = []
    for child in sorted(path.iterdir(), key=lambda item: _checkpoint_step(item), reverse=False):
        if child.is_dir() and (child / "params").exists():
            checkpoints.append(child)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {path}")
    return checkpoints


def checkpoint_step(path: pathlib.Path) -> int:
    return _checkpoint_step(path)


def build_eval_output_dir(input_path: pathlib.Path, output_dir: pathlib.Path | None = None) -> pathlib.Path:
    if output_dir is not None:
        return output_dir
    if (input_path / "params").exists():
        return input_path.parent / "eval"
    return input_path / "eval"


def load_model_and_data_config(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
) -> tuple[_model.BaseModel, _config.DataConfig]:
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load normalization stats for offline evaluation.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
    return model, data_config


def create_validation_loader(
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
    *,
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    dataset = _data_loader.create_torch_dataset(
        data_config,
        train_config.model,
        split="val",
        split_base_dir=train_config.assets_dirs / "episode_splits",
    )
    dataset = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)
    dataset = _data_loader.SafeDataset(dataset)

    mp_context = None
    if num_workers > 0:
        import multiprocessing

        mp_context = multiprocessing.get_context("spawn")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_data_loader._collate_fn,
        worker_init_fn=_data_loader._worker_init_fn,
        drop_last=False,
    )


def build_action_output_transform(data_config: _config.DataConfig) -> _transforms.DataTransformFn:
    return _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )


def to_jax_batch(batch: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(convert(v) for v in value)

        array = np.asarray(value)
        if array.dtype.kind in {"U", "S", "O"}:
            return value
        return jnp.asarray(array)

    return convert(batch)


def normalize_task_name(task_name: Any) -> str:
    if isinstance(task_name, bytes):
        return task_name.decode()
    if isinstance(task_name, np.ndarray):
        if task_name.shape == ():
            return normalize_task_name(task_name.item())
        raise ValueError(f"Expected scalar task name, got shape {task_name.shape}")
    return str(task_name)


@dataclasses.dataclass
class OfflineEvalAccumulator:
    action_dim: int
    horizon: int
    loss_sum: float = 0.0
    loss_count: int = 0
    per_task_loss_sum: dict[str, float] = dataclasses.field(default_factory=lambda: defaultdict(float))
    per_task_loss_count: dict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    action_abs_error_sum: np.ndarray | None = None
    action_abs_error_count: int = 0
    horizon_abs_error_sum: np.ndarray | None = None
    horizon_abs_error_count: int = 0
    gripper_binary_candidate: bool = True
    gripper_bce_sum: float = 0.0
    gripper_correct: int = 0
    gripper_count: int = 0

    def __post_init__(self) -> None:
        if self.action_abs_error_sum is None:
            self.action_abs_error_sum = np.zeros(self.action_dim, dtype=np.float64)
        if self.horizon_abs_error_sum is None:
            self.horizon_abs_error_sum = np.zeros(self.horizon, dtype=np.float64)

    def update(
        self,
        *,
        loss_per_example: np.ndarray,
        predicted_actions: np.ndarray,
        target_actions: np.ndarray,
        task_names: np.ndarray,
    ) -> None:
        abs_error = np.abs(predicted_actions - target_actions)
        self.loss_sum += float(loss_per_example.sum())
        self.loss_count += int(loss_per_example.shape[0])

        self.action_abs_error_sum += abs_error.sum(axis=(0, 1))
        self.action_abs_error_count += int(abs_error.shape[0] * abs_error.shape[1])

        self.horizon_abs_error_sum += abs_error.sum(axis=(0, 2))
        self.horizon_abs_error_count += int(abs_error.shape[0] * abs_error.shape[2])

        for task_name, loss_value in zip(task_names, loss_per_example, strict=True):
            task_key = normalize_task_name(task_name)
            self.per_task_loss_sum[task_key] += float(loss_value)
            self.per_task_loss_count[task_key] += 1

        if self.action_dim == 8:
            gripper_target = target_actions[..., -1]
            gripper_prediction = predicted_actions[..., -1]
            self.gripper_binary_candidate &= bool(
                np.all(np.isclose(gripper_target, 0.0, atol=1e-4) | np.isclose(gripper_target, 1.0, atol=1e-4))
            )

            gripper_prob = np.asarray(gripper_prediction)
            if np.any((gripper_prob < 0.0) | (gripper_prob > 1.0)):
                gripper_prob = 1.0 / (1.0 + np.exp(-gripper_prob))
            gripper_prob = np.clip(gripper_prob, 1e-6, 1.0 - 1e-6)
            gripper_target_binary = gripper_target >= 0.5

            self.gripper_bce_sum += float(
                (-gripper_target * np.log(gripper_prob) - (1.0 - gripper_target) * np.log(1.0 - gripper_prob)).sum()
            )
            self.gripper_correct += int(((gripper_prob >= 0.5) == gripper_target_binary).sum())
            self.gripper_count += int(gripper_target.size)

    def finalize(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "overall_val_loss": self.loss_sum / max(self.loss_count, 1),
            "per_task_val_loss": {
                task: self.per_task_loss_sum[task] / self.per_task_loss_count[task]
                for task in sorted(self.per_task_loss_sum)
            },
            "per_action_dim_mae": (self.action_abs_error_sum / max(self.action_abs_error_count, 1)).tolist(),
            "per_action_dim_mae_named": {
                f"dim_{index}": float(value)
                for index, value in enumerate(self.action_abs_error_sum / max(self.action_abs_error_count, 1))
            },
            "horizon_step_mae": (self.horizon_abs_error_sum / max(self.horizon_abs_error_count, 1)).tolist(),
        }

        horizon_mae = np.asarray(metrics["horizon_step_mae"], dtype=np.float64)
        horizon_slices = np.array_split(np.arange(self.horizon), 3)
        metrics["horizon_segment_mae"] = {
            name: float(horizon_mae[idx].mean()) if len(idx) > 0 else math.nan
            for name, idx in zip(("early", "mid", "late"), horizon_slices, strict=True)
        }

        if self.action_dim == 8:
            joint_mae = np.asarray(metrics["per_action_dim_mae"][:7], dtype=np.float64)
            metrics["joint_mae"] = {
                f"joint_{index}": float(value) for index, value in enumerate(joint_mae)
            }
            metrics["overall_joint_mae"] = float(joint_mae.mean())
            if self.gripper_count > 0:
                metrics["gripper"] = {
                    "is_binary_target": self.gripper_binary_candidate,
                    "accuracy": float(self.gripper_correct / self.gripper_count),
                    "bce": float(self.gripper_bce_sum / self.gripper_count),
                }
        return metrics


def write_metrics_json(output_path: pathlib.Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_summary_csv(output_path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_summary_row(checkpoint_path: pathlib.Path, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint_step": checkpoint_step(checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "overall_val_loss": metrics["overall_val_loss"],
    }

    for task_name, task_loss in metrics.get("per_task_val_loss", {}).items():
        row[f"task__{task_name}__val_loss"] = task_loss

    if "overall_joint_mae" in metrics:
        row["joint_mae"] = metrics["overall_joint_mae"]
    if "gripper" in metrics:
        row["gripper_accuracy"] = metrics["gripper"].get("accuracy")
        row["gripper_bce"] = metrics["gripper"].get("bce")

    for segment_name, segment_mae in metrics.get("horizon_segment_mae", {}).items():
        row[f"horizon_{segment_name}_mae"] = segment_mae

    return row


def _checkpoint_step(path: pathlib.Path) -> int:
    try:
        return int(path.name)
    except ValueError:
        return -1
