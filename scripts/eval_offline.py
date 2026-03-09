import dataclasses
import logging
import pathlib
from typing import Any

import jax
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import offline_eval as _offline_eval


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint: str
    output_dir: str | None = None
    batch_size: int | None = None
    num_workers: int = 0
    max_batches: int | None = None
    seed: int = 0
    teacher_force_time: float = 0.5


def _evaluate_checkpoint(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    *,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    seed: int,
    teacher_force_time: float,
) -> dict[str, Any]:
    model, data_config = _offline_eval.load_model_and_data_config(train_config, checkpoint_dir)
    if data_config.episode_split is None:
        raise ValueError("Offline evaluation requires an episode-level validation split.")

    if not hasattr(model, "compute_loss_per_example") or not hasattr(model, "teacher_force_actions"):
        raise ValueError(
            f"Model type {type(model).__name__} does not implement offline evaluation hooks."
        )

    val_loader = _offline_eval.create_validation_loader(
        train_config,
        data_config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    output_transform = _offline_eval.build_action_output_transform(data_config)
    loss_fn = nnx_utils.module_jit(model.compute_loss_per_example)  # type: ignore[arg-type]
    predict_fn = nnx_utils.module_jit(model.teacher_force_actions)  # type: ignore[arg-type]

    accumulator: _offline_eval.OfflineEvalAccumulator | None = None
    base_rng = jax.random.key(seed)

    for batch_index, batch in enumerate(val_loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        batch_size_current = np.asarray(batch["actions"]).shape[0]
        task_names = np.asarray(batch.get("task", np.full(batch_size_current, "unknown")))
        jax_batch = _offline_eval.to_jax_batch(batch)
        observation = _model.Observation.from_dict(jax_batch)
        actions = jax_batch["actions"]
        eval_rng = jax.random.fold_in(base_rng, batch_index)

        if "coarse_actions" in jax_batch:
            coarse_actions = jax_batch["coarse_actions"]
            loss_per_example = loss_fn(eval_rng, observation, actions, coarse_actions, train=False)
            predicted_actions = predict_fn(
                observation,
                actions,
                coarse_actions,
                time=teacher_force_time,
            )
        else:
            loss_per_example = loss_fn(eval_rng, observation, actions, train=False)
            predicted_actions = predict_fn(
                observation,
                actions,
                time=teacher_force_time,
            )

        state = np.asarray(jax.device_get(jax_batch["state"]))
        predicted_actions_np = np.asarray(jax.device_get(predicted_actions))
        target_actions_np = np.asarray(jax.device_get(actions))
        loss_per_example_np = np.asarray(jax.device_get(loss_per_example))

        raw_predicted_actions = output_transform(
            {"state": state.copy(), "actions": predicted_actions_np.copy()}
        )["actions"]
        raw_target_actions = output_transform(
            {"state": state.copy(), "actions": target_actions_np.copy()}
        )["actions"]

        if accumulator is None:
            accumulator = _offline_eval.OfflineEvalAccumulator(
                action_dim=raw_target_actions.shape[-1],
                horizon=raw_target_actions.shape[-2],
            )
        accumulator.update(
            loss_per_example=loss_per_example_np,
            predicted_actions=np.asarray(raw_predicted_actions),
            target_actions=np.asarray(raw_target_actions),
            task_names=task_names,
        )

    if accumulator is None:
        raise ValueError(f"No validation batches were produced for checkpoint {checkpoint_dir}")

    metrics = accumulator.finalize()
    metrics.update(
        {
            "checkpoint_path": str(checkpoint_dir),
            "checkpoint_step": _offline_eval.checkpoint_step(checkpoint_dir),
            "teacher_force_time": teacher_force_time,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "max_batches": max_batches,
        }
    )
    return metrics


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    train_config = _config.get_config(args.config_name)
    checkpoint_input = pathlib.Path(args.checkpoint).expanduser().resolve()
    output_dir = _offline_eval.build_eval_output_dir(
        checkpoint_input,
        pathlib.Path(args.output_dir).expanduser().resolve() if args.output_dir is not None else None,
    )
    checkpoint_dirs = _offline_eval.discover_checkpoints(checkpoint_input)

    batch_size = args.batch_size or train_config.batch_size
    summary_rows = []

    for checkpoint_dir in checkpoint_dirs:
        metrics = _evaluate_checkpoint(
            train_config,
            checkpoint_dir,
            batch_size=batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            seed=args.seed,
            teacher_force_time=args.teacher_force_time,
        )
        output_path = output_dir / f"checkpoint_{metrics['checkpoint_step']}.json"
        _offline_eval.write_metrics_json(output_path, metrics)
        summary_rows.append(_offline_eval.flatten_summary_row(checkpoint_dir, metrics))

        print(f"\nCheckpoint {metrics['checkpoint_step']}")
        print(f"  overall_val_loss: {metrics['overall_val_loss']:.6f}")
        if "overall_joint_mae" in metrics:
            print(f"  overall_joint_mae: {metrics['overall_joint_mae']:.6f}")
        if "gripper" in metrics:
            print(f"  gripper_accuracy: {metrics['gripper']['accuracy']:.6f}")
            print(f"  gripper_bce: {metrics['gripper']['bce']:.6f}")
        print(f"  horizon_segment_mae: {metrics['horizon_segment_mae']}")
        print(f"  per_task_val_loss: {metrics['per_task_val_loss']}")
        print(f"  wrote: {output_path}")

    summary_path = output_dir / "summary.csv"
    _offline_eval.write_summary_csv(summary_path, summary_rows)
    print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
