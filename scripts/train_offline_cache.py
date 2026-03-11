from __future__ import annotations

import dataclasses
import functools
import os
import pathlib
import platform
import sys
import time

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader_offline as _data_loader_offline
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.train as legacy_train


@dataclasses.dataclass
class BootstrapArgs:
    base_config_name: str
    cache_root: str


def _cli() -> tuple[BootstrapArgs, _config.TrainConfig]:
    bootstrap, remaining = tyro.cli(BootstrapArgs, return_unknown_args=True)
    base_config = _config.get_config(bootstrap.base_config_name)
    config = tyro.cli(type(base_config), args=remaining, default=base_config)
    if config.name != bootstrap.base_config_name:
        raise ValueError(
            f"Offline cache train path must keep config name fixed at {bootstrap.base_config_name}; got {config.name}"
        )
    return bootstrap, config


def main(config: _config.TrainConfig, *, cache_root: pathlib.Path) -> None:
    legacy_train.init_logging()
    legacy_train.logging.info("Running on: %s", platform.node())

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite if not os.getenv("DEBUG_MODE", default=False) == "true" else True,
        resume=config.resume,
    )
    legacy_train.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    legacy_train.configure_wandb_metrics()

    data_loader = _data_loader_offline.create_data_loader(
        config,
        cache_root=cache_root,
        split="train",
        sharding=data_sharding,
        shuffle=True,
    )
    action_output_transform = legacy_train.build_action_output_transform(data_loader.data_config())
    metrics_logger = legacy_train.JsonlMetricLogger(config.checkpoint_dir / "train_metrics.jsonl")
    data_iter = iter(data_loader)
    batch = next(data_iter)
    batch_metadata = legacy_train._last_batch_metadata(data_loader)
    legacy_train.logging.info("Initialized offline cache data loader:\n%s", training_utils.array_tree_to_info(batch))

    val_loader = None
    if config.val_interval is not None:
        val_loader = _data_loader_offline.create_data_loader(
            config,
            cache_root=cache_root,
            split="val",
            sharding=data_sharding,
            shuffle=False,
            num_batches=config.val_num_batches,
        )
        legacy_train.logging.info("Initialized offline cache validation loader with %d batches.", config.val_num_batches)

    images_to_log = [
        wandb.Image(np.concatenate([np.asarray(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = legacy_train.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    legacy_train.logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(train_state.params))
    num_params = training_utils.count_parameters(train_state.params)
    legacy_train.logging.info("Total number of parameters: %s", f"{num_params:,}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    trainable_params_sharding = sharding.fsdp_sharding(train_state.params.filter(config.trainable_filter), mesh)
    ptrain_batch_metrics = None
    if config.model.model_type in (_model.ModelType.ACOT_VLA_PI05, _model.ModelType.ACOT_VLA_PI0):
        pcompute_grads = jax.jit(
            functools.partial(legacy_train.acot_compute_grads, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
            out_shardings=(trainable_params_sharding, replicated_sharding),
        )
        ptrain_batch_metrics = jax.jit(
            functools.partial(legacy_train.acot_train_batch_metrics_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        )
        peval_step = jax.jit(
            functools.partial(legacy_train.acot_eval_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
    else:
        pcompute_grads = jax.jit(
            functools.partial(legacy_train.compute_grads, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
            out_shardings=(trainable_params_sharding, replicated_sharding),
        )
        ptrain_batch_metrics = jax.jit(
            functools.partial(legacy_train.train_batch_metrics_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        )
        peval_step = jax.jit(
            functools.partial(legacy_train.eval_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        if config.model.model_type == _model.ModelType.PI0_FAST:
            ptrain_batch_metrics = None
    papply_grads = jax.jit(
        functools.partial(legacy_train.apply_grads, config),
        in_shardings=(replicated_sharding, train_state_sharding, trainable_params_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1, 2),
    )

    start_step = int(train_state.step)
    print("\n--- Trainable Parameters ---")
    model = nnx.merge(train_state.model_def, train_state.params)
    trainable_state = nnx.state(model, config.trainable_filter)
    legacy_train.logging.info("%s", training_utils.array_tree_to_info(trainable_state))
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    lr_schedule = config.lr_schedule.create()
    train_start_time = time.perf_counter()
    last_log_time = train_start_time
    infos = []
    for step in pbar:
        accumulated_grads = None
        total_loss = None
        metric_batch = batch
        metric_batch_metadata = batch_metadata
        with sharding.set_mesh(mesh):
            for micro_step in range(config.grad_accum_steps):
                metric_batch = batch
                metric_batch_metadata = batch_metadata
                grads, loss = pcompute_grads(train_rng, train_state, batch, micro_step)
                total_loss = loss if total_loss is None else total_loss + loss
                accumulated_grads = grads if accumulated_grads is None else jax.tree.map(jnp.add, accumulated_grads, grads)
                batch = next(data_iter)
                batch_metadata = legacy_train._last_batch_metadata(data_loader)

            assert accumulated_grads is not None
            assert total_loss is not None
            avg_grads = jax.tree.map(lambda g: g / config.grad_accum_steps, accumulated_grads)
            train_state, info = papply_grads(train_rng, train_state, avg_grads)
            info["loss"] = total_loss / config.grad_accum_steps
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = {k: float(v) for k, v in jax.device_get(jax.tree.map(jnp.mean, stacked_infos)).items()}
            batch_metric_payload = {}
            if ptrain_batch_metrics is not None:
                with sharding.set_mesh(mesh):
                    batch_metrics = jax.device_get(ptrain_batch_metrics(train_rng, train_state, metric_batch))
                batch_metric_payload = legacy_train._summarize_train_batch_metrics(
                    batch=metric_batch,
                    batch_metrics=batch_metrics,
                    batch_metadata=metric_batch_metadata,
                    action_output_transform=action_output_transform,
                )

            train_state_step = int(jax.device_get(train_state.step))
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-6)
            last_log_time = now
            train_steps_per_sec = max(step - start_step + 1, 1) / max(now - train_start_time, 1e-6)
            reduced_info.update(batch_metric_payload)
            reduced_info["train/step"] = step
            reduced_info["train/train_state_step"] = train_state_step
            reduced_info["train/learning_rate"] = lr_schedule(train_state.step)
            reduced_info["train/steps_per_sec"] = train_steps_per_sec
            reduced_info["train/wall_time_sec"] = now - train_start_time
            reduced_info["train/log_interval_sec"] = elapsed
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Train {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            metrics_logger.log({"event": "train", **reduced_info})
            infos = []

        if val_loader is not None and (step % config.val_interval == 0 or step == config.num_train_steps - 1):
            val_infos = []
            with sharding.set_mesh(mesh):
                for val_batch in val_loader:
                    val_infos.append(peval_step(train_rng, train_state, val_batch))
            reduced_val_info = {k: float(v) for k, v in jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(val_infos))).items()}
            reduced_val_info["val/loss"] = reduced_val_info["val_loss"]
            reduced_val_info["train/step"] = step
            reduced_val_info["train/train_state_step"] = int(jax.device_get(train_state.step))
            reduced_val_info["train/wall_time_sec"] = time.perf_counter() - train_start_time
            val_info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_val_info.items())
            pbar.write(f"Validation {step}: {val_info_str}")
            wandb.log(reduced_val_info, step=step)
            metrics_logger.log({"event": "val", **reduced_val_info})

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            checkpoint_payload = {
                "checkpoint/step": step,
                "checkpoint/train_state_step": int(jax.device_get(train_state.step)),
                "checkpoint/wall_time_sec": time.perf_counter() - train_start_time,
            }
            wandb.log(checkpoint_payload, step=step)
            metrics_logger.log({"event": "checkpoint", **checkpoint_payload})
            pbar.write(f"Checkpoint {step}: wall={checkpoint_payload['checkpoint/wall_time_sec']:.1f}s")

    legacy_train.logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    bootstrap_args, parsed_config = _cli()
    main(parsed_config, cache_root=pathlib.Path(bootstrap_args.cache_root).expanduser().resolve())
