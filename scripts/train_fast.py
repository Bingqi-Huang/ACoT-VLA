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
import wandb

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader_fast as _data_loader_fast
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.train as legacy_train


def _host_preview_images(data_loader, batch) -> list[wandb.Image]:
    host_batch = getattr(getattr(data_loader, "_data_loader", None), "last_host_batch", None)
    if isinstance(host_batch, dict) and "image" in host_batch:
        image_tree = host_batch["image"]
        first_camera = next(iter(image_tree.values()))
        limit = min(5, len(first_camera))
        return [
            wandb.Image(np.concatenate([np.asarray(img[i]) for img in image_tree.values()], axis=1))
            for i in range(limit)
        ]

    return [
        wandb.Image(np.concatenate([np.asarray(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]


def main(config: _config.TrainConfig, *, r2a_cache_root: pathlib.Path | None = None):
    legacy_train.init_logging()
    legacy_train.logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    # Keep auxiliary thread pools contained in the fast-path process.
    if hasattr(legacy_train, "torch"):
        legacy_train.torch.set_num_threads(1)
        legacy_train.torch.set_num_interop_threads(1)

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

    preview_loader = _data_loader_fast.create_data_loader(
        config,
        split="train",
        sharding=data_sharding,
        shuffle=True,
        num_batches=1,
        prompt_cache_split="train",
        r2a_cache_root=r2a_cache_root,
    )
    action_output_transform = legacy_train.build_action_output_transform(preview_loader.data_config())
    metrics_logger = legacy_train.JsonlMetricLogger(config.checkpoint_dir / "train_metrics.jsonl")
    preview_iter = iter(preview_loader)
    preview_batch = next(preview_iter)
    legacy_train.logging.info(f"Initialized fast data loader:\n{training_utils.array_tree_to_info(preview_batch)}")

    val_loader = None
    if config.val_interval is not None:
        if preview_loader.data_config().episode_split is None:
            legacy_train.logging.warning("Validation requested but no episode split is configured; skipping validation.")
        else:
            val_loader = _data_loader_fast.create_data_loader(
                config,
                split="val",
                sharding=data_sharding,
                shuffle=False,
                num_batches=config.val_num_batches,
                prompt_cache_split="val",
                r2a_cache_root=r2a_cache_root,
            )
            legacy_train.logging.info("Initialized fast validation loader with %d batches per evaluation.", config.val_num_batches)

    images_to_log = _host_preview_images(preview_loader, preview_batch)
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = legacy_train.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    legacy_train.logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    num_params = training_utils.count_parameters(train_state.params)
    legacy_train.logging.info(f"Total number of parameters: {num_params:,}")

    if resuming:
        train_data_loader = _data_loader_fast.create_data_loader(
            config,
            split="train",
            sharding=data_sharding,
            shuffle=True,
            prompt_cache_split="train",
            r2a_cache_root=r2a_cache_root,
        )
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, train_data_loader)
    else:
        train_data_loader = _data_loader_fast.create_data_loader(
            config,
            split="train",
            sharding=data_sharding,
            shuffle=True,
            prompt_cache_split="train",
            r2a_cache_root=r2a_cache_root,
        )

    data_iter = iter(train_data_loader)
    batch = next(data_iter)
    batch_metadata = legacy_train._last_batch_metadata(train_data_loader)

    trainable_params_sharding = sharding.fsdp_sharding(train_state.params.filter(config.trainable_filter), mesh)
    ptrain_batch_metrics = None
    peval_step = None
    ptrain_step = None
    pcompute_grads = None
    papply_grads = None

    if config.model.model_type in (_model.ModelType.ACOT_VLA_PI05, _model.ModelType.ACOT_VLA_PI0):
        ptrain_batch_metrics = jax.jit(
            functools.partial(legacy_train.acot_train_batch_metrics_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        )
        peval_step = jax.jit(
            functools.partial(legacy_train.acot_eval_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        if config.grad_accum_steps == 1:
            ptrain_step = jax.jit(
                functools.partial(legacy_train.acot_train_step, config),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
                out_shardings=(train_state_sharding, replicated_sharding),
                donate_argnums=(1,),
            )
        else:
            pcompute_grads = jax.jit(
                functools.partial(legacy_train.acot_compute_grads, config),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
                out_shardings=(trainable_params_sharding, replicated_sharding),
            )
    else:
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
        if config.grad_accum_steps == 1:
            ptrain_step = jax.jit(
                functools.partial(legacy_train.train_step, config),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
                out_shardings=(train_state_sharding, replicated_sharding),
                donate_argnums=(1,),
            )
        else:
            pcompute_grads = jax.jit(
                functools.partial(legacy_train.compute_grads, config),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
                out_shardings=(trainable_params_sharding, replicated_sharding),
            )

    if config.grad_accum_steps > 1:
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
    legacy_train.logging.info(f"{training_utils.array_tree_to_info(trainable_state)}")
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
        metric_batch = batch
        metric_batch_metadata = batch_metadata
        with sharding.set_mesh(mesh):
            if config.grad_accum_steps == 1:
                assert ptrain_step is not None
                train_state, info = ptrain_step(train_rng, train_state, batch)
                batch = next(data_iter)
                batch_metadata = legacy_train._last_batch_metadata(train_data_loader)
            else:
                accumulated_grads = None
                total_loss = None
                for micro_step in range(config.grad_accum_steps):
                    metric_batch = batch
                    metric_batch_metadata = batch_metadata
                    assert pcompute_grads is not None
                    grads, loss = pcompute_grads(train_rng, train_state, batch, micro_step)
                    total_loss = loss if total_loss is None else total_loss + loss
                    accumulated_grads = grads if accumulated_grads is None else jax.tree.map(jnp.add, accumulated_grads, grads)
                    batch = next(data_iter)
                    batch_metadata = legacy_train._last_batch_metadata(train_data_loader)

                assert accumulated_grads is not None
                assert total_loss is not None
                avg_grads = jax.tree.map(lambda g: g / config.grad_accum_steps, accumulated_grads)
                assert papply_grads is not None
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
            wall_time_sec = now - train_start_time
            throughput_metrics = {
                "train/step": step,
                "train/train_state_step": train_state_step,
                "train/loss": reduced_info["loss"],
                "train/grad_norm": reduced_info["grad_norm"],
                "train/param_norm": reduced_info["param_norm"],
                "train/learning_rate": float(jax.device_get(lr_schedule(train_state_step))),
                "train/steps_per_sec": len(infos) / elapsed,
                "train/samples_per_sec": len(infos) * config.batch_size * config.grad_accum_steps / elapsed,
                "train/wall_time_sec": wall_time_sec,
            }
            log_payload = {
                **reduced_info,
                **throughput_metrics,
                **batch_metric_payload,
            }
            info_str = ", ".join(
                [
                    f"loss={log_payload['train/loss']:.4f}",
                    f"lr={log_payload['train/learning_rate']:.2e}",
                    f"grad_norm={log_payload['train/grad_norm']:.4f}",
                    f"steps/sec={log_payload['train/steps_per_sec']:.2f}",
                    f"wall={log_payload['train/wall_time_sec']:.1f}s",
                ]
            )
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(log_payload, step=step)
            metrics_logger.log({"event": "train", **log_payload})
            infos = []
            last_log_time = now

        if val_loader is not None and (step % config.val_interval == 0 or step == config.num_train_steps - 1):
            val_infos = []
            assert peval_step is not None
            with sharding.set_mesh(mesh):
                for val_batch in val_loader:
                    val_infos.append(peval_step(train_rng, train_state, val_batch))
            reduced_val_info = {
                k: float(v) for k, v in jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(val_infos))).items()
            }
            reduced_val_info["val/loss"] = reduced_val_info["val_loss"]
            reduced_val_info["train/step"] = step
            reduced_val_info["train/train_state_step"] = int(jax.device_get(train_state.step))
            reduced_val_info["train/wall_time_sec"] = time.perf_counter() - train_start_time
            val_info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_val_info.items())
            pbar.write(f"Validation {step}: {val_info_str}")
            wandb.log(reduced_val_info, step=step)
            metrics_logger.log({"event": "val", **reduced_val_info})

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, train_data_loader, step)
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


def _parse_r2a_cache_root(argv: list[str]) -> tuple[list[str], pathlib.Path | None]:
    remaining: list[str] = []
    cache_root: pathlib.Path | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--r2a-cache-root":
            if i + 1 >= len(argv):
                raise ValueError("--r2a-cache-root requires a value")
            cache_root = pathlib.Path(argv[i + 1]).expanduser()
            i += 2
            continue
        if arg.startswith("--r2a-cache-root="):
            cache_root = pathlib.Path(arg.split("=", 1)[1]).expanduser()
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return remaining, cache_root


if __name__ == "__main__":
    cli_args, cache_root = _parse_r2a_cache_root(sys.argv[1:])
    sys.argv = [sys.argv[0], *cli_args]
    main(_config.cli(), r2a_cache_root=cache_root)
