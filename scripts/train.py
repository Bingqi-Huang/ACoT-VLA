import dataclasses
import functools
import json
import logging
import platform
import re
import time
from typing import Any
import os
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.transforms as _transforms
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def build_action_output_transform(data_config: _config.DataConfig) -> _transforms.DataTransformFn:
    return _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )


def _normalize_task_name(task_name: Any) -> str:
    if isinstance(task_name, bytes):
        return task_name.decode()
    if isinstance(task_name, np.ndarray):
        if task_name.shape == ():
            return _normalize_task_name(task_name.item())
        raise ValueError(f"Expected scalar task name, got shape {task_name.shape}")
    return str(task_name)


def _sanitize_metric_component(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_")
    return sanitized or "unknown"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, jax.Array):
        return _json_ready(np.asarray(jax.device_get(value)))
    return value


class JsonlMetricLogger:
    def __init__(self, path: epath.PathLike):
        self._path = epath.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps(_json_ready(payload), sort_keys=True) + "\n")


def _last_batch_metadata(data_loader: _data_loader.DataLoader[Any]) -> dict[str, np.ndarray] | None:
    return data_loader.last_metadata()


def _summarize_train_batch_metrics(
    *,
    batch: tuple[_model.Observation, _model.Actions] | tuple[_model.Observation, _model.Actions, _model.CoarseActions],
    batch_metrics: dict[str, Any],
    batch_metadata: dict[str, np.ndarray] | None,
    action_output_transform: _transforms.DataTransformFn,
) -> dict[str, float]:
    observation = batch[0]
    target_actions = np.asarray(jax.device_get(batch[1]))
    predicted_actions = np.asarray(jax.device_get(batch_metrics["predicted_actions"]))
    loss_per_example = np.asarray(jax.device_get(batch_metrics["loss_per_example"]))
    state = np.asarray(jax.device_get(observation.state))

    raw_predicted_actions = np.asarray(
        action_output_transform({"state": state.copy(), "actions": predicted_actions.copy()})["actions"]
    )
    raw_target_actions = np.asarray(action_output_transform({"state": state.copy(), "actions": target_actions.copy()})["actions"])
    abs_error = np.abs(raw_predicted_actions - raw_target_actions)

    metrics: dict[str, float] = {
        "train/batch_loss_eval": float(loss_per_example.mean()),
        "train/action_mae/overall": float(abs_error.mean()),
    }

    if batch_metadata is not None and "task" in batch_metadata:
        task_names = np.asarray(batch_metadata["task"])
        if task_names.shape[0] == loss_per_example.shape[0]:
            for task_name in np.unique(task_names):
                task_mask = task_names == task_name
                task_key = _sanitize_metric_component(_normalize_task_name(task_name))
                metrics[f"train/task/{task_key}/loss"] = float(loss_per_example[task_mask].mean())

    per_action_dim_mae = abs_error.mean(axis=(0, 1))
    for index, value in enumerate(per_action_dim_mae):
        metrics[f"train/action_mae/dim_{index:02d}"] = float(value)

    if raw_target_actions.shape[-1] == 8:
        joint_mae = per_action_dim_mae[:7]
        metrics["train/joint_mae/overall"] = float(joint_mae.mean())
        for index, value in enumerate(joint_mae):
            metrics[f"train/joint_mae/joint_{index}"] = float(value)

        gripper_target = raw_target_actions[..., -1]
        if np.all(np.isclose(gripper_target, 0.0, atol=1e-4) | np.isclose(gripper_target, 1.0, atol=1e-4)):
            gripper_prediction = raw_predicted_actions[..., -1]
            if np.any((gripper_prediction < 0.0) | (gripper_prediction > 1.0)):
                gripper_prediction = 1.0 / (1.0 + np.exp(-gripper_prediction))
            gripper_prediction = np.clip(gripper_prediction, 1e-6, 1.0 - 1e-6)
            gripper_target_binary = gripper_target >= 0.5
            metrics["train/gripper_accuracy"] = float(((gripper_prediction >= 0.5) == gripper_target_binary).mean())
            metrics["train/gripper_bce"] = float(
                (-gripper_target * np.log(gripper_prediction) - (1.0 - gripper_target) * np.log(1.0 - gripper_prediction)).mean()
            )

    return metrics


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _fold_in_train_rng(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    micro_step: at.Int[at.ArrayLike, ""] = 0,
) -> at.KeyArrayLike:
    return jax.random.fold_in(rng, state.step * config.grad_accum_steps + micro_step)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def compute_grads(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    micro_step: at.Int[at.ArrayLike, ""],
) -> tuple[nnx.State, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, train_rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(train_rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = _fold_in_train_rng(config, rng, state, micro_step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    return grads, loss


@at.typecheck
def apply_grads(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    grads: nnx.State,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng

    model = nnx.merge(state.model_def, state.params)
    model.train()

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_trainable_params = optax.apply_updates(params, updates)

    nnx.update(model, new_trainable_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _eval_params(state: training_utils.TrainState) -> at.Params:
    return state.ema_params if state.ema_params is not None else state.params


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, _eval_params(state))
    model.eval()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, eval_rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(eval_rng, observation, actions, train=False)
        return jnp.mean(chunked_loss)

    eval_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    loss = loss_fn(model, eval_rng, observation, actions)
    return {"val_loss": loss}


@at.typecheck
def train_batch_metrics_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    del config
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    eval_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    typed_model = model
    loss_per_example = typed_model.compute_loss_per_example(eval_rng, observation, actions, train=False)  # type: ignore[attr-defined]
    predicted_actions = typed_model.teacher_force_actions(observation, actions, time=0.5)  # type: ignore[attr-defined]
    return {
        "loss_per_example": loss_per_example,
        "predicted_actions": predicted_actions,
    }

@at.typecheck
def acot_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.CoarseActions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions,
        coarse_actions: _model.CoarseActions
    ):
        return model.compute_loss(rng, observation, actions, coarse_actions, train=True)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, coarse_actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions, coarse_actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def acot_compute_grads(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.CoarseActions],
    micro_step: at.Int[at.ArrayLike, ""],
) -> tuple[nnx.State, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        train_rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
    ):
        return model.compute_loss(train_rng, observation, actions, coarse_actions, train=True)

    train_rng = _fold_in_train_rng(config, rng, state, micro_step)
    observation, actions, coarse_actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions, coarse_actions)

    return grads, loss


@at.typecheck
def acot_eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.CoarseActions],
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, _eval_params(state))
    model.eval()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        eval_rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
    ):
        return model.compute_loss(eval_rng, observation, actions, coarse_actions, train=False)

    eval_rng = jax.random.fold_in(rng, state.step)
    observation, actions, coarse_actions = batch
    loss = loss_fn(model, eval_rng, observation, actions, coarse_actions)
    return {"val_loss": loss}


@at.typecheck
def acot_train_batch_metrics_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.CoarseActions],
) -> dict[str, at.Array]:
    del config
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    eval_rng = jax.random.fold_in(rng, state.step)
    observation, actions, coarse_actions = batch
    typed_model = model
    loss_per_example = typed_model.compute_loss_per_example(  # type: ignore[attr-defined]
        eval_rng,
        observation,
        actions,
        coarse_actions,
        train=False,
    )
    predicted_actions = typed_model.teacher_force_actions(  # type: ignore[attr-defined]
        observation,
        actions,
        coarse_actions,
        time=0.5,
    )
    return {
        "loss_per_example": loss_per_example,
        "predicted_actions": predicted_actions,
    }


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

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
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        split="train",
        sharding=data_sharding,
        shuffle=True,
    )
    action_output_transform = build_action_output_transform(data_loader.data_config())
    metrics_logger = JsonlMetricLogger(config.checkpoint_dir / "train_metrics.jsonl")
    data_iter = iter(data_loader)
    batch = next(data_iter)
    batch_metadata = _last_batch_metadata(data_loader)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    val_loader = None
    if config.val_interval is not None:
        if data_loader.data_config().episode_split is None:
            logging.warning("Validation requested but no episode split is configured; skipping validation.")
        else:
            val_loader = _data_loader.create_data_loader(
                config,
                split="val",
                sharding=data_sharding,
                shuffle=False,
                num_batches=config.val_num_batches,
            )
            logging.info("Initialized validation loader with %d batches per evaluation.", config.val_num_batches)

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    num_params = training_utils.count_parameters(train_state.params)
    logging.info(f"Total number of parameters: {num_params:,}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    trainable_params_sharding = sharding.fsdp_sharding(train_state.params.filter(config.trainable_filter), mesh)
    ptrain_batch_metrics = None
    if config.model.model_type == _model.ModelType.ACOT_VLA_PI05 or config.model.model_type == _model.ModelType.ACOT_VLA_PI0:
        pcompute_grads = jax.jit(
            functools.partial(acot_compute_grads, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
            out_shardings=(trainable_params_sharding, replicated_sharding),
        )
        ptrain_batch_metrics = jax.jit(
            functools.partial(acot_train_batch_metrics_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        )
        peval_step = jax.jit(
            functools.partial(acot_eval_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
    else:
        pcompute_grads = jax.jit(
            functools.partial(compute_grads, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
            out_shardings=(trainable_params_sharding, replicated_sharding),
        )
        ptrain_batch_metrics = jax.jit(
            functools.partial(train_batch_metrics_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        )
        peval_step = jax.jit(
            functools.partial(eval_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        if config.model.model_type == _model.ModelType.PI0_FAST:
            ptrain_batch_metrics = None
    papply_grads = jax.jit(
        functools.partial(apply_grads, config),
        in_shardings=(replicated_sharding, train_state_sharding, trainable_params_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1, 2),
    )

    start_step = int(train_state.step)
    print("\n--- Trainable Parameters ---")
    model = nnx.merge(train_state.model_def, train_state.params)
    trainable_state = nnx.state(model, config.trainable_filter)
    logging.info(f"{training_utils.array_tree_to_info(trainable_state)}")
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
                batch_metadata = _last_batch_metadata(data_loader)

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
                batch_metric_payload = _summarize_train_batch_metrics(
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

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
