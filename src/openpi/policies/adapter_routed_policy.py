from collections.abc import Sequence
import copy
import logging
import os
import pathlib
import time
from typing import Any

from flax import nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import array_typing as at
from openpi.shared import download as _download
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import weight_loaders

logger = logging.getLogger(__name__)

PATHS_KEY = "__paths__"
VALUE_KEY_TEMPLATE = "param_{index:04d}"
# Route only tasks that have a prepared adapter in the current submission package.
# Other tasks fall through to _base.
TASK_ROUTING = {
    "clean_the_desktop": "clean_the_desktop_1500",
}


def _summarize_paths(paths: list[str], *, limit: int = 8) -> str:
    if len(paths) <= limit:
        return ", ".join(paths)
    return ", ".join(paths[:limit]) + f", ... (+{len(paths) - limit} more)"


def _path_to_tuple(path: str) -> tuple[str | int, ...]:
    # Adapter paths are serialized as '/'-joined strings; convert back to tuple
    # keys and preserve integer indices where applicable.
    out: list[str | int] = []
    for token in path.split("/"):
        if token.isdigit():
            out.append(int(token))
        else:
            out.append(token)
    return tuple(out)


def _normalize_task_name(task_name: Any) -> str | None:
    if isinstance(task_name, np.ndarray):
        if task_name.shape != ():
            return None
        task_name = task_name.item()
    if isinstance(task_name, bytes):
        task_name = task_name.decode()
    return task_name if isinstance(task_name, str) else None


def _build_policy_transforms(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    *,
    repack_transforms: _transforms.Group | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
) -> tuple[list[_transforms.DataTransformFn], list[_transforms.DataTransformFn]]:
    repack_transforms = repack_transforms or _transforms.Group()
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return (
        [
            *repack_transforms.inputs,
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )


def _load_adapter_file(adapter_path: pathlib.Path) -> dict[tuple[str | int, ...], np.ndarray]:
    def _to_infer_dtype(value: np.ndarray) -> jax.Array:
        arr = jnp.asarray(value)
        if jnp.issubdtype(arr.dtype, jnp.floating):
            return arr.astype(jnp.bfloat16)
        return arr

    with np.load(adapter_path, allow_pickle=False) as adapter_data:
        if PATHS_KEY in adapter_data.files:
            paths = adapter_data[PATHS_KEY].tolist()
            return {
                _path_to_tuple(path): _to_infer_dtype(np.asarray(adapter_data[VALUE_KEY_TEMPLATE.format(index=index)]))
                for index, path in enumerate(paths)
            }

        return {_path_to_tuple(path): _to_infer_dtype(np.asarray(adapter_data[path])) for path in adapter_data.files}


def create_adapter_routed_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    adapter_dir: pathlib.Path | str,
    *,
    repack_transforms: _transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
) -> "AdapterRoutedPolicy":
    checkpoint_dir = _download.maybe_download(str(checkpoint_dir))
    adapter_dir = _download.maybe_download(str(adapter_dir))

    # Build expected parameter tree from the configured model, then load checkpoint
    # with zero-initialized missing tensors (especially LoRA) so a baseline checkpoint
    # can serve as the routed base while adapters add task-specific deltas.
    def _init_params(rng: jax.Array):
        model = train_config.model.create(rng)
        return nnx.state(model).to_pure_dict()

    params_shape = jax.eval_shape(_init_params, jax.random.PRNGKey(0))
    base_params = weight_loaders.ACOTCheckpointWeightLoader(
        str(checkpoint_dir / "params"),
        missing_init="zeros",
    ).load(params_shape)
    # Match the fast checkpoint policy path: keep inference params in bfloat16 on JAX arrays.
    base_params = jax.tree.map(
        lambda x: (jnp.asarray(x).astype(jnp.bfloat16) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating) else jnp.asarray(x)),
        base_params,
    )

    transforms, output_transforms = _build_policy_transforms(
        train_config,
        checkpoint_dir,
        repack_transforms=repack_transforms,
        default_prompt=default_prompt,
        norm_stats=norm_stats,
    )
    return AdapterRoutedPolicy(
        train_config.model,
        base_params=base_params,
        adapter_dir=adapter_dir,
        transforms=transforms,
        output_transforms=output_transforms,
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )


class AdapterRoutedPolicy(_policy.Policy):
    def __init__(
        self,
        model_config: _model.BaseModelConfig,
        *,
        base_params: at.Params,
        adapter_dir: pathlib.Path,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._model_config = model_config
        self._adapter_dir = pathlib.Path(adapter_dir)
        base_model = self._model_config.load(base_params)
        self._base_graphdef, self._base_state = nnx.split(base_model)
        self._adapters = self._load_adapters(self._adapter_dir)
        self._state_cache: dict[str, nnx.State] = {}
        self._sampler_cache: dict[str, Any] = {}
        self._current_adapter_name: str | None = None
        self._sample_actions = nnx_utils.module_jit(base_model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._force_base = os.getenv("ACOT_ROUTE_FORCE_BASE", "").lower() in {"1", "true", "yes", "on"}

        if self._force_base:
            logger.warning("ACOT_ROUTE_FORCE_BASE is enabled; all tasks will use _base.")

        logger.info("Loaded adapters from %s: %s", self._adapter_dir, sorted(self._adapters.keys()))

        initial_adapter = "_base"
        self._activate_adapter(initial_adapter)

    def _load_adapters(self, adapter_dir: pathlib.Path) -> dict[str, dict[tuple[str | int, ...], np.ndarray]]:
        adapters = {}
        for adapter_path in sorted(adapter_dir.glob("*.npz")):
            adapters[adapter_path.stem] = _load_adapter_file(adapter_path)
        return adapters

    def _resolve_adapter_name(self, task_name: str | None) -> str:
        if self._force_base:
            return "_base"

        if task_name is None:
            return "_base"
        requested = TASK_ROUTING.get(task_name)
        if requested is None:
            return "_base"
        if requested in self._adapters:
            return requested

        fallback = "_base"
        logger.warning("Task '%s' routed to missing adapter '%s'; falling back to %s", task_name, requested, fallback)
        return fallback

    def _build_state(self, adapter_name: str) -> nnx.State:
        cached_state = self._state_cache.get(adapter_name)
        if cached_state is not None:
            return cached_state

        # Always flatten the concrete state instance we are about to update.
        # This avoids stale key-set assumptions if the internal state graph differs
        # from any cached flatten mapping.
        state = copy.deepcopy(self._base_state)
        state_flat = flax.traverse_util.flatten_dict(state.to_pure_dict())
        merged_params = dict(state_flat)
        adapter_params = self._adapters.get(adapter_name, {})
        if adapter_params:
            known_adapter_params = {}
            unknown_paths = []
            for key, value in adapter_params.items():
                if key in state_flat:
                    known_adapter_params[key] = value
                else:
                    unknown_paths.append("/".join(map(str, key)))

            if unknown_paths:
                unknown_paths.sort()
                logger.warning(
                    "Adapter '%s' has %d parameter(s) not present in base model; ignoring them. Sample: %s",
                    adapter_name,
                    len(unknown_paths),
                    _summarize_paths(unknown_paths),
                )

            merged_params.update(copy.deepcopy(known_adapter_params))

        state.replace_by_pure_dict(flax.traverse_util.unflatten_dict(merged_params))
        self._state_cache[adapter_name] = state
        return state

    def _activate_adapter(self, adapter_name: str) -> None:
        if adapter_name == self._current_adapter_name:
            return

        logger.info("Activating adapter: %s", adapter_name)
        cached_sampler = self._sampler_cache.get(adapter_name)
        if cached_sampler is None:
            state = self._build_state(adapter_name)
            module = nnx.merge(self._base_graphdef, state)
            cached_sampler = nnx_utils.module_jit(module.sample_actions)
            self._sampler_cache[adapter_name] = cached_sampler
        self._sample_actions = cached_sampler
        self._current_adapter_name = adapter_name

    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        raw_task_name = jax.tree.map(lambda x: x, obs).get("task_name", None)
        task_name = _normalize_task_name(raw_task_name)
        adapter_name = self._resolve_adapter_name(task_name)
        logger.info("Route decision raw_task_name=%r normalized_task_name=%r adapter=%s", raw_task_name, task_name, adapter_name)
        self._activate_adapter(adapter_name)

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {"state": inputs["state"]}
        result = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs)

        if isinstance(result, dict):
            outputs.update(result)
        else:
            outputs["actions"] = result

        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        outputs["adapter_name"] = self._current_adapter_name
        return self.post_process(obs, outputs)
