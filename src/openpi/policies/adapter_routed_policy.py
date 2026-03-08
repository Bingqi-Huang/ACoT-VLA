from collections.abc import Sequence
import copy
import logging
import pathlib
import time
from typing import Any

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

logger = logging.getLogger(__name__)

PATHS_KEY = "__paths__"
VALUE_KEY_TEMPLATE = "param_{index:04d}"
TASK_ROUTING = {
    "pour_workpiece": "pour_workpiece",
    "open_door": "open_door",
    "scoop_popcorn": "scoop_popcorn",
    "hold_pot": "hold_pot",
    "place_block_into_box": "place_block_into_box",
    "grab_toy": "place_block_into_box",
    "take_wrong_item_shelf": "take_wrong_item_shelf",
    "stock_and_straighten_shelf": "stock_and_straighten_shelf",
    "sorting_packages": "sorting_packages",
    "sorting_packages_continuous": "sorting_packages",
    "clean_the_desktop": "clean_the_desktop",
}


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


def _load_adapter_file(adapter_path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(adapter_path, allow_pickle=False) as adapter_data:
        if PATHS_KEY in adapter_data.files:
            paths = adapter_data[PATHS_KEY].tolist()
            return {
                path: np.asarray(adapter_data[VALUE_KEY_TEMPLATE.format(index=index)])
                for index, path in enumerate(paths)
            }

        return {path: np.asarray(adapter_data[path]) for path in adapter_data.files}


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
    base_params = _model.restore_params(checkpoint_dir / "params", restore_type=np.ndarray)
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
        self._base_params_flat = flax.traverse_util.flatten_dict(base_params, sep="/")
        self._adapters = self._load_adapters(self._adapter_dir)
        self._current_adapter_name: str | None = None
        self._current_model: _model.BaseModel | None = None
        self._sample_actions = None
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        initial_adapter = "_default" if "_default" in self._adapters else next(iter(self._adapters), "_base")
        self._activate_adapter(initial_adapter)

    def _load_adapters(self, adapter_dir: pathlib.Path) -> dict[str, dict[str, np.ndarray]]:
        adapters = {}
        for adapter_path in sorted(adapter_dir.glob("*.npz")):
            adapters[adapter_path.stem] = _load_adapter_file(adapter_path)
        return adapters

    def _resolve_adapter_name(self, task_name: str | None) -> str:
        if task_name is None:
            return "_default" if "_default" in self._adapters else "_base"
        return TASK_ROUTING.get(task_name, "_default" if "_default" in self._adapters else "_base")

    def _activate_adapter(self, adapter_name: str) -> None:
        if adapter_name == self._current_adapter_name:
            return

        merged_params = dict(self._base_params_flat)
        merged_params.update(copy.deepcopy(self._adapters.get(adapter_name, {})))
        model_params = flax.traverse_util.unflatten_dict(merged_params, sep="/")

        logger.info("Activating adapter: %s", adapter_name)
        self._current_model = self._model_config.load(model_params)
        self._sample_actions = nnx_utils.module_jit(self._current_model.sample_actions)
        self._current_adapter_name = adapter_name

    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        task_name = jax.tree.map(lambda x: x, obs).get("task_name", None)
        self._activate_adapter(self._resolve_adapter_name(task_name))

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {"state": inputs["state"]}
        assert self._sample_actions is not None
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
