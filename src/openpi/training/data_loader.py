from collections.abc import Iterator, Sequence
import multiprocessing
import os
import pathlib
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.episode_split as _episode_split
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class SafeDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: SupportsIndex):
        try:
            return self.dataset[index]
        except Exception as e:
            print(f"[Data Load Error] Skipping index {index} due to: {e}")
            return None
    
    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'dataset'")
        
        return getattr(self.dataset, name)


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def last_metadata(self) -> dict[str, np.ndarray] | None:
        """Get metadata for the last yielded batch, if available."""
        raise NotImplementedError("Subclasses of DataLoader should implement last_metadata.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        if not hasattr(self._dataset, "_datasets"):
            item = self._dataset[index]
            return self._transform(item)
        else:
            idx = index.__index__()
            for d in self._dataset._datasets:
                if idx < len(d):
                    item = d[idx]
                    return self._transform(item)
                idx -= len(d)
            raise IndexError("Index out of range")

    def __len__(self) -> int:
        if not hasattr(self._dataset, "_datasets"):
            length = len(self._dataset)
        else:
            length = 0
            for item in self._dataset._datasets:
                length += len(item)
        return length


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class MultiDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]):
        if not datasets:
            raise ValueError("At least one dataset is required.")
        self._datasets = list(datasets)
        self.disabled_features: set[str] = set()

        first_features = set(getattr(self._datasets[0], "features", []))
        if first_features:
            intersection_features = set(first_features)
            for ds in self._datasets[1:]:
                intersection_features.intersection_update(getattr(ds, "features", []))
            for ds in self._datasets:
                extra_keys = set(getattr(ds, "features", [])).difference(intersection_features)
                self.disabled_features.update(extra_keys)

    def __getitem__(self, index: SupportsIndex):
        idx = index.__index__()
        for dataset in self._datasets:
            if idx < len(dataset):
                item = dataset[idx]
                if self.disabled_features:
                    item = {k: v for k, v in item.items() if k not in self.disabled_features}
                return item
            idx -= len(dataset)
        raise IndexError("Index out of range")

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self._datasets)


class EpisodeSubsetCompatibleLeRobotDataset(Dataset):
    def __init__(
        self,
        dataset: lerobot_dataset.LeRobotDataset,
        *,
        required_camera_keys: set[str] | None = None,
    ):
        self._base_dataset = dataset
        episodes = getattr(dataset, "episodes", None)
        self._episode_id_to_local_idx = None
        if episodes is not None:
            self._episode_id_to_local_idx = {
                int(episode_id): local_idx for local_idx, episode_id in enumerate(episodes)
            }
        self._required_camera_keys = required_camera_keys

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, idx: SupportsIndex) -> dict:
        index = idx.__index__()
        item = self._base_dataset.hf_dataset[index]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self._base_dataset.delta_indices is not None:
            local_ep_idx = int(ep_idx)
            if self._episode_id_to_local_idx is not None:
                local_ep_idx = self._episode_id_to_local_idx.get(int(ep_idx))
                if local_ep_idx is None:
                    raise IndexError(
                        f"Episode index {ep_idx} not found in selected episodes for dataset {self._base_dataset.repo_id}"
                    )
            query_indices, padding = self._base_dataset._get_query_indices(index, local_ep_idx)
            query_result = self._base_dataset._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        video_keys = self._selected_video_keys()
        if video_keys:
            current_ts = item["timestamp"].item()
            query_timestamps = self._base_dataset._get_query_timestamps(current_ts, query_indices)
            query_timestamps = {key: value for key, value in query_timestamps.items() if key in video_keys}
            video_frames = self._base_dataset._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._base_dataset.image_transforms is not None:
            image_keys = self._selected_camera_keys()
            for cam in image_keys:
                if cam in item:
                    item[cam] = self._base_dataset.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self._base_dataset.meta.tasks[task_idx]
        return item

    def _selected_camera_keys(self) -> list[str]:
        camera_keys = list(self._base_dataset.meta.camera_keys)
        if self._required_camera_keys is None:
            return camera_keys
        return [key for key in camera_keys if key in self._required_camera_keys]

    def _selected_video_keys(self) -> list[str]:
        return [key for key in self._base_dataset.meta.video_keys if key in self._selected_camera_keys()]

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        base_dataset = self.__dict__.get("_base_dataset")
        if base_dataset is None:
            raise AttributeError(name)
        return getattr(base_dataset, name)

    def __getstate__(self):
        return {
            "_base_dataset": self._base_dataset,
            "_episode_id_to_local_idx": self._episode_id_to_local_idx,
            "_required_camera_keys": self._required_camera_keys,
        }

    def __setstate__(self, state):
        self._base_dataset = state["_base_dataset"]
        self._episode_id_to_local_idx = state["_episode_id_to_local_idx"]
        self._required_camera_keys = state["_required_camera_keys"]


def _required_camera_keys_from_data_config(data_config: _config.DataConfig) -> set[str] | None:
    required_camera_keys: set[str] = set()
    for transform in data_config.repack_transforms.inputs:
        if not isinstance(transform, _transforms.RepackTransform):
            continue
        for source_key in _transforms.flatten_dict(transform.structure).values():
            if isinstance(source_key, str) and source_key.startswith("observation.images."):
                required_camera_keys.add(source_key)
    return required_camera_keys or None


def _create_dataset_metadata(repo_id: str):
    repo_path = pathlib.Path(repo_id).expanduser()
    if repo_path.is_absolute():
        try:
            return lerobot_dataset.LeRobotDatasetMetadata(repo_path.name, root=repo_path)
        except TypeError:
            return lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    return lerobot_dataset.LeRobotDatasetMetadata(repo_id)


def _create_lerobot_dataset(
    repo_id: str,
    *,
    episodes: list[int] | None,
    delta_timestamps: dict[str, list[float]],
    tolerance_s: float,
):
    repo_path = pathlib.Path(repo_id).expanduser()
    if repo_path.is_absolute():
        return lerobot_dataset.LeRobotDataset(
            repo_path.name,
            root=repo_path,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
        )
    return lerobot_dataset.LeRobotDataset(
        repo_id,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        tolerance_s=tolerance_s,
    )


def _make_selected_episode_compatible_dataset(dataset, required_camera_keys: set[str] | None = None):
    sub_datasets = getattr(dataset, "_datasets", None)
    if sub_datasets is not None:
        dataset._datasets = [
            _make_selected_episode_compatible_dataset(sub_dataset, required_camera_keys) for sub_dataset in sub_datasets
        ]
        return dataset

    if not isinstance(dataset, lerobot_dataset.LeRobotDataset):
        return dataset

    needs_subset_mapping = False
    episodes = getattr(dataset, "episodes", None)
    if episodes is not None:
        selected_episode_ids = [int(ep_id) for ep_id in episodes]
        needs_subset_mapping = selected_episode_ids != list(range(len(selected_episode_ids)))

    needs_camera_filter = False
    if required_camera_keys is not None:
        video_keys = set(getattr(dataset.meta, "video_keys", []))
        needs_camera_filter = bool(video_keys - required_camera_keys)

    if not needs_subset_mapping and not needs_camera_filter:
        return dataset

    return EpisodeSubsetCompatibleLeRobotDataset(dataset, required_camera_keys=required_camera_keys)


def create_torch_dataset(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    *,
    split: str = "all",
    split_base_dir: str | os.PathLike[str] | None = None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    selected_episodes = None
    required_camera_keys = _required_camera_keys_from_data_config(data_config)
    if split != "all" and _episode_split.split_enabled(data_config):
        if split_base_dir is None:
            raise ValueError("split_base_dir is required when using episode-level splits.")
        manifest, manifest_path = _episode_split.get_or_create_manifest(
            data_config,
            base_output_dir=pathlib.Path(split_base_dir),
        )
        _episode_split.report_split(manifest, split=split, manifest_path=manifest_path)
        selected_episodes = _episode_split.episodes_for_split(manifest, split)

    if model_config.model_type == _model.ModelType.ACOT_VLA_PI0 or model_config.model_type == _model.ModelType.ACOT_VLA_PI05:

        acot_action_horizons = jnp.array((model_config.coarse_action_horizon, model_config.action_horizon))
        joint_action_shifts = jnp.array((data_config.joint_action_shifts))
        action_chunk_size = max(acot_action_horizons * joint_action_shifts).item()

    else:
        action_chunk_size = model_config.action_horizon

    if isinstance(repo_id, list):
        # If repo_id is a list, create a dataset for each repo_id and concatenate them.
        dataset_metas = [
            _create_dataset_metadata(r) for r in repo_id
        ]
        dataset = MultiDataset(
            [
                _create_lerobot_dataset(
                    repo_path,
                    episodes=typing.cast(dict[str, list[int]] | None, selected_episodes).get(repo_path)
                    if isinstance(selected_episodes, dict)
                    else None,
                    delta_timestamps={
                        key: [t / dataset_meta.fps for t in range(action_chunk_size)]
                        for key in data_config.action_sequence_keys
                    },
                    tolerance_s=data_config.video_tolerance_s,
                )
                for repo_path, dataset_meta in zip(repo_id, dataset_metas, strict=True)
            ]
        )
        dataset = _make_selected_episode_compatible_dataset(dataset, required_camera_keys)
        if data_config.prompt_from_task:
            for n, d in enumerate(dataset._datasets):
                dataset._datasets[n] = TransformedDataset(
                    d, [_transforms.PromptFromLeRobotTask(dataset_metas[n].tasks)]
                )
        if data_config.prompt_from_hl_instruction:
            for n, d in enumerate(dataset._datasets):
                dataset._datasets[n] = TransformedDataset(
                    d,[_transforms.PromptFromHighlevelInstruction(dataset_metas[n].info['instruction_segments'])]
                )
        for i, d in enumerate(dataset._datasets):
            print(f"Dataset {i} has {len(d)} frames.")

    else:
        dataset_meta = _create_dataset_metadata(repo_id)
        dataset = _create_lerobot_dataset(
            data_config.repo_id,
            episodes=typing.cast(list[int] | None, selected_episodes),
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_chunk_size)]
                for key in data_config.action_sequence_keys
            },
            tolerance_s=data_config.video_tolerance_s,
        )
        dataset = _make_selected_episode_compatible_dataset(dataset, required_camera_keys)

        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
        if data_config.prompt_from_hl_instruction:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromHighlevelInstruction(dataset_meta.info['instruction_segments'])])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    split: str = "train",
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training."""
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            split=split,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        split=split,
        split_base_dir=config.assets_dirs / "episode_splits",
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    split: str = "train",
    split_base_dir: str | os.PathLike[str] | None = None,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(
        data_config,
        model_config,
        split=split,
        split_base_dir=split_base_dir,
    )
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    sampler = None
    if data_config.dataloader_sampler != '':
        from openpi.training.sampler import FrameSampler
        sampler = FrameSampler(dataset, data_config.dataloader_sampler, shuffle=shuffle, seed=seed)
        shuffle = False

    dataset = SafeDataset(dataset)
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        sampler = sampler
    )

    if model_config.model_type == _model.ModelType.ACOT_VLA_PI0 or model_config.model_type == _model.ModelType.ACOT_VLA_PI05:
        return DataLoaderACOTImpl(data_config, data_loader)
    else:
        return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    split: str = "train",
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        split: Dataset split to load. RLDS datasets currently ignore this value.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    del split
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        sampler = None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches
        self._local_batch_size = local_batch_size
        self._last_host_batch = None
        self._partial_batch_warnings = 0

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            sampler=sampler,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    @property
    def last_host_batch(self):
        return self._last_host_batch

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                if batch is None:
                    continue
                actual_batch_size = _get_batch_size(batch)
                if actual_batch_size is None or actual_batch_size != self._local_batch_size:
                    if self._partial_batch_warnings < 5:
                        print(
                            "[Data Load Warning] Skipping incomplete batch with "
                            f"size {actual_batch_size}; expected {self._local_batch_size}."
                        )
                        self._partial_batch_warnings += 1
                    continue
                num_items += 1
                self._last_host_batch = batch
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    filter_items = [x for x in items if x is not None]
    if not filter_items:
        return None
    # return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *filter_items)

    def debug_stack(*args):
        arrays = [np.asarray(x) for x in args]
        try:
            return np.stack(arrays, axis=0)
        except ValueError as e:
            shapes = [x.shape for x in arrays]
            unique_shapes = set(shapes)
            print(f"\n======== DEBUG ERROR ========")
            print(f"Stacking failed!")
            print(f"Found varying shapes: {unique_shapes}")
            print(f"First 5 shapes: {shapes[:5]}")
            print(f"Sample data (first item): {arrays[0]}")
            print(f"=============================\n")
            raise e

    return jax.tree.map(debug_stack, *filter_items)


def _get_batch_size(batch) -> int | None:
    for leaf in jax.tree.leaves(batch):
        shape = getattr(leaf, "shape", None)
        if shape:
            return int(shape[0])
    return None


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader
        self._last_metadata: dict[str, np.ndarray] | None = None

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def last_metadata(self) -> dict[str, np.ndarray] | None:
        if self._last_metadata is None:
            return None
        return {key: np.asarray(value).copy() for key, value in self._last_metadata.items()}

    def __iter__(self):
        for batch in self._data_loader:
            host_batch = getattr(self._data_loader, "last_host_batch", None)
            self._last_metadata = _extract_batch_metadata(host_batch if host_batch is not None else batch)
            yield _model.Observation.from_dict(batch), batch["actions"]

class DataLoaderACOTImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader
        self._last_metadata: dict[str, np.ndarray] | None = None

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def last_metadata(self) -> dict[str, np.ndarray] | None:
        if self._last_metadata is None:
            return None
        return {key: np.asarray(value).copy() for key, value in self._last_metadata.items()}

    def __iter__(self):
        for batch in self._data_loader:
            host_batch = getattr(self._data_loader, "last_host_batch", None)
            self._last_metadata = _extract_batch_metadata(host_batch if host_batch is not None else batch)
            yield _model.Observation.from_dict(batch), batch["actions"], batch["coarse_actions"]


def _extract_batch_metadata(batch) -> dict[str, np.ndarray] | None:
    if batch is None:
        return None

    metadata = {}
    for key in ("task", "episode_index", "frame_index"):
        if isinstance(batch, dict) and key in batch:
            metadata[key] = np.asarray(batch[key])

    return metadata or None
