from __future__ import annotations

from collections import OrderedDict
import bisect
import multiprocessing
import pathlib
import random
import typing
from typing import Any

import jax
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as legacy_loader
import openpi.training.offline_cache as _offline_cache


class OfflineShardDataset:
    def __init__(
        self,
        cache_root,
        *,
        split: str,
        manifest: _offline_cache.CacheManifest | None = None,
        max_open_shards: int = _offline_cache.DEFAULT_MAX_OPEN_SHARDS,
    ):
        self._cache_root = str(cache_root)
        self._split = split
        self._manifest = manifest or _offline_cache.load_manifest(cache_root)
        if split not in self._manifest.splits:
            raise ValueError(f"Split `{split}` not found in offline cache.")
        self._split_info = self._manifest.splits[split]
        self._shard_starts = [shard.start_index for shard in self._split_info.shards]
        self._max_open_shards = max_open_shards
        self._open_shards: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return self._split_info.num_samples

    def __getitem__(self, index):
        sample_index = index.__index__()
        if sample_index < 0 or sample_index >= len(self):
            raise IndexError(f"Offline cache index out of range: {sample_index}")
        shard_pos = bisect.bisect_right(self._shard_starts, sample_index) - 1
        shard_info = self._split_info.shards[shard_pos]
        local_index = sample_index - shard_info.start_index
        shard_arrays = self._load_shard(shard_info)
        flat_sample = {key: np.asarray(value[local_index]) for key, value in shard_arrays.items()}
        return _offline_cache.reconstruct_cached_sample(flat_sample, self._manifest.task_vocab)

    @property
    def split_info(self) -> _offline_cache.SplitInfo:
        return self._split_info

    def _load_shard(self, shard_info: _offline_cache.ShardInfo) -> dict[str, np.ndarray]:
        shard_arrays = self._open_shards.get(shard_info.index)
        if shard_arrays is not None:
            self._open_shards.move_to_end(shard_info.index)
            return shard_arrays

        shard_dir = pathlib.Path(self._cache_root) / shard_info.relative_dir
        shard_arrays = {
            key: np.load(shard_dir / _offline_cache.field_filename(key), mmap_mode="r")
            for key in self._manifest.field_specs
        }
        self._open_shards[shard_info.index] = shard_arrays
        self._open_shards.move_to_end(shard_info.index)
        while len(self._open_shards) > self._max_open_shards:
            self._open_shards.popitem(last=False)
        return shard_arrays

    def __getstate__(self):
        return {
            "_cache_root": self._cache_root,
            "_split": self._split,
            "_manifest": self._manifest.to_dict(),
            "_max_open_shards": self._max_open_shards,
        }

    def __setstate__(self, state):
        self._cache_root = state["_cache_root"]
        self._split = state["_split"]
        self._manifest = _offline_cache.CacheManifest.from_dict(state["_manifest"])
        self._split_info = self._manifest.splits[self._split]
        self._shard_starts = [shard.start_index for shard in self._split_info.shards]
        self._max_open_shards = state["_max_open_shards"]
        self._open_shards = OrderedDict()


class ShardAwareSampler(torch.utils.data.Sampler[int]):
    def __init__(self, split_info: _offline_cache.SplitInfo, *, shuffle: bool, seed: int):
        self._split_info = split_info
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self._seed + self._epoch)
        self._epoch += 1
        shard_order = list(range(len(self._split_info.shards)))
        if self._shuffle:
            rng.shuffle(shard_order)
        ordered_indices: list[int] = []
        for shard_idx in shard_order:
            shard = self._split_info.shards[shard_idx]
            indices = list(range(shard.start_index, shard.end_index))
            if self._shuffle:
                rng.shuffle(indices)
            ordered_indices.extend(indices)
        return iter(ordered_indices)

    def __len__(self) -> int:
        return self._split_info.num_samples


class OfflineTorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        shuffle: bool = False,
        sampler=None,
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Offline cached data loading with multiple host processes is not supported.")
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
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
            collate_fn=legacy_loader._collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            sampler=sampler,
        )

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
                    break
                if batch is None:
                    continue
                actual_batch_size = legacy_loader._get_batch_size(batch)
                if actual_batch_size is None or actual_batch_size != self._local_batch_size:
                    if self._partial_batch_warnings < 5:
                        print(
                            "[Offline Cache Warning] Skipping incomplete batch with "
                            f"size {actual_batch_size}; expected {self._local_batch_size}."
                        )
                        self._partial_batch_warnings += 1
                    continue
                num_items += 1
                self._last_host_batch = batch
                device_batch = {
                    key: value
                    for key, value in batch.items()
                    if key not in _offline_cache.DEVICE_EXCLUDED_KEYS
                }
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), device_batch)


def create_data_loader(
    config: _config.TrainConfig,
    *,
    cache_root,
    split: str = "train",
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    data_config: _config.DataConfig | None = None,
):
    data_config = data_config or config.data.create(config.assets_dirs, config.model)
    manifest = _offline_cache.load_manifest(cache_root)
    _offline_cache.validate_manifest_for_config(
        manifest,
        train_config=config,
        data_config=data_config,
        split=split,
        split_base_dir=config.assets_dirs / "episode_splits",
    )

    dataset = OfflineShardDataset(cache_root, split=split, manifest=manifest)
    sampler = ShardAwareSampler(dataset.split_info, shuffle=shuffle, seed=config.seed)
    loader = OfflineTorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        shuffle=False,
        sampler=sampler,
    )

    if config.model.model_type in (_model.ModelType.ACOT_VLA_PI0, _model.ModelType.ACOT_VLA_PI05):
        return legacy_loader.DataLoaderACOTImpl(data_config, loader)
    return legacy_loader.DataLoaderImpl(data_config, loader)


def _worker_init_fn(worker_id: int) -> None:
    del worker_id
    legacy_loader._worker_init_fn(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
