from __future__ import annotations

import dataclasses
import pathlib
import random
from typing import Any

import numpy as np
import torch
import tyro

from openpi.training import config as _config
from openpi.training import data_loader as legacy_loader
from openpi.training import data_loader_offline as _data_loader_offline
from openpi.training import offline_cache as _offline_cache


@dataclasses.dataclass
class Args:
    base_config_name: str
    cache_root: str
    split: str = "all"
    sample_count: int = 512
    batch_count: int = 32
    batch_size: int | None = None
    seed: int = 0


class _MappedSampleSampler(torch.utils.data.Sampler[int]):
    def __init__(self, valid_indices: np.ndarray, split_info: _offline_cache.SplitInfo, *, shuffle: bool, seed: int):
        self._valid_indices = np.asarray(valid_indices, dtype=np.int64)
        self._base_sampler = _data_loader_offline.ShardAwareSampler(split_info, shuffle=shuffle, seed=seed)

    def __iter__(self):
        return iter([int(self._valid_indices[sample_index]) for sample_index in self._base_sampler])

    def __len__(self) -> int:
        return len(self._base_sampler)


def main(args: Args) -> None:
    train_config = _config.get_config(args.base_config_name)
    cache_root = pathlib.Path(args.cache_root).expanduser().resolve()
    manifest = _offline_cache.load_manifest(cache_root)
    splits = ["train", "val"] if args.split == "all" else [args.split]

    for split in splits:
        print(f"Verifying split `{split}`")
        data_config, source_dataset, valid_indices, _ = _offline_cache.resolve_source_dataset(
            train_config,
            split=split,
            skip_norm_stats=False,
        )
        _offline_cache.validate_manifest_for_config(
            manifest,
            train_config=train_config,
            data_config=data_config,
            split=split,
            split_base_dir=train_config.assets_dirs / "episode_splits",
        )
        cached_dataset = _data_loader_offline.OfflineShardDataset(cache_root, split=split, manifest=manifest)
        _verify_samples(
            source_dataset,
            valid_indices,
            cached_dataset,
            sample_count=min(args.sample_count, len(cached_dataset)),
            seed=args.seed,
        )
        _verify_batches(
            source_dataset,
            valid_indices,
            cached_dataset,
            split_info=manifest.splits[split],
            batch_count=args.batch_count,
            batch_size=args.batch_size or train_config.batch_size,
            seed=args.seed,
        )
        print(f"Split `{split}` passed.")


def _verify_samples(source_dataset, valid_indices: np.ndarray, cached_dataset, *, sample_count: int, seed: int) -> None:
    rng = random.Random(seed)
    sample_ids = list(range(len(cached_dataset)))
    rng.shuffle(sample_ids)
    for sample_id in sample_ids[:sample_count]:
        raw_sample = source_dataset[int(valid_indices[sample_id])]
        cached_sample = cached_dataset[sample_id]
        _assert_equal_tree(raw_sample, cached_sample, path=f"sample[{sample_id}]")


def _verify_batches(
    source_dataset,
    valid_indices: np.ndarray,
    cached_dataset,
    *,
    split_info: _offline_cache.SplitInfo,
    batch_count: int,
    batch_size: int,
    seed: int,
) -> None:
    raw_sampler = _MappedSampleSampler(valid_indices, split_info, shuffle=True, seed=seed)
    cached_sampler = _data_loader_offline.ShardAwareSampler(split_info, shuffle=True, seed=seed)
    raw_loader = torch.utils.data.DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=legacy_loader._collate_fn,
        drop_last=True,
        sampler=raw_sampler,
    )
    cached_loader = torch.utils.data.DataLoader(
        cached_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=legacy_loader._collate_fn,
        drop_last=True,
        sampler=cached_sampler,
    )
    for batch_index, (raw_batch, cached_batch) in enumerate(zip(raw_loader, cached_loader, strict=False)):
        if batch_index >= batch_count:
            break
        _assert_equal_tree(raw_batch, cached_batch, path=f"batch[{batch_index}]")


def _assert_equal_tree(expected: Any, actual: Any, *, path: str) -> None:
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"{path}: key mismatch expected={sorted(expected)} actual={sorted(actual)}")
        for key in expected:
            _assert_equal_tree(expected[key], actual[key], path=f"{path}.{key}")
        return

    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)
    if expected_arr.dtype.kind in {"U", "S", "O"} or actual_arr.dtype.kind in {"U", "S", "O"}:
        if not np.array_equal(expected_arr, actual_arr):
            raise AssertionError(f"{path}: string mismatch")
        return
    if not np.array_equal(expected_arr, actual_arr):
        raise AssertionError(
            f"{path}: value mismatch shape={expected_arr.shape} dtype={expected_arr.dtype}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
