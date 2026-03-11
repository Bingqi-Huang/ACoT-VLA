from __future__ import annotations

import pathlib
import random

import numpy as np
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as legacy_loader
import openpi.training.data_loader_fast as fast_loader
import openpi.training.data_loader_fast_r2a as r2a_fast_loader
import openpi.training.r2a_frame_cache as r2a_frame_cache


def _assert_array_equal(name: str, left, right) -> None:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape or left_arr.dtype != right_arr.dtype or not np.array_equal(left_arr, right_arr):
        raise AssertionError(
            f"Mismatch for {name}: left={left_arr.shape}@{left_arr.dtype}, right={right_arr.shape}@{right_arr.dtype}"
        )


def main(
    cache_root: str,
    config_name: str,
    split: str = "train",
    num_samples: int = 32,
    seed: int = 0,
):
    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    raw_dataset = legacy_loader.create_torch_dataset(
        data_config,
        train_config.model,
        split=split,
        split_base_dir=train_config.assets_dirs / "episode_splits",
    )
    cache_dataset = r2a_fast_loader.create_cache_backed_dataset(
        pathlib.Path(cache_root).expanduser(),
        data_config,
        train_config.model,
        split=split,
        split_base_dir=train_config.assets_dirs / "episode_splits",
    )

    if len(raw_dataset) != len(cache_dataset):
        raise AssertionError(f"Dataset length mismatch: raw={len(raw_dataset)} cache={len(cache_dataset)}")

    rng = random.Random(seed)
    chosen_indices = sorted(rng.sample(range(len(cache_dataset)), k=min(num_samples, len(cache_dataset))))

    for index in chosen_indices:
        raw_sample = raw_dataset[index]
        cache_sample = cache_dataset[index]
        expected_shard_sample, _ = r2a_frame_cache.source_sample_to_cache_sample(
            raw_sample,
            repo_name="dummy",
        )
        for camera_key in r2a_frame_cache._CAMERA_KEYS:
            _assert_array_equal(f"{index}:{camera_key}", expected_shard_sample[camera_key], cache_sample[camera_key])
        _assert_array_equal(f"{index}:state", expected_shard_sample["observation.state"], cache_sample["observation.state"])
        _assert_array_equal(f"{index}:action", expected_shard_sample["action"], cache_sample["action"])
        if str(raw_sample["prompt"]) != str(cache_sample["prompt"]):
            raise AssertionError(f"Prompt mismatch at index {index}")
        if str(raw_sample["task"]) != str(cache_sample["task"]):
            raise AssertionError(f"Task mismatch at index {index}")
        _assert_array_equal(f"{index}:episode_index", raw_sample["episode_index"], cache_sample["episode_index"])
        _assert_array_equal(f"{index}:frame_index", raw_sample["frame_index"], cache_sample["frame_index"])

    processor = fast_loader.FastBatchProcessor(
        data_config,
        prompt_cache_path_value=None,
        skip_norm_stats=False,
    )
    batch_size = min(4, len(chosen_indices))
    raw_items = [raw_dataset[i] for i in chosen_indices[:batch_size]]
    cache_items = [cache_dataset[i] for i in chosen_indices[:batch_size]]
    np.random.seed(seed)
    raw_batch = processor.process(raw_items, batch_size)
    np.random.seed(seed)
    cache_batch = processor.process(cache_items, batch_size)
    if raw_batch is None or cache_batch is None:
        raise AssertionError("Batch processor unexpectedly returned None during cache verification.")

    for key in raw_batch:
        _assert_array_equal(f"batch:{key}", raw_batch[key], cache_batch[key])

    print(
        f"Verified Reasoning2Action frame cache for {config_name} split={split} "
        f"on {len(chosen_indices)} samples and {batch_size} processed batch items."
    )


if __name__ == "__main__":
    tyro.cli(main)
