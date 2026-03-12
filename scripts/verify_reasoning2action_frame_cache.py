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


def _assert_array_equal(name: str, left, right, *, check_dtype: bool = True) -> None:
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise AssertionError(f"Mismatch for {name}: left type={type(left).__name__}, right type={type(right).__name__}")
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            raise AssertionError(f"Mismatch for {name}: left_keys={sorted(left_keys)}, right_keys={sorted(right_keys)}")
        for key in sorted(left_keys):
            _assert_array_equal(f"{name}.{key}", left[key], right[key], check_dtype=check_dtype)
        return

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            raise AssertionError(f"Mismatch for {name}: left type={type(left).__name__}, right type={type(right).__name__}")
        if len(left) != len(right):
            raise AssertionError(f"Mismatch for {name}: left_len={len(left)}, right_len={len(right)}")
        for idx, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_array_equal(f"{name}[{idx}]", left_item, right_item, check_dtype=check_dtype)
        return

    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    same_shape = left_arr.shape == right_arr.shape
    same_dtype = left_arr.dtype == right_arr.dtype
    same_values = np.array_equal(left_arr, right_arr)
    if same_shape and same_values and (same_dtype or not check_dtype):
        return

    details = [f"left={left_arr.shape}@{left_arr.dtype}", f"right={right_arr.shape}@{right_arr.dtype}"]
    if left_arr.size <= 4 and right_arr.size <= 4:
        details.append(f"left_value={left_arr.tolist()}")
        details.append(f"right_value={right_arr.tolist()}")
    raise AssertionError(f"Mismatch for {name}: {', '.join(details)}")


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
        _assert_array_equal(
            f"{index}:episode_index",
            raw_sample["episode_index"],
            cache_sample["episode_index"],
            check_dtype=False,
        )
        _assert_array_equal(
            f"{index}:frame_index",
            raw_sample["frame_index"],
            cache_sample["frame_index"],
            check_dtype=False,
        )

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
        _assert_array_equal(
            f"batch:{key}",
            raw_batch[key],
            cache_batch[key],
            check_dtype=key not in {"episode_index", "frame_index"},
        )

    print(
        f"Verified Reasoning2Action frame cache for {config_name} split={split} "
        f"on {len(chosen_indices)} samples and {batch_size} processed batch items."
    )


if __name__ == "__main__":
    tyro.cli(main)
