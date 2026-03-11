from __future__ import annotations

import pathlib

import numpy as np

import openpi.training.config as _config
import openpi.training.r2a_frame_cache as r2a_frame_cache


def _write_fake_cache(cache_root: pathlib.Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    r2a_frame_cache.index_dir(cache_root).mkdir(parents=True, exist_ok=True)
    r2a_frame_cache.shards_dir(cache_root).mkdir(parents=True, exist_ok=True)

    top = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    left = np.ones((3, 224, 224, 3), dtype=np.uint8)
    right = np.full((3, 224, 224, 3), 2, dtype=np.uint8)
    state = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    action = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)

    np.save(r2a_frame_cache.shard_field_path(cache_root, 0, "observation.images.top_head"), top)
    np.save(r2a_frame_cache.shard_field_path(cache_root, 0, "observation.images.hand_left"), left)
    np.save(r2a_frame_cache.shard_field_path(cache_root, 0, "observation.images.hand_right"), right)
    np.save(r2a_frame_cache.shard_field_path(cache_root, 0, "observation.state"), state)
    np.save(r2a_frame_cache.shard_field_path(cache_root, 0, "action"), action)

    index_arrays = {
        "repo_index": np.asarray([0, 0, 1], dtype=np.int32),
        "episode_index": np.asarray([0, 1, 0], dtype=np.int32),
        "frame_index": np.asarray([0, 1, 2], dtype=np.int32),
        "task_index": np.asarray([0, 1, 0], dtype=np.int32),
        "prompt_index": np.asarray([0, 1, 0], dtype=np.int32),
        "timestamp": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "shard_index": np.asarray([0, 0, 0], dtype=np.int32),
        "shard_offset": np.asarray([0, 1, 2], dtype=np.int32),
        "subtask_valid": np.asarray([True, False, True], dtype=np.bool_),
    }
    for name, array in index_arrays.items():
        np.save(r2a_frame_cache.index_array_path(cache_root, name), array)

    manifest = r2a_frame_cache.R2AFrameCacheManifest(
        version=1,
        dataset_family="Reasoning2Action-Sim",
        data_root_fingerprint="fake",
        max_action_chunk_size=5,
        repo_names=("task_a", "task_b"),
        task_vocab=("Task A", "Task B"),
        prompt_vocab=("Prompt A", "Prompt B"),
        data_fields=(
            r2a_frame_cache.FieldSpec("observation.images.top_head", "uint8", (224, 224, 3)),
            r2a_frame_cache.FieldSpec("observation.images.hand_left", "uint8", (224, 224, 3)),
            r2a_frame_cache.FieldSpec("observation.images.hand_right", "uint8", (224, 224, 3)),
            r2a_frame_cache.FieldSpec("observation.state", "float32", (8,)),
            r2a_frame_cache.FieldSpec("action", "float32", (5, 4)),
        ),
        index_fields=tuple(
            r2a_frame_cache.FieldSpec(name, str(array.dtype), tuple(array.shape[1:]))
            for name, array in index_arrays.items()
        ),
        shard_sizes=(3,),
        sample_count=3,
    )
    r2a_frame_cache.write_manifest(cache_root, manifest)


def test_r2a_frame_cache_dataset_filters_repo_and_metadata(tmp_path: pathlib.Path):
    cache_root = tmp_path / "cache"
    _write_fake_cache(cache_root)

    data_config = _config.DataConfig(repo_id=[str(tmp_path / "task_b")])
    dataset = r2a_frame_cache.R2AFrameCacheDataset(
        cache_root,
        data_config,
        split="all",
        split_base_dir=tmp_path,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["task"] == "Task A"
    assert sample["prompt"] == "Prompt A"
    assert sample["frame_index"] == 2
    assert sample["observation.images.hand_right"].shape == (224, 224, 3)
    np.testing.assert_array_equal(dataset.selected_subtask_indices(), np.asarray([0], dtype=np.int64))
