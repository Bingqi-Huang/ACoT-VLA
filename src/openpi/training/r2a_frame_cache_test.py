from __future__ import annotations

import json
import pathlib
import mmap

import numpy as np

import openpi.training.config as _config
import openpi.training.r2a_frame_cache as r2a_frame_cache


def _base_object(value):
    base = value
    while getattr(base, "base", None) is not None:
        base = base.base
    return base


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
        "episode_index": np.asarray([0, 1, 0], dtype=np.int64),
        "frame_index": np.asarray([0, 1, 2], dtype=np.int64),
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


def _write_fake_staged_repo_cache(cache_root: pathlib.Path) -> None:
    repo_name = "task_a"
    (r2a_frame_cache.staged_repo_dir(cache_root, repo_name) / "index").mkdir(parents=True, exist_ok=True)
    (r2a_frame_cache.staged_repo_dir(cache_root, repo_name) / "shards").mkdir(parents=True, exist_ok=True)

    top = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    left = np.ones((2, 224, 224, 3), dtype=np.uint8)
    right = np.full((2, 224, 224, 3), 2, dtype=np.uint8)
    state = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
    action = np.arange(2 * 5 * 4, dtype=np.float32).reshape(2, 5, 4)

    np.save(r2a_frame_cache.staged_repo_shard_path(cache_root, repo_name, 0, "observation.images.top_head"), top)
    np.save(r2a_frame_cache.staged_repo_shard_path(cache_root, repo_name, 0, "observation.images.hand_left"), left)
    np.save(r2a_frame_cache.staged_repo_shard_path(cache_root, repo_name, 0, "observation.images.hand_right"), right)
    np.save(r2a_frame_cache.staged_repo_shard_path(cache_root, repo_name, 0, "observation.state"), state)
    np.save(r2a_frame_cache.staged_repo_shard_path(cache_root, repo_name, 0, "action"), action)

    index_arrays = {
        "episode_index": np.asarray([3, 4], dtype=np.int64),
        "frame_index": np.asarray([11, 12], dtype=np.int64),
        "timestamp": np.asarray([0.1, 0.2], dtype=np.float32),
        "subtask_valid": np.asarray([True, False], dtype=np.bool_),
        "task": np.asarray(["Task A", "Task A"]),
        "prompt": np.asarray(["Prompt A", "Prompt A"]),
        "shard_index": np.asarray([0, 0], dtype=np.int32),
        "shard_offset": np.asarray([0, 1], dtype=np.int32),
    }
    for name, array in index_arrays.items():
        np.save(r2a_frame_cache.staged_repo_index_path(cache_root, repo_name, name), array)

    manifest = r2a_frame_cache.RepoStageManifest(
        repo_name=repo_name,
        sample_count=2,
        shard_sizes=(2,),
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
        action_chunk_size=5,
        video_tolerance_s=0.15,
        data_root_fingerprint="fake",
        complete=True,
    )
    r2a_frame_cache.write_repo_stage_manifest(cache_root, repo_name, manifest)


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
    assert np.asarray(sample["episode_index"]).dtype == np.int64
    assert np.asarray(sample["frame_index"]).dtype == np.int64
    assert sample["observation.state"].flags.owndata
    assert not isinstance(_base_object(sample["observation.state"]), mmap.mmap)
    assert sample["observation.images.hand_right"].flags.owndata
    assert not isinstance(_base_object(sample["observation.images.hand_right"]), mmap.mmap)
    assert sample["observation.images.hand_right"].shape == (224, 224, 3)
    np.testing.assert_array_equal(dataset.selected_subtask_indices(), np.asarray([0], dtype=np.int64))


def test_source_sample_to_cache_sample_preserves_index_dtype():
    sample = {
        "observation.images.top_head": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.hand_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.hand_right": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.state": np.zeros((8,), dtype=np.float32),
        "action": np.zeros((5, 4), dtype=np.float32),
        "prompt": "Prompt A",
        "task": "Task A",
        "episode_index": np.int64(7),
        "frame_index": np.int64(9),
        "timestamp": np.float32(0.1),
    }

    _, metadata = r2a_frame_cache.source_sample_to_cache_sample(sample, repo_name="task_a")

    assert np.asarray(metadata["episode_index"]).dtype == np.int64
    assert np.asarray(metadata["frame_index"]).dtype == np.int64


def test_r2a_frame_cache_dataset_builds_split_manifest_from_cache(tmp_path: pathlib.Path):
    cache_root = tmp_path / "cache"
    split_dir = tmp_path / "splits"
    _write_fake_cache(cache_root)

    data_config = _config.DataConfig(
        repo_id=[str(tmp_path / "missing_source_root" / "task_a")],
        episode_split=_config.EpisodeSplitConfig(seed=7, train_ratio=0.5, split_name="cache_only_split"),
    )
    train_dataset = r2a_frame_cache.R2AFrameCacheDataset(
        cache_root,
        data_config,
        split="train",
        split_base_dir=split_dir,
    )
    val_dataset = r2a_frame_cache.R2AFrameCacheDataset(
        cache_root,
        data_config,
        split="val",
        split_base_dir=split_dir,
    )

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert {int(train_dataset[0]["episode_index"]), int(val_dataset[0]["episode_index"])} == {0, 1}

    manifest_path = split_dir / "cache_only_split.json"
    payload = json.loads(manifest_path.read_text())
    assert payload["datasets"][0]["repo_id"] == str(tmp_path / "missing_source_root" / "task_a")
    assert payload["datasets"][0]["total_episodes"] == 2


def test_assemble_final_cache_preserves_index_dtype(tmp_path: pathlib.Path):
    cache_root = tmp_path / "cache"
    _write_fake_staged_repo_cache(cache_root)

    manifest = r2a_frame_cache._assemble_final_cache(
        cache_root=cache_root,
        repo_names=["task_a"],
        repo_vocab={"task_a": 0},
        data_root_fingerprint="fake",
        action_chunk_size=5,
    )

    assert manifest.sample_count == 2
    assert np.load(r2a_frame_cache.index_array_path(cache_root, "episode_index")).dtype == np.int64
    assert np.load(r2a_frame_cache.index_array_path(cache_root, "frame_index")).dtype == np.int64
