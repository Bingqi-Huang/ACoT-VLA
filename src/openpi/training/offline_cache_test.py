from __future__ import annotations

import numpy as np

from openpi.training import data_loader_offline as _data_loader_offline
from openpi.training import offline_cache as _offline_cache


def test_offline_cache_round_trip(tmp_path):
    batch = {
        "image": {
            "base_0_rgb": np.zeros((2, 224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.ones((2, 224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.full((2, 224, 224, 3), 2, dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.array([True, True]),
            "left_wrist_0_rgb": np.array([True, True]),
            "right_wrist_0_rgb": np.array([True, True]),
        },
        "state": np.zeros((2, 32), dtype=np.float32),
        "actions": np.zeros((2, 30, 32), dtype=np.float32),
        "coarse_actions": np.zeros((2, 30, 32), dtype=np.float32),
        "tokenized_prompt": np.zeros((2, 210), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((2, 210), dtype=bool),
        "task": np.array(["task_a", "task_b"]),
        "episode_index": np.array([1, 2], dtype=np.int64),
        "frame_index": np.array([10, 20], dtype=np.int64),
        "source_global_index": np.array([0, 1], dtype=np.int64),
    }

    writer = _offline_cache.CacheShardWriter(cache_root=tmp_path, split="train", shard_size=2)
    writer.add_batch(batch)
    split_info = writer.finalize()
    manifest = _offline_cache.CacheManifest(
        version=_offline_cache.CACHE_VERSION,
        base_config_name="dummy",
        sampler_type="subtask",
        config_fingerprint="cfg",
        model_fingerprint="model",
        episode_split_manifest_hash="split",
        norm_stats_fingerprint="norm",
        repo_ids=["repo"],
        task_vocab=writer.task_vocab,
        field_specs=writer.field_specs,
        splits={"train": split_info},
    )
    _offline_cache.save_manifest(tmp_path, manifest)

    dataset = _data_loader_offline.OfflineShardDataset(tmp_path, split="train")
    sample = dataset[1]

    assert sample["task"] == "task_b"
    assert int(sample["episode_index"]) == 2
    assert int(sample["frame_index"]) == 20
    assert sample["image"]["right_wrist_0_rgb"].shape == (224, 224, 3)
    assert sample["image"]["right_wrist_0_rgb"].dtype == np.uint8
