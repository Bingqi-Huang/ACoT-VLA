from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np

from openpi.training import data_loader_fast as _data_loader_fast


class _SequenceDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        return int(index)


class _BatchProcessor:
    def process(self, items: list[int], expected_batch_size: int):
        if len(items) != expected_batch_size:
            return None
        return {"value": np.asarray(items, dtype=np.int32)}


def test_fast_torch_data_loader_num_batches_restarts_dataset() -> None:
    data_loader = _data_loader_fast.FastTorchDataLoader(
        _SequenceDataset(),
        local_batch_size=2,
        batch_processor=_BatchProcessor(),
        shuffle=False,
        num_batches=3,
    )

    batches = [np.asarray(batch["value"]) for batch in data_loader]

    assert len(batches) == 3
    np.testing.assert_array_equal(batches[0], np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(batches[1], np.asarray([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(batches[2], np.asarray([0, 1], dtype=np.int32))


def test_load_cached_subtask_indices_rejects_mismatched_dataset_size(tmp_path: pathlib.Path) -> None:
    train_config = SimpleNamespace(name="unit_test_config", assets_dirs=tmp_path)
    expected_indices = np.asarray([1, 3, 5], dtype=np.int64)

    _data_loader_fast.save_cached_subtask_indices(
        train_config,
        "train",
        expected_indices,
        dataset_size=10,
        source_tag="raw",
    )

    loaded = _data_loader_fast.load_cached_subtask_indices(
        train_config,
        "train",
        source_tag="raw",
        expected_dataset_size=10,
    )
    assert loaded is not None
    np.testing.assert_array_equal(np.asarray(loaded), expected_indices)

    mismatched = _data_loader_fast.load_cached_subtask_indices(
        train_config,
        "train",
        source_tag="raw",
        expected_dataset_size=11,
    )
    assert mismatched is None
