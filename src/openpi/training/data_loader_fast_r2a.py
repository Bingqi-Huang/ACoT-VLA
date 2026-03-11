from __future__ import annotations

import pathlib

import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.r2a_frame_cache as r2a_frame_cache


def create_cache_backed_dataset(
    cache_root: pathlib.Path | str,
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    *,
    split: str,
    split_base_dir: pathlib.Path,
) -> r2a_frame_cache.R2AFrameCacheDataset:
    del model_config
    return r2a_frame_cache.R2AFrameCacheDataset(
        cache_root,
        data_config,
        split=split,
        split_base_dir=split_base_dir,
    )


def build_cached_subtask_indices(dataset: r2a_frame_cache.R2AFrameCacheDataset) -> np.ndarray:
    return dataset.selected_subtask_indices()
