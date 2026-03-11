from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import re
from typing import Any

import numpy as np
import torch
from openpi_client import image_tools as client_image_tools

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as legacy_loader
import openpi.training.episode_split as _episode_split
import openpi.training.sampler as legacy_sampler
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_CACHE_FAMILY = "Reasoning2Action-Sim"
_CAMERA_KEYS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)
_INDEX_FIELDS = (
    "repo_index",
    "episode_index",
    "frame_index",
    "task_index",
    "prompt_index",
    "timestamp",
    "shard_index",
    "shard_offset",
    "subtask_valid",
)


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FieldSpec":
        return cls(
            name=str(payload["name"]),
            dtype=str(payload["dtype"]),
            shape=tuple(int(x) for x in payload["shape"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclasses.dataclass(frozen=True)
class R2AFrameCacheManifest:
    version: int
    dataset_family: str
    data_root_fingerprint: str
    max_action_chunk_size: int
    repo_names: tuple[str, ...]
    task_vocab: tuple[str, ...]
    prompt_vocab: tuple[str, ...]
    data_fields: tuple[FieldSpec, ...]
    index_fields: tuple[FieldSpec, ...]
    shard_sizes: tuple[int, ...]
    sample_count: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "R2AFrameCacheManifest":
        return cls(
            version=int(payload["version"]),
            dataset_family=str(payload["dataset_family"]),
            data_root_fingerprint=str(payload["data_root_fingerprint"]),
            max_action_chunk_size=int(payload["max_action_chunk_size"]),
            repo_names=tuple(str(x) for x in payload["repo_names"]),
            task_vocab=tuple(str(x) for x in payload["task_vocab"]),
            prompt_vocab=tuple(str(x) for x in payload["prompt_vocab"]),
            data_fields=tuple(FieldSpec.from_dict(item) for item in payload["data_fields"]),
            index_fields=tuple(FieldSpec.from_dict(item) for item in payload["index_fields"]),
            shard_sizes=tuple(int(x) for x in payload["shard_sizes"]),
            sample_count=int(payload["sample_count"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_family": self.dataset_family,
            "data_root_fingerprint": self.data_root_fingerprint,
            "max_action_chunk_size": self.max_action_chunk_size,
            "repo_names": list(self.repo_names),
            "task_vocab": list(self.task_vocab),
            "prompt_vocab": list(self.prompt_vocab),
            "data_fields": [field.to_dict() for field in self.data_fields],
            "index_fields": [field.to_dict() for field in self.index_fields],
            "shard_sizes": list(self.shard_sizes),
            "sample_count": self.sample_count,
        }


def manifest_path(cache_root: pathlib.Path) -> pathlib.Path:
    return cache_root / "manifest.json"


def index_dir(cache_root: pathlib.Path) -> pathlib.Path:
    return cache_root / "index"


def shards_dir(cache_root: pathlib.Path) -> pathlib.Path:
    return cache_root / "shards"


def index_array_path(cache_root: pathlib.Path, name: str) -> pathlib.Path:
    return index_dir(cache_root) / f"{_sanitize_name(name)}.npy"


def shard_field_path(cache_root: pathlib.Path, shard_index: int, name: str) -> pathlib.Path:
    return shards_dir(cache_root) / f"shard_{shard_index:06d}__{_sanitize_name(name)}.npy"


def supported_reasoning2action_configs(
    data_root: pathlib.Path | str | None = None,
) -> list[_config.TrainConfig]:
    configs: list[_config.TrainConfig] = []
    resolved_data_root = pathlib.Path(
        os.path.expanduser(str(data_root or _config._REASONING2ACTION_DATA_ROOT))
    ).resolve()
    for train_config in _config._CONFIGS_DICT.values():
        if not isinstance(train_config.data, _config.LerobotACOTGo2DataConfig):
            continue
        repo_ids = _episode_split.resolve_repo_ids(getattr(train_config.data, "repo_id", None))
        if not repo_ids:
            continue
        repo_paths = [pathlib.Path(repo_id).expanduser().resolve() for repo_id in repo_ids]
        if not all(repo_path.parent == resolved_data_root for repo_path in repo_paths):
            continue
        configs.append(train_config)
    return configs


def supported_reasoning2action_repo_names(
    data_root: pathlib.Path | str | None = None,
) -> list[str]:
    repo_names: OrderedDict[str, None] = OrderedDict()
    for train_config in supported_reasoning2action_configs(data_root=data_root):
        for repo_id in _episode_split.resolve_repo_ids(getattr(train_config.data, "repo_id", None)):
            repo_names[pathlib.Path(repo_id).name] = None
    return list(repo_names)


def max_reasoning2action_action_chunk_size(
    data_root: pathlib.Path | str | None = None,
) -> int:
    max_chunk = 1
    for train_config in supported_reasoning2action_configs(data_root=data_root):
        max_chunk = max(max_chunk, required_action_chunk_size(train_config))
    return max_chunk


def required_action_chunk_size(train_config: _config.TrainConfig) -> int:
    if train_config.model.model_type in (_model.ModelType.ACOT_VLA_PI0, _model.ModelType.ACOT_VLA_PI05):
        shifts = tuple(int(x) for x in getattr(train_config.data, "joint_action_shifts", (1, 1)))
        coarse = int(getattr(train_config.model, "coarse_action_horizon", 1))
        fine = int(train_config.model.action_horizon)
        return max(coarse * shifts[0], fine * shifts[1])
    return int(train_config.model.action_horizon)


def load_manifest(cache_root: pathlib.Path | str) -> R2AFrameCacheManifest:
    cache_root = pathlib.Path(cache_root)
    payload = json.loads(manifest_path(cache_root).read_text())
    manifest = R2AFrameCacheManifest.from_dict(payload)
    if manifest.version != _CACHE_VERSION:
        raise ValueError(f"Unsupported R2A frame cache version: {manifest.version}")
    if manifest.dataset_family != _CACHE_FAMILY:
        raise ValueError(f"Unexpected cache family: {manifest.dataset_family}")
    return manifest


def write_manifest(cache_root: pathlib.Path | str, manifest: R2AFrameCacheManifest) -> pathlib.Path:
    cache_root = pathlib.Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = manifest_path(cache_root)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_index_array(cache_root: pathlib.Path | str, name: str, *, mmap: bool = True) -> np.ndarray:
    mode = "r" if mmap else None
    return np.load(index_array_path(pathlib.Path(cache_root), name), mmap_mode=mode, allow_pickle=False)


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "__", name).strip("_")


def _ensure_uint8_hwc(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "cpu"):
        image = image.cpu().numpy()
    else:
        image = np.asarray(image)

    if np.issubdtype(image.dtype, np.floating):
        image = (255.0 * image).round().clip(0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def source_sample_to_cache_sample(
    sample: dict[str, Any],
    *,
    repo_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    prompt = _normalize_string(sample["prompt"])
    task = _normalize_string(sample["task"])

    shard_sample = {
        camera_key: client_image_tools.resize_with_pad(_ensure_uint8_hwc(sample[camera_key]), 224, 224)
        for camera_key in _CAMERA_KEYS
    }
    shard_sample["observation.state"] = np.asarray(sample["observation.state"], dtype=np.float32)
    shard_sample["action"] = np.asarray(sample["action"], dtype=np.float32)

    metadata = {
        "repo_name": np.asarray(repo_name),
        "episode_index": np.int32(int(sample["episode_index"])),
        "frame_index": np.int32(int(sample["frame_index"])),
        "task": np.asarray(task),
        "prompt": np.asarray(prompt),
        "timestamp": np.float32(sample["timestamp"]),
    }
    return shard_sample, metadata


def create_reasoning2action_source_dataset(
    repo_id: str,
    *,
    action_chunk_size: int,
    video_tolerance_s: float,
) -> legacy_loader.Dataset:
    metadata = legacy_loader._create_dataset_metadata(repo_id)
    dataset = legacy_loader._create_lerobot_dataset(
        repo_id,
        episodes=None,
        delta_timestamps={"action": [t / metadata.fps for t in range(action_chunk_size)]},
        tolerance_s=video_tolerance_s,
    )
    dataset = legacy_loader._make_selected_episode_compatible_dataset(
        dataset, required_camera_keys=set(_CAMERA_KEYS)
    )
    dataset = legacy_loader.TransformedDataset(
        dataset,
        [_transforms.PromptFromHighlevelInstruction(metadata.info["instruction_segments"])],
    )
    return dataset


class _BuildDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: legacy_loader.Dataset,
        *,
        repo_name: str,
    ):
        self._dataset = dataset
        self._repo_name = repo_name

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        sample = self._dataset[index]
        return source_sample_to_cache_sample(sample, repo_name=self._repo_name)


def _first_item(items):
    return items[0]


class R2AFrameCacheDataset:
    def __init__(
        self,
        cache_root: pathlib.Path | str,
        data_config: _config.DataConfig,
        *,
        split: str,
        split_base_dir: pathlib.Path,
        max_open_shards: int = 8,
    ):
        self._cache_root = pathlib.Path(cache_root)
        self._manifest = load_manifest(self._cache_root)
        self._data_config = data_config
        self._split = split
        self._split_base_dir = pathlib.Path(split_base_dir)
        self._max_open_shards = max_open_shards

        self._task_vocab = list(self._manifest.task_vocab)
        self._prompt_vocab = list(self._manifest.prompt_vocab)
        self._repo_names = list(self._manifest.repo_names)
        self._repo_name_to_index = {name: idx for idx, name in enumerate(self._repo_names)}

        self._index_arrays = {name: load_index_array(self._cache_root, name) for name in _INDEX_FIELDS}
        self._selected_indices = self._build_selected_indices()
        self._shard_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return int(self._selected_indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        local_index = int(index)
        sample_index = int(self._selected_indices[local_index])
        shard_index = int(self._index_arrays["shard_index"][sample_index])
        shard_offset = int(self._index_arrays["shard_offset"][sample_index])
        shard_arrays = self._load_shard(shard_index)

        task_index = int(self._index_arrays["task_index"][sample_index])
        prompt_index = int(self._index_arrays["prompt_index"][sample_index])
        sample = {
            "observation.state": np.asarray(shard_arrays["observation.state"][shard_offset]),
            "action": np.asarray(shard_arrays["action"][shard_offset]),
            "timestamp": np.float32(self._index_arrays["timestamp"][sample_index]),
            "episode_index": np.int32(self._index_arrays["episode_index"][sample_index]),
            "frame_index": np.int32(self._index_arrays["frame_index"][sample_index]),
            "task_index": np.int32(task_index),
            "task": self._task_vocab[task_index],
            "prompt": self._prompt_vocab[prompt_index],
        }
        for camera_key in _CAMERA_KEYS:
            sample[camera_key] = np.asarray(shard_arrays[camera_key][shard_offset])
        return sample

    def selected_subtask_indices(self) -> np.ndarray:
        selected_valid = self._index_arrays["subtask_valid"][self._selected_indices]
        return np.flatnonzero(np.asarray(selected_valid, dtype=bool)).astype(np.int64)

    @property
    def selected_indices(self) -> np.ndarray:
        return self._selected_indices

    def _build_selected_indices(self) -> np.ndarray:
        repo_ids = _episode_split.resolve_repo_ids(self._data_config.repo_id)
        repo_array = np.asarray(self._index_arrays["repo_index"], dtype=np.int64)
        episode_array = np.asarray(self._index_arrays["episode_index"], dtype=np.int64)

        split_episode_lookup: dict[str, np.ndarray] | None = None
        if self._split != "all" and _episode_split.split_enabled(self._data_config):
            manifest, _ = _episode_split.get_or_create_manifest(
                self._data_config,
                base_output_dir=self._split_base_dir,
            )
            selected_episodes = _episode_split.episodes_for_split(manifest, self._split)
            if isinstance(selected_episodes, dict):
                split_episode_lookup = {
                    pathlib.Path(repo_id).name: np.asarray(episodes, dtype=np.int64)
                    for repo_id, episodes in selected_episodes.items()
                }
            else:
                split_episode_lookup = {
                    pathlib.Path(repo_ids[0]).name: np.asarray(selected_episodes, dtype=np.int64)
                }

        selected_parts: list[np.ndarray] = []
        for repo_id in repo_ids:
            repo_name = pathlib.Path(repo_id).name
            repo_index = self._repo_name_to_index.get(repo_name)
            if repo_index is None:
                continue
            repo_mask = repo_array == repo_index
            if split_episode_lookup is not None:
                episodes = split_episode_lookup.get(repo_name)
                if episodes is None:
                    continue
                repo_mask &= np.isin(episode_array, episodes)
            selected_parts.append(np.flatnonzero(repo_mask).astype(np.int64))

        if not selected_parts:
            return np.asarray([], dtype=np.int64)
        return np.concatenate(selected_parts, axis=0)

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        shard_arrays = self._shard_cache.get(shard_index)
        if shard_arrays is not None:
            self._shard_cache.move_to_end(shard_index)
            return shard_arrays

        shard_arrays = {
            field.name: np.load(
                shard_field_path(self._cache_root, shard_index, field.name),
                mmap_mode="r",
                allow_pickle=False,
            )
            for field in self._manifest.data_fields
        }
        self._shard_cache[shard_index] = shard_arrays
        while len(self._shard_cache) > self._max_open_shards:
            self._shard_cache.popitem(last=False)
        return shard_arrays


def build_reasoning2action_frame_cache(
    *,
    cache_root: pathlib.Path | str,
    data_root: pathlib.Path | str | None = None,
    shard_size: int = 2048,
    num_workers: int = 0,
) -> R2AFrameCacheManifest:
    cache_root = pathlib.Path(cache_root)
    data_root = pathlib.Path(data_root or os.path.expanduser(_config._REASONING2ACTION_DATA_ROOT)).expanduser().resolve()

    repo_names = supported_reasoning2action_repo_names(data_root=data_root)
    repo_vocab = {repo_name: idx for idx, repo_name in enumerate(repo_names)}
    prompt_vocab: dict[str, int] = {}
    task_vocab: dict[str, int] = {}
    action_chunk_size = max_reasoning2action_action_chunk_size(data_root=data_root)
    video_tolerance_s = max(
        float(getattr(train_config.data.base_config, "video_tolerance_s", 1e-4) or 1e-4)
        for train_config in supported_reasoning2action_configs(data_root=data_root)
    )

    cache_root.mkdir(parents=True, exist_ok=True)
    index_dir(cache_root).mkdir(parents=True, exist_ok=True)
    shards_dir(cache_root).mkdir(parents=True, exist_ok=True)

    shard_sizes: list[int] = []
    shard_rows: list[dict[str, list[np.ndarray]]] = []
    index_rows: dict[str, list[np.ndarray]] = {name: [] for name in _INDEX_FIELDS if name not in ("shard_index", "shard_offset")}
    shard_index_rows: list[np.ndarray] = []
    shard_offset_rows: list[np.ndarray] = []
    data_fields: tuple[FieldSpec, ...] | None = None

    current_shard: dict[str, list[np.ndarray]] = {}
    current_shard_size = 0
    current_shard_index = 0
    total_samples = 0

    def flush_current_shard() -> None:
        nonlocal current_shard, current_shard_size, current_shard_index, data_fields
        if current_shard_size == 0:
            return

        shard_sizes.append(current_shard_size)
        shard_offset_rows.append(np.arange(current_shard_size, dtype=np.int32))
        shard_index_rows.append(np.full(current_shard_size, current_shard_index, dtype=np.int32))

        stacked = {
            name: np.stack(values, axis=0)
            for name, values in current_shard.items()
        }
        if data_fields is None:
            data_fields = tuple(
                FieldSpec(name=name, dtype=str(array.dtype), shape=tuple(array.shape[1:]))
                for name, array in stacked.items()
            )

        for name, array in stacked.items():
            np.save(shard_field_path(cache_root, current_shard_index, name), array, allow_pickle=False)

        current_shard_index += 1
        current_shard = {}
        current_shard_size = 0

    for repo_name in repo_names:
        repo_path = data_root / repo_name
        logger.info("Building cache for repo: %s", repo_path)
        source_dataset = create_reasoning2action_source_dataset(
            str(repo_path),
            action_chunk_size=action_chunk_size,
            video_tolerance_s=video_tolerance_s,
        )
        valid_indices = legacy_sampler.FrameSampler(source_dataset, "subtask", shuffle=False, seed=0).valid_indices
        valid_index_set = set(valid_indices)

        build_dataset = _BuildDataset(
            source_dataset,
            repo_name=repo_name,
        )
        loader = torch.utils.data.DataLoader(
            build_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_first_item,
        )
        for local_index, item in enumerate(loader):
            sample, metadata = item
            if current_shard_size == 0:
                current_shard = {name: [] for name in sample}
            for name, value in sample.items():
                current_shard.setdefault(name, []).append(np.asarray(value))
            prompt = _normalize_string(metadata.pop("prompt"))
            task = _normalize_string(metadata.pop("task"))
            repo_name = _normalize_string(metadata.pop("repo_name"))
            index_rows["prompt_index"].append(np.int32(prompt_vocab.setdefault(prompt, len(prompt_vocab))))
            index_rows["task_index"].append(np.int32(task_vocab.setdefault(task, len(task_vocab))))
            index_rows["repo_index"].append(np.int32(repo_vocab[repo_name]))
            for name, value in metadata.items():
                index_rows[name].append(np.asarray(value))
            index_rows["subtask_valid"].append(np.asarray(local_index in valid_index_set, dtype=np.bool_))
            current_shard_size += 1
            total_samples += 1

            if current_shard_size >= shard_size:
                flush_current_shard()

    flush_current_shard()

    if data_fields is None:
        raise RuntimeError("No Reasoning2Action samples were written to the cache.")

    stacked_indices = {
        name: np.asarray(values)
        for name, values in index_rows.items()
    }
    stacked_indices["shard_index"] = np.concatenate(shard_index_rows, axis=0)
    stacked_indices["shard_offset"] = np.concatenate(shard_offset_rows, axis=0)

    if any(array.shape[0] != total_samples for array in stacked_indices.values()):
        raise RuntimeError("Index array length mismatch while building R2A frame cache.")

    index_fields = tuple(
        FieldSpec(name=name, dtype=str(array.dtype), shape=tuple(array.shape[1:]))
        for name, array in stacked_indices.items()
    )
    for name, array in stacked_indices.items():
        np.save(index_array_path(cache_root, name), array, allow_pickle=False)

    manifest = R2AFrameCacheManifest(
        version=_CACHE_VERSION,
        dataset_family=_CACHE_FAMILY,
        data_root_fingerprint=_fingerprint_data_root(data_root, repo_names),
        max_action_chunk_size=action_chunk_size,
        repo_names=tuple(repo_names),
        task_vocab=tuple(prompt for prompt, _ in sorted(task_vocab.items(), key=lambda item: item[1])),
        prompt_vocab=tuple(prompt for prompt, _ in sorted(prompt_vocab.items(), key=lambda item: item[1])),
        data_fields=data_fields,
        index_fields=index_fields,
        shard_sizes=tuple(shard_sizes),
        sample_count=total_samples,
    )
    write_manifest(cache_root, manifest)
    logger.info(
        "Built Reasoning2Action frame cache at %s with %d samples across %d shards.",
        cache_root,
        total_samples,
        len(shard_sizes),
    )
    return manifest


def _fingerprint_data_root(data_root: pathlib.Path, repo_names: Sequence[str]) -> str:
    payload = {"data_root": str(data_root), "repo_names": list(repo_names)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _normalize_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _normalize_string(value.item())
        raise ValueError(f"Expected scalar string-like array, got {value.shape}")
    return str(value)
