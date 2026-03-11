from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as legacy_loader
import openpi.training.episode_split as _episode_split
import openpi.training.sampler as legacy_sampler
import openpi.transforms as _transforms


CACHE_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
DEFAULT_MAX_OPEN_SHARDS = 2

TASK_KEY = "task"
TASK_ID_KEY = "task_id"
SOURCE_GLOBAL_INDEX_KEY = "source_global_index"
METADATA_KEYS = (TASK_KEY, "episode_index", "frame_index", SOURCE_GLOBAL_INDEX_KEY)
DEVICE_EXCLUDED_KEYS = (TASK_KEY, TASK_ID_KEY, "episode_index", "frame_index", SOURCE_GLOBAL_INDEX_KEY)


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    dtype: str
    shape: list[int]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FieldSpec":
        return cls(
            dtype=str(payload["dtype"]),
            shape=[int(value) for value in payload["shape"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


@dataclasses.dataclass(frozen=True)
class ShardInfo:
    index: int
    relative_dir: str
    start_index: int
    num_samples: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShardInfo":
        return cls(
            index=int(payload["index"]),
            relative_dir=str(payload["relative_dir"]),
            start_index=int(payload["start_index"]),
            num_samples=int(payload["num_samples"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def end_index(self) -> int:
        return self.start_index + self.num_samples


@dataclasses.dataclass(frozen=True)
class SplitInfo:
    name: str
    num_samples: int
    shard_size: int
    shards: list[ShardInfo]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitInfo":
        return cls(
            name=str(payload["name"]),
            num_samples=int(payload["num_samples"]),
            shard_size=int(payload["shard_size"]),
            shards=[ShardInfo.from_dict(item) for item in payload["shards"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_samples": self.num_samples,
            "shard_size": self.shard_size,
            "shards": [item.to_dict() for item in self.shards],
        }


@dataclasses.dataclass(frozen=True)
class CacheManifest:
    version: int
    base_config_name: str
    sampler_type: str
    config_fingerprint: str
    model_fingerprint: str
    episode_split_manifest_hash: str
    norm_stats_fingerprint: str
    repo_ids: list[str]
    task_vocab: list[str]
    field_specs: dict[str, FieldSpec]
    splits: dict[str, SplitInfo]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CacheManifest":
        return cls(
            version=int(payload["version"]),
            base_config_name=str(payload["base_config_name"]),
            sampler_type=str(payload["sampler_type"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            model_fingerprint=str(payload["model_fingerprint"]),
            episode_split_manifest_hash=str(payload["episode_split_manifest_hash"]),
            norm_stats_fingerprint=str(payload["norm_stats_fingerprint"]),
            repo_ids=[str(item) for item in payload["repo_ids"]],
            task_vocab=[str(item) for item in payload["task_vocab"]],
            field_specs={key: FieldSpec.from_dict(value) for key, value in payload["field_specs"].items()},
            splits={key: SplitInfo.from_dict(value) for key, value in payload["splits"].items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "base_config_name": self.base_config_name,
            "sampler_type": self.sampler_type,
            "config_fingerprint": self.config_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "episode_split_manifest_hash": self.episode_split_manifest_hash,
            "norm_stats_fingerprint": self.norm_stats_fingerprint,
            "repo_ids": list(self.repo_ids),
            "task_vocab": list(self.task_vocab),
            "field_specs": {key: value.to_dict() for key, value in sorted(self.field_specs.items())},
            "splits": {key: value.to_dict() for key, value in sorted(self.splits.items())},
        }


class MetadataPreservingTransformedDataset:
    def __init__(
        self,
        dataset,
        data_config: _config.DataConfig,
        *,
        skip_norm_stats: bool = False,
    ):
        self._dataset = dataset
        self._data_config = data_config
        norm_stats = None if skip_norm_stats else data_config.norm_stats
        if not skip_norm_stats and data_config.repo_id != "fake" and norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        self._normalize = _transforms.Normalize(norm_stats or {}, use_quantiles=data_config.use_quantile_norm)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        raw_item = self._dataset[index]
        metadata = extract_sample_metadata(raw_item, source_global_index=index.__index__())
        rng_state = np.random.get_state()
        np.random.seed(index.__index__() % (2**32))
        try:
            data = raw_item
            for transform in self._data_config.repack_transforms.inputs:
                data = transform(data)
            for transform in self._data_config.data_transforms.inputs:
                data = transform(data)
            data = self._normalize(data)
            for transform in self._data_config.model_transforms.inputs:
                data = transform(data)
        finally:
            np.random.set_state(rng_state)
        data.update(metadata)
        return data

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._dataset, name)


class FixedOrderSampler:
    def __init__(self, indices: Sequence[int]):
        self._indices = np.asarray(indices, dtype=np.int64)

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices.tolist())

    def __len__(self) -> int:
        return int(self._indices.shape[0])


class CacheShardWriter:
    def __init__(
        self,
        *,
        cache_root: pathlib.Path,
        split: str,
        shard_size: int,
        task_vocab: Sequence[str] | None = None,
        field_specs: Mapping[str, FieldSpec] | None = None,
        starting_shard_index: int = 0,
        starting_sample_index: int = 0,
    ):
        self._cache_root = cache_root
        self._split = split
        self._shard_size = shard_size
        self._task_vocab = list(task_vocab or [])
        self._task_to_id = {task: idx for idx, task in enumerate(self._task_vocab)}
        self._field_specs = dict(field_specs or {})
        self._starting_shard_index = starting_shard_index
        self._current_shard_index = starting_shard_index
        self._next_sample_index = starting_sample_index
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._current_count = 0
        self._shards: list[ShardInfo] = []

    @property
    def task_vocab(self) -> list[str]:
        return list(self._task_vocab)

    @property
    def field_specs(self) -> dict[str, FieldSpec]:
        return dict(self._field_specs)

    def add_batch(self, batch: dict[str, Any]) -> None:
        flat_batch = flatten_batched_sample(batch)
        self._normalize_task_ids(flat_batch)
        self._initialize_field_specs(flat_batch)

        batch_size = batch_size_from_flat_batch(flat_batch)
        offset = 0
        while offset < batch_size:
            take = min(self._shard_size - self._current_count, batch_size - offset)
            for key, value in flat_batch.items():
                self._buffers.setdefault(key, []).append(np.asarray(value[offset : offset + take]))
            self._current_count += take
            offset += take
            if self._current_count == self._shard_size:
                self._flush_current()

    def finalize(self) -> SplitInfo:
        if self._current_count > 0:
            self._flush_current()
        return SplitInfo(
            name=self._split,
            num_samples=self._next_sample_index,
            shard_size=self._shard_size,
            shards=list(self._shards),
        )

    def _normalize_task_ids(self, flat_batch: dict[str, np.ndarray]) -> None:
        task_values = flat_batch.pop(TASK_KEY, None)
        if task_values is None:
            raise ValueError("Expected `task` metadata in cache builder batch.")

        normalized = [normalize_task_name(value) for value in np.asarray(task_values)]
        task_ids = np.empty(len(normalized), dtype=np.int32)
        for idx, task in enumerate(normalized):
            task_ids[idx] = self._task_to_id.setdefault(task, len(self._task_vocab))
            if task_ids[idx] == len(self._task_vocab):
                self._task_vocab.append(task)
        flat_batch[TASK_ID_KEY] = task_ids

    def _initialize_field_specs(self, flat_batch: Mapping[str, np.ndarray]) -> None:
        for key, value in flat_batch.items():
            array = np.asarray(value)
            spec = FieldSpec(dtype=np.dtype(array.dtype).str, shape=[int(dim) for dim in array.shape[1:]])
            existing = self._field_specs.get(key)
            if existing is None:
                self._field_specs[key] = spec
                continue
            if existing != spec:
                raise ValueError(f"Field spec mismatch for {key}: existing={existing}, current={spec}")

    def _flush_current(self) -> None:
        shard_dir = self._cache_root / self._split / f"shard_{self._current_shard_index:06d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for key, chunks in self._buffers.items():
            np.save(shard_dir / field_filename(key), np.concatenate(chunks, axis=0))
        self._shards.append(
            ShardInfo(
                index=self._current_shard_index,
                relative_dir=str(shard_dir.relative_to(self._cache_root)),
                start_index=self._next_sample_index,
                num_samples=self._current_count,
            )
        )
        self._next_sample_index += self._current_count
        self._current_shard_index += 1
        self._buffers = {}
        self._current_count = 0


def manifest_path(cache_root: pathlib.Path | str) -> pathlib.Path:
    return pathlib.Path(cache_root) / MANIFEST_FILENAME


def load_manifest(cache_root: pathlib.Path | str) -> CacheManifest:
    path = manifest_path(cache_root)
    payload = json.loads(path.read_text())
    manifest = CacheManifest.from_dict(payload)
    if manifest.version != CACHE_VERSION:
        raise ValueError(f"Unsupported offline cache version: {manifest.version}")
    return manifest


def save_manifest(cache_root: pathlib.Path | str, manifest: CacheManifest) -> pathlib.Path:
    path = manifest_path(cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def repo_id_basenames(repo_id: str | Sequence[str] | None) -> list[str]:
    if repo_id is None:
        return []
    if isinstance(repo_id, str):
        return [pathlib.Path(repo_id).name]
    return [pathlib.Path(item).name for item in repo_id]


def normalize_task_name(task_name: Any) -> str:
    if isinstance(task_name, bytes):
        return task_name.decode()
    if isinstance(task_name, np.ndarray):
        if task_name.shape == ():
            return normalize_task_name(task_name.item())
        raise ValueError(f"Expected scalar task name, got shape {task_name.shape}")
    return str(task_name)


def extract_sample_metadata(sample: Mapping[str, Any], *, source_global_index: int) -> dict[str, Any]:
    metadata = {SOURCE_GLOBAL_INDEX_KEY: np.int64(source_global_index)}
    for key in (TASK_KEY, "episode_index", "frame_index"):
        if key not in sample:
            raise KeyError(f"Expected metadata key `{key}` in raw sample.")
        metadata[key] = sample[key]
    return metadata


def field_filename(flat_key: str) -> str:
    return flat_key.replace("/", "__") + ".npy"


def flatten_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                next_prefix = f"{prefix}/{key}" if prefix else str(key)
                visit(next_prefix, nested_value)
            return
        output[prefix] = value

    visit("", sample)
    return output


def flatten_batched_sample(batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
    flat = flatten_sample(batch)
    return {key: np.asarray(value) for key, value in flat.items()}


def unflatten_sample(flat_sample: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in flat_sample.items():
        parts = key.split("/")
        target = output
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return output


def reconstruct_cached_sample(flat_sample: Mapping[str, Any], task_vocab: Sequence[str]) -> dict[str, Any]:
    sample = dict(flat_sample)
    task_id = int(np.asarray(sample.pop(TASK_ID_KEY)).item())
    sample[TASK_KEY] = task_vocab[task_id]
    reconstructed = unflatten_sample(sample)
    return reconstructed


def batch_size_from_flat_batch(flat_batch: Mapping[str, np.ndarray]) -> int:
    for value in flat_batch.values():
        array = np.asarray(value)
        if array.ndim < 1:
            continue
        return int(array.shape[0])
    raise ValueError("Unable to infer batch size from flat batch.")


def config_fingerprint(train_config: _config.TrainConfig, data_config: _config.DataConfig) -> str:
    payload = {
        "base_config_name": train_config.name,
        "repo_ids": repo_id_basenames(data_config.repo_id),
        "asset_id": data_config.asset_id,
        "use_quantile_norm": data_config.use_quantile_norm,
        "action_sequence_keys": list(data_config.action_sequence_keys),
        "prompt_from_task": data_config.prompt_from_task,
        "prompt_from_hl_instruction": data_config.prompt_from_hl_instruction,
        "dataloader_sampler": data_config.dataloader_sampler,
        "video_tolerance_s": data_config.video_tolerance_s,
        "repack_transforms": [repr(item) for item in data_config.repack_transforms.inputs],
        "data_transforms": [repr(item) for item in data_config.data_transforms.inputs],
        "model_transforms": [repr(item) for item in data_config.model_transforms.inputs],
    }
    return _sha256_json(payload)


def model_fingerprint(model_config: Any) -> str:
    payload = {
        "model_type": getattr(model_config, "model_type", None).value if getattr(model_config, "model_type", None) else None,
        "repr": repr(model_config),
    }
    return _sha256_json(payload)


def norm_stats_fingerprint(norm_stats: Any) -> str:
    hasher = hashlib.sha256()
    _update_hash_with_value(hasher, norm_stats)
    return hasher.hexdigest()


def split_manifest_hash(data_config: _config.DataConfig, *, base_output_dir: pathlib.Path) -> str:
    if not _episode_split.split_enabled(data_config):
        return ""
    _, split_path = _episode_split.get_or_create_manifest(data_config, base_output_dir=base_output_dir)
    return hashlib.sha256(split_path.read_bytes()).hexdigest()


def create_manifest_for_config(
    *,
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
    episode_split_hash: str,
    task_vocab: Sequence[str],
    field_specs: Mapping[str, FieldSpec],
    splits: Mapping[str, SplitInfo],
) -> CacheManifest:
    return CacheManifest(
        version=CACHE_VERSION,
        base_config_name=train_config.name,
        sampler_type=str(data_config.dataloader_sampler or ""),
        config_fingerprint=config_fingerprint(train_config, data_config),
        model_fingerprint=model_fingerprint(train_config.model),
        episode_split_manifest_hash=episode_split_hash,
        norm_stats_fingerprint=norm_stats_fingerprint(data_config.norm_stats),
        repo_ids=repo_id_basenames(data_config.repo_id),
        task_vocab=list(task_vocab),
        field_specs=dict(field_specs),
        splits=dict(splits),
    )


def resolve_source_dataset(
    train_config: _config.TrainConfig,
    *,
    split: str,
    skip_norm_stats: bool = False,
) -> tuple[_config.DataConfig, MetadataPreservingTransformedDataset, np.ndarray, str]:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    raw_dataset = legacy_loader.create_torch_dataset(
        data_config,
        train_config.model,
        split=split,
        split_base_dir=train_config.assets_dirs / "episode_splits",
    )
    dataset = MetadataPreservingTransformedDataset(
        raw_dataset,
        data_config,
        skip_norm_stats=skip_norm_stats,
    )
    if data_config.dataloader_sampler:
        sampler = legacy_sampler.FrameSampler(dataset, data_config.dataloader_sampler, shuffle=False, seed=train_config.seed)
        indices = np.asarray(sampler.valid_indices, dtype=np.int64)
    else:
        indices = np.arange(len(dataset), dtype=np.int64)
    manifest_hash = split_manifest_hash(data_config, base_output_dir=train_config.assets_dirs / "episode_splits")
    return data_config, dataset, indices, manifest_hash


def validate_manifest_for_config(
    manifest: CacheManifest,
    *,
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
    split: str,
    split_base_dir: pathlib.Path,
) -> None:
    if manifest.base_config_name != train_config.name:
        raise ValueError(
            f"Offline cache config mismatch: cache built for {manifest.base_config_name}, "
            f"requested {train_config.name}."
        )
    current_config_fp = config_fingerprint(train_config, data_config)
    if manifest.config_fingerprint != current_config_fp:
        raise ValueError("Offline cache data/config fingerprint mismatch.")
    current_model_fp = model_fingerprint(train_config.model)
    if manifest.model_fingerprint != current_model_fp:
        raise ValueError("Offline cache model fingerprint mismatch.")
    current_norm_fp = norm_stats_fingerprint(data_config.norm_stats)
    if manifest.norm_stats_fingerprint != current_norm_fp:
        raise ValueError("Offline cache normalization stats fingerprint mismatch.")
    current_repos = repo_id_basenames(data_config.repo_id)
    if manifest.repo_ids != current_repos:
        raise ValueError(f"Offline cache repo mismatch: cache={manifest.repo_ids}, current={current_repos}")
    current_split_hash = split_manifest_hash(data_config, base_output_dir=split_base_dir)
    if manifest.episode_split_manifest_hash != current_split_hash:
        raise ValueError("Offline cache episode split manifest mismatch.")
    if split not in manifest.splits:
        raise ValueError(f"Offline cache split `{split}` not found in cache.")


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _update_hash_with_value(hasher: hashlib._Hash, value: Any) -> None:
    if value is None:
        hasher.update(b"null")
        return
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            hasher.update(field.name.encode())
            _update_hash_with_value(hasher, getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        for key in sorted(value):
            hasher.update(str(key).encode())
            _update_hash_with_value(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(str(len(value)).encode())
        for item in value:
            _update_hash_with_value(hasher, item)
        return
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        hasher.update(np.dtype(array.dtype).str.encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
        return
    hasher.update(repr(value).encode())
