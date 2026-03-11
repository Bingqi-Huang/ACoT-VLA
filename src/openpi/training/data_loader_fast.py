from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
import pathlib
import queue
import random
import threading
from typing import Any

import jax
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as legacy_loader
import openpi.training.sampler as legacy_sampler
import openpi.transforms as _transforms


def fast_cache_dir(train_config: _config.TrainConfig) -> pathlib.Path:
    return train_config.assets_dirs / "fast_cache"


def subtask_index_cache_path(train_config: _config.TrainConfig, split: str) -> pathlib.Path:
    return fast_cache_dir(train_config) / f"{train_config.name}_{split}_subtask_indices.npy"


def subtask_index_metadata_path(train_config: _config.TrainConfig, split: str) -> pathlib.Path:
    return fast_cache_dir(train_config) / f"{train_config.name}_{split}_subtask_indices.meta.json"


def prompt_token_cache_path(train_config: _config.TrainConfig, split: str) -> pathlib.Path:
    return fast_cache_dir(train_config) / f"{train_config.name}_{split}_prompt_tokens.npz"


class CachedIndexSampler(torch.utils.data.Sampler[int]):
    def __init__(self, indices: np.ndarray, *, shuffle: bool, seed: int):
        self._indices = np.asarray(indices, dtype=np.int64)
        self._shuffle = shuffle
        self._seed = seed

    def __iter__(self):
        indices = self._indices.copy()
        if self._shuffle:
            random.Random(self._seed).shuffle(indices)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return int(self._indices.shape[0])


class PromptTokenCache:
    def __init__(self, cached_tokens: dict[str, tuple[np.ndarray, np.ndarray]] | None = None):
        self._cached_tokens = cached_tokens or {}

    @classmethod
    def from_path(cls, path: pathlib.Path | None) -> "PromptTokenCache":
        if path is None or not path.exists():
            return cls()

        data = np.load(path, allow_pickle=True)
        prompts = [str(item) for item in data["prompts"].tolist()]
        tokens = data["tokens"]
        masks = data["masks"]
        cached = {prompt: (tokens[idx], masks[idx]) for idx, prompt in enumerate(prompts)}
        return cls(cached)

    def get(self, prompt: str, tokenizer) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cached_tokens.get(prompt)
        if cached is not None:
            return cached

        tokens, masks = tokenizer.tokenize(prompt)
        self._cached_tokens[prompt] = (tokens, masks)
        return tokens, masks


class FastBatchProcessor:
    def __init__(
        self,
        data_config: _config.DataConfig,
        *,
        prompt_cache_path_value: pathlib.Path | None,
        skip_norm_stats: bool = False,
    ):
        self._data_config = data_config
        self._sample_input_transforms = list(data_config.data_transforms.inputs)
        self._normalize = _transforms.Normalize(
            None if skip_norm_stats else data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
        self._resize = None
        self._inject_default_prompt = None
        self._tokenizer = None
        self._pad = None
        for transform in data_config.model_transforms.inputs:
            if isinstance(transform, _transforms.InjectDefaultPrompt):
                self._inject_default_prompt = transform
            elif isinstance(transform, _transforms.ResizeImages):
                self._resize = transform
            elif isinstance(transform, _transforms.TokenizePrompt):
                self._tokenizer = transform.tokenizer
            elif isinstance(transform, (_transforms.ACOTPadStatesAndActions, _transforms.PadStatesAndActions)):
                self._pad = transform
            else:
                raise ValueError(f"Unsupported fast-path model transform: {type(transform).__name__}")

        self._prompt_cache = PromptTokenCache.from_path(prompt_cache_path_value)

    def process(self, items: list[dict[str, Any] | None], expected_batch_size: int) -> dict[str, Any] | None:
        filtered_items = [item for item in items if item is not None]
        if len(filtered_items) != expected_batch_size:
            return None

        processed_items = []
        for item in filtered_items:
            metadata = {
                key: item[key]
                for key in ("task", "episode_index", "frame_index")
                if key in item
            }
            data = item
            for transform in self._sample_input_transforms:
                data = transform(data)
            if self._inject_default_prompt is not None:
                data = self._inject_default_prompt(data)
            data.update(metadata)
            processed_items.append(data)

        batch = _stack_samples(processed_items)
        batch = self._normalize(batch)
        if self._resize is not None:
            batch = self._resize(batch)
        batch = self._tokenize_prompts(batch)
        if self._pad is not None:
            batch = self._pad(batch)
        return batch

    def _tokenize_prompts(self, batch: dict[str, Any]) -> dict[str, Any]:
        prompt_values = batch.pop("prompt", None)
        if prompt_values is None:
            if self._tokenizer is None:
                return batch
            raise ValueError("Prompt is required for tokenization in fast loader.")

        if self._tokenizer is None:
            return batch

        prompts = [self._normalize_prompt_value(value) for value in prompt_values]
        tokens, masks = zip(*(self._prompt_cache.get(prompt, self._tokenizer) for prompt in prompts), strict=True)
        batch["tokenized_prompt"] = np.stack(tokens, axis=0)
        batch["tokenized_prompt_mask"] = np.stack(masks, axis=0)
        return batch

    @staticmethod
    def _normalize_prompt_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return FastBatchProcessor._normalize_prompt_value(value.item())
            raise ValueError(f"Expected scalar prompt value, got shape {value.shape}")
        return str(value)


class FastTorchDataLoader:
    def __init__(
        self,
        dataset,
        *,
        local_batch_size: int,
        batch_processor: FastBatchProcessor,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        sampler=None,
        seed: int = 0,
        prefetch_batches: int = 2,
    ):
        if jax.process_count() > 1:
            raise NotImplementedError("Fast data loading with multiple host processes is not supported.")
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches
        self._local_batch_size = local_batch_size
        self._batch_processor = batch_processor
        self._prefetch_batches = max(prefetch_batches, 1)
        self._last_host_batch = None

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._torch_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda items: items,
            drop_last=True,
            generator=generator,
            sampler=sampler,
        )

    @property
    def last_host_batch(self):
        return self._last_host_batch

    def __iter__(self):
        stop_event = threading.Event()
        producer_done = object()
        result_queue: queue.Queue[Any] = queue.Queue(maxsize=self._prefetch_batches)

        def producer():
            try:
                while not stop_event.is_set():
                    for raw_items in self._torch_loader:
                        if stop_event.is_set():
                            break
                        batch = self._batch_processor.process(raw_items, self._local_batch_size)
                        if batch is None:
                            continue
                        result_queue.put(batch)
                    if self._num_batches is not None:
                        break
            except Exception as exc:  # pragma: no cover - surfaced in consumer
                result_queue.put(exc)
            finally:
                result_queue.put(producer_done)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        num_items = 0
        try:
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return

                next_item = result_queue.get()
                if next_item is producer_done:
                    return
                if isinstance(next_item, Exception):
                    raise next_item

                num_items += 1
                self._last_host_batch = next_item
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                    next_item,
                )
        finally:
            stop_event.set()
            producer_thread.join(timeout=1.0)


def load_cached_subtask_indices(train_config: _config.TrainConfig, split: str) -> np.ndarray | None:
    cache_path = subtask_index_cache_path(train_config, split)
    metadata_path = subtask_index_metadata_path(train_config, split)
    if not cache_path.exists() or not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("config_name") != train_config.name or metadata.get("split") != split:
        return None
    return np.load(cache_path, mmap_mode="r")


def save_cached_subtask_indices(
    train_config: _config.TrainConfig,
    split: str,
    indices: Sequence[int],
    *,
    dataset_size: int,
) -> pathlib.Path:
    cache_dir = fast_cache_dir(train_config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = subtask_index_cache_path(train_config, split)
    metadata_path = subtask_index_metadata_path(train_config, split)
    np.save(cache_path, np.asarray(indices, dtype=np.int64))
    metadata_path.write_text(
        json.dumps(
            {
                "config_name": train_config.name,
                "split": split,
                "dataset_size": int(dataset_size),
                "num_indices": int(len(indices)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return cache_path


def build_subtask_indices(
    dataset,
    *,
    sampler_type: str,
    shuffle: bool,
    seed: int,
) -> np.ndarray:
    sampler = legacy_sampler.FrameSampler(dataset, sampler_type, shuffle=shuffle, seed=seed)
    return np.asarray(sampler.valid_indices, dtype=np.int64)


def create_data_loader(
    config: _config.TrainConfig,
    *,
    split: str = "train",
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    use_cached_subtask_indices: bool = True,
    prompt_cache_split: str | None = None,
    prefetch_batches: int = 2,
):
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("Fast data loader currently supports only the LeRobot path.")

    raw_dataset = legacy_loader.create_torch_dataset(
        data_config,
        config.model,
        split=split,
        split_base_dir=config.assets_dirs / "episode_splits",
    )
    raw_dataset = legacy_loader.SafeDataset(raw_dataset)

    sampler = None
    if data_config.dataloader_sampler:
        indices = None
        if use_cached_subtask_indices:
            indices = load_cached_subtask_indices(config, split)
        if indices is None:
            indices = build_subtask_indices(
                raw_dataset,
                sampler_type=data_config.dataloader_sampler,
                shuffle=shuffle,
                seed=config.seed,
            )
        sampler = CachedIndexSampler(indices, shuffle=shuffle, seed=config.seed)
        shuffle = False

    prompt_cache_path_value = None
    if prompt_cache_split is not None:
        candidate = prompt_token_cache_path(config, prompt_cache_split)
        if candidate.exists():
            prompt_cache_path_value = candidate

    batch_processor = FastBatchProcessor(
        data_config,
        prompt_cache_path_value=prompt_cache_path_value,
        skip_norm_stats=skip_norm_stats,
    )
    fast_loader = FastTorchDataLoader(
        raw_dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        batch_processor=batch_processor,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        sampler=sampler,
        seed=config.seed,
        prefetch_batches=prefetch_batches,
    )

    if config.model.model_type in (_model.ModelType.ACOT_VLA_PI0, _model.ModelType.ACOT_VLA_PI05):
        return legacy_loader.DataLoaderACOTImpl(data_config, fast_loader)
    return legacy_loader.DataLoaderImpl(data_config, fast_loader)


def _stack_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot stack an empty sample list.")

    sample = samples[0]
    output = {}
    for key in sample:
        values = [item[key] for item in samples]
        output[key] = _stack_value(values)
    return output


def _stack_value(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _stack_value([value[key] for value in values]) for key in first}
    if isinstance(first, np.ndarray):
        return np.stack(values, axis=0)
    if isinstance(first, (np.generic, bool, int, float)):
        return np.asarray(values)
    if isinstance(first, (str, bytes)):
        return np.asarray(values)
    if hasattr(first, "cpu"):
        return np.stack([np.asarray(value.cpu().numpy()) for value in values], axis=0)
    return np.asarray(values)
