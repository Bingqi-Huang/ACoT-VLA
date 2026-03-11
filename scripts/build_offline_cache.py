from __future__ import annotations

import dataclasses
import math
import pathlib

import torch
import tqdm.auto as tqdm
import tyro

from openpi.training import config as _config
from openpi.training import data_loader as legacy_loader
from openpi.training import offline_cache as _offline_cache


@dataclasses.dataclass
class Args:
    base_config_name: str
    cache_root: str
    split: str = "all"
    shard_size: int = 2048
    num_workers: int = 0


def main(args: Args) -> None:
    train_config = _config.get_config(args.base_config_name)
    cache_root = pathlib.Path(args.cache_root).expanduser().resolve()
    splits = _resolve_splits(args.split)

    existing_manifest = _offline_cache.load_manifest(cache_root) if _offline_cache.manifest_path(cache_root).exists() else None
    split_records = dict(existing_manifest.splits) if existing_manifest is not None else {}
    task_vocab = list(existing_manifest.task_vocab) if existing_manifest is not None else []
    field_specs = dict(existing_manifest.field_specs) if existing_manifest is not None else {}
    episode_split_hash = existing_manifest.episode_split_manifest_hash if existing_manifest is not None else None

    for split in splits:
        if split in split_records:
            raise FileExistsError(f"Offline cache split `{split}` already exists in {cache_root}")

        data_config, dataset, valid_indices, split_hash = _offline_cache.resolve_source_dataset(
            train_config,
            split=split,
            skip_norm_stats=False,
        )
        if episode_split_hash is None:
            episode_split_hash = split_hash
        elif episode_split_hash != split_hash:
            raise ValueError("Episode split hash changed across splits while building offline cache.")

        if existing_manifest is not None:
            _offline_cache.validate_manifest_for_config(
                existing_manifest,
                train_config=train_config,
                data_config=data_config,
                split=next(iter(existing_manifest.splits)),
                split_base_dir=train_config.assets_dirs / "episode_splits",
            )

        batch_size = min(max(train_config.batch_size, 1), 32)
        sampler = _offline_cache.FixedOrderSampler(valid_indices)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            multiprocessing_context=torch.multiprocessing.get_context("spawn") if args.num_workers > 0 else None,
            collate_fn=legacy_loader._collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=False,
            sampler=sampler,
        )

        writer = _offline_cache.CacheShardWriter(
            cache_root=cache_root,
            split=split,
            shard_size=args.shard_size,
            task_vocab=task_vocab,
            field_specs=field_specs,
        )
        total_batches = math.ceil(len(valid_indices) / batch_size)
        progress = tqdm.tqdm(total=total_batches, desc=f"Build offline cache [{split}]", dynamic_ncols=True)
        try:
            for batch in data_loader:
                writer.add_batch(batch)
                progress.update(1)
        finally:
            progress.close()

        split_records[split] = writer.finalize()
        task_vocab = writer.task_vocab
        field_specs = writer.field_specs
        manifest = _offline_cache.create_manifest_for_config(
            train_config=train_config,
            data_config=data_config,
            episode_split_hash=episode_split_hash or "",
            task_vocab=task_vocab,
            field_specs=field_specs,
            splits=split_records,
        )
        _offline_cache.save_manifest(cache_root, manifest)
        print(
            f"Built offline cache split `{split}`: samples={split_records[split].num_samples}, "
            f"shards={len(split_records[split].shards)}"
        )


def _resolve_splits(split: str) -> list[str]:
    if split == "all":
        return ["train", "val"]
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")
    return [split]


def _worker_init_fn(worker_id: int) -> None:
    legacy_loader._worker_init_fn(worker_id)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
