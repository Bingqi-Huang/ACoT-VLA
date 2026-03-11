from __future__ import annotations

import pathlib

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as legacy_loader
import openpi.training.data_loader_fast_r2a as r2a_fast_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _default_output_dir(config: _config.TrainConfig) -> pathlib.Path:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.asset_id is None:
        raise ValueError("Data config must define an asset_id to save normalization stats.")
    assets_base_dir = config.data.assets.assets_dir or config.assets_dirs
    return pathlib.Path(assets_base_dir) / data_config.asset_id


def create_cache_dataloader(
    config: _config.TrainConfig,
    *,
    cache_root: pathlib.Path,
    split: str,
    max_frames: int | None,
) -> tuple[legacy_loader.DataLoader, int]:
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = r2a_fast_loader.create_cache_backed_dataset(
        cache_root,
        data_config,
        config.model,
        split=split,
        split_base_dir=config.assets_dirs / "episode_splits",
    )
    dataset = legacy_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.ResizeImages(224, 224),
            RemoveStrings(),
        ],
    )
    dataset = legacy_loader.SafeDataset(dataset)
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // config.batch_size
    else:
        num_batches = len(dataset) // config.batch_size

    data_loader = legacy_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    r2a_cache_root: str,
    max_frames: int | None = None,
    output_dir: str | None = None,
    split: str = "train",
):
    config = _config.get_config(config_name)
    data_loader, num_batches = create_cache_dataloader(
        config,
        cache_root=pathlib.Path(r2a_cache_root).expanduser(),
        split=split,
        max_frames=max_frames,
    )

    keys = ["state", "actions", "coarse_actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    sample_ratio = 0.1
    max_batches = max(1, int(num_batches * sample_ratio))

    data_iter = iter(data_loader)
    pbar = tqdm.tqdm(total=max_batches, desc="Computing stats from R2A cache")
    valid_batches = 0
    while valid_batches < max_batches:
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        except Exception as exc:
            print(f"\n[Warning] Skipped a bad batch due to error: {exc}")
            continue

        for key in keys:
            if key not in batch:
                continue
            values = np.asarray(batch[key])
            stats[key].update(values.reshape(-1, values.shape[-1]))

        pbar.update(1)
        valid_batches += 1

    pbar.close()
    if valid_batches == 0:
        raise RuntimeError("No valid batches were processed while computing norm stats from cache.")

    norm_stats = {}
    for key, key_stats in stats.items():
        try:
            norm_stats[key] = key_stats.get_statistics()
        except ValueError as exc:
            raise RuntimeError(
                f"Failed to compute normalization stats for `{key}` after {valid_batches} valid batches."
            ) from exc

    output_path = pathlib.Path(output_dir) if output_dir is not None else _default_output_dir(config)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
