import json
import pathlib

import openpi.training.config as _config
import openpi.training.data_loader_fast as _data_loader_fast
import tyro


def main(
    config_name: str,
    *,
    split: str = "train",
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if not data_config.dataloader_sampler:
        raise ValueError(f"Config `{config_name}` does not use a sampler cacheable by this script.")

    dataset = _data_loader_fast.legacy_loader.create_torch_dataset(
        data_config,
        config.model,
        split=split,
        split_base_dir=config.assets_dirs / "episode_splits",
    )
    dataset = _data_loader_fast.legacy_loader.SafeDataset(dataset)
    indices = _data_loader_fast.build_subtask_indices(
        dataset,
        sampler_type=data_config.dataloader_sampler,
        shuffle=False,
        seed=config.seed,
    )
    cache_path = _data_loader_fast.save_cached_subtask_indices(
        config,
        split,
        indices,
        dataset_size=len(dataset),
    )
    meta_path = _data_loader_fast.subtask_index_metadata_path(config, split)
    metadata = json.loads(meta_path.read_text())
    print(f"Saved {metadata['num_indices']} indices for `{config_name}` split `{split}` to {cache_path}")


if __name__ == "__main__":
    tyro.cli(main)
