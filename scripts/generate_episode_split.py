import json

import tyro

from openpi.training import config as _config
from openpi.training import episode_split as _episode_split


def main(config_name: str) -> None:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.episode_split is None:
        raise ValueError(f"Config `{config_name}` does not define an episode split.")

    manifest, manifest_path = _episode_split.get_or_create_manifest(
        data_config,
        base_output_dir=config.assets_dirs / "episode_splits",
    )
    _episode_split.report_split(manifest, split="train", manifest_path=manifest_path)
    _episode_split.report_split(manifest, split="val", manifest_path=manifest_path)

    print(f"\nManifest path: {manifest_path}")
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    tyro.cli(main)
