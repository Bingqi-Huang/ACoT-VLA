from __future__ import annotations

import logging
import os
import pathlib

import tyro

import openpi.training.r2a_frame_cache as r2a_frame_cache


def main(
    cache_root: str,
    data_root: str | None = None,
    shard_size: int = 2048,
    num_workers: int = 0,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    resolved_data_root = data_root or os.path.expanduser(
        os.getenv("ACOT_CHALLENGE_DATA_ROOT", "~/Datasets/lerobot/Reasoning2Action-Sim")
    )
    manifest = r2a_frame_cache.build_reasoning2action_frame_cache(
        cache_root=pathlib.Path(cache_root).expanduser(),
        data_root=pathlib.Path(resolved_data_root).expanduser(),
        shard_size=shard_size,
        num_workers=num_workers,
    )
    print(
        f"Built Reasoning2Action frame cache with {manifest.sample_count} samples "
        f"across {len(manifest.shard_sizes)} shards at {cache_root}"
    )


if __name__ == "__main__":
    tyro.cli(main)
