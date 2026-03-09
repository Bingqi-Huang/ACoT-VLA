from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
import random
from typing import Any

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import openpi.training.config as _config

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1


@dataclasses.dataclass(frozen=True)
class DatasetSplit:
    repo_id: str
    task_name: str
    total_episodes: int
    total_frames: int
    episode_lengths: dict[str, int]
    train_episode_indices: list[int]
    val_episode_indices: list[int]
    train_num_frames: int
    val_num_frames: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSplit":
        return cls(
            repo_id=payload["repo_id"],
            task_name=payload["task_name"],
            total_episodes=int(payload["total_episodes"]),
            total_frames=int(payload["total_frames"]),
            episode_lengths={str(k): int(v) for k, v in payload["episode_lengths"].items()},
            train_episode_indices=[int(x) for x in payload["train_episode_indices"]],
            val_episode_indices=[int(x) for x in payload["val_episode_indices"]],
            train_num_frames=int(payload["train_num_frames"]),
            val_num_frames=int(payload["val_num_frames"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class EpisodeSplitManifest:
    version: int
    seed: int
    train_ratio: float
    datasets: list[DatasetSplit]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeSplitManifest":
        return cls(
            version=int(payload["version"]),
            seed=int(payload["seed"]),
            train_ratio=float(payload["train_ratio"]),
            datasets=[DatasetSplit.from_dict(item) for item in payload["datasets"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "datasets": [item.to_dict() for item in self.datasets],
        }

    def dataset_by_repo(self) -> dict[str, DatasetSplit]:
        return {dataset.repo_id: dataset for dataset in self.datasets}


def split_enabled(data_config: _config.DataConfig) -> bool:
    return data_config.episode_split is not None and data_config.repo_id not in (None, "fake")


def resolve_repo_ids(repo_id: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if repo_id is None:
        return []
    if isinstance(repo_id, str):
        return [repo_id]
    return list(repo_id)


def get_or_create_manifest(
    data_config: _config.DataConfig,
    *,
    base_output_dir: pathlib.Path,
) -> tuple[EpisodeSplitManifest, pathlib.Path]:
    if data_config.episode_split is None:
        raise ValueError("Episode split is not configured for this dataset.")

    manifest_path = _manifest_path(data_config, base_output_dir=base_output_dir)
    if manifest_path.exists():
        manifest = EpisodeSplitManifest.from_dict(json.loads(manifest_path.read_text()))
        _validate_manifest(data_config, manifest)
        return manifest, manifest_path

    manifest = _build_manifest(data_config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    logger.info("Wrote episode split manifest: %s", manifest_path)
    return manifest, manifest_path


def episodes_for_split(
    manifest: EpisodeSplitManifest,
    split: str,
) -> dict[str, list[int]] | list[int]:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    datasets = manifest.dataset_by_repo()
    if len(datasets) == 1:
        dataset = next(iter(datasets.values()))
        return list(dataset.train_episode_indices if split == "train" else dataset.val_episode_indices)

    return {
        repo_id: list(dataset.train_episode_indices if split == "train" else dataset.val_episode_indices)
        for repo_id, dataset in datasets.items()
    }


def report_split(
    manifest: EpisodeSplitManifest,
    *,
    split: str,
    manifest_path: pathlib.Path | None = None,
) -> None:
    suffix = f" ({manifest_path})" if manifest_path is not None else ""
    print(f"Episode split summary for `{split}`{suffix}")
    total_episodes = 0
    total_frames = 0
    for dataset in manifest.datasets:
        if split == "train":
            num_episodes = len(dataset.train_episode_indices)
            num_frames = dataset.train_num_frames
        elif split == "val":
            num_episodes = len(dataset.val_episode_indices)
            num_frames = dataset.val_num_frames
        else:
            raise ValueError(f"Unsupported split: {split}")
        total_episodes += num_episodes
        total_frames += num_frames
        print(
            f"  {dataset.task_name} | repo={dataset.repo_id} | episodes={num_episodes} | frames={num_frames}"
        )
    print(f"  total | episodes={total_episodes} | frames={total_frames}")


def _build_manifest(data_config: _config.DataConfig) -> EpisodeSplitManifest:
    assert data_config.episode_split is not None
    repo_ids = resolve_repo_ids(data_config.repo_id)
    datasets = [_build_dataset_split(repo_id, data_config.episode_split) for repo_id in repo_ids]
    return EpisodeSplitManifest(
        version=_MANIFEST_VERSION,
        seed=data_config.episode_split.seed,
        train_ratio=data_config.episode_split.train_ratio,
        datasets=datasets,
    )


def _build_dataset_split(repo_id: str, split_config: _config.EpisodeSplitConfig) -> DatasetSplit:
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    episode_lengths = {
        int(ep_idx): int(meta.episodes[ep_idx]["length"])
        for ep_idx in sorted(meta.episodes)
    }
    episode_indices = list(episode_lengths)
    shuffled = list(episode_indices)
    random.Random(_dataset_seed(split_config.seed, repo_id)).shuffle(shuffled)

    train_count = _train_count(len(shuffled), split_config.train_ratio)
    train_episode_indices = sorted(shuffled[:train_count])
    val_episode_indices = sorted(shuffled[train_count:])
    train_num_frames = sum(episode_lengths[ep_idx] for ep_idx in train_episode_indices)
    val_num_frames = sum(episode_lengths[ep_idx] for ep_idx in val_episode_indices)

    return DatasetSplit(
        repo_id=repo_id,
        task_name=pathlib.Path(repo_id).name,
        total_episodes=len(episode_indices),
        total_frames=sum(episode_lengths.values()),
        episode_lengths={str(k): v for k, v in episode_lengths.items()},
        train_episode_indices=train_episode_indices,
        val_episode_indices=val_episode_indices,
        train_num_frames=train_num_frames,
        val_num_frames=val_num_frames,
    )


def _train_count(num_episodes: int, train_ratio: float) -> int:
    if num_episodes < 1:
        raise ValueError("Each dataset must contain at least one episode.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1); got {train_ratio}")
    if num_episodes == 1:
        return 1
    count = int(num_episodes * train_ratio)
    return min(max(count, 1), num_episodes - 1)


def _manifest_path(data_config: _config.DataConfig, *, base_output_dir: pathlib.Path) -> pathlib.Path:
    assert data_config.episode_split is not None
    repo_ids = resolve_repo_ids(data_config.repo_id)
    output_dir = pathlib.Path(data_config.episode_split.output_dir or base_output_dir)
    name = data_config.episode_split.split_name or _default_split_name(
        repo_ids,
        seed=data_config.episode_split.seed,
        train_ratio=data_config.episode_split.train_ratio,
    )
    return output_dir / f"{name}.json"


def _default_split_name(repo_ids: list[str], *, seed: int, train_ratio: float) -> str:
    if len(repo_ids) == 1:
        prefix = pathlib.Path(repo_ids[0]).name
    elif len(repo_ids) <= 3:
        prefix = "__".join(pathlib.Path(repo_id).name for repo_id in repo_ids)
    else:
        prefix = f"{pathlib.Path(repo_ids[0]).name}__plus_{len(repo_ids) - 1}_tasks"
    digest = hashlib.sha256("||".join(repo_ids).encode()).hexdigest()[:8]
    ratio_pct = int(round(train_ratio * 100))
    return f"{prefix}__train{ratio_pct}_seed{seed}_{digest}"


def _dataset_seed(seed: int, repo_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{repo_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _validate_manifest(data_config: _config.DataConfig, manifest: EpisodeSplitManifest) -> None:
    assert data_config.episode_split is not None
    if manifest.version != _MANIFEST_VERSION:
        raise ValueError(f"Unsupported episode split manifest version: {manifest.version}")

    if manifest.seed != data_config.episode_split.seed:
        raise ValueError(
            f"Episode split seed mismatch: manifest={manifest.seed}, config={data_config.episode_split.seed}"
        )

    if abs(manifest.train_ratio - data_config.episode_split.train_ratio) > 1e-9:
        raise ValueError(
            "Episode split ratio mismatch: "
            f"manifest={manifest.train_ratio}, config={data_config.episode_split.train_ratio}"
        )

    expected_repo_ids = resolve_repo_ids(data_config.repo_id)
    actual_repo_ids = [dataset.repo_id for dataset in manifest.datasets]
    if expected_repo_ids != actual_repo_ids:
        raise ValueError(
            f"Episode split repo mismatch: expected {expected_repo_ids}, got {actual_repo_ids}"
        )
