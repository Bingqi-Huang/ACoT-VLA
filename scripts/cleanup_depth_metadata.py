"""Remove depth feature references from LeRobot metadata.

This is intended for RGB-only ACoT-VLA workflows after depth video files have
been deleted from the underlying dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def _is_depth_feature(feature_name: str, feature_spec: dict[str, Any]) -> bool:
    video_info = feature_spec.get("video_info", {})
    return bool(
        feature_name.endswith("_depth")
        or video_info.get("video.is_depth_map") is True
    )


def cleanup_info_payload(info: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cleaned = dict(info)
    features = cleaned.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("Expected `features` in info.json to be a dictionary.")

    removed_depth_features = [
        feature_name
        for feature_name, feature_spec in features.items()
        if isinstance(feature_spec, dict) and _is_depth_feature(feature_name, feature_spec)
    ]
    cleaned["features"] = {
        feature_name: feature_spec
        for feature_name, feature_spec in features.items()
        if feature_name not in removed_depth_features
    }

    remaining_video_features = sum(
        1
        for feature_spec in cleaned["features"].values()
        if isinstance(feature_spec, dict) and feature_spec.get("dtype") == "video"
    )
    total_episodes = cleaned.get("total_episodes")
    if isinstance(total_episodes, int):
        cleaned["total_videos"] = total_episodes * remaining_video_features

    return cleaned, removed_depth_features


def cleanup_episode_stats_record(
    record: dict[str, Any], removed_depth_features: set[str]
) -> tuple[dict[str, Any], int]:
    cleaned = dict(record)
    stats = cleaned.get("stats", {})
    if not isinstance(stats, dict):
        raise ValueError("Expected `stats` in episodes_stats.jsonl to be a dictionary.")

    removed_count = sum(1 for feature_name in removed_depth_features if feature_name in stats)
    cleaned["stats"] = {
        feature_name: feature_stats
        for feature_name, feature_stats in stats.items()
        if feature_name not in removed_depth_features
    }
    return cleaned, removed_count


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _rewrite_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=4) + "\n")


def _rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(record) + "\n" for record in records)
    _atomic_write_text(path, content)


def cleanup_dataset_dir(dataset_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    meta_dir = dataset_dir / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {info_path}")

    try:
        info_payload = json.loads(info_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {info_path}: {exc}") from exc
    cleaned_info, removed_depth_features = cleanup_info_payload(info_payload)

    episode_stats_path = meta_dir / "episodes_stats.jsonl"
    updated_episode_stats = 0
    episode_stats_records: list[dict[str, Any]] = []
    if episode_stats_path.exists():
        with episode_stats_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cleaned_record, removed_count = cleanup_episode_stats_record(
                    record, set(removed_depth_features)
                )
                updated_episode_stats += removed_count
                episode_stats_records.append(cleaned_record)

    if not dry_run:
        _rewrite_json(info_path, cleaned_info)
        if episode_stats_path.exists():
            # Rewrite line-by-line metadata atomically after filtering depth stats.
            _rewrite_jsonl(episode_stats_path, episode_stats_records)

    return {
        "dataset_dir": str(dataset_dir),
        "removed_depth_features": removed_depth_features,
        "remaining_features": len(cleaned_info.get("features", {})),
        "total_videos": cleaned_info.get("total_videos"),
        "updated_episode_stats": updated_episode_stats,
        "dry_run": dry_run,
    }


def find_dataset_dirs(dataset_root: Path) -> list[Path]:
    dataset_dirs: list[Path] = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "meta" / "info.json").exists():
            dataset_dirs.append(child)
    return dataset_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove depth feature references from LeRobot metadata."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory containing per-task LeRobot dataset folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first malformed dataset folder instead of skipping it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    dataset_dirs = find_dataset_dirs(dataset_root)
    if not dataset_dirs:
        raise SystemExit(f"No dataset folders with meta/info.json found under {dataset_root}")

    total_removed_features = 0
    total_episode_stat_entries = 0
    failures: list[tuple[Path, str]] = []
    for dataset_dir in dataset_dirs:
        try:
            summary = cleanup_dataset_dir(dataset_dir, dry_run=args.dry_run)
        except Exception as exc:
            if args.strict:
                raise
            failures.append((dataset_dir, str(exc)))
            print(f"Skipped {dataset_dir}: {exc}")
            continue
        total_removed_features += len(summary["removed_depth_features"])
        total_episode_stat_entries += summary["updated_episode_stats"]
        removed = ", ".join(summary["removed_depth_features"]) or "none"
        action = "Would update" if args.dry_run else "Updated"
        print(
            f"{action} {summary['dataset_dir']}: "
            f"removed [{removed}], total_videos={summary['total_videos']}, "
            f"episode_stats_entries_removed={summary['updated_episode_stats']}"
        )

    mode = "Dry run complete" if args.dry_run else "Cleanup complete"
    print(
        f"{mode}: processed {len(dataset_dirs)} dataset folders, "
        f"removed {total_removed_features} depth feature declarations, "
        f"removed {total_episode_stat_entries} depth stats entries."
    )
    if failures:
        print(f"Skipped {len(failures)} dataset folders due to malformed metadata.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
