"""Remove bad LeRobot episodes identified in a compute_norm_stats log.

This rewrites each affected task dataset so that:
- bad episodes are dropped
- remaining episodes are reindexed contiguously from 0
- parquet files are rewritten with updated `episode_index` and global `index`
- video filenames are renamed to match the new episode indices
- metadata files under `meta/` are rewritten consistently

Use this only when you intentionally want a destructive in-place cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


LOG_PATTERN = re.compile(
    r"Skipping index (?P<idx>\d+) due to:.*?"
    r"queried timestamps: tensor\(\[(?P<query>[0-9.]+)\]\).*?"
    r"loaded timestamps: tensor\(\[(?P<loaded>[0-9.]+)\]\).*?"
    r"video: (?P<video>[^\n]+)",
    re.S,
)

VIDEO_PATTERN = re.compile(
    r"/Reasoning2Action-Sim/(?P<task>[^/]+)/videos/chunk-\d+/(?P<camera>[^/]+)/episode_(?P<episode>\d+)\.mp4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Path to compute_norm_stats log file.")
    parser.add_argument("dataset_root", type=Path, help="Root directory containing per-task LeRobot datasets.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of tasks to rewrite. Defaults to every task found in the log.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned removals without rewriting any dataset.",
    )
    return parser.parse_args()


def parse_bad_episodes(log_text: str) -> dict[str, set[int]]:
    bad_by_task: dict[str, set[int]] = {}
    for match in LOG_PATTERN.finditer(log_text):
        video = match.group("video").strip()
        video_match = VIDEO_PATTERN.search(video)
        if video_match is None:
            continue
        task = video_match.group("task")
        episode = int(video_match.group("episode"))
        bad_by_task.setdefault(task, set()).add(episode)
    return bad_by_task


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _relative_from_template(path_template: str, expected_prefix: str) -> Path:
    path = Path(path_template)
    parts = path.parts
    if not parts or parts[0] != expected_prefix:
        raise ValueError(f"Expected path template to start with `{expected_prefix}`: {path_template}")
    return Path(*parts[1:])


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Column `{name}` not found in parquet schema.")
    return table.set_column(index, name, values)


def rewrite_dataset(dataset_dir: Path, bad_episodes: set[int], *, dry_run: bool = False) -> dict[str, Any]:
    meta_dir = dataset_dir / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    episode_stats_path = meta_dir / "episodes_stats.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"

    info = _load_json(info_path)
    episodes = _load_jsonl(episodes_path)
    episode_stats = _load_jsonl(episode_stats_path) if episode_stats_path.exists() else []
    tasks = _load_jsonl(tasks_path)

    total_episodes = int(info["total_episodes"])
    all_episode_ids = list(range(total_episodes))
    keep_episode_ids = [episode for episode in all_episode_ids if episode not in bad_episodes]
    if not keep_episode_ids:
        raise ValueError(f"Refusing to remove all episodes from {dataset_dir}")

    old_to_new = {old: new for new, old in enumerate(keep_episode_ids)}
    video_keys = [
        feature_name
        for feature_name, feature_spec in info["features"].items()
        if isinstance(feature_spec, dict) and feature_spec.get("dtype") == "video"
    ]
    chunks_size = int(info["chunks_size"])

    kept_episode_records = [record for record in episodes if int(record["episode_index"]) in old_to_new]
    kept_stats_records = [record for record in episode_stats if int(record["episode_index"]) in old_to_new]
    total_frames = sum(int(record["length"]) for record in kept_episode_records)
    total_videos = len(keep_episode_ids) * len(video_keys)
    total_chunks = ((len(keep_episode_ids) - 1) // chunks_size) + 1

    summary = {
        "dataset_dir": str(dataset_dir),
        "task_name": dataset_dir.name,
        "removed_episodes": sorted(bad_episodes),
        "removed_episode_count": len(bad_episodes),
        "kept_episode_count": len(keep_episode_ids),
        "old_total_episodes": total_episodes,
        "new_total_episodes": len(keep_episode_ids),
        "new_total_frames": total_frames,
        "new_total_videos": total_videos,
    }
    if dry_run:
        return summary

    temp_root = Path(tempfile.mkdtemp(prefix=f"{dataset_dir.name}_rewrite_", dir=str(dataset_dir.parent)))
    try:
        temp_data = temp_root / "data"
        temp_videos = temp_root / "videos"
        temp_meta = temp_root / "meta"
        temp_meta.mkdir(parents=True, exist_ok=True)

        global_frame_index = 0
        for episode_record in kept_episode_records:
            old_episode = int(episode_record["episode_index"])
            new_episode = old_to_new[old_episode]
            old_chunk = old_episode // chunks_size
            new_chunk = new_episode // chunks_size

            src_parquet = dataset_dir / info["data_path"].format(
                episode_chunk=old_chunk,
                episode_index=old_episode,
            )
            dst_parquet = temp_data / Path(
                _relative_from_template(
                    info["data_path"].format(
                        episode_chunk=new_chunk,
                        episode_index=new_episode,
                    ),
                    "data",
                )
            )
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)

            table = pq.read_table(src_parquet)
            num_rows = table.num_rows
            table = _replace_column(table, "episode_index", pa.array([new_episode] * num_rows, type=pa.int64()))
            table = _replace_column(table, "index", pa.array(range(global_frame_index, global_frame_index + num_rows), type=pa.int64()))
            pq.write_table(table, dst_parquet)
            global_frame_index += num_rows

            for video_key in video_keys:
                src_video = dataset_dir / info["video_path"].format(
                    episode_chunk=old_chunk,
                    video_key=video_key,
                    episode_index=old_episode,
                )
                if not src_video.exists():
                    raise FileNotFoundError(f"Missing video file: {src_video}")
                dst_video = temp_videos / Path(
                    _relative_from_template(
                        info["video_path"].format(
                            episode_chunk=new_chunk,
                            video_key=video_key,
                            episode_index=new_episode,
                        ),
                        "videos",
                    )
                )
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_video)

        rewritten_episode_records: list[dict[str, Any]] = []
        for record in kept_episode_records:
            updated = dict(record)
            updated["episode_index"] = old_to_new[int(record["episode_index"])]
            rewritten_episode_records.append(updated)

        rewritten_stats_records: list[dict[str, Any]] = []
        for record in kept_stats_records:
            updated = dict(record)
            updated["episode_index"] = old_to_new[int(record["episode_index"])]
            rewritten_stats_records.append(updated)

        updated_info = dict(info)
        updated_info["total_episodes"] = len(keep_episode_ids)
        updated_info["total_frames"] = total_frames
        updated_info["total_videos"] = total_videos
        updated_info["total_chunks"] = total_chunks
        updated_info["splits"] = {"train": f"0:{len(keep_episode_ids)}"}

        _write_json(temp_meta / "info.json", updated_info)
        _write_jsonl(temp_meta / "episodes.jsonl", rewritten_episode_records)
        if episode_stats_path.exists():
            _write_jsonl(temp_meta / "episodes_stats.jsonl", rewritten_stats_records)
        _write_jsonl(temp_meta / "tasks.jsonl", tasks)

        old_data = dataset_dir / "data"
        old_videos = dataset_dir / "videos"
        old_meta = dataset_dir / "meta"
        backup_root = dataset_dir / ".rewrite_backup"
        if backup_root.exists():
            shutil.rmtree(backup_root)
        backup_root.mkdir(parents=True)

        shutil.move(str(old_data), str(backup_root / "data"))
        shutil.move(str(old_videos), str(backup_root / "videos"))
        shutil.move(str(old_meta), str(backup_root / "meta"))

        shutil.move(str(temp_data), str(dataset_dir / "data"))
        shutil.move(str(temp_videos), str(dataset_dir / "videos"))
        shutil.move(str(temp_meta), str(dataset_dir / "meta"))

        shutil.rmtree(backup_root)
        shutil.rmtree(temp_root, ignore_errors=True)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    return summary


def main() -> int:
    args = parse_args()
    bad_by_task = parse_bad_episodes(args.log_path.read_text())
    if args.tasks is not None:
        requested = set(args.tasks)
        bad_by_task = {task: episodes for task, episodes in bad_by_task.items() if task in requested}

    if not bad_by_task:
        raise SystemExit("No bad episodes parsed from the log with the requested task filter.")

    summaries: list[dict[str, Any]] = []
    for task_name in sorted(bad_by_task):
        dataset_dir = args.dataset_root / task_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found for task `{task_name}`: {dataset_dir}")
        summary = rewrite_dataset(dataset_dir, bad_by_task[task_name], dry_run=args.dry_run)
        summaries.append(summary)
        action = "Would rewrite" if args.dry_run else "Rewrote"
        print(
            f"{action} {task_name}: removed {summary['removed_episode_count']} episodes, "
            f"{summary['old_total_episodes']} -> {summary['new_total_episodes']} episodes, "
            f"new_total_frames={summary['new_total_frames']}"
        )

    print("Done.")
    print("Important: any existing episode split manifests are now stale. Regenerate or delete them before recomputing norm stats/training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
