"""Detect systematic timestamp mismatches between parquet timestamps and video frame PTS.

This script scans LeRobot episodes and reports whether a task/camera suffers from
fixed one-frame or multi-frame timing offsets that can cause item loading failures.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from lerobot.common.datasets.video_utils import decode_video_frames_torchvision


@dataclass
class EpisodeCameraSummary:
    episode_index: int
    camera_key: str
    total_queries: int
    checked_queries: int
    mismatch_count: int
    mismatch_ratio: float
    max_diff_s: float
    dominant_frame_offsets: list[int]
    mismatch_row_span: tuple[int, int]
    first_query_s: float
    first_loaded_s: float
    last_query_s: float
    last_loaded_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "camera_key": self.camera_key,
            "total_queries": self.total_queries,
            "checked_queries": self.checked_queries,
            "mismatch_count": self.mismatch_count,
            "mismatch_ratio": self.mismatch_ratio,
            "max_diff_s": self.max_diff_s,
            "dominant_frame_offsets": self.dominant_frame_offsets,
            "mismatch_row_span": list(self.mismatch_row_span),
            "first_query_s": self.first_query_s,
            "first_loaded_s": self.first_loaded_s,
            "last_query_s": self.last_query_s,
            "last_loaded_s": self.last_loaded_s,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Single LeRobot task directory.")
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        default=None,
        help="Camera key(s) to scan. Defaults to all video keys in metadata.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit episode indices to scan.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Scan only the first N selected episodes.",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Mismatch threshold in seconds.",
    )
    parser.add_argument(
        "--backend",
        default="pyav",
        choices=["pyav", "video_reader"],
        help="Torchvision video backend used for PTS extraction.",
    )
    parser.add_argument(
        "--check-mode",
        default="seek",
        choices=["seek", "pts"],
        help="`seek` reproduces the actual decode path; `pts` is faster but less faithful.",
    )
    parser.add_argument(
        "--scan-order",
        default="tail",
        choices=["head", "tail"],
        help="Query timestamps from the beginning or end of each episode.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Check every Nth timestamp.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save machine-readable results.",
    )
    return parser.parse_args()


def load_episode_timestamps(data_path: Path) -> np.ndarray:
    table = pq.read_table(data_path, columns=["timestamp"])
    column = table.column("timestamp")
    return np.asarray(column.to_numpy(), dtype=np.float64)


def load_video_pts(video_path: Path, backend: str) -> np.ndarray:
    torchvision.set_video_backend(backend)
    reader = torchvision.io.VideoReader(str(video_path), "video")
    pts: list[float] = []
    for frame in reader:
        pts.append(float(frame["pts"]))
    if backend == "pyav":
        reader.container.close()
    return np.asarray(pts, dtype=np.float64)


def _parse_tensor_values(raw: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", raw)]


def parse_decode_assertion(exc: AssertionError) -> tuple[list[float], list[float]]:
    message = str(exc)
    queried_match = re.search(r"queried timestamps: tensor\(\[([^\]]+)\]\)", message)
    loaded_match = re.search(r"loaded timestamps: tensor\(\[([^\]]+)\]\)", message)
    if queried_match is None or loaded_match is None:
        raise ValueError(f"Unexpected decode assertion format: {message}")
    return _parse_tensor_values(queried_match.group(1)), _parse_tensor_values(loaded_match.group(1))


def nearest_loaded_pts(query_ts: np.ndarray, loaded_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if loaded_ts.size == 0:
        raise ValueError("Loaded video contains no frames.")

    right = np.searchsorted(loaded_ts, query_ts, side="left")
    left = np.clip(right - 1, 0, loaded_ts.size - 1)
    right = np.clip(right, 0, loaded_ts.size - 1)

    left_diff = np.abs(query_ts - loaded_ts[left])
    right_diff = np.abs(query_ts - loaded_ts[right])
    choose_right = right_diff < left_diff
    nearest_idx = np.where(choose_right, right, left)
    return loaded_ts[nearest_idx], nearest_idx


def summarize_episode_camera(
    *,
    episode_index: int,
    camera_key: str,
    query_ts: np.ndarray,
    loaded_ts: np.ndarray,
    fps: float,
    tolerance_s: float,
) -> EpisodeCameraSummary | None:
    nearest_ts, _ = nearest_loaded_pts(query_ts, loaded_ts)
    diffs = np.abs(query_ts - nearest_ts)
    mismatch_mask = diffs > tolerance_s
    if not mismatch_mask.any():
        return None

    mismatch_rows = np.flatnonzero(mismatch_mask)
    mismatch_diffs = diffs[mismatch_mask]
    mismatch_frame_offsets = np.rint(mismatch_diffs * fps).astype(int)
    offset_counter = Counter(int(x) for x in mismatch_frame_offsets)
    dominant_offsets = [offset for offset, _ in offset_counter.most_common(3)]

    first_row = int(mismatch_rows[0])
    last_row = int(mismatch_rows[-1])
    return EpisodeCameraSummary(
        episode_index=episode_index,
        camera_key=camera_key,
        total_queries=int(query_ts.size),
        checked_queries=int(query_ts.size),
        mismatch_count=int(mismatch_mask.sum()),
        mismatch_ratio=float(mismatch_mask.mean()),
        max_diff_s=float(mismatch_diffs.max()),
        dominant_frame_offsets=dominant_offsets,
        mismatch_row_span=(first_row, last_row),
        first_query_s=float(query_ts[first_row]),
        first_loaded_s=float(nearest_ts[first_row]),
        last_query_s=float(query_ts[last_row]),
        last_loaded_s=float(nearest_ts[last_row]),
    )


def summarize_episode_camera_seek(
    *,
    episode_index: int,
    camera_key: str,
    query_ts: np.ndarray,
    video_path: Path,
    fps: float,
    tolerance_s: float,
    backend: str,
    scan_order: str,
    stride: int,
) -> EpisodeCameraSummary | None:
    indices = np.arange(query_ts.size)
    if stride > 1:
        indices = indices[::stride]
    if scan_order == "tail":
        indices = indices[::-1]

    checked = 0
    for row_index in indices:
        checked += 1
        ts = float(query_ts[int(row_index)])
        try:
            decode_video_frames_torchvision(video_path, [ts], tolerance_s, backend=backend)
        except AssertionError as exc:
            queried_ts, loaded_ts = parse_decode_assertion(exc)
            if len(queried_ts) != 1 or len(loaded_ts) != 1:
                raise ValueError(f"Expected one queried/loaded timestamp, got: {exc}") from exc
            diff_s = abs(queried_ts[0] - loaded_ts[0])
            offset_frames = int(round(diff_s * fps))
            return EpisodeCameraSummary(
                episode_index=episode_index,
                camera_key=camera_key,
                total_queries=int(query_ts.size),
                checked_queries=checked,
                mismatch_count=1,
                mismatch_ratio=1.0 / max(checked, 1),
                max_diff_s=diff_s,
                dominant_frame_offsets=[offset_frames],
                mismatch_row_span=(int(row_index), int(row_index)),
                first_query_s=queried_ts[0],
                first_loaded_s=loaded_ts[0],
                last_query_s=queried_ts[0],
                last_loaded_s=loaded_ts[0],
            )
    return None


def decide_recommendation(
    *,
    total_episodes: int,
    affected_episodes: int,
    camera_counter: Counter[str],
    frame_offset_counter: Counter[int],
) -> str:
    if affected_episodes == 0:
        return "No timing mismatches detected; no repair needed."

    affected_ratio = affected_episodes / total_episodes
    dominant_cameras = [camera for camera, _ in camera_counter.most_common(2)]
    dominant_offsets = [offset for offset, _ in frame_offset_counter.most_common(3)]
    small_discrete_offsets = dominant_offsets and set(dominant_offsets).issubset({1, 2})

    if len(camera_counter) == 1 and small_discrete_offsets:
        if affected_ratio <= 0.05:
            return (
                "Low-coverage, single-camera, 1-2 frame mismatch. Prefer excluding the listed episodes "
                "from train/val splits instead of changing global tolerance."
            )
        if affected_ratio <= 0.20:
            return (
                "Moderate-coverage, single-camera, 1-2 frame mismatch. Prefer repairing or re-encoding the "
                f"{dominant_cameras[0]} stream for affected episodes; excluding episodes is the fallback."
            )
        return (
            "High-coverage, single-camera, 1-2 frame mismatch. This looks systematic. Do not edit global "
            "parquet timestamps; repair the affected camera stream or add a task/camera-specific load-time offset."
        )

    return (
        "Mismatches span multiple cameras or have variable offsets. Treat this as a broader synchronization "
        "issue; avoid global tolerance changes unless you confirm the pattern is safe."
    )


def scan_dataset(
    dataset_dir: Path,
    *,
    cameras: list[str] | None,
    episodes: list[int] | None,
    max_episodes: int | None,
    tolerance_s: float,
    backend: str,
    check_mode: str,
    scan_order: str,
    stride: int,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    meta = lerobot_dataset.LeRobotDatasetMetadata(dataset_dir.name, root=dataset_dir)
    selected_cameras = cameras or list(meta.video_keys)
    selected_episodes = episodes or list(range(meta.total_episodes))
    if max_episodes is not None:
        selected_episodes = selected_episodes[:max_episodes]

    per_episode: list[EpisodeCameraSummary] = []
    camera_counter: Counter[str] = Counter()
    frame_offset_counter: Counter[int] = Counter()
    camera_episode_counter: dict[str, set[int]] = defaultdict(set)

    for episode_index in selected_episodes:
        data_path = dataset_dir / meta.get_data_file_path(episode_index)
        query_ts = load_episode_timestamps(data_path)

        for camera_key in selected_cameras:
            video_path = dataset_dir / meta.get_video_file_path(episode_index, camera_key)
            if not video_path.is_file():
                continue
            if check_mode == "pts":
                loaded_ts = load_video_pts(video_path, backend)
                summary = summarize_episode_camera(
                    episode_index=episode_index,
                    camera_key=camera_key,
                    query_ts=query_ts,
                    loaded_ts=loaded_ts,
                    fps=float(meta.fps),
                    tolerance_s=tolerance_s,
                )
            else:
                summary = summarize_episode_camera_seek(
                    episode_index=episode_index,
                    camera_key=camera_key,
                    query_ts=query_ts,
                    video_path=video_path,
                    fps=float(meta.fps),
                    tolerance_s=tolerance_s,
                    backend=backend,
                    scan_order=scan_order,
                    stride=stride,
                )
            if summary is None:
                continue
            per_episode.append(summary)
            camera_counter[camera_key] += 1
            camera_episode_counter[camera_key].add(episode_index)
            for offset in summary.dominant_frame_offsets:
                frame_offset_counter[offset] += 1

    affected_episode_ids = sorted({item.episode_index for item in per_episode})
    result = {
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_dir.name,
        "fps": float(meta.fps),
        "tolerance_s": tolerance_s,
        "backend": backend,
        "check_mode": check_mode,
        "scanned_episodes": len(selected_episodes),
        "affected_episodes": len(affected_episode_ids),
        "affected_episode_ratio": (
            len(affected_episode_ids) / len(selected_episodes) if selected_episodes else 0.0
        ),
        "affected_episode_ids": affected_episode_ids,
        "camera_summary": {
            camera: {
                "affected_episodes": len(camera_episode_counter[camera]),
                "affected_episode_ids": sorted(camera_episode_counter[camera]),
            }
            for camera in selected_cameras
            if camera in camera_episode_counter
        },
        "frame_offset_histogram": dict(sorted(frame_offset_counter.items())),
        "sample_failures": [item.to_dict() for item in per_episode[:50]],
    }
    result["recommendation"] = decide_recommendation(
        total_episodes=len(selected_episodes),
        affected_episodes=len(affected_episode_ids),
        camera_counter=camera_counter,
        frame_offset_counter=frame_offset_counter,
    )
    return result


def main() -> int:
    args = parse_args()
    result = scan_dataset(
        args.dataset_dir,
        cameras=args.cameras,
        episodes=args.episodes,
        max_episodes=args.max_episodes,
        tolerance_s=args.tolerance_s,
        backend=args.backend,
        check_mode=args.check_mode,
        scan_order=args.scan_order,
        stride=args.stride,
    )

    print(f"Dataset: {result['dataset_name']}")
    print(
        "Affected episodes: "
        f"{result['affected_episodes']}/{result['scanned_episodes']} "
        f"({result['affected_episode_ratio']:.2%})"
    )
    print(f"Frame offset histogram: {result['frame_offset_histogram']}")
    print(f"Recommendation: {result['recommendation']}")

    if result["camera_summary"]:
        print("Camera summary:")
        for camera_key, summary in result["camera_summary"].items():
            print(
                f"  {camera_key}: {summary['affected_episodes']} affected episodes; "
                f"sample={summary['affected_episode_ids'][:10]}"
            )

    if result["sample_failures"]:
        print("Sample failures:")
        for item in result["sample_failures"][:10]:
            print(
                "  ep={episode_index} cam={camera_key} mismatches={mismatch_count}/{total_queries} "
                "checked={checked_queries} max_diff={max_diff_s:.4f}s offsets={dominant_frame_offsets} "
                "first={first_query_s:.4f}->{first_loaded_s:.4f} "
                "last={last_query_s:.4f}->{last_loaded_s:.4f}".format(**item)
            )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote JSON summary to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
