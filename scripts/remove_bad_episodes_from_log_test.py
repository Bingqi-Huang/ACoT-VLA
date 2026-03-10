import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import remove_bad_episodes_from_log as script


class RemoveBadEpisodesFromLogTest(unittest.TestCase):
    def _write_dataset(self, root: Path) -> Path:
        dataset_dir = root / "place_block_into_box"
        (dataset_dir / "meta").mkdir(parents=True)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "g2a",
            "total_episodes": 3,
            "total_frames": 6,
            "total_tasks": 1,
            "total_videos": 3,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 30,
            "splits": {"train": "0:3"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.top_head": {"dtype": "video"},
                "episode_index": {"dtype": "int64"},
                "index": {"dtype": "int64"},
                "timestamp": {"dtype": "float32"},
            },
        }
        (dataset_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
        (dataset_dir / "meta" / "episodes.jsonl").write_text(
            "".join(
                json.dumps({"episode_index": i, "tasks": ["task"], "length": 2}) + "\n"
                for i in range(3)
            )
        )
        (dataset_dir / "meta" / "episodes_stats.jsonl").write_text(
            "".join(json.dumps({"episode_index": i, "stats": {}}) + "\n" for i in range(3))
        )
        (dataset_dir / "meta" / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "task"}) + "\n"
        )

        for ep in range(3):
            data_path = dataset_dir / f"data/chunk-000/episode_{ep:06d}.parquet"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.table(
                {
                    "episode_index": [ep, ep],
                    "index": [ep * 2, ep * 2 + 1],
                    "timestamp": [0.0, 0.0333333],
                }
            )
            pq.write_table(table, data_path)

            video_path = dataset_dir / f"videos/chunk-000/observation.images.top_head/episode_{ep:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"fake")
        return dataset_dir

    def test_parse_bad_episodes(self) -> None:
        log_text = """
[Data Load Error] Skipping index 1 due to: foo
queried timestamps: tensor([8.3000])
loaded timestamps: tensor([8.3333])
video: /ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim/take_wrong_item_shelf/videos/chunk-000/observation.images.top_head/episode_000003.mp4
[Data Load Error] Skipping index 2 due to: foo
queried timestamps: tensor([8.3000])
loaded timestamps: tensor([8.3333])
video: /ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim/place_block_into_box/videos/chunk-000/observation.images.top_head/episode_000051.mp4
"""
        parsed = script.parse_bad_episodes(log_text)
        self.assertEqual(parsed["place_block_into_box"], {51})
        self.assertEqual(parsed["take_wrong_item_shelf"], {3})

    def test_rewrite_dataset_reindexes_remaining_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = self._write_dataset(Path(temp_dir))
            summary = script.rewrite_dataset(dataset_dir, {1}, dry_run=False)

            self.assertEqual(summary["new_total_episodes"], 2)
            info = json.loads((dataset_dir / "meta" / "info.json").read_text())
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["splits"], {"train": "0:2"})

            episodes_lines = (dataset_dir / "meta" / "episodes.jsonl").read_text().splitlines()
            episode_indices = [json.loads(line)["episode_index"] for line in episodes_lines]
            self.assertEqual(episode_indices, [0, 1])

            second_ep = pq.read_table(dataset_dir / "data/chunk-000/episode_000001.parquet").to_pydict()
            self.assertEqual(second_ep["episode_index"], [1, 1])
            self.assertEqual(second_ep["index"], [2, 3])


if __name__ == "__main__":
    unittest.main()
