import json
from pathlib import Path
import tempfile
import unittest

from scripts import cleanup_depth_metadata


class CleanupDepthMetadataTest(unittest.TestCase):
    def _write_dataset(self, root: Path) -> Path:
        dataset_dir = root / "open_door"
        meta_dir = dataset_dir / "meta"
        meta_dir.mkdir(parents=True)

        info_payload = {
            "total_episodes": 2,
            "total_videos": 6,
            "features": {
                "observation.images.top_head": {
                    "dtype": "video",
                    "video_info": {"video.is_depth_map": False},
                },
                "observation.images.hand_left": {
                    "dtype": "video",
                    "video_info": {"video.is_depth_map": False},
                },
                "observation.images.head_depth": {
                    "dtype": "video",
                    "video_info": {"video.is_depth_map": True},
                },
                "observation.state": {"dtype": "float32"},
            },
        }
        (meta_dir / "info.json").write_text(json.dumps(info_payload, indent=4) + "\n")

        records = [
            {
                "episode_index": 0,
                "stats": {
                    "observation.images.top_head": {"count": [0]},
                    "observation.images.head_depth": {"count": [0]},
                    "observation.state": {"count": [10]},
                },
            },
            {
                "episode_index": 1,
                "stats": {
                    "observation.images.top_head": {"count": [0]},
                    "observation.images.head_depth": {"count": [0]},
                    "observation.state": {"count": [10]},
                },
            },
        ]
        (meta_dir / "episodes_stats.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        return dataset_dir

    def test_cleanup_dataset_dir_rewrites_info_and_episode_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = self._write_dataset(Path(temp_dir))

            summary = cleanup_depth_metadata.cleanup_dataset_dir(dataset_dir)

            self.assertEqual(
                summary["removed_depth_features"],
                ["observation.images.head_depth"],
            )
            info_payload = json.loads((dataset_dir / "meta" / "info.json").read_text())
            self.assertNotIn("observation.images.head_depth", info_payload["features"])
            self.assertEqual(info_payload["total_videos"], 4)

            stats_lines = (dataset_dir / "meta" / "episodes_stats.jsonl").read_text().splitlines()
            for line in stats_lines:
                record = json.loads(line)
                self.assertNotIn("observation.images.head_depth", record["stats"])
                self.assertIn("observation.images.top_head", record["stats"])

    def test_cleanup_dataset_dir_dry_run_keeps_files_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = self._write_dataset(Path(temp_dir))
            info_path = dataset_dir / "meta" / "info.json"
            stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
            original_info = info_path.read_text()
            original_stats = stats_path.read_text()

            cleanup_depth_metadata.cleanup_dataset_dir(dataset_dir, dry_run=True)

            self.assertEqual(info_path.read_text(), original_info)
            self.assertEqual(stats_path.read_text(), original_stats)


if __name__ == "__main__":
    unittest.main()
