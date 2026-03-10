import unittest

from scripts import detect_video_timestamp_mismatch as script


class DetectVideoTimestampMismatchTest(unittest.TestCase):
    def test_recommend_exclude_for_sparse_single_camera_offsets(self) -> None:
        recommendation = script.decide_recommendation(
            total_episodes=100,
            affected_episodes=3,
            camera_counter=script.Counter({"observation.images.top_head": 3}),
            frame_offset_counter=script.Counter({1: 3, 2: 1}),
        )

        self.assertIn("excluding the listed episodes", recommendation)

    def test_recommend_repair_for_moderate_single_camera_offsets(self) -> None:
        recommendation = script.decide_recommendation(
            total_episodes=100,
            affected_episodes=12,
            camera_counter=script.Counter({"observation.images.top_head": 12}),
            frame_offset_counter=script.Counter({1: 8, 2: 4}),
        )

        self.assertIn("repairing or re-encoding", recommendation)


if __name__ == "__main__":
    unittest.main()
