from __future__ import annotations

import numpy as np

from scripts.verify_reasoning2action_frame_cache import _assert_array_equal


def test_assert_array_equal_handles_nested_dicts() -> None:
    left = {
        "image": {
            "base_0_rgb": np.zeros((2, 3, 4, 4), dtype=np.float32),
            "left_wrist_0_rgb": np.ones((2, 3, 4, 4), dtype=np.float32),
        }
    }
    right = {
        "image": {
            "base_0_rgb": np.zeros((2, 3, 4, 4), dtype=np.float32),
            "left_wrist_0_rgb": np.ones((2, 3, 4, 4), dtype=np.float32),
        }
    }

    _assert_array_equal("batch:image", left, right)


def test_assert_array_equal_can_ignore_dtype_for_metadata() -> None:
    _assert_array_equal(
        "batch:episode_index",
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([1, 2, 3], dtype=np.int32),
        check_dtype=False,
    )
