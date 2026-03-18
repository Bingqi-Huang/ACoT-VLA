import json

import torch

from openpi.training import config as _config
from openpi.training import episode_split as _episode_split
from openpi.training import sampler as _sampler


class _FakeMeta:
    def __init__(self, repo_id: str):
        self.episodes = {
            0: {"length": 10},
            1: {"length": 20},
            2: {"length": 30},
            3: {"length": 40},
        }
        self.info = {"instruction_segments": {}}


def test_episode_split_manifest_is_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr(_episode_split.lerobot_dataset, "LeRobotDatasetMetadata", _FakeMeta)
    data_config = _config.DataConfig(
        repo_id=["task_a", "task_b"],
        episode_split=_config.EpisodeSplitConfig(seed=7, train_ratio=0.8, split_name="unit_test_split"),
    )

    manifest_a, path_a = _episode_split.get_or_create_manifest(data_config, base_output_dir=tmp_path)
    manifest_b, path_b = _episode_split.get_or_create_manifest(data_config, base_output_dir=tmp_path)

    assert path_a == path_b
    assert manifest_a == manifest_b
    assert path_a.exists()

    payload = json.loads(path_a.read_text())
    assert payload["seed"] == 7
    assert payload["train_ratio"] == 0.8
    assert [dataset["repo_id"] for dataset in payload["datasets"]] == ["task_a", "task_b"]
    assert all(len(dataset["train_episode_indices"]) == 3 for dataset in payload["datasets"])
    assert all(len(dataset["val_episode_indices"]) == 1 for dataset in payload["datasets"])


class _FakeSubDataset:
    def __init__(self):
        self.meta = type(
            "Meta",
            (),
            {
                "info": {
                    "instruction_segments": {
                        "5": [
                            {
                                "start_frame_index": 1,
                                "success_frame_index": 3,
                                "instruction": "pick object",
                            }
                        ],
                        "9": [
                            {
                                "start_frame_index": 0,
                                "success_frame_index": 2,
                                "instruction": "reset pose",
                            }
                        ],
                    }
                }
            },
        )()
        self.episode_data_index = {
            "from": torch.tensor([0, 10]),
            "to": torch.tensor([10, 20]),
        }
        self.episodes = [5, 9]

    def __len__(self):
        return 20


def test_sample_subtask_uses_selected_episode_ids():
    dataset = _FakeSubDataset()
    intervals = _sampler.sample_subtask(dataset)

    assert intervals == [(1, 3), (10, 12)]
