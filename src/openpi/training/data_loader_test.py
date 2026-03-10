import dataclasses
import pickle
import pathlib

import jax
import torch

from openpi.models import pi0
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


def test_torch_data_loader():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


class _FakeLeRobotDataset:
    class _Meta:
        def __init__(self):
            self.video_keys = ["observation.images.hand_right", "observation.images.hand_right_depth"]
            self.camera_keys = ["observation.images.hand_right", "observation.images.hand_right_depth"]
            self.tasks = {0: "fake_task"}

    def __init__(self, episodes, episode_index=0):
        self.episodes = episodes
        self.repo_id = "fake_repo"
        self.delta_indices = {"actions": [0]}
        self.meta = self._Meta()
        self.image_transforms = None
        self.hf_dataset = [
            {
                "episode_index": torch.tensor(episode_index),
                "task_index": torch.tensor(0),
                "timestamp": torch.tensor(0.0),
                "observation.images.hand_right": torch.ones((4, 4, 3)),
            }
        ]
        self.calls = []
        self.video_calls = []

    def _get_query_indices(self, idx, ep_idx):
        self.calls.append((idx, ep_idx))
        return {"actions": [idx]}, {"actions_is_pad": torch.BoolTensor([False])}

    def _query_hf_dataset(self, query_indices):
        return {"actions": torch.zeros((1, 8))}

    def _get_query_timestamps(self, current_ts, query_indices=None):
        del current_ts, query_indices
        return {
            "observation.images.hand_right": [0.0],
            "observation.images.hand_right_depth": [0.0],
        }

    def _query_videos(self, query_timestamps, ep_idx):
        self.video_calls.append((query_timestamps, ep_idx))
        return {key: torch.ones((1, 4, 4, 3)) for key in query_timestamps}

    def __len__(self):
        return len(self.hf_dataset)


class _FakeMultiDataset:
    def __init__(self, datasets):
        self._datasets = datasets


class _TinyDataset:
    def __init__(self, items, *, features=None):
        self._items = items
        self.features = features or list(items[0].keys())

    def __getitem__(self, idx):
        return self._items[idx]

    def __len__(self):
        return len(self._items)


def test_episode_subset_compatible_lerobot_dataset_maps_global_to_local(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    dataset = _FakeLeRobotDataset([5, 9], episode_index=9)
    wrapped = _data_loader.EpisodeSubsetCompatibleLeRobotDataset(
        dataset,
        required_camera_keys={"observation.images.hand_right"},
    )

    item = wrapped[0]

    assert dataset.calls == [(0, 1)]
    assert dataset.video_calls == [({"observation.images.hand_right": [0.0]}, 9)]
    assert item["task"] == "fake_task"
    assert item["episode_index"].item() == 9
    assert "observation.images.hand_right_depth" not in item


def test_episode_subset_compatible_lerobot_dataset_is_pickle_safe(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    wrapped = _data_loader.EpisodeSubsetCompatibleLeRobotDataset(
        _FakeLeRobotDataset([3, 7], episode_index=7),
        required_camera_keys={"observation.images.hand_right"},
    )

    restored = pickle.loads(pickle.dumps(wrapped))
    restored[0]

    assert restored._base_dataset.calls == [(0, 1)]
    assert restored.repo_id == "fake_repo"


def test_episode_subset_compatible_lerobot_dataset_uninitialized_getattr_is_safe():
    wrapped = _data_loader.EpisodeSubsetCompatibleLeRobotDataset.__new__(
        _data_loader.EpisodeSubsetCompatibleLeRobotDataset
    )

    try:
        _ = wrapped.repo_id
    except AttributeError:
        pass
    else:
        raise AssertionError("Expected AttributeError when _base_dataset is not initialized")


def test_make_selected_episode_compatible_dataset_recurses_into_multi_dataset(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    left = _FakeLeRobotDataset([3, 7], episode_index=7)
    right = _FakeLeRobotDataset([10, 20], episode_index=20)
    dataset = _FakeMultiDataset([left, right])

    wrapped = _data_loader._make_selected_episode_compatible_dataset(dataset)
    wrapped._datasets[0][0]
    wrapped._datasets[1][0]

    assert isinstance(wrapped._datasets[0], _data_loader.EpisodeSubsetCompatibleLeRobotDataset)
    assert isinstance(wrapped._datasets[1], _data_loader.EpisodeSubsetCompatibleLeRobotDataset)
    assert left.calls == [(0, 1)]
    assert right.calls == [(0, 1)]


def test_make_selected_episode_compatible_dataset_leaves_contiguous_subset_unchanged(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    dataset = _FakeLeRobotDataset([0, 1], episode_index=1)

    wrapped = _data_loader._make_selected_episode_compatible_dataset(dataset)

    assert isinstance(wrapped, _data_loader.EpisodeSubsetCompatibleLeRobotDataset)


def test_required_camera_keys_from_data_config_extracts_only_used_image_sources():
    data_config = _config.DataConfig(
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "rgb": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                    }
                )
            ]
        )
    )

    required_camera_keys = _data_loader._required_camera_keys_from_data_config(data_config)

    assert required_camera_keys == {"observation.images.hand_right"}


def test_create_dataset_metadata_uses_root_for_absolute_paths(monkeypatch):
    calls = []

    class _FakeMeta:
        def __init__(self, repo_id, root=None, revision=None, force_cache_sync=False):
            del revision, force_cache_sync
            calls.append((repo_id, root))

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _FakeMeta)

    _data_loader._create_dataset_metadata("/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim/pour_workpiece")

    assert calls == [
        (
            "pour_workpiece",
            pathlib.Path("/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim/pour_workpiece"),
        )
    ]


def test_create_dataset_metadata_preserves_hf_repo_ids(monkeypatch):
    calls = []

    class _FakeMeta:
        def __init__(self, repo_id, root=None, revision=None, force_cache_sync=False):
            del revision, force_cache_sync
            calls.append((repo_id, root))

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _FakeMeta)

    _data_loader._create_dataset_metadata("lerobot/aloha_sim_transfer_cube_human")

    assert calls == [("lerobot/aloha_sim_transfer_cube_human", None)]


def test_create_lerobot_dataset_uses_root_for_absolute_paths(monkeypatch):
    calls = []

    class _FakeDatasetCtor:
        def __init__(self, repo_id, root=None, episodes=None, delta_timestamps=None, tolerance_s=None):
            calls.append((repo_id, root, episodes, delta_timestamps, tolerance_s))

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeDatasetCtor)

    _data_loader._create_lerobot_dataset(
        "/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim/pour_workpiece",
        episodes=[1, 2],
        delta_timestamps={"action": [0.0, 0.1]},
        tolerance_s=0.15,
    )

    assert calls == [
        (
            "pour_workpiece",
            pathlib.Path("/ssd_workspace/huggingface/lerobot/Reasoning2Action-Sim/pour_workpiece"),
            [1, 2],
            {"action": [0.0, 0.1]},
            0.15,
        )
    ]


def test_multi_dataset_indexes_across_subdatasets():
    dataset = _data_loader.MultiDataset(
        [
            _TinyDataset([{"x": 1}, {"x": 2}]),
            _TinyDataset([{"x": 3}]),
        ]
    )

    assert len(dataset) == 3
    assert dataset[0] == {"x": 1}
    assert dataset[2] == {"x": 3}
