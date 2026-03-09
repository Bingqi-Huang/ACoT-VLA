import dataclasses
import pickle

import jax
import torch

from openpi.models import pi0
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


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
            self.video_keys = []
            self.camera_keys = []
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
            }
        ]
        self.calls = []

    def _get_query_indices(self, idx, ep_idx):
        self.calls.append((idx, ep_idx))
        return {"actions": [idx]}, {"actions_is_pad": torch.BoolTensor([False])}

    def _query_hf_dataset(self, query_indices):
        return {"actions": torch.zeros((1, 8))}

    def _get_query_timestamps(self, current_ts, query_indices=None):
        del current_ts, query_indices
        return {}

    def _query_videos(self, query_timestamps, ep_idx):
        del query_timestamps, ep_idx
        return {}

    def __len__(self):
        return len(self.hf_dataset)


class _FakeMultiDataset:
    def __init__(self, datasets):
        self._datasets = datasets


def test_episode_subset_compatible_lerobot_dataset_maps_global_to_local(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    dataset = _FakeLeRobotDataset([5, 9], episode_index=9)
    wrapped = _data_loader.EpisodeSubsetCompatibleLeRobotDataset(dataset)

    item = wrapped[0]

    assert dataset.calls == [(0, 1)]
    assert item["task"] == "fake_task"
    assert item["episode_index"].item() == 9


def test_episode_subset_compatible_lerobot_dataset_is_pickle_safe(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    wrapped = _data_loader.EpisodeSubsetCompatibleLeRobotDataset(_FakeLeRobotDataset([3, 7], episode_index=7))

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

    assert wrapped is dataset
