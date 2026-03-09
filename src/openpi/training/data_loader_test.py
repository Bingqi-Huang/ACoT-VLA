import dataclasses
import types

import jax

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
    def __init__(self, episodes):
        self.episodes = episodes
        self.repo_id = "fake_repo"
        self.calls = []

    def _get_query_indices(self, idx, ep_idx):
        self.calls.append((idx, ep_idx))
        return {"query": [idx]}, {"pad": [False]}


def test_patch_lerobot_selected_episode_query_indices_maps_global_to_local(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    dataset = _FakeLeRobotDataset([5, 9])

    _data_loader._patch_lerobot_selected_episode_query_indices(dataset)
    dataset._get_query_indices(42, 9)

    assert dataset.calls == [(42, 1)]
    assert dataset._openpi_episode_subset_patch_applied is True


def test_patch_lerobot_selected_episode_query_indices_recurses_into_multi_dataset(monkeypatch):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    left = _FakeLeRobotDataset([3, 7])
    right = _FakeLeRobotDataset([10, 20])
    dataset = types.SimpleNamespace(_datasets=[left, right])

    _data_loader._patch_lerobot_selected_episode_query_indices(dataset)
    left._get_query_indices(1, 7)
    right._get_query_indices(2, 20)

    assert left.calls == [(1, 1)]
    assert right.calls == [(2, 1)]
