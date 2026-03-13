import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training import weight_loaders as _weight_loaders


def test_merge_params_normalizes_numbered_checkpoint_keys():
    params = {
        "implicit_action_reasoner": {
            "query_params": {
                0: jnp.zeros((2,), dtype=jnp.float32),
                1: jnp.zeros((2,), dtype=jnp.float32),
            }
        }
    }
    loaded = {
        "implicit_action_reasoner": {
            "query_params": {
                "0": np.ones((2,), dtype=np.float32),
                "1": np.full((2,), 2.0, dtype=np.float32),
            }
        }
    }

    merged = _weight_loaders._merge_params(loaded, params, missing_regex=".*", strict=True)

    assert list(merged["implicit_action_reasoner"]["query_params"].keys()) == [0, 1]
    np.testing.assert_allclose(merged["implicit_action_reasoner"]["query_params"][0], [1.0, 1.0])
    np.testing.assert_allclose(merged["implicit_action_reasoner"]["query_params"][1], [2.0, 2.0])


def test_merge_params_strict_raises_on_missing_params():
    params = {"foo": {"bar": jnp.zeros((2,), dtype=jnp.float32)}}

    with pytest.raises(ValueError, match="foo/bar"):
        _weight_loaders._merge_params({}, params, missing_regex=".*", strict=True)


def test_merge_params_strict_raises_on_shape_mismatch():
    params = {"foo": {"bar": jnp.zeros((2,), dtype=jnp.float32)}}
    loaded = {"foo": {"bar": np.zeros((1,), dtype=np.float32)}}

    with pytest.raises(ValueError, match="foo/bar"):
        _weight_loaders._merge_params(loaded, params, missing_regex="^$", strict=True)
