"""Create a zero-LoRA adapter NPZ for use as the _default adapter.

When the adapter-routed policy falls through to _default, this zero adapter
zeroes out all LoRA contributions, making inference identical to the baseline
(pure frozen weights, no LoRA perturbation).

Usage:
    python scripts/create_zero_lora_adapter.py \
        --checkpoint <baseline_checkpoint>/params \
        --output adapters/_default.npz
"""

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import flax.traverse_util
import jax
import numpy as np
import tyro
from flax import nnx

from openpi.training import config as _config
from openpi.training.weight_loaders import ACOTCheckpointWeightLoader

PATHS_KEY = "__paths__"
VALUE_KEY_TEMPLATE = "param_{index:04d}"


def main(
    checkpoint: str,
    output: str,
    config_name: str = "acot_challenge_lora_conservative",
) -> None:
    """Create a zero-LoRA adapter from a baseline checkpoint.

    Args:
        checkpoint: Path to baseline checkpoint params directory
                    (e.g. checkpoints/pi05_base/params).
        output:     Output path for the NPZ file (e.g. adapters/_default.npz).
        config_name: Training config that defines the LoRA model architecture.
    """
    cfg = _config.get_config(config_name)

    print(f"Building model param shapes from config '{config_name}' ...")

    def _init(rng):
        model = cfg.model.create(rng)
        return nnx.state(model).to_pure_dict()

    params_shape = jax.eval_shape(_init, jax.random.PRNGKey(0))

    print(f"Loading baseline checkpoint with missing_init='zeros': {checkpoint}")
    loader = ACOTCheckpointWeightLoader(checkpoint, missing_init="zeros")
    loaded = loader.load(params_shape)

    flat = flax.traverse_util.flatten_dict(loaded, sep="/")

    lora_params = {k: np.asarray(v) for k, v in flat.items() if "lora" in k}
    if not lora_params:
        raise ValueError("No LoRA parameters found — check that the config uses a LoRA model.")

    # Verify all values are zero (sanity check).
    non_zero = {k for k, v in lora_params.items() if not np.all(v == 0)}
    if non_zero:
        print(f"[WARN] {len(non_zero)} LoRA tensors are not zero — they came from the baseline checkpoint.")
        print(f"       Non-zero keys: {list(non_zero)[:5]} ...")

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = list(lora_params.keys())
    payload = {
        PATHS_KEY: np.asarray(paths, dtype=str),
        **{VALUE_KEY_TEMPLATE.format(index=i): v for i, v in enumerate(lora_params.values())},
    }
    np.savez(output_path, **payload)

    total_params = sum(v.size for v in lora_params.values())
    print(f"Wrote {len(paths)} zero LoRA tensors ({total_params:,} params) to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
