import pathlib

import flax.traverse_util
import numpy as np
import tyro

from openpi.models import model as _model

ADAPTER_PATTERNS = (
    "lora",
    "coarse_action_in_proj",
    "action_in_proj",
    "coarse_action_out_proj",
    "action_out_proj",
    "coarse_time_mlp_in",
    "coarse_time_mlp_out",
    "time_mlp_in",
    "time_mlp_out",
    "explicit_action_reasoner",
    "implicit_action_reasoner",
    "implicit_action_reasoner_interact",
    "action_reasoning_fusion",
    "explicit_action_reason_proj",
    "implicit_action_reason_proj",
)
LORA_ONLY_PATTERNS = ("lora",)
PATHS_KEY = "__paths__"
VALUE_KEY_TEMPLATE = "param_{index:04d}"


def main(checkpoint: str, output: str, lora_only: bool = False) -> None:
    patterns = LORA_ONLY_PATTERNS if lora_only else ADAPTER_PATTERNS
    checkpoint_path = pathlib.Path(checkpoint)
    params = _model.restore_params(checkpoint_path / "params", restore_type=np.ndarray)
    flat_params = flax.traverse_util.flatten_dict(params, sep="/")

    filtered = {path: value for path, value in flat_params.items() if any(pattern in path for pattern in patterns)}
    if not filtered:
        raise ValueError(f"No adapter parameters matched under checkpoint: {checkpoint_path}")

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        PATHS_KEY: np.asarray(list(filtered.keys()), dtype=str),
        **{VALUE_KEY_TEMPLATE.format(index=index): value for index, value in enumerate(filtered.values())},
    }
    np.savez(output_path, **payload)
    print(f"Wrote {len(filtered)} adapter tensors to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
