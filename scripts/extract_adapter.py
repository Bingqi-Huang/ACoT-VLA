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
# When dual action experts are fully trained (freeze_dual_ae=[False, False]),
# their weights must be included in the adapter so the routing system can
# overlay them on top of the base checkpoint.
#
# Historical checkpoints do not namespace these under "llm_1/llm_2". Instead,
# dual-expert parameters are encoded as suffixed submodules such as
# q_einsum_1/q_einsum_2, mlp_1/mlp_2, final_norm_1/final_norm_2, etc.
#
# Keep "llm_1" and "llm_2" for backward compatibility in case older/newer
# checkpoints use explicit llm_1/llm_2 path prefixes.
DUAL_AE_PATTERNS = (
    "llm_1",
    "llm_2",
    "attn_vec_einsum_1",
    "attn_vec_einsum_2",
    "kv_einsum_1",
    "kv_einsum_2",
    "q_einsum_1",
    "q_einsum_2",
    "mlp_1",
    "mlp_2",
    "pre_attention_norm_1",
    "pre_attention_norm_2",
    "pre_ffw_norm_1",
    "pre_ffw_norm_2",
    "final_norm_1",
    "final_norm_2",
)
LORA_ONLY_PATTERNS = ("lora",)
PATHS_KEY = "__paths__"
VALUE_KEY_TEMPLATE = "param_{index:04d}"


def main(checkpoint: str, output: str, lora_only: bool = False, include_dual_ae: bool = False) -> None:
    """Extract adapter weights from a checkpoint.

    Args:
        checkpoint: Path to the checkpoint directory (parent of the params/ sub-dir).
        output: Output .npz file path.
        lora_only: Extract only LoRA weights (smallest possible adapter).
        include_dual_ae: Also extract fully-trained dual action expert weights
            (llm_1/llm_2-style or *_1/*_2-suffixed submodules). Required when the model was trained with
            freeze_dual_ae=[False, False] (e.g. acot_challenge_generalist_continued).
    """
    if lora_only:
        patterns = LORA_ONLY_PATTERNS
    elif include_dual_ae:
        patterns = ADAPTER_PATTERNS + DUAL_AE_PATTERNS
    else:
        patterns = ADAPTER_PATTERNS
    checkpoint_path = pathlib.Path(checkpoint)
    params = _model.restore_params(checkpoint_path / "params", restore_type=np.ndarray)
    flat_params = flax.traverse_util.flatten_dict(params, sep="/")

    filtered = {path: value for path, value in flat_params.items() if any(pattern in path for pattern in patterns)}
    if include_dual_ae:
        dual_matches = [path for path in flat_params if any(pattern in path for pattern in DUAL_AE_PATTERNS)]
        if not dual_matches:
            print(
                "WARNING: --include-dual-ae was requested, but no dual-AE tensor paths matched. "
                "Verify checkpoint naming and DUAL_AE_PATTERNS."
            )
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
