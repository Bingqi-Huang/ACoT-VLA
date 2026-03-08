import flax.nnx as nnx
import jax
from typing_extensions import get_args

import openpi.models.acot_vla as _acot_vla
import openpi.models.gemma as _gemma


def _get_frozen_state(config: _acot_vla.ACOTConfig, freeze_filter: nnx.filterlib.Filter) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_gemma_variant_includes_300m_lora():
    assert "gemma_300m_lora" in get_args(_gemma.Variant)


def test_acot_all_lora_keeps_coarse_expert_lora_trainable():
    config = _acot_vla.ACOTConfig(
        paligemma_variant="gemma_2b_lora",
        coarse_action_expert_variant="gemma_300m_lora",
        action_expert_variant="gemma_300m_lora",
    )
    freeze_filter = config.get_freeze_filter(
        freeze_vision=True,
        freeze_llm=True,
        freeze_llm_embedder=True,
        freeze_dual_ae=[True, True],
    )
    frozen_state = _get_frozen_state(config, freeze_filter)

    assert len(frozen_state) > 0
    assert all("lora" not in "/".join(map(str, path)) for path in frozen_state)

    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    trainable_state = nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(freeze_filter))).flat_state()

    assert any("lora" in "/".join(map(str, path)) for path in trainable_state)
    assert any("_1" in "/".join(map(str, path)) and "lora" in "/".join(map(str, path)) for path in trainable_state)
