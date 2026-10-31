# SPDX-License-Identifier: Apache-2.0
"""Model-specific contracts for paged attention."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal


class QKNormPlacement(Enum):
    """Point in Q/K preparation where normalization is applied."""

    BEFORE_ROPE = auto()  # Default: per-head normalization
    AFTER_ROPE = auto()  # Hunyuan: per-head normalization after RoPE
    BEFORE_HEAD_SPLIT = auto()  # MiniMax, OLMo 2/3/E: full-projection normalization


@dataclass(frozen=True, slots=True)
class AttentionContract:
    """Architecture-specific behavior consumed by the paged SDPA path."""

    qk_norm_placement: QKNormPlacement = QKNormPlacement.BEFORE_ROPE
    derive_scale_from_query: bool = False
    use_rope: bool = True
    # Ministral 3: attention is called as ``(x, attn_scale, mask, cache)`` and
    # scales queries by a position-dependent Llama 4 factor.
    llama4_query_scaling: bool = False
    # Muse Glimmer: a weightless Q/K RMSNorm (inline with the module's ``eps`` in
    # mlx-lm, one shared ``qk_norm`` module in mlx-vlm), normed queries scaled
    # by ``qk_scale_factor`` before RoPE, and a ``sigmoid(gate_proj(x))`` gate.
    scaleless_qk_norm: bool = False
    shared_qk_norm: bool = False
    post_norm_query_scale: bool = False
    sigmoid_output_gate: Literal["elementwise", "per_head"] | None = None
    # Module attribute holding learned per-head attention sinks, if any.
    sinks_attr: str = "sinks"


DEFAULT_ATTENTION_CONTRACT = AttentionContract()
_ATTENTION_CONTRACTS: dict[str, AttentionContract] = {
    "mlx_vlm.models.lfm3.language": AttentionContract(sigmoid_output_gate="per_head"),
    "mlx_lm.models.hunyuan_v1_dense": AttentionContract(
        qk_norm_placement=QKNormPlacement.AFTER_ROPE
    ),
    "mlx_lm.models.minimax": AttentionContract(
        qk_norm_placement=QKNormPlacement.BEFORE_HEAD_SPLIT
    ),
    "mlx_lm.models.olmo2": AttentionContract(
        qk_norm_placement=QKNormPlacement.BEFORE_HEAD_SPLIT
    ),
    "mlx_lm.models.olmo3": AttentionContract(
        qk_norm_placement=QKNormPlacement.BEFORE_HEAD_SPLIT
    ),
    "mlx_lm.models.olmoe": AttentionContract(
        qk_norm_placement=QKNormPlacement.BEFORE_HEAD_SPLIT
    ),
    "mlx_lm.models.stablelm": AttentionContract(derive_scale_from_query=True),
    # Phi computes the standard scale inline in its forward and exposes no
    # scale attribute, exactly like StableLM.
    "mlx_lm.models.phi": AttentionContract(derive_scale_from_query=True),
    # Nemotron-H attention is position-free: no rotary module, none expected.
    "mlx_lm.models.nemotron_h": AttentionContract(use_rope=False),
    "mlx_lm.models.ministral3": AttentionContract(llama4_query_scaling=True),
    # Muse Glimmer's Q/K norm, query scale and output gate are inline in its forward.
    "mlx_lm.models.muse_glimmer": AttentionContract(
        scaleless_qk_norm=True,
        post_norm_query_scale=True,
        sigmoid_output_gate="elementwise",
    ),
    # VLLM_METAL_MULTIMODAL_MODE=multimodal-native loads mlx-vlm's port.
    "mlx_vlm.models.muse_glimmer.language": AttentionContract(
        shared_qk_norm=True,
        post_norm_query_scale=True,
        sigmoid_output_gate="elementwise",
    ),
}


def attention_contract_for(module: object) -> AttentionContract:
    """Return the model's attention contract or the standard default."""
    module_name = type(module).__module__
    if module_name == "mlx_lm.models.granitemoehybrid":
        # Granite selects RoPE or position-free attention from the checkpoint.
        return AttentionContract(use_rope=module.rope is not None)
    if module_name == "mlx_lm.models.flint":
        # Flint's full-attention layers are position-free; every layer has
        # learned sinks named ``softmax_offset``.
        return AttentionContract(
            use_rope=module.rope is not None, sinks_attr="softmax_offset"
        )
    return _ATTENTION_CONTRACTS.get(module_name, DEFAULT_ATTENTION_CONTRACT)
