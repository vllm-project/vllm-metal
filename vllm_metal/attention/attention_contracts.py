# SPDX-License-Identifier: Apache-2.0
"""Model-specific contracts for paged attention."""

from dataclasses import dataclass
from enum import Enum, auto


class QKNormPlacement(Enum):
    """Placement of per-head Q/K normalization relative to RoPE."""

    BEFORE_ROPE = auto()
    AFTER_ROPE = auto()


@dataclass(frozen=True, slots=True)
class AttentionContract:
    """Architecture-specific behavior consumed by the paged SDPA path."""

    qk_norm_placement: QKNormPlacement = QKNormPlacement.BEFORE_ROPE
    derive_scale_from_query: bool = False
    use_rope: bool = True


DEFAULT_ATTENTION_CONTRACT = AttentionContract()
_ATTENTION_CONTRACTS: dict[str, AttentionContract] = {
    "mlx_lm.models.hunyuan_v1_dense": AttentionContract(
        qk_norm_placement=QKNormPlacement.AFTER_ROPE
    ),
    "mlx_lm.models.stablelm": AttentionContract(derive_scale_from_query=True),
    # Phi computes the standard scale inline in its forward and exposes no
    # scale attribute, exactly like StableLM.
    "mlx_lm.models.phi": AttentionContract(derive_scale_from_query=True),
    # Nemotron-H attention is position-free: no rotary module, none expected.
    "mlx_lm.models.nemotron_h": AttentionContract(use_rope=False),
}


def attention_contract_for(module: object) -> AttentionContract:
    """Return the model's attention contract or the standard default."""
    return _ATTENTION_CONTRACTS.get(type(module).__module__, DEFAULT_ATTENTION_CONTRACT)
