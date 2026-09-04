# SPDX-License-Identifier: Apache-2.0
"""Model-specific contracts for paged attention."""

from dataclasses import dataclass
from enum import Enum, auto


class QKNormPlacement(Enum):
    """Placement of Q/K normalization relative to RoPE."""

    BEFORE_ROPE = auto()
    AFTER_ROPE = auto()


class QKNormLayout(Enum):
    """Tensor layout expected by an attention module's Q/K norms."""

    PER_HEAD = auto()
    FULL_PROJECTION = auto()


@dataclass(frozen=True, slots=True)
class AttentionContract:
    """Architecture-specific behavior consumed by the paged SDPA path."""

    qk_norm_placement: QKNormPlacement = QKNormPlacement.BEFORE_ROPE
    qk_norm_layout: QKNormLayout = QKNormLayout.PER_HEAD


DEFAULT_ATTENTION_CONTRACT = AttentionContract()
_ATTENTION_CONTRACTS: dict[str, AttentionContract] = {
    "mlx_lm.models.hunyuan_v1_dense": AttentionContract(
        qk_norm_placement=QKNormPlacement.AFTER_ROPE
    ),
    "mlx_lm.models.olmo3": AttentionContract(
        qk_norm_layout=QKNormLayout.FULL_PROJECTION
    ),
}


def attention_contract_for(module: object) -> AttentionContract:
    """Return the model's attention contract or the standard default."""
    return _ATTENTION_CONTRACTS.get(type(module).__module__, DEFAULT_ATTENTION_CONTRACT)
