# SPDX-License-Identifier: Apache-2.0
"""Native MTP head registry keyed by draft ``model_type``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

if TYPE_CHECKING:
    import mlx.core as mx
    from vllm.config import VllmConfig

    from vllm_metal.v1.proposer import MetalProposer
    from vllm_metal.v1.spec_decode import SpeculativeDecodeController


@dataclass(frozen=True, slots=True)
class NativeMTPBuildContext:
    """Inputs a native MTP head uses to build its proposer."""

    speculative_config: Any
    controller: SpeculativeDecodeController
    vllm_config: VllmConfig
    target_config: Any
    dtype: mx.Dtype | None


class NativeMTPHead(Protocol):
    """Native MTP head that owns validation and proposer construction."""

    model_type: str

    def build_proposer(self, context: NativeMTPBuildContext) -> MetalProposer: ...


class NativeMTPHeadRegistry:
    """Registry for native MTP head implementations."""

    _heads: ClassVar[dict[str, NativeMTPHead]] = {}

    @classmethod
    def register(cls, head: NativeMTPHead) -> None:
        cls._heads[head.model_type] = head

    @classmethod
    def find(cls, speculative_config: Any) -> NativeMTPHead | None:
        model_type = cls._draft_model_type(speculative_config)
        return cls._heads.get(model_type)

    @classmethod
    def registered_types(cls) -> list[str]:
        return sorted(cls._heads)

    @classmethod
    def unsupported_message(cls, speculative_config: Any) -> str:
        model_type = cls._draft_model_type(speculative_config)
        return (
            f"No MTP head on Metal for draft model_type {model_type!r} "
            f"(registered native heads: {cls.registered_types()}; plus "
            "Gemma4 MTP via the in-model assistant)."
        )

    @staticmethod
    def _draft_model_type(speculative_config: Any) -> str:
        return speculative_config.draft_model_config.hf_config.model_type


__all__ = [
    "NativeMTPBuildContext",
    "NativeMTPHead",
    "NativeMTPHeadRegistry",
]
