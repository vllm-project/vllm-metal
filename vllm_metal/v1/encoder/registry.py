# SPDX-License-Identifier: Apache-2.0
"""Registry for dense encoder embedding families."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from vllm_metal.v1.encoder.backend import (
    EncoderEmbeddingBackend,
    EncoderPoolingPolicy,
)


class EncoderEmbeddingFamily(Protocol):
    """One dense encoder family: match config, load weights, build a backend."""

    @staticmethod
    def matches_config(model_config: Any) -> bool:
        """Return whether this family owns ``model_config``."""

    @staticmethod
    def pooling_policy() -> EncoderPoolingPolicy:
        """Return pooling/cache policy for this family."""

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        tokenizer_config: dict[str, Any] | None = None,
        lazy: bool = False,
    ) -> tuple[Any, Any, EncoderEmbeddingBackend]:
        """Load weights and return ``(model, tokenizer, backend)``."""

    @classmethod
    def from_loaded_model(cls, model: Any) -> EncoderEmbeddingBackend | None:
        """Return a backend when ``model`` was loaded by this family."""


def _families() -> Sequence[type[EncoderEmbeddingFamily]]:
    # Imported lazily so family modules can depend on backend.py without cycles.
    from vllm_metal.v1.encoder.xlm_roberta_family import XLMRobertaEmbeddingFamily

    return (XLMRobertaEmbeddingFamily,)


def encoder_family_for_config(
    model_config: Any,
) -> type[EncoderEmbeddingFamily] | None:
    """Return the registered family that matches ``model_config``, if any."""
    for family in _families():
        if family.matches_config(model_config):
            return family
    return None


def encoder_pooling_policy(model_config: Any) -> EncoderPoolingPolicy | None:
    """Return the family pooling policy for ``model_config``, if any."""
    family = encoder_family_for_config(model_config)
    if family is None:
        return None
    return family.pooling_policy()


def load_encoder_backend(
    model_name: str,
    model_config: Any,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    lazy: bool = False,
) -> tuple[Any, Any, EncoderEmbeddingBackend]:
    """Load the encoder family that matches ``model_config``."""
    family = encoder_family_for_config(model_config)
    if family is None:
        raise ValueError(
            "No encoder embedding family registered for this model config."
        )
    return family.load(
        model_name,
        tokenizer_config=tokenizer_config,
        lazy=lazy,
    )


def backend_from_loaded_model(model: Any) -> EncoderEmbeddingBackend | None:
    """Return a backend when ``model`` was produced by a registered family."""
    for family in _families():
        backend = family.from_loaded_model(model)
        if backend is not None:
            return backend
    return None
