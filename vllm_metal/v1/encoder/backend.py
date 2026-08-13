# SPDX-License-Identifier: Apache-2.0
"""Typed encoder embedding backend used by the shared pooling path.

Family-specific loaders and architectures live in per-family modules. Pooling,
lifecycle, cache setup, and the runner only depend on this contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx


@dataclass(frozen=True, slots=True)
class EncoderPoolingPolicy:
    """Pooling and cache behavior owned by one encoder family."""

    default_sequence_pooling_type: str
    allowed_sequence_pooling_types: tuple[str | None, ...]
    skip_paged_attention_patch: bool


class EncoderEmbeddingBackend(Protocol):
    """Runtime backend for one loaded dense encoder embedding family."""

    pooling_policy: EncoderPoolingPolicy

    def forward_sequence_hidden_states(
        self,
        input_ids: mx.array,
        *,
        segment_lengths: list[int] | None = None,
        model_label: str = "encoder-embedding",
    ) -> mx.array:
        """Return packed ``[1, tokens, hidden]`` states for pooling."""


class EncoderSequenceBody:
    """Callable transformer body returning ``[1, tokens, hidden]`` states."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(self, input_ids: mx.array, cache: Any = None) -> mx.array:
        del cache  # Encoder embeddings are bidirectional and cache-free.
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        return self._model(input_ids, attention_mask=attention_mask)


class EncoderSequenceModel:
    """Weight wrapper exposing a Metal-compatible ``.model`` body."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.config = getattr(model, "config", None)
        self.args = self.config
        self.model = EncoderSequenceBody(model)

    def modules(self) -> list[Any]:
        """Expose child modules for vLLM engine cleanup hooks."""
        inner_modules = getattr(self._model, "modules", None)
        if callable(inner_modules):
            return list(inner_modules())
        return [self._model]


def forward_packed_encoder_segments(
    body: Any,
    input_ids: mx.array,
    *,
    segment_lengths: Sequence[int] | None,
    model_label: str,
) -> mx.array:
    """Forward each packed prompt independently (bidirectional attention)."""
    flat = input_ids.reshape(-1)
    total_tokens = int(flat.shape[0])
    lengths = list(segment_lengths) if segment_lengths else [total_tokens]
    if sum(lengths) != total_tokens:
        raise ValueError(
            "Encoder pooling segment_lengths "
            f"{lengths!r} do not cover input tokens "
            f"{total_tokens} for model={model_label}."
        )

    parts: list[mx.array] = []
    offset = 0
    for length in lengths:
        segment = flat[offset : offset + length].reshape(1, length)
        parts.append(body(segment))
        offset += length
    if len(parts) == 1:
        return parts[0]
    return mx.concatenate(parts, axis=1)
