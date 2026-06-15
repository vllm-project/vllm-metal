# SPDX-License-Identifier: Apache-2.0
"""The runtime contract the GGUF-aware wrappers depend on.

A small, format-neutral protocol so an MLX-native quantized tensor (the only
implementer today, ``GGUFMLXQuantizedTensor``) and a future raw-block/K-quant
tensor can be consumed by the same Linear/Embedding wrappers without
backend-specific dispatch branches. Introduced at the reviewer's request in the
PR 1 review.

This module imports no ``gguf`` and no MLX at runtime — the ``mx`` annotations
are only evaluated under ``TYPE_CHECKING`` — so it stays dependency-light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import mlx.core as mx


@runtime_checkable
class GGUFTensor(Protocol):
    """A GGUF weight that knows how to matmul and gather embedding rows.

    ``GGUFMLXQuantizedTensor`` satisfies this structurally (no inheritance);
    ``isinstance(tensor, GGUFTensor)`` only checks the members below are present.
    """

    @property
    def out_features(self) -> int:
        """Logical output dim — rows of the logical ``(out, in)`` weight."""
        ...

    @property
    def in_features(self) -> int:
        """Logical input dim — cols of the logical ``(out, in)`` weight."""
        ...

    def matmul(self, x: mx.array) -> mx.array:
        """``x @ W.T`` without dequantizing; preserves leading dims, returns x.dtype."""
        ...

    def embedding(self, ids: mx.array, output_dtype: mx.Dtype) -> mx.array:
        """Gather and dequantize the rows selected by ``ids``; returns output_dtype."""
        ...
