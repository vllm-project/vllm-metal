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

    def gather_matmul(
        self,
        x: mx.array,
        indices: mx.array,
        *,
        num_experts: int,
        sorted_indices: bool = False,
    ) -> mx.array:
        """Compute x @ dequantize(W_e).T for each expert e that indices selects.
        Treats this tensor as a stack of experts.

        ``sorted_indices`` may allow for a faster implementation if passed indices
        are sorted.

        The following shapes are a guideline for use in a MoE forward pass:

        When ``sorted_indices`` is False:
        - gate/up: ``x`` (B, S, 1, 1, D), ``indices`` (B, S, k) -> (B, S, k, 1, N)
        - down: ``x`` (B, S, k, 1, D), ``indices`` (B, S, k) -> (B, S, k, 1, N)

        When ``sorted_indices`` is True:
        - gate/up/down: ``x`` (B*S*k, 1, D), ``indices`` (B*S*k,) -> (B*S*k, 1, N)

        B = batch, S = sequence length, k = experts per token, D = ``in_features``,
        N = ``out_features // num_experts``.
        """
        ...

    def embedding(self, ids: mx.array, output_dtype: mx.Dtype) -> mx.array:
        """Gather and dequantize the rows selected by ``ids``; returns output_dtype."""
        ...

    def eval_arrays(self) -> None:
        """Materialize any non-parameter arrays the tensor owns."""
        ...
