# SPDX-License-Identifier: Apache-2.0
"""GGUF tensors whose logical layout differs from their stored blocks.

A packed block can hold input columns that must move apart (a 256-wide Q6_K
superblock spans heads that are not neighbours in the model's order), so a
quantized tensor keeps its stored blocks and the activation is gathered into the
stored column order instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from vllm_metal.gguf.tensor import GGUFTensor


@dataclass(frozen=True, eq=False)
class GGUFColumnPermutedTensor:
    """A GGUF tensor whose logical weight is ``tensor``'s ``stored[:, column_index]``.

    ``column_index`` is a permutation of ``range(in_features)``. Satisfies
    :class:`~vllm_metal.gguf.tensor.GGUFTensor`.
    """

    tensor: GGUFTensor
    column_index: mx.array
    _stored_order: mx.array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_stored_order", mx.argsort(self.column_index))

    @property
    def out_features(self) -> int:
        return self.tensor.out_features

    @property
    def in_features(self) -> int:
        return self.tensor.in_features

    def matmul(self, x: mx.array) -> mx.array:
        """``x @ W.T`` for the logical weight, gathering ``x`` into stored order."""
        return self.tensor.matmul(mx.take(x, self._stored_order, axis=-1))

    def embedding(self, ids: mx.array, output_dtype: mx.Dtype) -> mx.array:
        """Gather and dequantize logical rows: stored rows, columns reordered."""
        rows = self.tensor.embedding(ids, output_dtype)
        return mx.take(rows, self.column_index, axis=-1)

    def eval_arrays(self) -> None:
        """Materialize the stored tensor and both column orders."""
        self.tensor.eval_arrays()
        mx.eval(self.column_index, self._stored_order)
