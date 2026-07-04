# SPDX-License-Identifier: Apache-2.0
"""GGUF-aware Linear and Embedding modules backed by a GGUFTensor.

Thin ``nn.Module`` drop-ins that delegate all quantized math to the tensor's
``matmul`` / ``embedding``. They add only the ``nn.Module`` surface, an optional
dense bias, and the tied-lm_head ``as_linear`` hook — no quant logic and no
re-validation of what the tensor already guarantees. Unsupported qtypes are
rejected when the tensor is built (PR 1's ``GGUFMLXQuantizedTensor``), so a
wrapper never sees an invalid one.

The quantized arrays are owned by the tensor, not by these modules, so they are
intentionally invisible to ``Module.parameters()`` (a feature: a dtype cast or
parameter sweep won't touch packed uint32). The forward path runs through MLX's
lazy evaluation of the output, which does not need the arrays to be parameters.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.gguf.tensor import GGUFTensor


class GGUFLinear(nn.Module):
    """A quantized linear layer: ``x @ W.T`` (plus an optional bias).

    Mirrors ``mlx.nn.QuantizedLinear`` — the bias is a dense array added after
    the matmul. It is cast to the activation dtype so the output follows ``x``'s
    dtype (GGUF biases are stored as float32, which would otherwise promote a
    float16/bfloat16 result).
    """

    def __init__(self, tensor: GGUFTensor, bias: mx.array | None = None) -> None:
        super().__init__()
        self.tensor = tensor
        if bias is not None:
            self.bias = self._validate_bias(tensor, bias)
        self.freeze()

    @staticmethod
    def _validate_bias(tensor: GGUFTensor, bias: mx.array) -> mx.array:
        expected = (tensor.out_features,)
        if (
            not isinstance(bias, mx.array)
            or bias.shape != expected
            or not mx.issubdtype(bias.dtype, mx.floating)
        ):
            got = (
                f"{bias.dtype} {bias.shape}"
                if isinstance(bias, mx.array)
                else type(bias)
            )
            raise ValueError(
                f"GGUFLinear bias must be a floating mx.array of shape "
                f"{expected}, got {got}"
            )
        return bias

    @property
    def out_features(self) -> int:
        """Output feature count, surfacing the tensor's already-public value.

        Lets shape-introspecting callers (e.g. the attention backend deriving
        head_dim) read the projection's output dim without a dense ``.weight``,
        which this wrapper does not expose (the packed weight lives on the
        tensor).
        """
        return self.tensor.out_features

    def __call__(self, x: mx.array) -> mx.array:
        out = self.tensor.matmul(x)
        if "bias" in self:
            out = out + self["bias"].astype(out.dtype)
        return out

    def eval_arrays(self) -> None:
        """Materialize the packed quant arrays hidden behind the wrapper."""
        self.tensor.eval_arrays()

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.tensor.in_features}, "
            f"output_dims={self.tensor.out_features}, bias={'bias' in self}"
        )


class GGUFEmbedding(nn.Module):
    """A quantized embedding table; ``as_linear`` reuses it for a tied lm_head.

    Mirrors ``mlx.nn.QuantizedEmbedding``: ``__call__`` gathers and dequantizes
    rows, ``as_linear`` runs the table as the output projection.
    """

    def __init__(self, tensor: GGUFTensor, output_dtype: mx.Dtype) -> None:
        super().__init__()
        self.tensor = tensor
        self.output_dtype = self._validate_output_dtype(output_dtype)
        self.freeze()

    @staticmethod
    def _validate_output_dtype(output_dtype: mx.Dtype) -> mx.Dtype:
        # A non-floating output_dtype would silently truncate dequantized rows to
        # integers; reject it at this public wrapper boundary.
        if not isinstance(output_dtype, mx.Dtype) or not mx.issubdtype(
            output_dtype, mx.floating
        ):
            raise ValueError(
                "GGUFEmbedding output_dtype must be a floating mx.Dtype, got "
                f"{output_dtype!r}"
            )
        return output_dtype

    def __call__(self, ids: mx.array) -> mx.array:
        return self.tensor.embedding(ids, self.output_dtype)

    def as_linear(self, x: mx.array) -> mx.array:
        return self.tensor.matmul(x)

    def eval_arrays(self) -> None:
        """Materialize the packed quant arrays hidden behind the wrapper."""
        self.tensor.eval_arrays()

    def _extra_repr(self) -> str:
        # Bare "num_embeddings, dims" mirrors mlx.nn.QuantizedEmbedding's repr.
        return f"{self.tensor.out_features}, {self.tensor.in_features}"
