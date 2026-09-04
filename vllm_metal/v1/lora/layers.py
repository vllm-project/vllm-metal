# SPDX-License-Identifier: Apache-2.0
"""MLX layer wrappers for LoRA and QLoRA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from .punica_wrapper import PunicaWrapperMLX


def can_wrap(module: Any) -> bool:
    return isinstance(module, nn.Linear) and not isinstance(
        module, getattr(nn, "QuantizedLinear", ())
    )


def can_wrap_qlora(module: Any) -> bool:
    quantized_embedding_cls = getattr(nn, "QuantizedEmbedding", None)
    if quantized_embedding_cls is not None and isinstance(
        module, quantized_embedding_cls
    ):
        return False

    weight = getattr(module, "weight", None)
    scales = getattr(module, "scales", None)
    return (
        not isinstance(module, nn.Linear)
        and getattr(weight, "ndim", None) == 2
        and getattr(scales, "ndim", None) == 2
        and hasattr(module, "bits")
        and hasattr(module, "group_size")
        and callable(module)
    )


def _infer_qlora_dims(module: Any) -> tuple[int, int]:
    scales = getattr(module, "scales", None)
    if scales is None:
        raise TypeError(
            f"Cannot infer QLoRA dims from {type(module).__name__}: "
            "the layer has no 'scales' attribute."
        )
    group_size = int(module.group_size)
    out_dim = int(scales.shape[0])
    in_dim = int(scales.shape[1]) * group_size
    return in_dim, out_dim


class _LoRALinearBase(nn.Module):
    _lora_name = "LoRA"

    def __init__(
        self,
        base_layer: Any,
        input_size: int,
        output_size: int,
        max_loras: int,
        max_lora_rank: int,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.max_loras, self.max_lora_rank = max_loras, max_lora_rank
        self.input_size, self.output_size = input_size, output_size
        slots = max_loras + 1
        self.lora_a_stacked = mx.zeros((slots, max_lora_rank, input_size), dtype)
        self.lora_b_stacked = mx.zeros((slots, output_size, max_lora_rank), dtype)
        # MLX tracks lists assigned through Module.__setattr__; ranks are metadata.
        object.__setattr__(self, "_lora_ranks", [0] * slots)
        self.punica_wrapper: PunicaWrapperMLX | None = None

    def set_mapping(self, punica_wrapper: PunicaWrapperMLX) -> None:
        self.punica_wrapper = punica_wrapper

    @property
    def weight(self) -> mx.array:
        return self.base_layer.weight

    @property
    def bias(self) -> mx.array:
        return self.base_layer.bias

    def prepare_lora_weights(
        self, slot: int, lora_a: mx.array, lora_b: mx.array
    ) -> tuple[mx.array, mx.array]:
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(
                f"{self._lora_name} weight shape mismatch for slot {slot}: "
                f"A and B must be 2-D, got A.ndim={lora_a.ndim}, "
                f"B.ndim={lora_b.ndim} "
                "(expected A=(rank, in), B=(out, rank))"
            )
        rank, in_ = int(lora_a.shape[0]), int(lora_a.shape[1])
        out, b_rank = int(lora_b.shape[0]), int(lora_b.shape[1])
        if b_rank != rank:
            raise ValueError(
                f"{self._lora_name} weight shape mismatch for slot {slot}: "
                f"A rank {rank} (A=({rank},{in_})) does not match "
                f"B rank {b_rank} (B=({out},{b_rank})); "
                "A=(rank, in), B=(out, rank)"
            )
        if (in_, out) != (self.input_size, self.output_size):
            raise ValueError(
                f"{self._lora_name} weight shape mismatch for slot {slot}: "
                f"A=({rank},{in_}), B=({out},{rank}); expected "
                f"in={self.input_size}, out={self.output_size}"
            )
        if rank > self.max_lora_rank:
            raise ValueError(
                f"{self._lora_name} rank {rank} exceeds "
                f"max_lora_rank {self.max_lora_rank}"
            )
        a = mx.zeros_like(self.lora_a_stacked[slot])
        b = mx.zeros_like(self.lora_b_stacked[slot])
        a[:rank, :] = lora_a.astype(a.dtype)
        b[:, :rank] = lora_b.astype(b.dtype)
        return a, b

    def set_lora(self, slot: int, lora_a: mx.array, lora_b: mx.array) -> None:
        prepared = self.prepare_lora_weights(slot, lora_a, lora_b)
        self.set_prepared_lora(slot, *prepared, rank=int(lora_a.shape[0]))

    def set_prepared_lora(
        self, slot: int, lora_a: mx.array, lora_b: mx.array, *, rank: int
    ) -> None:
        self.lora_a_stacked[slot], self.lora_b_stacked[slot] = lora_a, lora_b
        self._lora_ranks[slot] = rank

    def reset_lora(self, slot: int) -> None:
        self.lora_a_stacked[slot] = mx.zeros_like(self.lora_a_stacked[slot])
        self.lora_b_stacked[slot] = mx.zeros_like(self.lora_b_stacked[slot])
        self._lora_ranks[slot] = 0


class MLXLinearWithLoRA(_LoRALinearBase):
    def __init__(
        self, base_layer: nn.Linear, max_loras: int, max_lora_rank: int, dtype: mx.Dtype
    ) -> None:
        out, in_ = base_layer.weight.shape[-2:]
        super().__init__(
            base_layer,
            int(in_),
            int(out),
            max_loras,
            max_lora_rank,
            dtype,
        )

    @_LoRALinearBase.weight.setter
    def weight(self, value: mx.array) -> None:
        self.base_layer.weight = value

    @_LoRALinearBase.bias.setter
    def bias(self, value: mx.array) -> None:
        self.base_layer.bias = value

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base_layer(x)
        if self.punica_wrapper is None or self.punica_wrapper.no_lora:
            return y

        # Punica expects (n_tokens, dim); collapse leading dims if needed.
        output_shape = y.shape
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        if x.dtype != self.lora_a_stacked.dtype:
            x = x.astype(self.lora_a_stacked.dtype)
        output = self.punica_wrapper.add_lora_linear(
            y,
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            scale=1.0,
            lora_ranks=self._lora_ranks,
        )
        return output if output.shape == output_shape else output.reshape(output_shape)


class MLXQuantizedLinearWithLoRA(_LoRALinearBase):
    _lora_name = "QLoRA"

    def __init__(
        self,
        base_layer: Any,
        max_loras: int,
        max_lora_rank: int,
        dtype: mx.Dtype,
    ) -> None:
        input_size, output_size = _infer_qlora_dims(base_layer)
        super().__init__(
            base_layer,
            input_size,
            output_size,
            max_loras,
            max_lora_rank,
            dtype,
        )

    @property
    def in_features(self) -> int:
        return self.input_size

    @property
    def out_features(self) -> int:
        return self.output_size

    def __call__(self, x: mx.array) -> mx.array:
        # Quantized base forward — runs at the layer's native precision.
        y = self.base_layer(x)
        if self.punica_wrapper is None or self.punica_wrapper.no_lora:
            return y

        output_shape = y.shape
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        if x.dtype != self.lora_a_stacked.dtype:
            x = x.astype(self.lora_a_stacked.dtype)
        output = self.punica_wrapper.add_lora_linear(
            y,
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            scale=1.0,
            lora_ranks=self._lora_ranks,
        )
        return output if output.shape == output_shape else output.reshape(output_shape)
