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


class MLXLinearWithLoRA(nn.Module):
    def __init__(
        self, base_layer: nn.Linear, max_loras: int, max_lora_rank: int, dtype: mx.Dtype
    ):
        super().__init__()
        self.base_layer = base_layer
        self.max_loras, self.max_lora_rank = max_loras, max_lora_rank
        out, in_ = base_layer.weight.shape[-2:]
        self.input_size, self.output_size = int(in_), int(out)
        # +1 trailing slot is the null slot (see punica_wrapper).
        slots = max_loras + 1
        self.lora_a_stacked = mx.zeros((slots, max_lora_rank, self.input_size), dtype)
        self.lora_b_stacked = mx.zeros((slots, self.output_size, max_lora_rank), dtype)
        self.punica_wrapper: PunicaWrapperMLX | None = None

    @property
    def weight(self) -> mx.array:
        return self.base_layer.weight

    @weight.setter
    def weight(self, value: mx.array) -> None:
        self.base_layer.weight = value

    @property
    def bias(self) -> mx.array:
        return self.base_layer.bias

    @bias.setter
    def bias(self, value: mx.array) -> None:
        self.base_layer.bias = value

    def set_mapping(self, punica_wrapper: PunicaWrapperMLX) -> None:
        self.punica_wrapper = punica_wrapper

    def prepare_lora_weights(
        self, slot: int, lora_a: mx.array, lora_b: mx.array
    ) -> tuple[mx.array, mx.array]:
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(
                f"LoRA weight shape mismatch for slot {slot}: A and B must be "
                f"2-D, got A.ndim={lora_a.ndim}, B.ndim={lora_b.ndim} "
                "(expected A=(rank, in), B=(out, rank))"
            )
        rank, in_ = int(lora_a.shape[0]), int(lora_a.shape[1])
        out, b_rank = int(lora_b.shape[0]), int(lora_b.shape[1])
        if b_rank != rank:
            raise ValueError(
                f"LoRA weight shape mismatch for slot {slot}: A rank {rank} "
                f"(A=({rank},{in_})) does not match B rank {b_rank} "
                f"(B=({out},{b_rank})); A=(rank, in), B=(out, rank)"
            )
        if (in_, out) != (self.input_size, self.output_size):
            raise ValueError(
                f"LoRA weight shape mismatch for slot {slot}: A=({rank},{in_}), "
                f"B=({out},{rank}); expected in={self.input_size}, out={self.output_size}"
            )
        if rank > self.max_lora_rank:
            raise ValueError(
                f"LoRA rank {rank} exceeds max_lora_rank {self.max_lora_rank}"
            )
        a = mx.zeros_like(self.lora_a_stacked[slot])
        b = mx.zeros_like(self.lora_b_stacked[slot])
        a[:rank, :] = lora_a.astype(a.dtype)
        b[:, :rank] = lora_b.astype(b.dtype)
        return a, b

    def set_lora(self, slot: int, lora_a: mx.array, lora_b: mx.array) -> None:
        a, b = self.prepare_lora_weights(slot, lora_a, lora_b)
        self.set_prepared_lora(slot, a, b)

    def set_prepared_lora(self, slot: int, lora_a: mx.array, lora_b: mx.array) -> None:
        self.lora_a_stacked[slot], self.lora_b_stacked[slot] = lora_a, lora_b

    def reset_lora(self, slot: int) -> None:
        self.lora_a_stacked[slot] = mx.zeros_like(self.lora_a_stacked[slot])
        self.lora_b_stacked[slot] = mx.zeros_like(self.lora_b_stacked[slot])

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base_layer(x)
        if self.punica_wrapper is None or self.punica_wrapper.no_lora:
            return y

        # Punica expects (n_tokens, dim); collapse leading dims if needed.
        shape = y.shape
        x2, y2 = (
            (x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]))
            if x.ndim > 2
            else (x, y)
        )
        out = self.punica_wrapper.add_lora_linear(
            y2, x2, self.lora_a_stacked, self.lora_b_stacked, scale=1.0
        )
        return out if out is y else out.reshape(shape)


class MLXQuantizedLinearWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: Any,
        max_loras: int,
        max_lora_rank: int,
        dtype: mx.Dtype,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.max_loras, self.max_lora_rank = max_loras, max_lora_rank
        self.input_size, self.output_size = _infer_qlora_dims(base_layer)
        # +1 trailing slot is the null slot (see punica_wrapper).
        slots = max_loras + 1
        self.lora_a_stacked = mx.zeros((slots, max_lora_rank, self.input_size), dtype)
        self.lora_b_stacked = mx.zeros((slots, self.output_size, max_lora_rank), dtype)
        self.punica_wrapper: PunicaWrapperMLX | None = None

    @property
    def weight(self) -> mx.array:
        return self.base_layer.weight

    @property
    def bias(self) -> mx.array:
        return self.base_layer.bias

    @property
    def in_features(self) -> int:
        return self.input_size

    @property
    def out_features(self) -> int:
        return self.output_size

    def set_mapping(self, punica_wrapper: PunicaWrapperMLX) -> None:
        self.punica_wrapper = punica_wrapper

    def prepare_lora_weights(
        self, slot: int, lora_a: mx.array, lora_b: mx.array
    ) -> tuple[mx.array, mx.array]:
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(
                f"QLoRA weight shape mismatch for slot {slot}: A and B must be "
                f"2-D, got A.ndim={lora_a.ndim}, B.ndim={lora_b.ndim} "
                "(expected A=(rank, in), B=(out, rank))"
            )
        rank, in_ = int(lora_a.shape[0]), int(lora_a.shape[1])
        out, b_rank = int(lora_b.shape[0]), int(lora_b.shape[1])
        if b_rank != rank:
            raise ValueError(
                f"QLoRA weight shape mismatch for slot {slot}: A rank {rank} "
                f"(A=({rank},{in_})) does not match B rank {b_rank} "
                f"(B=({out},{b_rank})); A=(rank, in), B=(out, rank)"
            )
        if (in_, out) != (self.input_size, self.output_size):
            raise ValueError(
                f"QLoRA weight shape mismatch for slot {slot}: A=({rank},{in_}), "
                f"B=({out},{rank}); expected in={self.input_size}, out={self.output_size}"
            )
        if rank > self.max_lora_rank:
            raise ValueError(
                f"QLoRA rank {rank} exceeds max_lora_rank {self.max_lora_rank}"
            )
        a = mx.zeros_like(self.lora_a_stacked[slot])
        b = mx.zeros_like(self.lora_b_stacked[slot])
        a[:rank, :] = lora_a.astype(a.dtype)
        b[:, :rank] = lora_b.astype(b.dtype)
        return a, b

    def set_lora(self, slot: int, lora_a: mx.array, lora_b: mx.array) -> None:
        a, b = self.prepare_lora_weights(slot, lora_a, lora_b)
        self.set_prepared_lora(slot, a, b)

    def set_prepared_lora(self, slot: int, lora_a: mx.array, lora_b: mx.array) -> None:
        self.lora_a_stacked[slot], self.lora_b_stacked[slot] = lora_a, lora_b

    def reset_lora(self, slot: int) -> None:
        self.lora_a_stacked[slot] = mx.zeros_like(self.lora_a_stacked[slot])
        self.lora_b_stacked[slot] = mx.zeros_like(self.lora_b_stacked[slot])

    def __call__(self, x: mx.array) -> mx.array:
        # Quantized base forward — runs at the layer's native precision.
        y = self.base_layer(x)
        if self.punica_wrapper is None or self.punica_wrapper.no_lora:
            return y

        # Run the LoRA delta in the adapter dtype regardless of the quantized
        # base output dtype, then cast back so downstream layers are unaffected.
        lora_dtype = self.lora_a_stacked.dtype
        shape = y.shape
        x_flat = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        y_flat = y.reshape(-1, y.shape[-1]) if y.ndim > 2 else y
        out = self.punica_wrapper.add_lora_linear(
            y_flat.astype(lora_dtype),
            x_flat.astype(lora_dtype),
            self.lora_a_stacked,
            self.lora_b_stacked,
            scale=1.0,
        )
        out = out.astype(y.dtype)
        return out if out.shape == shape else out.reshape(shape)
