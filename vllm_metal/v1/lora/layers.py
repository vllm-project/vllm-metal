# SPDX-License-Identifier: Apache-2.0
"""MLX layer wrappers for LoRA and QLoRA."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU, SwitchLinear

if TYPE_CHECKING:
    from .punica_wrapper import EncoderPunicaWrapperMLX, PunicaWrapperMLX

SWITCH_GLU_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


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


def can_wrap_switch_glu(module: Any) -> bool:
    return isinstance(module, SwitchGLU) and all(
        isinstance(getattr(module, p), (SwitchLinear, QuantizedSwitchLinear))
        for p in SWITCH_GLU_PROJECTIONS
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
        self.punica_wrapper: PunicaWrapperMLX | EncoderPunicaWrapperMLX | None = None

    def set_mapping(
        self, punica_wrapper: PunicaWrapperMLX | EncoderPunicaWrapperMLX
    ) -> None:
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


class MLXSwitchLinearWithLoRA(nn.Module):
    """Per-expert LoRA slots for one ``SwitchGLU`` projection.

    Flattened, slot ``s`` of expert ``e`` is row ``e * max_loras + s``, so rows
    sorted by expert are also sorted for the LoRA gathers. The projection
    itself stays in the ``SwitchGLU``.
    """

    _lora_name = "MoE expert LoRA"

    def __init__(
        self,
        base_layer: SwitchLinear | QuantizedSwitchLinear,
        max_loras: int,
        max_lora_rank: int,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.num_experts = int(base_layer.num_experts)
        self.input_size = int(base_layer.input_dims)
        self.output_size = int(base_layer.output_dims)
        self.max_loras, self.max_lora_rank = max_loras, max_lora_rank
        slots = (self.num_experts, max_loras)
        self.lora_a_stacked = mx.zeros((*slots, max_lora_rank, self.input_size), dtype)
        self.lora_b_stacked = mx.zeros((*slots, self.output_size, max_lora_rank), dtype)
        object.__setattr__(self, "_lora_ranks", [0] * max_loras)

    def has_lora(self, slots: Iterable[int]) -> bool:
        return any(self._lora_ranks[slot] for slot in slots)

    def lora_delta(self, x: mx.array, keys: mx.array, sorted_indices: bool) -> mx.array:
        rows = self.num_experts * self.max_loras
        lora_a = self.lora_a_stacked.reshape(rows, self.max_lora_rank, self.input_size)
        lora_b = self.lora_b_stacked.reshape(rows, self.output_size, self.max_lora_rank)
        if x.dtype != lora_a.dtype:
            x = x.astype(lora_a.dtype)
        h = mx.gather_mm(
            x, lora_a.swapaxes(-1, -2), rhs_indices=keys, sorted_indices=sorted_indices
        )
        return mx.gather_mm(
            h, lora_b.swapaxes(-1, -2), rhs_indices=keys, sorted_indices=sorted_indices
        )

    def prepare_lora_weights(
        self, slot: int, lora_a: mx.array, lora_b: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Pad per-expert A ``(E, rank, in)`` and B ``(E, out, rank)`` to the slot."""
        if lora_a.ndim != 3 or lora_b.ndim != 3:
            raise ValueError(
                f"{self._lora_name} weight shape mismatch for slot {slot}: "
                f"expected A=(experts, rank, in) and B=(experts, out, rank), got "
                f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
            )
        experts, rank, in_ = (int(d) for d in lora_a.shape)
        b_experts, out, b_rank = (int(d) for d in lora_b.shape)
        if (experts, b_experts, in_, out, b_rank) != (
            self.num_experts,
            self.num_experts,
            self.input_size,
            self.output_size,
            rank,
        ):
            raise ValueError(
                f"{self._lora_name} weight shape mismatch for slot {slot}: "
                f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}; expected "
                f"experts={self.num_experts}, in={self.input_size}, "
                f"out={self.output_size} and matching ranks"
            )
        if rank > self.max_lora_rank:
            raise ValueError(
                f"{self._lora_name} rank {rank} exceeds "
                f"max_lora_rank {self.max_lora_rank}"
            )
        a = mx.zeros_like(self.lora_a_stacked[:, slot])
        b = mx.zeros_like(self.lora_b_stacked[:, slot])
        a[:, :rank, :] = lora_a.astype(a.dtype)
        b[:, :, :rank] = lora_b.astype(b.dtype)
        return a, b

    def set_prepared_lora(
        self, slot: int, lora_a: mx.array, lora_b: mx.array, *, rank: int
    ) -> None:
        self.lora_a_stacked[:, slot], self.lora_b_stacked[:, slot] = lora_a, lora_b
        self._lora_ranks[slot] = rank

    def reset_lora(self, slot: int) -> None:
        self.lora_a_stacked[:, slot] = mx.zeros_like(self.lora_a_stacked[:, slot])
        self.lora_b_stacked[:, slot] = mx.zeros_like(self.lora_b_stacked[:, slot])
        self._lora_ranks[slot] = 0


class MLXSwitchGLUWithLoRA(nn.Module):
    """``SwitchGLU`` that adds each token's adapter delta to its experts.

    Without an active expert adapter it runs mlx_lm's ``SwitchGLU`` unchanged.
    Otherwise rows are sorted by ``expert * max_loras + slot``: the base
    gathers see sorted experts and the LoRA gathers see sorted slot rows.
    """

    def __init__(
        self,
        base_layer: SwitchGLU,
        lora: dict[str, MLXSwitchLinearWithLoRA],
        punica_wrapper: PunicaWrapperMLX,
        max_loras: int,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora = lora
        self.max_loras = max_loras
        self.punica_wrapper = punica_wrapper

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        base, punica = self.base_layer, self.punica_wrapper
        if punica.no_lora or not any(
            proj.has_lora(punica.active_slots) for proj in self.lora.values()
        ):
            return base(x, indices)

        slots = punica.token_slots
        if slots.size * indices.shape[-1] != indices.size:
            raise ValueError(
                "MoE LoRA routing row count mismatch: "
                f"metadata={slots.size}, indices={tuple(indices.shape)}"
            )
        slots = slots.reshape(indices.shape[:-1])
        keys = indices  # one slot: the key is the expert
        if self.max_loras > 1:
            keys = indices.astype(mx.int32) * self.max_loras
            keys = keys + mx.maximum(slots, 0)[..., None]
        adapted = slots >= 0 if punica.has_unadapted_tokens else None
        x = mx.expand_dims(x, (-2, -3))
        # Same threshold as SwitchGLU, so the base gathers see the same layout.
        do_sort = indices.size >= 64
        experts = indices
        inv_order = None
        if do_sort:
            order = mx.argsort(keys.flatten())
            inv_order = mx.argsort(order)
            tokens = order // indices.shape[-1]
            x = x.flatten(0, -3)[tokens]
            keys = keys.flatten()[order]
            experts = keys // self.max_loras if self.max_loras > 1 else keys
            if adapted is not None:
                adapted = adapted.flatten()[tokens][:, None, None]
        elif adapted is not None:
            adapted = adapted[..., None, None, None]

        def project(name: str, h: mx.array) -> mx.array:
            y = getattr(base, name)(h, experts, sorted_indices=do_sort)
            if (lora := self.lora.get(name)) is not None:
                delta = lora.lora_delta(h, keys, do_sort)
                if adapted is not None:
                    delta = mx.where(adapted, delta, 0)
                y = y + delta.astype(y.dtype)
            return y

        h = base.activation(project("up_proj", x), project("gate_proj", x))
        y = project("down_proj", h)
        if inv_order is not None:
            y = mx.unflatten(y[inv_order], 0, indices.shape)
        return y.squeeze(-2)
