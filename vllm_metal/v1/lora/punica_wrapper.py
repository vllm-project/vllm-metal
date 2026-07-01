# SPDX-License-Identifier: Apache-2.0
"""MLX PunicaWrapper — gather + batched matmul for the rank-r LoRA delta.

Slot ``max_loras`` is the permanently-zero null slot: no-LoRA tokens are
remapped to it so the kernel runs branch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from vllm.lora.layers import LoRAMapping


class PunicaWrapperMLX:
    def __init__(self, max_num_batched_tokens: int, max_batches: int, max_loras: int):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_batches = max_batches
        self.max_loras = max_loras
        self._token_slot_indices: mx.array | None = None
        self._sampler_slot_indices: mx.array | None = None
        self._no_lora = True
        self.is_prefill = False

    @property
    def no_lora(self) -> bool:
        return self._no_lora

    @property
    def token_slot_indices(self) -> mx.array | None:
        return self._token_slot_indices

    @property
    def sampler_slot_indices(self) -> mx.array | None:
        return self._sampler_slot_indices

    def update_metadata(
        self, mapping: LoRAMapping, lora_index_to_id: list[int | None]
    ) -> None:
        null = self.max_loras
        slot_of = {aid: i for i, aid in enumerate(lora_index_to_id) if aid is not None}
        tok = [slot_of.get(t, null) for t in mapping.index_mapping]
        prm = [slot_of.get(t, null) for t in mapping.prompt_mapping]
        self._token_slot_indices = mx.array(tok, dtype=mx.int32)
        self._sampler_slot_indices = mx.array(prm, dtype=mx.int32)
        self._no_lora = all(s == null for s in tok)
        self.is_prefill = mapping.is_prefill

    def reset(self) -> None:
        self._token_slot_indices = self._sampler_slot_indices = None
        self._no_lora, self.is_prefill = True, False

    def add_lora_linear(
        self,
        y: mx.array,
        x: mx.array,
        lora_a_stacked: mx.array,
        lora_b_stacked: mx.array,
        scale: float,
    ) -> mx.array:
        """``y += (B[idx] @ A[idx] @ x) * scale``  per token."""
        if self._no_lora or self._token_slot_indices is None:
            return y
        idx = self._token_slot_indices
        a, b = (
            mx.take(lora_a_stacked, idx, axis=0),
            mx.take(lora_b_stacked, idx, axis=0),
        )
        return y + mx.matmul(b, mx.matmul(a, x[:, :, None])).squeeze(-1) * scale
