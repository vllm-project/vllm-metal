# SPDX-License-Identifier: Apache-2.0
"""MLX PunicaWrapper — route LoRA deltas by adapter slot.

Rows are grouped by adapter, so each step runs one LoRA matmul pair per
active adapter rather than gathering a copy of the weights for every row.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from vllm.lora.layers import LoRAMapping


class PunicaWrapperMLX:
    def __init__(self, max_num_batched_tokens: int, max_batches: int, max_loras: int):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_batches = max_batches
        self.max_loras = max_loras
        self._contiguous_runs: tuple[tuple[int | None, int, int], ...] = ()
        self._token_indices_by_slot: tuple[tuple[int, mx.array], ...] = ()
        self._num_tokens = 0
        self._no_lora = True

    @property
    def no_lora(self) -> bool:
        return self._no_lora

    def update_metadata(
        self, mapping: LoRAMapping, lora_index_to_id: list[int | None]
    ) -> None:
        slot_of = {aid: i for i, aid in enumerate(lora_index_to_id) if aid is not None}
        self._num_tokens = len(mapping.index_mapping)

        runs: list[tuple[int | None, int, int]] = []
        active_slots: set[int] = set()
        run_slot: int | None = None
        run_start = 0
        for token_index, adapter_id in enumerate(mapping.index_mapping):
            slot = slot_of.get(adapter_id)
            if slot is not None:
                active_slots.add(slot)
            if token_index == 0:
                run_slot = slot
                continue
            if slot != run_slot:
                runs.append((run_slot, run_start, token_index))
                run_slot = slot
                run_start = token_index
        if mapping.index_mapping:
            runs.append((run_slot, run_start, len(mapping.index_mapping)))

        lora_run_count = sum(1 for slot, _, _ in runs if slot is not None)
        use_contiguous_runs = bool(active_slots) and lora_run_count <= len(active_slots)
        if use_contiguous_runs:
            self._contiguous_runs = tuple(runs)
            self._token_indices_by_slot = ()
        else:
            self._contiguous_runs = ()
            token_indices_by_slot: dict[int, list[int]] = {}
            for token_index, adapter_id in enumerate(mapping.index_mapping):
                slot = slot_of.get(adapter_id)
                if slot is not None:
                    token_indices_by_slot.setdefault(slot, []).append(token_index)
            self._token_indices_by_slot = tuple(
                (slot, mx.array(indices, dtype=mx.int32))
                for slot, indices in sorted(token_indices_by_slot.items())
            )
        self._no_lora = not active_slots

    def add_lora_linear(
        self,
        y: mx.array,
        x: mx.array,
        lora_a_stacked: mx.array,
        lora_b_stacked: mx.array,
        scale: float,
        lora_ranks: list[int] | None = None,
    ) -> mx.array:
        """Apply LoRA deltas once per active adapter slot."""
        return self._add_lora(
            y, x, lora_a_stacked, lora_b_stacked, scale, lora_ranks, _shrink_linear
        )

    def add_lora_embedding(
        self,
        y: mx.array,
        token_ids: mx.array,
        lora_a_stacked: mx.array,
        lora_b_stacked: mx.array,
        lora_ranks: list[int] | None = None,
    ) -> mx.array:
        """Apply embedding LoRA deltas, ``A.T[ids] @ B.T``, per adapter slot."""
        return self._add_lora(
            y,
            token_ids,
            lora_a_stacked,
            lora_b_stacked,
            1.0,
            lora_ranks,
            _shrink_embedding,
        )

    def _add_lora(
        self,
        y: mx.array,
        x: mx.array,
        lora_a_stacked: mx.array,
        lora_b_stacked: mx.array,
        scale: float,
        lora_ranks: list[int] | None,
        shrink: Callable[[mx.array, mx.array], mx.array],
    ) -> mx.array:
        if self._no_lora:
            return y
        if int(x.shape[0]) != self._num_tokens or int(y.shape[0]) != self._num_tokens:
            raise ValueError(
                "LoRA routing row count mismatch: "
                f"metadata={self._num_tokens}, x={x.shape[0]}, y={y.shape[0]}"
            )

        max_rank = int(lora_a_stacked.shape[1])
        ranks = lora_ranks if lora_ranks is not None else [max_rank] * self.max_loras

        if self._contiguous_runs:
            outputs: list[mx.array] = []
            has_lora_delta = False
            for slot, start, end in self._contiguous_runs:
                y_run = y[start:end]
                if slot is None:
                    outputs.append(y_run)
                    continue
                rank = ranks[slot]
                if rank == 0:
                    outputs.append(y_run)
                    continue
                lora_a = lora_a_stacked[slot, :rank]
                lora_b = lora_b_stacked[slot, :, :rank]
                x_run = x[start:end]
                delta = mx.matmul(shrink(x_run, lora_a), lora_b.T)
                if scale != 1.0:
                    delta = delta * scale
                if delta.dtype != y.dtype:
                    delta = delta.astype(y.dtype)
                has_lora_delta = True
                outputs.append(y_run + delta)
            if not has_lora_delta:
                return y
            return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)

        output = y
        for slot, token_indices in self._token_indices_by_slot:
            rank = ranks[slot]
            if rank == 0:
                continue
            lora_a = lora_a_stacked[slot, :rank]
            lora_b = lora_b_stacked[slot, :, :rank]
            x_for_slot = mx.take(x, token_indices, axis=0)
            delta = mx.matmul(shrink(x_for_slot, lora_a), lora_b.T)
            if scale != 1.0:
                delta = delta * scale
            if delta.dtype != y.dtype:
                delta = delta.astype(y.dtype)
            output = output.at[token_indices].add(delta)
        return output


def _shrink_linear(x: mx.array, lora_a: mx.array) -> mx.array:
    return mx.matmul(x, lora_a.T)


def _shrink_embedding(token_ids: mx.array, lora_a: mx.array) -> mx.array:
    return mx.take(lora_a, token_ids, axis=1).T


class EncoderPunicaWrapperMLX:
    """Route every row of an encoder call through one adapter slot.

    The runner encodes one request's features per call, so each row of a
    tower or connector projection belongs to the same adapter.
    """

    def __init__(self) -> None:
        self._slot: int | None = None

    @property
    def no_lora(self) -> bool:
        return self._slot is None

    @property
    def slot(self) -> int | None:
        return self._slot

    def set_slot(self, slot: int | None) -> None:
        self._slot = slot

    def add_lora_linear(
        self,
        y: mx.array,
        x: mx.array,
        lora_a_stacked: mx.array,
        lora_b_stacked: mx.array,
        scale: float,
        lora_ranks: list[int] | None = None,
    ) -> mx.array:
        slot = self._slot
        if slot is None:
            return y
        rank = int(lora_a_stacked.shape[1]) if lora_ranks is None else lora_ranks[slot]
        if rank == 0:
            return y
        lora_a = lora_a_stacked[slot, :rank]
        lora_b = lora_b_stacked[slot, :, :rank]
        delta = mx.matmul(mx.matmul(x, lora_a.T), lora_b.T)
        if scale != 1.0:
            delta = delta * scale
        if delta.dtype != y.dtype:
            delta = delta.astype(y.dtype)
        return y + delta
