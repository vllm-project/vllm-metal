# SPDX-License-Identifier: Apache-2.0
"""MLX PunicaWrapper — route LoRA deltas by adapter slot.

Prefill batches avoid per-token weight expansion; decode keeps the compact
batched path that works best for small token counts.
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
        self._contiguous_runs: tuple[tuple[int | None, int, int], ...] = ()
        self._token_indices_by_slot: tuple[tuple[int, mx.array], ...] = ()
        self._expanded_slot_indices: mx.array | None = None
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

        if not mapping.is_prefill:
            null_slot = self.max_loras
            token_slots = [
                slot_of.get(adapter_id, null_slot)
                for adapter_id in mapping.index_mapping
            ]
            self._expanded_slot_indices = mx.array(token_slots, dtype=mx.int32)
            self._contiguous_runs = ()
            self._token_indices_by_slot = ()
            self._no_lora = all(slot == null_slot for slot in token_slots)
            return

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
            self._expanded_slot_indices = None
            self._contiguous_runs = tuple(runs)
            self._token_indices_by_slot = ()
        else:
            self._expanded_slot_indices = None
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
        if self._no_lora:
            return y
        if int(x.shape[0]) != self._num_tokens or int(y.shape[0]) != self._num_tokens:
            raise ValueError(
                "LoRA routing row count mismatch: "
                f"metadata={self._num_tokens}, x={x.shape[0]}, y={y.shape[0]}"
            )

        if self._expanded_slot_indices is not None:
            lora_a = mx.take(lora_a_stacked, self._expanded_slot_indices, axis=0)
            lora_b = mx.take(lora_b_stacked, self._expanded_slot_indices, axis=0)
            delta = mx.matmul(lora_b, mx.matmul(lora_a, x[:, :, None])).squeeze(-1)
            if scale != 1.0:
                delta = delta * scale
            if delta.dtype != y.dtype:
                delta = delta.astype(y.dtype)
            return y + delta

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
                delta = mx.matmul(mx.matmul(x_run, lora_a.T), lora_b.T)
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
            delta = mx.matmul(mx.matmul(x_for_slot, lora_a.T), lora_b.T)
            if scale != 1.0:
                delta = delta * scale
            if delta.dtype != y.dtype:
                delta = delta.astype(y.dtype)
            output = output.at[token_indices].add(delta)
        return output
