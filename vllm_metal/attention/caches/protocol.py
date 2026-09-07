# SPDX-License-Identifier: Apache-2.0
"""State-pool operations shared by hybrid runtimes and state managers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import mlx.core as mx


class PagedStateCache(Protocol):
    """Own state slabs without exposing a family's tensor layout or kernels."""

    max_seqs: int
    allocated_seqs: int

    @property
    def num_state_pools(self) -> int: ...

    def ensure_capacity(self, num_seqs: int) -> None: ...

    def reset_slot(self, slot: int) -> None: ...

    def set_layer_layout(
        self, group_ordinals: list[int], pool_ordinals: list[int]
    ) -> None: ...

    def layers_for_group_ordinal(self, ordinal: int) -> list[int]: ...

    def copy_slots(
        self, src_ids: list[int], dst_ids: list[int], layer_indices: list[int]
    ) -> None: ...

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None: ...

    def zero_slots(self, slot_ids: list[int], layer_indices: list[int]) -> None: ...

    def apply_pending_states(self) -> None: ...

    def updated_state_arrays(self) -> list[mx.array]: ...
