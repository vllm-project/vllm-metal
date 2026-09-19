# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS expert parallelism: count-split experts, replicated routing."""

from __future__ import annotations

import os


def expert_partition(world_size: int, num_experts: int) -> list[int]:
    """Per-rank expert counts from VLLM_METAL_EXPERT_PARTITION or an even split."""
    raw = os.getenv("VLLM_METAL_EXPERT_PARTITION")
    if raw is None:
        if num_experts % world_size:
            raise ValueError(
                f"{num_experts} experts do not divide evenly across {world_size} "
                "ranks; set VLLM_METAL_EXPERT_PARTITION."
            )
        return [num_experts // world_size] * world_size
    try:
        counts = [int(part) for part in raw.split(",")]
    except ValueError:
        counts = []
    if (
        len(counts) != world_size
        or sum(counts) != num_experts
        or any(count <= 0 for count in counts)
    ):
        raise ValueError(
            f"VLLM_METAL_EXPERT_PARTITION must be {world_size} positive integers "
            f"summing to {num_experts}; got {raw!r}."
        )
    return counts
