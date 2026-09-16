"""Where the drafter's context lives, and how much of it there is.

Pure policy: which backend a configuration selects, and the pool's page size. It holds no Metal handles so the memory planner can ask these
questions before any shader is loaded, and so the planner and the proposer answer them
from one place instead of each carrying its own copy of the rule.
"""

from __future__ import annotations

from vllm_metal import envs

from .config import DSparkConfig

KERNEL_HEAD_SIZES = (64, 80, 96, 112, 128, 192, 256, 512)
"""Head sizes the paged attention kernel is instantiated for.

From the instantiate_paged_attention_heads macro in
vllm_metal/metal/kernels_v2/pagedattention.metal. A drafter outside this set cannot
use the pool; the caller keeps the private arena for it rather than failing at the
first drafting step with a missing Metal function.
"""

# Kernel page sizes are 8, 16 and 32. Sixteen matches the target cache's own page,
# which is what lets the drafter's context share the target's block machinery.
PAGED_BLOCK_SIZE = 16

__all__ = [
    "KERNEL_HEAD_SIZES",
    "PAGED_BLOCK_SIZE",
    "paged_context_enabled",
    "require_prefix_caching_off",
]


def paged_context_enabled(config: DSparkConfig) -> bool:
    """Whether this drafter's context will live in the paged pool.

    Asked once, by the memory planner: its answer is recorded as ``DSparkMemoryPlan.paged``
    and the proposer follows that rather than asking again, so the backend the planner
    sized for and the one the proposer builds cannot differ.
    """
    return bool(envs.VLLM_METAL_DSPARK_PAGED_CONTEXT) and (
        config.attn_head_dim in KERNEL_HEAD_SIZES
    )


def require_prefix_caching_off(enable_prefix_caching: bool) -> None:
    """Refuse a scheduler-owned drafter context under target prefix caching.

    The drafter's committed group is scheduler-hashed on token ids, like every group.
    A target prefix-cache hit skips the target forward for the matched positions, so no
    hidden states exist to build the drafter's K/V there -- yet the scheduler marks
    those pages computed, uniformly for every group. A separate draft model has no
    such hole because it forwards its own token ids. Until the drafter can recompute
    a hit prefix or opt its group out of hashing, the two cannot be combined, and that
    is said at startup rather than discovered as stale draft KV.
    """
    if enable_prefix_caching:
        raise ValueError(
            "VLLM_METAL_DSPARK_PAGED_CONTEXT keeps the drafter's context in a "
            "scheduler-owned KV-cache group, which target prefix caching would mark "
            "computed at positions the target never forwarded; pass "
            "--no-enable-prefix-caching, or leave the paged context off"
        )
