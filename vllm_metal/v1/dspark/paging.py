"""Where the drafter's context lives, and how much of it there is.

Pure policy: which backend a configuration selects, the pool's page size, and how many
pages a pool needs. It holds no Metal handles so the memory planner can ask these
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
    "pool_blocks",
]


def paged_context_enabled(config: DSparkConfig) -> bool:
    """Whether this drafter's context will live in the paged pool.

    Asked by the memory planner before the proposer exists, and by the proposer when it
    builds the context. Both must agree: a planner that sized for one backend while the
    proposer built the other would reserve the wrong thing.
    """
    return bool(envs.VLLM_METAL_DSPARK_PAGED_CONTEXT) and (
        config.attn_head_dim in KERNEL_HEAD_SIZES
    )


def pool_blocks(max_contexts: int, max_context_tokens: int, draft_block: int) -> int:
    """Pages for a pool that can still house the arena's worst case.

    The arena reserved `max_contexts * max_context_tokens` up front because every slot
    was sized for the whole model length. The pool holds the same worst case but hands
    pages out as contexts grow, so a server that never reaches that length never touches
    most of them and one request's unused tail is available to another. The extra page is
    the padding sink that ragged block tables point at.
    """
    per_context = (
        max_context_tokens + draft_block + PAGED_BLOCK_SIZE - 1
    ) // PAGED_BLOCK_SIZE
    return max_contexts * per_context + 1
