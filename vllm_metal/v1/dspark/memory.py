# SPDX-License-Identifier: Apache-2.0
"""Conservative, enforced storage/workspace bounds for standalone DSpark.

This is a request-local reservation, not an autoregressive scheduler KV group.
The target cache planner subtracts the capture staging and the per-step workspace
before sizing its pool; the context arena is allocated when the drafter loads and is
already accounted in the profiled model memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import DSparkConfig

MAX_CONTEXTS = 32
CONTEXT_ALIGNMENT = 256
KERNEL_RESERVE_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class DSparkMemoryPlan:
    max_contexts: int
    max_context_tokens: int
    max_step_tokens: int
    kv_bytes_per_token: int
    capture_bytes: int
    workspace_bytes: int
    # Scratch positions per context slot for the drafted block's own K/V
    # (the arena stores them right after the committed context).
    block_size: int = 0

    @property
    def context_bytes(self) -> int:
        return (
            self.max_contexts
            * (self.max_context_tokens + self.block_size)
            * self.kv_bytes_per_token
        )

    @property
    def reserve_bytes(self) -> int:
        """Everything the drafter will hold: checked before the arena exists."""
        return self.context_bytes + self.capture_bytes + self.workspace_bytes

    @property
    def planning_reserve_bytes(self) -> int:
        """What the target cache planner subtracts once the drafter is loaded.

        The context arena is allocated at load, so by the time the target KV
        cache is sized it is already part of the measured model memory; only
        the capture staging and the per-step workspace remain to be reserved.
        """
        return self.capture_bytes + self.workspace_bytes

    @classmethod
    def build(
        cls,
        config: DSparkConfig,
        *,
        itemsize: int,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        max_contexts: int = MAX_CONTEXTS,
    ) -> DSparkMemoryPlan:
        if min(max_num_seqs, max_model_len, max_num_batched_tokens) <= 0:
            raise ValueError(
                "DSpark memory planning requires positive scheduler limits"
            )
        if max_contexts <= 0:
            raise ValueError("DSpark requires at least one draft context slot")
        if itemsize not in (2, 4):
            raise ValueError(
                "DSpark context requires two- or four-byte floating values"
            )
        rows = min(max_num_seqs, max_contexts)
        length = (
            (max_model_len + CONTEXT_ALIGNMENT - 1)
            // CONTEXT_ALIGNMENT
            * CONTEXT_ALIGNMENT
        )
        tokens = max_num_batched_tokens
        kv_width = config.n_kv_heads * config.attn_head_dim
        kv_bytes = 2 * config.num_hidden_layers * kv_width * itemsize
        context = rows * length * kv_bytes
        # Captured layer outputs and their concatenation can coexist. Use FP32
        # sizing even for a two-byte target, including target/draft dtype casts.
        capture = 2 * tokens * len(config.target_layer_ids) * config.hidden_size * 4
        # One transient copy of the context arena: an in-place update that
        # cannot reuse its buffer (a view still alive) rewrites the whole arena
        # once. Drafting attends rows in place, so no padded batch copy exists.
        copies = context
        ingest = (
            tokens
            * (4 * config.hidden_size + 4 * config.num_hidden_layers * kv_width)
            * 4
        )
        block = config.block_size
        draft = (
            rows
            * block
            * (
                8 * config.hidden_size
                + 3 * config.intermediate_size
                + 4 * config.num_attention_heads * config.attn_head_dim
                + 2 * config.num_hidden_layers * kv_width
            )
            * 4
        )
        scores = rows * config.num_attention_heads * block * (length + block) * 4
        # Base logits, sequential Markov correction and the corrected output.
        logits = 3 * rows * block * config.vocab_size * 4
        # Stochastic proposals keep one float32 distribution per drafted
        # position until verification, and build each position's rows through
        # scaled, sorted, masked, softmax and cumulative-sum temporaries.
        proposals = rows * (block + 6) * config.vocab_size * 4
        return cls(
            rows,
            length,
            tokens,
            kv_bytes,
            capture,
            copies
            + ingest
            + draft
            + scores
            + logits
            + proposals
            + KERNEL_RESERVE_BYTES,
            block,
        )
