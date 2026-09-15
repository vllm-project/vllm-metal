# SPDX-License-Identifier: Apache-2.0
"""Conservative, enforced storage/workspace bounds for standalone DSpark.

This is a request-local reservation, not an autoregressive scheduler KV group.
The target cache planner subtracts the whole reservation before sizing its pool.
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

    @property
    def context_bytes(self) -> int:
        return self.max_contexts * self.max_context_tokens * self.kv_bytes_per_token

    @property
    def reserve_bytes(self) -> int:
        return self.context_bytes + self.capture_bytes + self.workspace_bytes

    @classmethod
    def build(
        cls,
        config: DSparkConfig,
        *,
        itemsize: int,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
    ) -> DSparkMemoryPlan:
        if min(max_num_seqs, max_model_len, max_num_batched_tokens) <= 0:
            raise ValueError(
                "DSpark memory planning requires positive scheduler limits"
            )
        if itemsize not in (2, 4):
            raise ValueError(
                "DSpark context requires two- or four-byte floating values"
            )
        rows = min(max_num_seqs, MAX_CONTEXTS)
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
        # Reserve a complete copy-on-write buffer set, a padded batch and the
        # context+block attention inputs. This covers lazy graph overlap without
        # depending on a particular MLX buffer-reuse optimization.
        copies = 3 * context
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
        return cls(
            rows,
            length,
            tokens,
            kv_bytes,
            capture,
            copies + ingest + draft + scores + logits + KERNEL_RESERVE_BYTES,
        )
