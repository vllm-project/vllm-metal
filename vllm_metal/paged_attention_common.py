# SPDX-License-Identifier: Apache-2.0
"""Paged attention shared utilities — context, prepare functions, and helpers.

Provides the thread-local ``PagedAttentionContext`` and ``OffsetCache`` used by
both the Metal kernel paged attention backend and the model runner.

Usage:
    1. Before each forward pass call ``prepare_unified()``
    2. Run ``model(input_ids, cache=offset_caches)`` as normal
    3. The attention wrapper reads ``get_context()`` for paged metadata
    4. Call ``clear_context()`` after the forward pass
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

# ---------------------------------------------------------------------------
# Global context (thread-local)
# ---------------------------------------------------------------------------

# Thread-local storage used to pass per-request metadata (slot_mapping,
# block_tables, etc.) to attention wrappers buried inside the model.
# We cannot add extra arguments to the mlx_lm forward signature, so
# instead: prepare_unified() stashes context here before the
# forward pass, each attention wrapper reads it via get_context(), and
# clear_context() cleans up afterwards.
_thread_local = threading.local()


@dataclass
class PagedAttentionContext:
    """Context set before each forward pass, read by patched attention.

    All forward passes use the varlen kernel with ``cu_seqlens`` to handle
    variable-length subsequences (both prefill and decode tokens packed
    into a single flat sequence).
    """

    slot_mapping: list[int]
    block_tables: list[list[int]] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)
    # Per-segment RoPE offsets: 0 for fresh prefill, seq_len for decode.
    offsets: list[int] = field(default_factory=list)
    # Cumulative sequence length array: [0, len0, len0+len1, ...]
    # (length = num_requests + 1).
    cu_seqlens: list[int] | None = None
    # GDN state pool slot mapping: request batch position → stable slot ID.
    # Populated by model_runner for hybrid models; None for non-hybrid.
    gdn_slot_mapping: list[int] | None = None

    # Lazy MLX views — converted once per forward pass on first access, then
    # reused across all attention layers. Eliminates the N-layer redundancy of
    # re-converting the same Python lists on every attention wrapper call.
    _slot_mapping_mx: mx.array | None = field(default=None, init=False, repr=False)
    _block_tables_mx: mx.array | None = field(default=None, init=False, repr=False)
    _context_lens_mx: mx.array | None = field(default=None, init=False, repr=False)
    _cu_seqlens_mx: mx.array | None = field(default=None, init=False, repr=False)
    _gdn_slot_mapping_mx: mx.array | None = field(
        default=None, init=False, repr=False
    )

    @property
    def slot_mapping_mx(self) -> mx.array:
        if self._slot_mapping_mx is None:
            self._slot_mapping_mx = mx.array(self.slot_mapping, dtype=mx.int64)
        return self._slot_mapping_mx

    @property
    def block_tables_mx(self) -> mx.array:
        """Dense padded [num_reqs, max_blocks] int32 tensor.

        Rows shorter than max_blocks are right-padded with 0. Callers that need
        a per-request length should read from ``context_lens`` or derive from
        ``block_tables``.
        """
        if self._block_tables_mx is None:
            if not self.block_tables:
                self._block_tables_mx = mx.zeros((0, 0), dtype=mx.int32)
            else:
                max_blocks = max(len(bt) for bt in self.block_tables)
                padded = [
                    bt + [0] * (max_blocks - len(bt)) for bt in self.block_tables
                ]
                self._block_tables_mx = mx.array(padded, dtype=mx.int32)
        return self._block_tables_mx

    @property
    def context_lens_mx(self) -> mx.array:
        if self._context_lens_mx is None:
            self._context_lens_mx = mx.array(self.context_lens, dtype=mx.int32)
        return self._context_lens_mx

    @property
    def cu_seqlens_mx(self) -> mx.array:
        if self._cu_seqlens_mx is None:
            self._cu_seqlens_mx = mx.array(self.cu_seqlens, dtype=mx.int32)
        return self._cu_seqlens_mx

    @property
    def gdn_slot_mapping_mx(self) -> mx.array | None:
        if self.gdn_slot_mapping is None:
            return None
        if self._gdn_slot_mapping_mx is None:
            self._gdn_slot_mapping_mx = mx.array(
                self.gdn_slot_mapping, dtype=mx.int32
            )
        return self._gdn_slot_mapping_mx

    def _compute_slot_mapping_mx(self, block_size: int) -> None:
        """Populate ``_slot_mapping_mx`` via vectorized MLX ops.

        Upstream reference:
        ``vllm/v1/worker/gpu/block_table.py::_compute_slot_mappings_kernel``
        (Triton, ~line 213). Upstream runs the same math on-GPU per step;
        we express it via MLX broadcast/indexing so MLX's scheduler runs
        it on Metal. Avoids the per-token Python loop that
        ``prepare_unified`` would otherwise execute for every prefill token.

        MLX lacks ``searchsorted`` as of 0.31.x, so ``per_token_req`` is
        derived via a broadcast compare + sum instead of the single
        ``tl.load`` upstream does from ``idx_mapping``. Equivalent result.
        """
        if (
            self.cu_seqlens is None
            or not self.block_tables
            or self.cu_seqlens[-1] == 0
        ):
            self._slot_mapping_mx = mx.zeros((0,), dtype=mx.int64)
            return

        total_tokens = self.cu_seqlens[-1]
        cu = self.cu_seqlens_mx                         # [num_reqs + 1]
        offs = mx.array(self.offsets, dtype=mx.int32)   # [num_reqs]
        bt = self.block_tables_mx                       # [num_reqs, max_blocks]

        positions = mx.arange(total_tokens, dtype=mx.int32)
        cu_starts = cu[:-1]                             # [num_reqs]
        # per_token_req: index of the request each packed token belongs to.
        # Upstream reads this directly from idx_mapping; we count how many
        # cu_seqlens boundaries each position has crossed.
        per_token_req = (
            (positions[:, None] >= cu_starts[None, :]).sum(axis=1) - 1
        )
        # Absolute sequence position within the request:
        #   batch_pos − req_start + rope_offset
        local_pos = (
            positions - cu_starts[per_token_req] + offs[per_token_req]
        )
        block_indices = local_pos // block_size
        block_offsets = local_pos % block_size
        block_numbers = bt[per_token_req, block_indices]
        self._slot_mapping_mx = (
            block_numbers * block_size + block_offsets
        ).astype(mx.int64)


def set_context(ctx: PagedAttentionContext) -> None:
    _thread_local.paged_ctx = ctx


def get_context() -> PagedAttentionContext | None:
    return getattr(_thread_local, "paged_ctx", None)


def clear_context() -> None:
    _thread_local.paged_ctx = None


# ---------------------------------------------------------------------------
# OffsetCache — thin shim so the model's create_attention_mask / RoPE work
# ---------------------------------------------------------------------------


class OffsetCache:
    """Fake KV cache that stores no data — only satisfies mlx_lm's protocol.

    The mlx_lm model expects ``cache=`` to be a list of cache objects (one
    per layer) and reads two things from each:
    - ``cache.offset``: RoPE position index for the current token(s).
    - ``cache.make_mask(N)``: attention mask (``"causal"`` for multi-token
      prefill, ``None`` for single-token decode where no mask is needed).

    With paged attention, real K/V lives in the MPS paged cache (managed by
    the attention wrapper), NOT in these objects.  OffsetCache is a shim so
    that mlx_lm's RoPE and masking logic work without changes.

    Note: during batched decode the model runner passes a single shared
    ``OffsetCache(max_offset)`` per layer.  The actual per-request RoPE
    offsets come from ``ctx.offsets`` inside the attention wrapper, not
    from this object.
    """

    def __init__(self, offset: int) -> None:
        self.offset = offset

    # --- satisfy KVCache protocol expected by create_attention_mask ---------

    def make_mask(
        self,
        N: int,  # noqa: N803
        return_array: bool = False,
        window_size: int | None = None,
    ) -> Any:
        if N == 1:
            return None
        if return_array:
            return create_causal_mask(N, self.offset, window_size=window_size)
        return "causal"


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------


def find_layers(model: Any) -> list[Any]:
    """Find transformer layers in an mlx_lm / mlx-vlm model.

    Supports model structures like:
        model.language_model.model.layers   (VLMs)
        model.model.layers
        model.layers
    """
    # Unwrap VLM wrapper (e.g. LLaVA, Pixtral via mlx-vlm)
    root = getattr(model, "language_model", model)
    # Try root.model.layers (Qwen3 Model wrapper)
    layers_container = getattr(root, "model", root)
    if hasattr(layers_container, "layers"):
        return layers_container.layers
    elif hasattr(root, "layers"):
        return root.layers
    else:
        raise ValueError(
            f"Cannot find transformer layers in model of type {type(model)}"
        )


# Attribute names to probe on each layer, in priority order.
_ATTN_ATTR_NAMES = ("self_attn", "linear_attn", "attention")


def find_attn_attr(layer: Any) -> str | None:
    """Return the attention attribute name for a single layer, or None."""
    for name in _ATTN_ATTR_NAMES:
        if hasattr(layer, name):
            return name
    return None


# ---------------------------------------------------------------------------
# Prepare functions — called before each forward pass
# ---------------------------------------------------------------------------


def prepare_unified(
    decode_requests: list[tuple[list[int], int]],
    prefill_requests: list[tuple[list[int], int, int]],
    block_size: int,
) -> None:
    """Compute metadata for a unified prefill + decode forward pass.

    Packs decode tokens (1 per request) followed by prefill tokens into a
    single flattened sequence.  ``cu_seqlens`` marks request boundaries so
    the varlen kernel handles both decode (length-1) and prefill (length-N)
    subsequences in one dispatch.

    Args:
        decode_requests: list of ``(block_ids, seq_len)`` for decode requests.
            ``seq_len`` = tokens already cached before this step.
        prefill_requests: list of ``(block_ids, num_tokens, start_pos)`` for
            prefill.  ``start_pos`` is the position of the first token in this
            chunk (0 for a fresh prefill, >0 for continuation chunks).
        block_size: tokens per KV cache block.
    """
    cu_seqlens: list[int] = [0]
    block_tables: list[list[int]] = []
    context_lens: list[int] = []
    offsets: list[int] = []

    # Decode requests first (1 token each).
    for block_ids, seq_len in decode_requests:
        cu_seqlens.append(cu_seqlens[-1] + 1)
        block_tables.append(block_ids)
        context_lens.append(seq_len + 1)  # including new token
        offsets.append(seq_len)  # RoPE position

    # Prefill requests (variable tokens each, starting at start_pos).
    for block_ids, num_tokens, start_pos in prefill_requests:
        cu_seqlens.append(cu_seqlens[-1] + num_tokens)
        block_tables.append(block_ids)
        context_lens.append(start_pos + num_tokens)
        offsets.append(start_pos)

    # slot_mapping is computed via vectorized MLX ops in
    # ``_compute_slot_mapping_mx`` — see upstream
    # ``vllm/v1/worker/gpu/block_table.py::_compute_slot_mappings_kernel``.
    # The Python list stays empty; consumers read ``ctx.slot_mapping_mx``.
    ctx = PagedAttentionContext(
        slot_mapping=[],
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens=cu_seqlens,
        offsets=offsets,
    )
    ctx._compute_slot_mapping_mx(block_size)
    set_context(ctx)
