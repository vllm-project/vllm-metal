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
import numpy as np
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


class PagedPrepBuffers:
    """Private working storage for ``prepare_unified`` — do not access fields
    from outside this module.

    The dense ``block_table`` is sized to the worst case once at runner
    init; per-step the active slice ``[:nreqs, :max_blocks]`` is a
    rectangle with short rows right-padded by 0, and kernels mask off the
    tail via ``context_lens``. The other four arrays are 1D contiguous.
    """

    __slots__ = (
        "slot_mapping",
        "block_table",
        "cu_seqlens",
        "context_lens",
        "offsets",
    )

    def __init__(
        self,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
    ) -> None:
        self.slot_mapping = np.zeros(max_num_batched_tokens, dtype=np.int64)
        self.block_table = np.zeros(
            (max_num_reqs, max_num_blocks_per_req), dtype=np.int32
        )
        self.cu_seqlens = np.zeros(max_num_reqs + 1, dtype=np.int32)
        self.context_lens = np.zeros(max_num_reqs, dtype=np.int32)
        self.offsets = np.zeros(max_num_reqs, dtype=np.int32)


@dataclass
class PagedAttentionContext:
    """Context set before each forward pass, read by patched attention.

    All forward passes use the varlen kernel with ``cu_seqlens`` to handle
    variable-length subsequences (both prefill and decode tokens packed
    into a single flat sequence).

    Each numpy field is a view into a ``PagedPrepBuffers`` slice for the
    current step's active extent. The view does not own memory; the
    underlying buffer outlives the context.
    """

    slot_mapping: np.ndarray  # int64, shape [total_tokens]
    block_tables: np.ndarray  # int32, shape [num_reqs, max_blocks_this_step]
    context_lens: np.ndarray  # int32, shape [num_reqs]
    # Per-segment RoPE offsets: 0 for fresh prefill, seq_len for decode.
    offsets: np.ndarray  # int32, shape [num_reqs]
    # Cumulative sequence length array: [0, len0, len0+len1, ...]
    # (length = num_requests + 1).
    cu_seqlens: np.ndarray  # int32, shape [num_reqs + 1]
    # GDN state pool slot mapping: request batch position → stable slot ID.
    # Populated by model_runner for hybrid models; None for non-hybrid.
    # Tiny (one int per request) so a Python list is fine.
    gdn_slot_mapping: list[int] | None = None

    _slot_mapping_mx: mx.array | None = field(default=None, init=False, repr=False)
    _block_tables_mx: mx.array | None = field(default=None, init=False, repr=False)
    _context_lens_mx: mx.array | None = field(default=None, init=False, repr=False)
    _cu_seqlens_mx: mx.array | None = field(default=None, init=False, repr=False)
    _gdn_slot_mapping_mx: mx.array | None = field(default=None, init=False, repr=False)
    _cu_seqlens_list: list[int] | None = field(default=None, init=False, repr=False)
    _offsets_list: list[int] | None = field(default=None, init=False, repr=False)

    @property
    def slot_mapping_mx(self) -> mx.array:
        if self._slot_mapping_mx is None:
            self._slot_mapping_mx = mx.array(self.slot_mapping)
        return self._slot_mapping_mx

    @property
    def block_tables_mx(self) -> mx.array:
        if self._block_tables_mx is None:
            self._block_tables_mx = mx.array(self.block_tables)
        return self._block_tables_mx

    @property
    def context_lens_mx(self) -> mx.array:
        if self._context_lens_mx is None:
            self._context_lens_mx = mx.array(self.context_lens)
        return self._context_lens_mx

    @property
    def cu_seqlens_mx(self) -> mx.array:
        if self._cu_seqlens_mx is None:
            self._cu_seqlens_mx = mx.array(self.cu_seqlens)
        return self._cu_seqlens_mx

    @property
    def gdn_slot_mapping_mx(self) -> mx.array | None:
        if self.gdn_slot_mapping is None:
            return None
        if self._gdn_slot_mapping_mx is None:
            self._gdn_slot_mapping_mx = mx.array(self.gdn_slot_mapping, dtype=mx.int32)
        return self._gdn_slot_mapping_mx

    @property
    def cu_seqlens_list(self) -> list[int]:
        """Python ints for per-request loops in attention wrappers.

        MLX slice indices reject ``np.int32``; the per-layer per-request
        ``int(cu_seqlens[i])`` coercion is hot enough at high batch/layer
        counts to matter, so cache one ``tolist()`` per forward pass.
        """
        if self._cu_seqlens_list is None:
            self._cu_seqlens_list = self.cu_seqlens.tolist()
        return self._cu_seqlens_list

    @property
    def offsets_list(self) -> list[int]:
        if self._offsets_list is None:
            self._offsets_list = self.offsets.tolist()
        return self._offsets_list

    @classmethod
    def from_lists(
        cls,
        *,
        slot_mapping: list[int],
        block_tables: list[list[int]],
        context_lens: list[int],
        cu_seqlens: list[int],
        offsets: list[int],
        gdn_slot_mapping: list[int] | None = None,
    ) -> PagedAttentionContext:
        """Build a context from raw Python lists — for tests and ad-hoc use.

        Mirrors the dense-padding ``prepare_unified`` applies to
        ``block_tables``: shorter rows get trailing zeros so the result is a
        contiguous rectangle.
        """
        if not block_tables:
            bt = np.zeros((0, 0), dtype=np.int32)
        else:
            max_blocks = max(len(b) for b in block_tables)
            padded = [b + [0] * (max_blocks - len(b)) for b in block_tables]
            bt = np.array(padded, dtype=np.int32)
        return cls(
            slot_mapping=np.array(slot_mapping, dtype=np.int64),
            block_tables=bt,
            context_lens=np.array(context_lens, dtype=np.int32),
            cu_seqlens=np.array(cu_seqlens, dtype=np.int32),
            offsets=np.array(offsets, dtype=np.int32),
            gdn_slot_mapping=gdn_slot_mapping,
        )


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
    buffers: PagedPrepBuffers | None = None,
) -> None:
    """Compute metadata for a unified prefill + decode forward pass.

    Packs decode tokens (1 per request) followed by prefill tokens into a
    single flattened sequence.  ``cu_seqlens`` marks request boundaries so
    the varlen kernel handles both decode (length-1) and prefill (length-N)
    subsequences in one dispatch.

    Writes are slice-assignments into ``buffers`` (persistent across steps).
    The constructed context holds *views* of the active extent; the buffer
    must outlive the context. When ``buffers`` is ``None`` an ad-hoc
    instance is allocated, sized to this call's inputs — used by tests; in
    production the runner passes a persistent buffer.

    Args:
        decode_requests: list of ``(block_ids, seq_len)`` for decode requests.
            ``seq_len`` = tokens already cached before this step.
        prefill_requests: list of ``(block_ids, num_tokens, start_pos)`` for
            prefill.  ``start_pos`` is the position of the first token in this
            chunk (0 for a fresh prefill, >0 for continuation chunks).
        block_size: tokens per KV cache block.
        buffers: persistent storage; auto-allocated when ``None``.
    """
    num_decode = len(decode_requests)
    num_reqs = num_decode + len(prefill_requests)

    total_tokens = num_decode + sum(num for _, num, _ in prefill_requests)
    max_blocks = 0
    for block_ids, _ in decode_requests:
        if len(block_ids) > max_blocks:
            max_blocks = len(block_ids)
    for block_ids, _, _ in prefill_requests:
        if len(block_ids) > max_blocks:
            max_blocks = len(block_ids)

    if buffers is None:
        buffers = PagedPrepBuffers(
            max_num_reqs=max(num_reqs, 1),
            max_num_blocks_per_req=max(max_blocks, 1),
            max_num_batched_tokens=max(total_tokens, 1),
        )

    # Buffer rows are reused across steps. A long row from step N would
    # leak stale tail entries into step N+1's shorter row through the
    # dense slice — clear the active rectangle once per step.
    if num_reqs > 0 and max_blocks > 0:
        buffers.block_table[:num_reqs, :max_blocks] = 0

    buffers.cu_seqlens[0] = 0
    cu = 0
    req_idx = 0

    for block_ids, seq_len in decode_requests:
        new_pos = seq_len
        block_idx = block_ids[new_pos // block_size]
        slot = block_idx * block_size + (new_pos % block_size)
        buffers.slot_mapping[cu] = slot
        n_blocks = len(block_ids)
        if n_blocks:
            buffers.block_table[req_idx, :n_blocks] = block_ids
        buffers.context_lens[req_idx] = seq_len + 1
        buffers.offsets[req_idx] = seq_len
        cu += 1
        buffers.cu_seqlens[req_idx + 1] = cu
        req_idx += 1

    for block_ids, num_tokens, start_pos in prefill_requests:
        if num_tokens > 0:
            positions = np.arange(start_pos, start_pos + num_tokens, dtype=np.int64)
            block_idx = positions // block_size
            offset_in_block = positions % block_size
            block_arr = np.asarray(block_ids, dtype=np.int64)
            slots = block_arr[block_idx] * block_size + offset_in_block
            buffers.slot_mapping[cu : cu + num_tokens] = slots
        n_blocks = len(block_ids)
        if n_blocks:
            buffers.block_table[req_idx, :n_blocks] = block_ids
        buffers.context_lens[req_idx] = start_pos + num_tokens
        buffers.offsets[req_idx] = start_pos
        cu += num_tokens
        buffers.cu_seqlens[req_idx + 1] = cu
        req_idx += 1

    set_context(
        PagedAttentionContext(
            slot_mapping=buffers.slot_mapping[:cu],
            block_tables=buffers.block_table[:num_reqs, :max_blocks],
            context_lens=buffers.context_lens[:num_reqs],
            offsets=buffers.offsets[:num_reqs],
            cu_seqlens=buffers.cu_seqlens[: num_reqs + 1],
        )
    )
