# SPDX-License-Identifier: Apache-2.0
"""Scaled dot-product attention (SDPA) on Metal.

Supports MHA, GQA, and MQA as variants of the same kernel — the head ratio
between ``n_heads`` (queries) and ``n_kv_heads`` (keys/values) is handled
transparently by the Metal paged attention kernel.

Handles models whose attention module exposes:
- ``q_proj``, ``k_proj``, ``o_proj`` linear projections (``v_proj`` optional —
  see K-eq-V variant below)
- ``rope`` or ``rotary_emb`` for rotary position embeddings
- ``n_heads``, ``n_kv_heads`` head counts
- Optionally ``q_norm``, ``k_norm``, ``v_norm`` per-head RMSNorms

Gemma4 variants (see :func:`prepare_sdpa_qkv`):
- **YOCO**: later layers reuse K/V from a reference layer via ``shared_kv``.
- **K-eq-V**: 26B/31B drop ``v_proj`` and reuse ``keys`` as ``values``.
- **Variable head_dim**: sliding vs. full-attention layers use different
  head_dim; Q/K/V are zero-padded up to the cache's allocated head_dim
  via :func:`pad_qkv_to_cache_head_dim`.

Covers: Qwen3, Qwen3.5, Llama, Mistral, Gemma, Gemma4, and other
RoPE-based transformer architectures.

All operations use MLX arrays end-to-end — no PyTorch MPS bridge.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.metal import get_ops
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.metal_kernel_backend.packed_prefill_compat import (
    apply_packed_rope,
)
from vllm_metal.paged_attention_common import PagedAttentionContext

# === Metal kernel block-size support ===
# The paged attention Metal kernel is template-instantiated for these block
# sizes only.  Sorted descending so _pick_kernel_block_size selects the
# largest valid divisor first, minimising the block-table expansion ratio.
_KERNEL_BLOCK_SIZES = (32, 16, 8)


def is_sdpa(module: nn.Module) -> bool:
    """Return True if *module* is an SDPA attention layer (MHA, GQA, or MQA).

    Requires ``q_proj`` / ``k_proj`` / ``o_proj``, plus EITHER ``v_proj``
    OR the explicit ``use_k_eq_v = True`` opt-in.  The latter admits
    Gemma4 26B / 31B full-attention layers which share the K projection
    for values and never define ``v_proj`` (``prepare_sdpa_qkv`` handles
    that branch symmetrically).

    Keeping this classifier tight matters because
    :meth:`HybridPagedAttentionBackend.patch_model` uses ``is_sdpa`` as
    the dispatch predicate — loose matching would send arbitrary Q/K/O
    modules through the SDPA path.
    """
    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and hasattr(module, "o_proj")
        and (hasattr(module, "v_proj") or getattr(module, "use_k_eq_v", False))
    )


# === Block-size translation helpers ===


def _pick_kernel_block_size(cache_block_size: int) -> int:
    """Pick the largest kernel-supported block size that divides evenly."""
    for kbs in _KERNEL_BLOCK_SIZES:
        if cache_block_size % kbs == 0:
            return kbs
    raise ValueError(
        f"Cache block_size={cache_block_size} is not divisible by any "
        f"supported kernel block size {_KERNEL_BLOCK_SIZES}. "
        "Adjust VLLM_METAL_BLOCK_SIZE or the hybrid page alignment."
    )


def _build_block_tables(
    raw_block_tables: list[list[int]],
    cache_block_size: int,
) -> tuple[mx.array, int]:
    """Build kernel-compatible block tables, translating if necessary.

    When ``cache_block_size`` exceeds the kernel's compiled block sizes,
    each vLLM block ``b`` is expanded into ``ratio`` kernel blocks
    ``[b*ratio, b*ratio+ratio)``.  The cache is reshaped later to
    match (zero-copy).

    Returns:
        (block_tables, kernel_block_size)
    """
    if not raw_block_tables:
        return mx.zeros((0, 0), dtype=mx.int32), cache_block_size

    if cache_block_size in _KERNEL_BLOCK_SIZES:
        # Fast path — no translation needed.
        max_blocks = max(len(bt) for bt in raw_block_tables)
        padded = [bt + [0] * (max_blocks - len(bt)) for bt in raw_block_tables]
        return mx.array(padded, dtype=mx.int32), cache_block_size

    # Hybrid path — translate large block_size to a kernel-compatible one.
    # Vectorized: each vLLM block b → [b*ratio, b*ratio+1, …, b*ratio+ratio-1].
    kernel_bs = _pick_kernel_block_size(cache_block_size)
    ratio = cache_block_size // kernel_bs

    max_blocks = max(len(bt) for bt in raw_block_tables)
    padded = [bt + [0] * (max_blocks - len(bt)) for bt in raw_block_tables]
    bt_arr = mx.array(padded, dtype=mx.int32)  # [num_seqs, max_blocks]
    offsets = mx.arange(ratio, dtype=mx.int32)  # [ratio]
    # [num_seqs, max_blocks, 1] * ratio + [1, 1, ratio] → [num_seqs, max_blocks, ratio]
    expanded = (bt_arr[:, :, None] * ratio + offsets[None, None, :]).reshape(
        bt_arr.shape[0], -1
    )
    return expanded, kernel_bs


# === Q/K/V preparation (YOCO, K-eq-V, v_norm variants) ===


def prepare_sdpa_qkv(
    inner: nn.Module,
    x: mx.array,
    ctx: PagedAttentionContext,
    n_heads: int,
    n_kv_heads: int,
    shared_kv: tuple[mx.array, mx.array] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array | None, tuple[mx.array, mx.array]]:
    """Project ``x`` into Q/K/V with norms, RoPE and Gemma4 variants.

    Handles three Gemma4-specific branches:

    - **YOCO** (``shared_kv`` given): reuse K/V from a prior layer; skip
      projection and only apply Q norm + RoPE.
    - **K-eq-V** (no ``inner.v_proj``): 26B/31B checkpoints share the
      projection so ``values`` references the same tensor as ``keys``.
    - **v_norm** (``inner.v_norm`` present): apply per-head RMSNorm to
      values alongside q_norm and k_norm.

    Args:
        inner: mlx_lm Attention module (or compatible).
        x: Input hidden states shaped ``(B, L, D)``.
        ctx: Paged attention context (supplies ``cu_seqlens`` / offsets
            for per-request RoPE).
        n_heads: Query head count.
        n_kv_heads: K/V head count.
        shared_kv: Optional ``(keys, values)`` from a reference layer,
            already normed and RoPE'd, in ``(B, H, L, head_dim)`` layout.

    Returns:
        Tuple ``(queries, keys, values, gate, kv_for_sharing)``:

        - ``queries``, ``keys``, ``values``: ``(B, H, L, head_dim)`` tensors
          ready for the Metal kernel.
        - ``gate``: optional gate tensor for gated attention (Qwen3.5
          Qwen3Next style), else ``None``.
        - ``kv_for_sharing``: the post-norm+RoPE ``(keys, values)`` pair so
          the caller can forward them to the next YOCO layer.

    Raises:
        NotImplementedError: If ``inner`` has neither ``rope`` nor
            ``rotary_emb`` (only RoPE-based models are supported).
    """
    B, L, _ = x.shape  # noqa: N806

    # Projections + reshape.  Qwen3.5 uses gated q_proj (2x head_dim).
    q_proj_out = inner.q_proj(x)
    gate: mx.array | None = None
    # head_dim has two architectural sources in our supported models:
    #   - self.head_dim instance attr (gemma*, llama, mistral, qwen3_5+)
    #   - k_proj.weight.shape (qwen3, qwen3_moe never set self.head_dim)
    # KV-shared Gemma 4 layers have head_dim but no k_proj. If neither
    # is present, raise — silently propagating a wrong head_dim would
    # corrupt downstream kernel shapes.
    if hasattr(inner, "head_dim"):
        head_dim = inner.head_dim
    elif hasattr(inner, "k_proj"):
        head_dim = inner.k_proj.weight.shape[0] // n_kv_heads
    else:
        raise AttributeError(
            f"Cannot determine head_dim for "
            f"{type(inner).__module__}.{type(inner).__name__}: "
            "neither 'head_dim' nor 'k_proj' attribute present"
        )
    q_full_head = q_proj_out.shape[-1] // n_heads
    if q_full_head == 2 * head_dim:
        q_reshaped = q_proj_out.reshape(B, L, n_heads, q_full_head)
        queries, gate = mx.split(q_reshaped, 2, axis=-1)
        gate = gate.reshape(B, L, -1)
    else:
        queries = q_proj_out.reshape(B, L, n_heads, -1)

    if shared_kv is not None:
        # YOCO: reuse K/V from a reference layer.  Q still needs norm + RoPE.
        keys, values = shared_kv
        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        queries = queries.transpose(0, 2, 1, 3)
        if not hasattr(inner, "rope") and not hasattr(inner, "rotary_emb"):
            raise NotImplementedError(
                f"Attention module {type(inner).__name__} does not have a "
                "'rope' or 'rotary_emb' attribute."
            )
        queries, _ = apply_packed_rope(
            inner,
            queries,
            keys,
            ctx.cu_seqlens,
            offsets=ctx.offsets if ctx.offsets else None,
            apply_keys=False,
        )
    else:
        keys = inner.k_proj(x).reshape(B, L, n_kv_heads, -1)
        # K-eq-V variant (Gemma4 26B/31B): no v_proj, values = keys.
        if hasattr(inner, "v_proj"):
            values = inner.v_proj(x).reshape(B, L, n_kv_heads, -1)
        else:
            values = keys

        # Per-head RMSNorm (Qwen3, Qwen3.5, Gemma4).
        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)
        if hasattr(inner, "v_norm"):
            values = inner.v_norm(values)

        # Transpose to (B, H, L, head_dim).
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if not hasattr(inner, "rope") and not hasattr(inner, "rotary_emb"):
            raise NotImplementedError(
                f"Attention module {type(inner).__name__} does not have a "
                "'rope' or 'rotary_emb' attribute."
            )
        queries, keys = apply_packed_rope(
            inner,
            queries,
            keys,
            ctx.cu_seqlens,
            offsets=ctx.offsets if ctx.offsets else None,
        )

    kv_for_sharing = (keys, values)
    return queries, keys, values, gate, kv_for_sharing


# === Variable head_dim helpers (Gemma4) ===


def pad_qkv_to_cache_head_dim(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    head_dim: int,
    cache_head_dim: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Zero-pad Q/K/V on the last axis up to ``cache_head_dim``.

    Variable head_dim models (e.g. Gemma4 sliding=256, full=512) allocate
    the paged KV cache at the max head_dim.  Layers with smaller head_dim
    are padded so scatter writes and the kernel both operate at the cache's
    native head_dim.  Zero-padded positions do not affect QK dot products
    or V aggregation.  No-op when ``head_dim == cache_head_dim``.

    Args:
        queries, keys, values: Tensors shaped ``(B, H, L, head_dim)``.
        head_dim: Current layer's head_dim.
        cache_head_dim: Cache's allocated head_dim (the target).

    Returns:
        Padded ``(queries, keys, values)``.

    Raises:
        ValueError: If ``head_dim > cache_head_dim`` (unsupported), or if
            ``queries`` / ``keys`` / ``values`` do not share the same last
            dimension (caller invariant).
    """
    if not (queries.shape[-1] == keys.shape[-1] == values.shape[-1] == head_dim):
        raise ValueError(
            "Q/K/V last-dim mismatch: "
            f"q={queries.shape[-1]}, k={keys.shape[-1]}, "
            f"v={values.shape[-1]}, head_dim={head_dim}"
        )
    if head_dim == cache_head_dim:
        return queries, keys, values
    if head_dim > cache_head_dim:
        raise ValueError(
            f"head_dim={head_dim} exceeds cache_head_dim={cache_head_dim}; "
            f"cache must be sized for the largest per-layer head_dim."
        )
    pad_spec = [(0, 0), (0, 0), (0, 0), (0, cache_head_dim - head_dim)]
    return (
        mx.pad(queries, pad_spec),
        mx.pad(keys, pad_spec),
        mx.pad(values, pad_spec),
    )


def truncate_padded_output(
    out: mx.array,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    cache_head_dim: int,
    actual_head_dim: int,
) -> mx.array:
    """Reshape kernel output and strip padding back to ``actual_head_dim``.

    Inverse of :func:`pad_qkv_to_cache_head_dim`: before the output goes to
    ``o_proj``, we slice off the zero-padded tail so the trailing
    projection sees the layer's real head_dim.  No-op when the layer was
    never padded (``actual_head_dim == cache_head_dim``).

    Args:
        out: Kernel output shaped ``(seq_len, n_heads, cache_head_dim)``.
        batch_size: Batch size (typically 1 for packed sequences).
        seq_len: Total tokens in the packed sequence.
        n_heads: Number of query heads.
        cache_head_dim: Head_dim the kernel operated on.
        actual_head_dim: Layer's original head_dim before padding.

    Returns:
        Flat output shaped ``(batch_size, seq_len, n_heads * actual_head_dim)``.
    """
    if actual_head_dim == cache_head_dim:
        return out.reshape(batch_size, seq_len, n_heads * cache_head_dim)
    out = out.reshape(batch_size, seq_len, n_heads, cache_head_dim)[
        ..., :actual_head_dim
    ]
    return out.reshape(batch_size, seq_len, n_heads * actual_head_dim)


# === SDPA forward ===


def sdpa_forward(
    inner: nn.Module,
    x: mx.array,
    ctx: PagedAttentionContext,
    kv_cache: MetalPagedKVCache,
    layer_idx: int,
    shared_kv: tuple[mx.array, mx.array] | None = None,
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    """Full SDPA forward pass: project → norm → RoPE → Metal kernel.

    Handles MHA, GQA, and MQA uniformly — the head ratio between
    query and KV heads is passed to the Metal kernel which handles
    the broadcast internally.

    Returns:
        Tuple of (output, kv_pair) where kv_pair is (keys, values)
        after norm + RoPE, for YOCO KV sharing across layers.
    """
    B, L, _ = x.shape  # noqa: N806

    # Resolve head counts — mlx_lm uses different attribute names:
    #   Qwen3/Llama/Gemma/Gemma4: n_heads, n_kv_heads
    #   Qwen3.5 (Qwen3Next):      num_attention_heads, num_key_value_heads
    n_heads = getattr(inner, "n_heads", None) or inner.num_attention_heads
    n_kv_heads = getattr(inner, "n_kv_heads", None) or inner.num_key_value_heads

    queries, keys, values, gate, kv_for_sharing = prepare_sdpa_qkv(
        inner, x, ctx, n_heads, n_kv_heads, shared_kv
    )

    # --- Metal kernel dispatch ---
    n_heads = queries.shape[1]
    head_dim = queries.shape[3]

    # Per-layer cache properties: shape and sliding window.
    cache_kv_heads = kv_cache.kv_heads_per_layer[layer_idx]
    cache_head_dim = kv_cache.head_dim_per_layer[layer_idx]
    layer_sliding_window = kv_cache.sliding_window_per_layer[layer_idx]
    actual_head_dim = head_dim
    queries, keys, values = pad_qkv_to_cache_head_dim(
        queries, keys, values, head_dim, cache_head_dim
    )
    head_dim = cache_head_dim

    # Reshape to 3D: (1, heads, L, hd) → (L, heads, hd)
    q_3d = mx.contiguous(queries[0].transpose(1, 0, 2).astype(kv_cache.dtype))
    k_3d = mx.contiguous(keys[0].transpose(1, 0, 2).astype(kv_cache.dtype))
    v_3d = mx.contiguous(values[0].transpose(1, 0, 2).astype(kv_cache.dtype))

    slot_mapping = mx.array(ctx.slot_mapping, dtype=mx.int64)
    seq_lens = mx.array(ctx.context_lens, dtype=mx.int32)
    cu_seqlens_q = mx.array(ctx.cu_seqlens, dtype=mx.int32)
    max_seq_len = max(ctx.context_lens)

    # --- Block tables (with hybrid block-size translation) ---
    # vLLM may inflate block_size (e.g. 544) to align attention pages with
    # mamba pages in hybrid models.  The Metal kernel only supports small
    # block sizes (8, 16, 32).  _build_block_tables handles the translation:
    # it expands each vLLM block into multiple kernel blocks and returns the
    # kernel-compatible block_size.  The cache is reshaped to match (zero-copy).
    block_tables, kernel_block_size = _build_block_tables(
        ctx.block_tables, kv_cache.block_size
    )

    if shared_kv is not None:
        # YOCO shared layer: the reference layer already scattered the
        # authoritative K/V (and, under TurboQuant, the scale/zero caches
        # too).  Skip the write to avoid (a) redundant compute and (b)
        # silent byte-drift if this layer's raw K/V differs marginally
        # from the reference's — re-encoding would produce non-identical
        # packed bytes.
        new_k_cache = kv_cache.key_caches[layer_idx]
        new_v_cache = kv_cache.value_caches[layer_idx]
        if kv_cache.turboquant:
            new_key_scale_cache = kv_cache.key_scale_caches[layer_idx]
            new_value_scale_cache = kv_cache.value_scale_caches[layer_idx]
            new_key_zero_cache = kv_cache.key_zero_caches[layer_idx]
    elif kv_cache.turboquant:
        # --- TurboQuant cache write: fused Metal encode + scatter ---
        # Single dispatch replaces Python turbo_quant_encode + 5 MLX scatters.
        # Supports the full QUANT_PARAMS matrix: signed q8_0/int8 at k_bits=8
        # and unsigned uint8/q5_0/q4_0/int4/uint4/int2/uint2 at k_bits in
        # {2, 3, 4, 5, 8}.
        from vllm_metal.metal_kernel_backend.turboquant import (
            QUANT_PARAMS,
            get_v_centroids,
        )

        v_centroids = get_v_centroids(kv_cache.v_bits)
        k_signed = bool(QUANT_PARAMS[kv_cache.k_quant]["signed"])
        # tq_encode is a proper MLX Primitive: it returns five NEW array
        # objects that alias the input cache buffers in place but carry
        # fresh graph provenance pointing at the primitive.  The subsequent
        # paged_attention_primitive (separate command buffer) depends on
        # these outputs through the lazy graph, which is what lets MLX
        # insert the fence that serialises reader-after-writer.  Using the
        # original cache arrays here instead would silently race on the
        # first real forward pass (EngineCore crash).  We must also rebind
        # kv_cache.<cache>[layer_idx] to the new arrays so the next decode
        # step's tq_encode input reads through this primitive's output.
        (
            new_k_cache,
            new_v_cache,
            new_key_scale_cache,
            new_value_scale_cache,
            new_key_zero_cache,
        ) = get_ops().tq_encode(
            k_3d,
            v_3d,
            kv_cache.key_caches[layer_idx],
            kv_cache.value_caches[layer_idx],
            kv_cache.key_scale_caches[layer_idx],
            kv_cache.value_scale_caches[layer_idx],
            kv_cache.key_zero_caches[layer_idx],
            slot_mapping,
            v_centroids,
            kv_cache.v_bits,
            kv_cache.k_bits,
            k_signed,
        )
        kv_cache.key_caches[layer_idx] = new_k_cache
        kv_cache.value_caches[layer_idx] = new_v_cache
        kv_cache.key_scale_caches[layer_idx] = new_key_scale_cache
        kv_cache.value_scale_caches[layer_idx] = new_value_scale_cache
        kv_cache.key_zero_caches[layer_idx] = new_key_zero_cache
    else:
        flat_k = kv_cache.key_caches[layer_idx].reshape(-1, cache_kv_heads, head_dim)
        flat_k[slot_mapping] = k_3d
        new_k_cache = flat_k.reshape(kv_cache.key_caches[layer_idx].shape)

        flat_v = kv_cache.value_caches[layer_idx].reshape(-1, cache_kv_heads, head_dim)
        flat_v[slot_mapping] = v_3d
        new_v_cache = flat_v.reshape(kv_cache.value_caches[layer_idx].shape)

        # Rebind so next layer / decode step uses the updated cache
        kv_cache.key_caches[layer_idx] = new_k_cache
        kv_cache.value_caches[layer_idx] = new_v_cache

    # --- Attention: paged attention primitive (read-only, fully lazy) ---
    # No per-layer eval or sync.  The primitive participates in MLX's lazy
    # graph and is evaluated by the model runner at the end of the forward
    # pass.  Fence-based synchronisation across command buffer boundaries
    # works correctly because eval_gpu skips add_temporary (which would
    # remove buffers from the encoder's fence tracking).
    #
    # When block-size translation is active (hybrid models), reshape the
    # cache so the kernel sees kernel_block_size-token blocks.  This is a
    # zero-copy view over the same physical memory.
    kernel_k_cache = new_k_cache
    kernel_v_cache = new_v_cache
    if kernel_block_size != kv_cache.block_size:
        # Use the cache's actual last-axis size rather than the logical
        # ``head_dim``.  Under TurboQuant the K/V caches are stored in
        # packed form (``packed_head_dim = packed_dim(head_dim, bits)``)
        # which differs from ``head_dim`` for all bitwidths except 8-bit K.
        # Mirrors the ``sg = ...shape[-1]`` idiom used for the scale/zero
        # reshape below.
        kernel_k_cache = new_k_cache.reshape(
            -1, kernel_block_size, cache_kv_heads, new_k_cache.shape[-1]
        )
        kernel_v_cache = new_v_cache.reshape(
            -1, kernel_block_size, cache_kv_heads, new_v_cache.shape[-1]
        )

    ops = get_ops()
    out = mx.array(0)
    if kv_cache.turboquant:
        # Reshape scale/zero caches for kernel block size
        kernel_key_scale = new_key_scale_cache
        kernel_value_scale = new_value_scale_cache
        kernel_key_zero = new_key_zero_cache
        if kernel_block_size != kv_cache.block_size:
            sg = new_key_scale_cache.shape[-1]
            kernel_key_scale = new_key_scale_cache.reshape(
                -1, kernel_block_size, kv_cache.num_kv_heads, sg
            )
            kernel_value_scale = new_value_scale_cache.reshape(
                -1, kernel_block_size, kv_cache.num_kv_heads, sg
            )
            kernel_key_zero = new_key_zero_cache.reshape(
                -1, kernel_block_size, kv_cache.num_kv_heads, sg
            )
        # Get Lloyd-Max centroids for V quantization (lazily computed, cached)
        from vllm_metal.metal_kernel_backend.turboquant import get_v_centroids

        v_centroids = get_v_centroids(kv_cache.v_bits)
        ops.paged_attention_primitive(
            q_3d,
            kernel_k_cache,
            kernel_v_cache,
            cache_kv_heads,
            inner.scale,
            0.0,  # softcap (0 = disabled)
            block_tables,
            seq_lens,
            cu_seqlens_q,
            kernel_block_size,
            max_seq_len,
            layer_sliding_window,
            out,
            key_scale_cache=kernel_key_scale,
            value_scale_cache=kernel_value_scale,
            key_zero_cache=kernel_key_zero,
            v_centroids=v_centroids,
            use_turboquant=True,
            quant_type=kv_cache.k_quant,
            v_bits=kv_cache.v_bits,
        )
    else:
        ops.paged_attention_primitive(
            q_3d,
            kernel_k_cache,
            kernel_v_cache,
            cache_kv_heads,
            inner.scale,
            0.0,  # softcap (0 = disabled)
            block_tables,
            seq_lens,
            cu_seqlens_q,
            kernel_block_size,
            max_seq_len,
            layer_sliding_window,
            out,
        )

    # Reshape + strip padding back to actual head_dim before o_proj.
    out = truncate_padded_output(out, B, L, n_heads, cache_head_dim, actual_head_dim)
    if gate is not None:
        out = out * mx.sigmoid(gate)
    return inner.o_proj(out), kv_for_sharing
