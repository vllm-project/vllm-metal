# SPDX-License-Identifier: Apache-2.0
"""Scaled dot-product attention (SDPA) on Metal.

Supports MHA, GQA, and MQA as variants of the same kernel — the head ratio
between ``n_heads`` (queries) and ``n_kv_heads`` (keys/values) is handled
transparently by the Metal paged attention kernel.

Handles models whose attention module exposes:
- ``q_proj``, ``k_proj``, ``o_proj`` / ``out_proj`` linear projections
  (``v_proj`` optional — see K-eq-V variant below)
- ``rope`` / ``rotary_emb`` for rotary position embeddings, or precomputed
  ``position_embeddings`` supplied by the caller
- ``n_heads``, ``n_kv_heads`` head counts
- Optionally ``q_norm``, ``k_norm``, ``v_norm`` RMSNorms
- Optionally ``g_proj`` (+ ``gating=True``) for Laguna-style per-head
  softplus attention-output gating (see :func:`apply_g_proj_gate`)

Gemma4 variants (see :func:`prepare_sdpa_qkv`):
- **YOCO**: later layers reuse K/V from a reference layer via ``shared_kv``.
- **K-eq-V**: 26B/31B drop ``v_proj`` and reuse ``keys`` as ``values``.
- **Variable head_dim**: sliding vs. full-attention layers use different
  head_dim; Q/K/V are zero-padded up to the cache's allocated head_dim
  via :func:`pad_qkv_to_cache_head_dim`.

Covers: Qwen3, Qwen3.5, Llama, Mistral, Gemma, Gemma4, Laguna, and other
RoPE-based transformer architectures.

All operations use MLX arrays end-to-end — no PyTorch MPS bridge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from vllm.logger import init_logger

from vllm_metal.attention.attention_contracts import (
    DEFAULT_ATTENTION_CONTRACT,
    AttentionContract,
    QKNormPlacement,
)
from vllm_metal.attention.block_tables import build_block_tables
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.turboquant_materialize import (
    materialize_turboquant_pages,
)
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.bidi_prefill import apply_bidirectional_segments
from vllm_metal.attention.impls.mm_prefix import (
    build_mm_prefix_rows,
    image_block_path,
)
from vllm_metal.attention.impls.turboquant_prefill import (
    _AttentionBatch,
    _turboquant_prefill_plan,
    _TurboQuantPrefillPlan,
    unsupported_reason,
)
from vllm_metal.attention.impls.varlen_rope_compat import (
    apply_attention_rope,
)
from vllm_metal.metal import get_ops, paged_attention_capabilities

logger = init_logger(__name__)


def _has_packed_qkv_sdpa_contract(module: nn.Module) -> bool:
    """Return True when *module* exposes the packed Phi-style SDPA contract."""
    has_rotary = hasattr(module, "rope") or hasattr(module, "rotary_emb")
    return (
        hasattr(module, "qkv_proj")
        and hasattr(module, "o_proj")
        and hasattr(module, "n_heads")
        and hasattr(module, "n_kv_heads")
        and hasattr(module, "head_dim")
        and hasattr(module, "scale")
        and has_rotary
    )


def _projection_out_features(proj: nn.Module) -> int:
    """Output feature count of an attention projection.

    Prefers an explicit ``out_features`` when the projection exposes one — e.g.
    a quantized wrapper that hides its packed weight behind a tensor and has no
    dense ``.weight`` — and falls back to the dense weight's row count
    otherwise. mlx ``nn.Linear`` and ``nn.QuantizedLinear`` carry no
    ``out_features`` attribute, so dense and AWQ projections keep the
    ``weight.shape[0]`` path unchanged.
    """
    out_features = getattr(proj, "out_features", None)
    if out_features is not None:
        return int(out_features)
    return proj.weight.shape[0]


def is_sdpa(module: nn.Module) -> bool:
    """Return True if *module* is an SDPA attention layer (MHA, GQA, or MQA).

    Accepts two contracts:

    - Split-projection SDPA: ``q_proj`` / ``k_proj`` / output projection
      (``o_proj``, ``dense`` in Phi, or ``out_proj`` in LFM2), plus
      EITHER ``v_proj`` OR the explicit ``use_k_eq_v = True`` opt-in.
      The latter admits Gemma4 26B / 31B full-attention layers which
      share the K projection for values and never define ``v_proj``
      (``prepare_sdpa_qkv`` handles that branch symmetrically).
    - Packed Phi-style SDPA: ``qkv_proj`` / ``o_proj`` plus the runtime
      metadata ``n_heads`` / ``n_kv_heads`` / ``head_dim`` / ``scale``
      and RoPE exposure via ``rope`` or ``rotary_emb``.

    Keeping this classifier tight matters because
    :meth:`HybridPagedAttentionRuntime.patch_model` uses ``is_sdpa`` as
    the dispatch predicate — loose matching would send arbitrary Q/K/O
    modules through the SDPA path.
    """
    if _has_packed_qkv_sdpa_contract(module):
        return True

    return (
        hasattr(module, "q_proj")
        and hasattr(module, "k_proj")
        and _output_projection(module) is not None
        and (hasattr(module, "v_proj") or getattr(module, "use_k_eq_v", False))
    )


# === Block-size translation helpers ===


@dataclass(eq=False)
class _KernelMetadata:
    """Kernel-format copies and mutable routing memo for one forward/group.

    ``eq=False``: the generated ``__eq__`` would compare mx arrays, which
    raises on ``bool()``; identity comparison is the only meaningful one.
    """

    slot_mapping: mx.array
    seq_lens: mx.array
    cu_seqlens_q: mx.array
    block_tables: mx.array
    block_size: int
    max_seq_len: int
    # Same forward/group lifetime as the existing kernel metadata. Only CPU
    # routing and gather indices are cached, never materialized K/V buffers.
    tq_prefill_plans: dict[tuple[int, ...], _TurboQuantPrefillPlan | None] = field(
        default_factory=dict
    )
    tq_prefill_workspace_bytes: int = 0


def _kernel_metadata(
    ctx: PagedAttentionContext,
    group_index: int | None,
    raw_slot_mapping: list[int],
    raw_block_tables: list[list[int]],
    cache_block_size: int,
) -> _KernelMetadata:
    """Kernel-format metadata for one KV group, built once per forward.

    The paged context is fixed for the duration of one forward pass and
    every layer of a KV group needs the same converted arrays, so the
    conversion runs on the group's first layer and is cached on the
    context; the remaining layers reuse it instead of re-serializing
    O(rows × blocks) Python lists per layer.  The context dies with the
    forward, so entries can never go stale.
    """
    key = (group_index, cache_block_size)
    meta = ctx.kernel_metadata_cache.get(key)
    if meta is None:
        block_tables, kernel_block_size = build_block_tables(
            raw_block_tables, cache_block_size
        )
        meta = _KernelMetadata(
            slot_mapping=mx.array(raw_slot_mapping, dtype=mx.int64),
            seq_lens=mx.array(ctx.context_lens, dtype=mx.int32),
            cu_seqlens_q=mx.array(ctx.cu_seqlens, dtype=mx.int32),
            block_tables=block_tables,
            block_size=kernel_block_size,
            max_seq_len=max(ctx.context_lens),
            tq_prefill_workspace_bytes=ctx.tq_prefill_workspace_bytes,
        )
        ctx.kernel_metadata_cache[key] = meta
    return meta


def _mm_prefix_rows(ctx: PagedAttentionContext) -> mx.array | None:
    """Per-row block ranges for the kernel, built once per forward (see context)."""
    if not ctx.mm_prefix_rows_built:
        assert ctx.segment_bidi_ranges is not None
        assert ctx.cu_seqlens is not None
        rows = build_mm_prefix_rows(
            ctx.cu_seqlens, ctx.context_lens, ctx.segment_bidi_ranges
        )
        if rows is not None:
            ctx.mm_prefix_row_count = int((rows[:, 0] >= 0).sum())
            ctx.mm_prefix_rows = mx.array(rows)
        ctx.mm_prefix_rows_built = True
    return ctx.mm_prefix_rows


def _named_norm(module: nn.Module, *names: str) -> nn.Module | None:
    """Return the first Q/K norm present on *module* among *names*.

    mlx_lm spells the Q/K norms differently per architecture:
    Qwen3/Qwen3.5/Gemma4/OLMo use ``q_norm``/``k_norm``, while Hunyuan
    (``hunyuan_v1_dense``) uses ``query_layernorm``/``key_layernorm``.
    Probing by name keeps the caller from silently skipping the norm on
    architectures that use the second spelling, which produces wrong
    output rather than an error.
    """
    for name in names:
        norm = getattr(module, name, None)
        if norm is not None:
            return norm
    return None


def _output_projection(module: nn.Module) -> nn.Module | None:
    """Return the attention-output projection on *module*.

    Nearly every mlx_lm SDPA architecture calls it ``o_proj``; Phi-1/1.5
    (``mlx_lm.models.phi``) spell theirs ``dense`` and LFM2 uses ``out_proj``.
    Probing by name keeps ``is_sdpa`` and the forward path aligned on the
    same alias set instead
    of the forward breaking on architectures that already pass dispatch.
    """
    for name in ("o_proj", "dense", "out_proj"):
        proj = getattr(module, name, None)
        if proj is not None:
            return proj
    return None


def _apply_qk_norms(
    queries: mx.array,
    keys: mx.array,
    q_norm: nn.Module | None,
    k_norm: nn.Module | None = None,
) -> tuple[mx.array, mx.array]:
    if q_norm is not None:
        queries = q_norm(queries)
    if k_norm is not None:
        keys = k_norm(keys)
    return queries, keys


# === Q/K/V preparation (YOCO, K-eq-V, v_norm variants) ===


def prepare_sdpa_qkv(
    inner: nn.Module,
    x: mx.array,
    ctx: PagedAttentionContext,
    n_heads: int,
    n_kv_heads: int,
    shared_kv: tuple[mx.array, mx.array] | None = None,
    *,
    read_existing_kv: bool = False,
    position_embeddings: tuple[mx.array, mx.array] | None = None,
    attention_contract: AttentionContract = DEFAULT_ATTENTION_CONTRACT,
) -> tuple[mx.array, mx.array, mx.array, mx.array | None, tuple[mx.array, mx.array]]:
    """Project ``x`` into Q/K/V with architecture-aware norms and RoPE.

    Handles three Gemma4-specific branches:

    - **YOCO** (``shared_kv`` given): reuse K/V from a prior layer; skip
      projection and only apply Q norm + RoPE.
    - **Read-existing KV** (``read_existing_kv=True``): prepare Q only and
      let the paged-attention kernel read K/V already present in the cache.
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
        read_existing_kv: If true, skip K/V projection and cache writes.

    Returns:
        Tuple ``(queries, keys, values, gate, kv_for_sharing)``:

        - ``queries``, ``keys``, ``values``: ``(B, H, L, head_dim)`` tensors
          ready for the Metal kernel.
        - ``gate``: optional gate tensor for gated attention (Qwen3.5
          Qwen3Next style), else ``None``.
        - ``kv_for_sharing``: the post-norm+RoPE ``(keys, values)`` pair so
          the caller can forward them to the next YOCO layer.

    Raises:
        NotImplementedError: If no precomputed rotary embeddings are provided,
            and ``inner`` has neither ``rope`` nor ``rotary_emb`` and does not
            explicitly disable RoPE.
    """
    B, L, _ = x.shape  # noqa: N806
    norm_placement = attention_contract.qk_norm_placement
    q_norm = _named_norm(inner, "q_norm", "query_layernorm", "q_layernorm")
    k_norm = _named_norm(inner, "k_norm", "key_layernorm", "k_layernorm")
    if shared_kv is not None and read_existing_kv:
        raise ValueError("shared_kv and read_existing_kv are mutually exclusive")

    gate: mx.array | None = None
    packed_qkv = _has_packed_qkv_sdpa_contract(inner)
    if read_existing_kv and packed_qkv:
        raise NotImplementedError(
            "read_existing_kv requires split Q/K/V projections so Q can be "
            "prepared without projecting new K/V tensors."
        )
    # head_dim has two architectural sources in our supported models:
    #   - self.head_dim instance attr (gemma*, llama, mistral, qwen3_5+, phi3)
    #   - k_proj output features (qwen3, qwen3_moe never set self.head_dim) —
    #     read via out_features when the projection exposes it (quantized
    #     wrappers with no dense .weight), else from the dense weight rows.
    # KV-shared Gemma 4 layers have head_dim but no k_proj. If neither
    # is present, raise — silently propagating a wrong head_dim would
    # corrupt downstream kernel shapes.
    if hasattr(inner, "head_dim"):
        head_dim = inner.head_dim
    elif hasattr(inner, "k_proj"):
        head_dim = _projection_out_features(inner.k_proj) // n_kv_heads
    else:
        raise AttributeError(
            f"Cannot determine head_dim for "
            f"{type(inner).__module__}.{type(inner).__name__}: "
            "neither 'head_dim' nor 'k_proj' attribute present"
        )

    if packed_qkv:
        qkv = inner.qkv_proj(x)
        q_width = n_heads * head_dim
        kv_width = n_kv_heads * head_dim
        queries, keys, values = mx.split(qkv, [q_width, q_width + kv_width], axis=-1)
        if norm_placement is QKNormPlacement.BEFORE_HEAD_SPLIT:
            queries, keys = _apply_qk_norms(queries, keys, q_norm, k_norm)
        queries = queries.reshape(B, L, n_heads, head_dim)
        keys = keys.reshape(B, L, n_kv_heads, head_dim)
        values = values.reshape(B, L, n_kv_heads, head_dim)
    else:
        # Projections + reshape.  Qwen3.5 uses gated q_proj (2x head_dim).
        q_proj_out = inner.q_proj(x)
        if norm_placement is QKNormPlacement.BEFORE_HEAD_SPLIT and q_norm is not None:
            q_proj_out = q_norm(q_proj_out)
        q_full_head = q_proj_out.shape[-1] // n_heads
        if q_full_head == 2 * head_dim:
            q_reshaped = q_proj_out.reshape(B, L, n_heads, q_full_head)
            queries, gate = mx.split(q_reshaped, 2, axis=-1)
            gate = gate.reshape(B, L, -1)
        else:
            queries = q_proj_out.reshape(B, L, n_heads, -1)

    if shared_kv is not None or read_existing_kv:
        # YOCO/reuse-cache paths: Q still needs norm + RoPE, but K/V
        # projection is skipped.  For read_existing_kv, local K/V tensors
        # only satisfy RoPE/padding shape contracts; sdpa_forward uses the
        # explicit flag to read the authoritative K/V from the paged cache.
        if shared_kv is None:
            keys = mx.zeros((B, n_kv_heads, L, head_dim), dtype=x.dtype)
            values = keys
        else:
            keys, values = shared_kv
        if norm_placement is QKNormPlacement.BEFORE_ROPE:
            queries, keys = _apply_qk_norms(queries, keys, q_norm)
        queries = queries.transpose(0, 2, 1, 3)
        queries, _ = apply_attention_rope(
            inner,
            queries,
            keys,
            ctx.cu_seqlens,
            offsets=ctx.offsets if ctx.offsets else None,
            apply_keys=False,
            positions=ctx.segment_positions,
            position_embeddings=position_embeddings,
            use_rope=attention_contract.use_rope,
        )
        if norm_placement is QKNormPlacement.AFTER_ROPE:
            queries, keys = _apply_qk_norms(queries, keys, q_norm)
    else:
        if not packed_qkv:
            k_proj_out = inner.k_proj(x)
            if (
                norm_placement is QKNormPlacement.BEFORE_HEAD_SPLIT
                and k_norm is not None
            ):
                k_proj_out = k_norm(k_proj_out)
            keys = k_proj_out.reshape(B, L, n_kv_heads, -1)
            # K-eq-V variant (Gemma4 26B/31B): no v_proj, values = keys.
            if hasattr(inner, "v_proj"):
                values = inner.v_proj(x).reshape(B, L, n_kv_heads, -1)
            else:
                values = keys

        # Per-head RMSNorm (Qwen3, Qwen3.5, Gemma4, Phi3/Phi4 when present).
        if norm_placement is QKNormPlacement.BEFORE_ROPE:
            queries, keys = _apply_qk_norms(queries, keys, q_norm, k_norm)
        if hasattr(inner, "v_norm"):
            values = inner.v_norm(values)

        # Transpose to (B, H, L, head_dim).
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        queries, keys = apply_attention_rope(
            inner,
            queries,
            keys,
            ctx.cu_seqlens,
            offsets=ctx.offsets if ctx.offsets else None,
            positions=ctx.segment_positions,
            position_embeddings=position_embeddings,
            use_rope=attention_contract.use_rope,
        )
        if norm_placement is QKNormPlacement.AFTER_ROPE:
            queries, keys = _apply_qk_norms(queries, keys, q_norm, k_norm)

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


def _float32_sinks(inner: nn.Module) -> mx.array | None:
    """The module's attention sinks as float32, cast once per parameter.

    The widened copy is stored outside the parameter tree, so the checkpoint
    weight keeps its dtype for the non-paged fallback and for anything that
    walks the parameters. It is keyed by the original array, so a reloaded
    parameter is cast again.
    """
    sinks = getattr(inner, "sinks", None)
    if sinks is None or sinks.dtype == mx.float32:
        return sinks
    cached = getattr(inner, "_vllm_metal_sinks_f32", None)
    if cached is None or cached[0] is not sinks:
        cached = (sinks, sinks.astype(mx.float32))
        object.__setattr__(inner, "_vllm_metal_sinks_f32", cached)
    return cached[1]


def sdpa_forward(
    inner: nn.Module,
    x: mx.array,
    ctx: PagedAttentionContext,
    kv_cache: MetalPagedKVCache,
    layer_idx: int,
    shared_kv: tuple[mx.array, mx.array] | None = None,
    *,
    read_existing_kv: bool = False,
    position_embeddings: tuple[mx.array, mx.array] | None = None,
    attention_contract: AttentionContract = DEFAULT_ATTENTION_CONTRACT,
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    """Full SDPA forward pass: project → norm/RoPE → Metal kernel.

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
    #   StableLM:                 num_heads, num_key_value_heads
    #   DeepseekAttention:        num_attention_heads, num_kv_heads
    n_heads = (
        getattr(inner, "n_heads", None)
        or getattr(inner, "num_attention_heads", None)
        or inner.num_heads
    )
    n_kv_heads = (
        getattr(inner, "n_kv_heads", None)
        or getattr(inner, "num_kv_heads", None)
        or inner.num_key_value_heads
    )

    # Softmax scale — GPT-OSS names it sm_scale rather than scale.
    attn_scale = getattr(inner, "scale", None)
    if attn_scale is None:
        attn_scale = getattr(inner, "sm_scale", None)

    # Attention logit softcapping (Gemma 2: ``attn_logit_softcapping``).
    # mlx_lm applies ``tanh(qk / cap) * cap`` to the pre-softmax scores; the
    # Metal kernel implements the same clamp and treats any value <= 0 as
    # disabled, so architectures that do not define the attribute keep the
    # previous uncapped path unchanged.
    attn_softcap = float(getattr(inner, "attn_logit_softcapping", None) or 0.0)

    # Attention sinks: a learned per-head logit that joins the softmax
    # denominator without contributing a value row (GPT-OSS). Models without
    # sinks leave this None and the kernel keeps its plain-softmax path.
    # The kernel reads them as device float; the cast is memoized off the
    # module's parameter tree, since this is the per-layer hot path.
    sinks = _float32_sinks(inner)

    queries, keys, values, gate, kv_for_sharing = prepare_sdpa_qkv(
        inner,
        x,
        ctx,
        n_heads,
        n_kv_heads,
        shared_kv,
        read_existing_kv=read_existing_kv,
        position_embeddings=position_embeddings,
        attention_contract=attention_contract,
    )
    if attn_scale is None:
        if not attention_contract.derive_scale_from_query:
            raise AttributeError(
                f"{type(inner).__module__}.{type(inner).__name__} exposes "
                "neither 'scale' nor 'sm_scale'"
            )
        # StableLM computes the standard scale inline and exposes no scale
        # attribute. Use the projected query width before any cache padding.
        attn_scale = math.sqrt(1 / queries.shape[-1])

    # --- Metal kernel dispatch ---
    n_heads = queries.shape[1]
    head_dim = queries.shape[3]

    # Per-layer cache properties: shape and sliding window.
    cache_kv_heads = kv_cache.kv_heads_per_layer[layer_idx]
    cache_head_dim = kv_cache.head_dim_per_layer[layer_idx]
    layer_sliding_window = kv_cache.sliding_window_per_layer[layer_idx]
    group_index = kv_cache.group_index_for_layer(layer_idx)
    if ctx.kv_groups is None:
        raw_slot_mapping = ctx.slot_mapping
        raw_block_tables = ctx.block_tables
        cache_block_size = kv_cache.block_size_for_layer(layer_idx)
    else:
        group = ctx.kv_groups[group_index]
        raw_slot_mapping = group.slot_mapping
        raw_block_tables = group.block_tables
        cache_block_size = group.block_size
    actual_head_dim = head_dim
    queries, keys, values = pad_qkv_to_cache_head_dim(
        queries, keys, values, head_dim, cache_head_dim
    )
    head_dim = cache_head_dim

    # Reshape to 3D: (1, heads, L, hd) → (L, heads, hd)
    q_3d = mx.contiguous(queries[0].transpose(1, 0, 2).astype(kv_cache.dtype))
    k_3d = mx.contiguous(keys[0].transpose(1, 0, 2).astype(kv_cache.dtype))
    v_3d = mx.contiguous(values[0].transpose(1, 0, 2).astype(kv_cache.dtype))

    # --- Kernel-format metadata (memoized per forward) ---
    # Converted on the group's first layer and reused by the rest (see
    # _kernel_metadata).  Includes the hybrid block-size translation:
    # vLLM may inflate block_size (e.g. 544) to align attention pages with
    # mamba pages in hybrid models, while the Metal kernel only supports
    # small block sizes (8, 16, 32); build_block_tables expands each vLLM
    # block into multiple kernel blocks and returns the kernel-compatible
    # block_size.
    meta = _kernel_metadata(
        ctx,
        None if ctx.kv_groups is None else group_index,
        raw_slot_mapping,
        raw_block_tables,
        cache_block_size,
    )
    slot_mapping = meta.slot_mapping
    seq_lens = meta.seq_lens
    cu_seqlens_q = meta.cu_seqlens_q
    block_tables, kernel_block_size = meta.block_tables, meta.block_size
    max_seq_len = meta.max_seq_len

    v_centroids: mx.array | None = None
    if shared_kv is not None or read_existing_kv:
        # YOCO shared layer / MTP read-existing layer: the authoritative K/V
        # already lives in the paged cache.  Skip writes to avoid redundant
        # compute and to keep the target cache read-only for assistant use.
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
        from vllm_metal.attention.caches.turboquant import (
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
        # Fused K/V paged scatter: one Metal dispatch (reshape_and_cache) writes
        # both K and V into the paged cache by slot_mapping, replacing the two
        # per-layer MLX scatters. The outputs alias the input cache buffers in
        # place and carry graph provenance for the paged_attention read below.
        new_k_cache, new_v_cache = get_ops().reshape_and_cache(
            k_3d,
            v_3d,
            kv_cache.key_caches[layer_idx],
            kv_cache.value_caches[layer_idx],
            slot_mapping,
        )
        # Rebind so next layer / decode step uses the updated cache
        kv_cache.replace_layer_cache(layer_idx, new_k_cache, new_v_cache)

    # --- Attention: paged attention primitive (read-only) ---
    # The primitive normally participates in MLX's lazy graph and is
    # evaluated by the model runner at the end of the forward
    # pass.  Fence-based synchronisation across command buffer boundaries
    # works correctly because eval_gpu skips add_temporary (which would
    # remove buffers from the encoder's fence tracking).
    #
    # TurboQuant caches must expose the kernel block size because their packed
    # K/V and scale layouts carry separate strides.
    kernel_k_cache = new_k_cache
    kernel_v_cache = new_v_cache
    if kernel_block_size != cache_block_size and kv_cache.turboquant:
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
    # Gemma 4 vision: rows of image blocks attend bidirectionally on the
    # adapter's layer kinds.  The kernel path hands the per-row block ranges
    # to the tiled prefill kernel; the recompute path (the reference)
    # recomputes those rows after the kernel with MLX SDPA.
    mm_prefix_ranges: mx.array | None = None
    recompute_after_kernel = False
    if ctx.segment_bidi_ranges is not None:
        kind = "sliding" if layer_sliding_window >= 0 else "full"
        if kind in ctx.bidi_layer_kinds:
            assert ctx.cu_seqlens is not None
            float32_cache = kernel_k_cache.dtype == mx.float32
            if image_block_path(ops, float32_cache=float32_cache) == "kernel":
                mm_prefix_ranges = _mm_prefix_rows(ctx)
                if mm_prefix_ranges is not None and not ctx.bidi_logged:
                    ctx.bidi_logged = True
                    logger.info(
                        "Metal: mm_prefix ranges on %d row(s)",
                        ctx.mm_prefix_row_count,
                    )
            else:
                recompute_after_kernel = True
    # Only the kernel path passes the ranges, so ops that predate the keyword
    # still run the kernel for text rows and leave image rows to the recompute.
    mm_kwargs = (
        {} if mm_prefix_ranges is None else {"mm_prefix_ranges": mm_prefix_ranges}
    )
    out = mx.array(0)
    if kv_cache.turboquant:
        # Preserve the compressed primitive's layout and rejection contracts
        # for decode, short continuations, verification, sinks and image rows.
        kernel_key_scale = new_key_scale_cache
        kernel_value_scale = new_value_scale_cache
        kernel_key_zero = new_key_zero_cache
        if kernel_block_size != cache_block_size:
            sg = new_key_scale_cache.shape[-1]
            kernel_key_scale = new_key_scale_cache.reshape(
                -1, kernel_block_size, cache_kv_heads, sg
            )
            kernel_value_scale = new_value_scale_cache.reshape(
                -1, kernel_block_size, cache_kv_heads, sg
            )
            kernel_key_zero = new_key_zero_cache.reshape(
                -1, kernel_block_size, cache_kv_heads, sg
            )
        if v_centroids is None:
            from vllm_metal.attention.caches.turboquant import get_v_centroids

            v_centroids = get_v_centroids(kv_cache.v_bits)

        def quantized_attention(
            query: mx.array, batch: _AttentionBatch | _KernelMetadata
        ) -> mx.array:
            result = mx.array(0)
            ops.paged_attention_primitive(
                query,
                kernel_k_cache,
                kernel_v_cache,
                cache_kv_heads,
                attn_scale,
                attn_softcap,
                batch.block_tables,
                batch.seq_lens,
                batch.cu_seqlens_q,
                kernel_block_size,
                batch.max_seq_len,
                layer_sliding_window,
                result,
                sinks=sinks,
                key_scale_cache=kernel_key_scale,
                value_scale_cache=kernel_value_scale,
                key_zero_cache=kernel_key_zero,
                v_centroids=v_centroids,
                use_turboquant=True,
                quant_type=kv_cache.k_quant,
                v_bits=kv_cache.v_bits,
                window_seqlen_q=ctx.verify_window_q,
                **mm_kwargs,
            )
            return result

        plan = None
        has_prefill = q_3d.shape[0] > len(ctx.context_lens)
        reason = unsupported_reason(
            dtype=q_3d.dtype,
            head_dim=q_3d.shape[2],
            kernel_block_size=kernel_block_size,
            cache_block_size=cache_block_size,
            stored_block_size=new_k_cache.shape[1],
        )
        if has_prefill and reason and ctx.tq_prefill_workspace_bytes:
            logger.info_once("Metal: TurboQuant prefill stays compressed: %s.", reason)
        if (
            has_prefill
            and ctx.verify_window_q == 1
            and sinks is None
            and not mm_kwargs
            and not recompute_after_kernel
            # Windowed layers already skip old KV in the compressed path;
            # materializing that history would undo the saving.
            and layer_sliding_window < 0
            and reason is None
        ):
            plan = _turboquant_prefill_plan(
                ctx,
                meta,
                raw_block_tables,
                cache_block_size,
                q_3d.shape[1],
                cache_kv_heads,
                q_3d.shape[2],
            )

        if plan is None:
            out = quantized_attention(q_3d, meta)
        else:
            logger.info_once("Metal: bounded TurboQuant prefill lane active.")
            logger.debug(
                "TurboQuant prefill: %d requests, %d gathered tokens, "
                "%d workspace bytes, %d compressed requests",
                plan.prefill.seq_lens.shape[0],
                plan.pool_pages.shape[0],
                plan.workspace_bytes,
                0 if plan.fallback is None else plan.fallback.seq_lens.shape[0],
            )

            # Fresh writer handles preserve encode -> gather dependencies.
            # The fused read respects padded upstream views and writes only
            # the final K/V pair, without packed or FP32 temporary arrays.
            k16, v16 = materialize_turboquant_pages(
                new_k_cache,
                new_v_cache,
                new_key_scale_cache,
                new_key_zero_cache,
                new_value_scale_cache,
                plan.pool_pages,
                plan.pool_offsets,
                v_centroids,
                head_dim=q_3d.shape[2],
                key_quant_type=kv_cache.k_quant,
                value_bits=kv_cache.v_bits,
                output_dtype=q_3d.dtype,
            )
            batch = plan.prefill
            query = q_3d if batch.query_indices is None else q_3d[batch.query_indices]
            ops.paged_attention_primitive(
                query,
                k16.reshape(-1, kernel_block_size, cache_kv_heads, q_3d.shape[2]),
                v16.reshape(-1, kernel_block_size, cache_kv_heads, q_3d.shape[2]),
                cache_kv_heads,
                attn_scale,
                attn_softcap,
                batch.block_tables,
                batch.seq_lens,
                batch.cu_seqlens_q,
                kernel_block_size,
                batch.max_seq_len,
                layer_sliding_window,
                out,
                window_seqlen_q=ctx.verify_window_q,
            )
            if plan.fallback is not None:
                fallback = plan.fallback
                rest = quantized_attention(q_3d[fallback.query_indices], fallback)
                out = mx.concatenate((out, rest), axis=0)[plan.restore_indices]
            # Finish this lane before building the next layer. Otherwise MLX
            # can keep several materialized K/V pairs in flight, multiplying
            # the single workspace reserved by WorkerCachePlanner. Decode and
            # the compressed fallback retain their fully lazy execution.
            mx.eval(out)
            # eval waits for the result event; Metal's completion handlers can
            # still retain input buffers. Drain the stream before another layer
            # allocates its K/V pair against the same workspace reservation.
            mx.synchronize()
            del k16, v16
    else:
        # Whole-batch decode routing belongs to the ordinary cache path. TQ
        # uses its own sub-batch metadata and stays outside native decode split.
        paged_kwargs: dict[str, int | mx.array] = dict(mm_kwargs)
        if ctx.paged_native_capabilities is None:
            ctx.paged_native_capabilities = paged_attention_capabilities(ops)
        capabilities = ctx.paged_native_capabilities
        # Omit the new keyword on the default path for older native builds.
        if ctx.gqa_disabled:
            if capabilities["gqa_disable"]:
                paged_kwargs["gqa_disabled"] = True
            elif capabilities["gqa_decode"]:
                # Never silently ignore a kill switch on an unrecognized GQA build.
                raise RuntimeError(
                    "Loaded native GQA build does not advertise disable support; "
                    "rebuild the vllm-metal native extension."
                )
            # Pre-GQA native builds already use the established attention path.
        if capabilities["decode_routing_metadata"]:
            paged_kwargs.update(
                num_decode_requests=ctx.num_decode_requests,
                num_decode_tokens=ctx.num_decode_tokens,
                max_decode_context_len=ctx.max_decode_context_len,
            )
        ops.paged_attention_primitive(
            q_3d,
            kernel_k_cache,
            kernel_v_cache,
            cache_kv_heads,
            attn_scale,
            attn_softcap,
            block_tables,
            seq_lens,
            cu_seqlens_q,
            kernel_block_size,
            max_seq_len,
            layer_sliding_window,
            out,
            window_seqlen_q=ctx.verify_window_q,
            sinks=sinks,
            **paged_kwargs,
        )

    if recompute_after_kernel:
        assert ctx.cu_seqlens is not None
        out = apply_bidirectional_segments(
            out,
            q_3d,
            kernel_k_cache,
            kernel_v_cache,
            block_tables=block_tables,
            block_size=kernel_block_size,
            cu_seqlens=ctx.cu_seqlens,
            context_lens=ctx.context_lens,
            ctx=ctx,
            window=layer_sliding_window if layer_sliding_window >= 0 else None,
            scale=attn_scale,
            head_dim=actual_head_dim,
            softcap=attn_softcap,
            sinks=sinks,
            turboquant=kv_cache.turboquant,
        )

    # Reshape + strip padding back to actual head_dim before o_proj.
    out = truncate_padded_output(out, B, L, n_heads, cache_head_dim, actual_head_dim)
    if gate is not None:
        out = out * mx.sigmoid(gate)
    out = apply_g_proj_gate(inner, out, x, n_heads, actual_head_dim)
    return _output_projection(inner)(out), kv_for_sharing


def apply_g_proj_gate(
    inner: nn.Module,
    out: mx.array,
    x: mx.array,
    n_heads: int,
    head_dim: int,
) -> mx.array:
    """Apply Laguna-style per-head attention-output gating.

    Laguna projects the layer input ``x`` through a dedicated ``g_proj``
    linear to a per-head scalar, passes it through ``softplus`` (in
    float32 for numerical stability) and multiplies the attention output
    of each head by its gate value before ``o_proj`` — matching
    ``mlx_lm.models.laguna.Attention``::

        gate = softplus(g_proj(x))                # (B, L, n_heads)
        out  = (out.reshape(B, L, H, hd) * gate[..., None])

    This is distinct from the Qwen3.5/Qwen3Next gate (a split of the
    ``q_proj`` output combined via ``sigmoid``), which is handled
    separately in :func:`prepare_sdpa_qkv` / :func:`sdpa_forward`.

    No-op for modules without a ``g_proj`` (or with gating disabled), so
    every other SDPA model is unaffected.
    """
    if not getattr(inner, "gating", False) or not hasattr(inner, "g_proj"):
        return out
    B, L, _ = out.shape  # noqa: N806
    gate = nn.softplus(inner.g_proj(x).astype(mx.float32)).astype(out.dtype)
    out = out.reshape(B, L, n_heads, head_dim)
    out = (out * gate[..., None]).reshape(B, L, -1)
    return out
