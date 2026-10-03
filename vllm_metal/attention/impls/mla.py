# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention

from vllm_metal import envs
from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.context import PagedAttentionContext, get_context
from vllm_metal.attention.impls.varlen_rope_compat import apply_packed_rope
from vllm_metal.metal.constants import MLA_KERNEL_BLOCK_SIZES

# Default rope head dim for GLM/DeepSeek-V2 lineage models.
# Used as fallback when qk_rope_head_dim is absent from model config.
MLA_DEFAULT_QK_ROPE_HEAD_DIM = 64


@dataclass(frozen=True, eq=False)
class MLAKernelMetadata:
    """Single-pass kernel inputs: padded block tables and lengths."""

    block_tables: mx.array  # [num_seqs, max_blocks] int32, zero-padded
    context_lens: mx.array  # uint32
    cu_seqlens_q: mx.array  # int32


@dataclass(eq=False)
class MLAForwardMetadata:
    """Kernel-format copies of the per-forward MLA metadata.

    The paged context lives exactly one forward pass, so these fields
    never go stale: the first MLA layer converts the Python lists and
    every later layer reuses the same arrays. ``eq=False`` for the same
    reason as ``impls.sdpa._KernelMetadata`` — the generated ``__eq__``
    would compare mx arrays, which raises on ``bool()``. The
    path-specific fields are filled lazily on first use because only one
    attention path runs per forward.
    """

    slot_mapping: mx.array  # int64, latent-cache scatter
    block_table_rows: tuple[mx.array, ...] | None = None  # per-request int32, SDPA loop
    kernel: MLAKernelMetadata | None = None
    # Batched absorbed decode: eligible row list (empty once computed when
    # the gates reject) and its on-device index array, both built once per
    # forward and shared by every layer.
    decode_batch_rows: list[int] | None = None
    decode_batch_idx: mx.array | None = None


def _mla_metadata(ctx: PagedAttentionContext) -> MLAForwardMetadata:
    """Per-forward MLA metadata, converted once and reused by all layers."""
    meta = ctx.mla_metadata
    if meta is None:
        meta = MLAForwardMetadata(
            slot_mapping=mx.array(ctx.slot_mapping, dtype=mx.int64),
        )
        ctx.mla_metadata = meta
    return meta


def _block_table_rows(ctx: PagedAttentionContext) -> tuple[mx.array, ...]:
    """Per-request int32 block tables for the SDPA loop, built once per forward."""
    meta = _mla_metadata(ctx)
    if meta.block_table_rows is None:
        meta.block_table_rows = tuple(
            mx.array(bt, dtype=mx.int32) for bt in ctx.block_tables
        )
    return meta.block_table_rows


def _kernel_inputs(ctx: PagedAttentionContext) -> MLAKernelMetadata:
    """Padded single-pass kernel inputs, built once per forward."""
    meta = _mla_metadata(ctx)
    if meta.kernel is None:
        # Pad block_tables (list[list[int]]) into a 2D [num_seqs, max_blocks]
        # int32 array. The kernel reads block_table_row[0..n_context_blocks-1];
        # padding entries beyond n_context_blocks are never read.
        bts = ctx.block_tables
        max_blocks = max(len(bt) for bt in bts)
        padded = [bt + [0] * (max_blocks - len(bt)) for bt in bts]
        meta.kernel = MLAKernelMetadata(
            block_tables=mx.array(padded, dtype=mx.int32),
            context_lens=mx.array(list(ctx.context_lens), dtype=mx.uint32),
            cu_seqlens_q=mx.array(list(ctx.cu_seqlens), dtype=mx.int32),
        )
    return meta.kernel


# Measured crossovers sit above the FLOP break-even (MLX kernel efficiency
# differs between the paths); 1.25x rounded up to a multiple of 64 matches the
# wrapper benchmark on DeepSeek-V2-Lite and GLM-4.7-Flash dims, fp16-4bit and
# bf16, 2k/8k cached context.
_MATERIALIZED_CROSSOVER_MARGIN = 1.25
_MATERIALIZED_THRESHOLD_ROUNDING = 64

# Batched absorbed decode (_absorbed_decode_batch): one padded gather + one
# SDPA amortizes the per-request dispatches when many one-token rows share a
# short context, but loses when a long context inflates the padded gather.
# Rows past the per-row context cap keep the per-segment loop so one long
# context cannot pad the whole batch; the token cap bounds the padded volume.
_DECODE_BATCH_MIN_ROWS = 16
_DECODE_BATCH_MAX_CTX = 1024
_DECODE_BATCH_MAX_TOKENS = 65536


def materialized_min_new_tokens_with_past(
    *,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
) -> int | None:
    """Smallest chunk (new tokens) over cached context for which materialized
    prefill beats the absorbed loop, or ``None`` if it never does.

    Per head and cached token, the absorbed loop spends ``num_new *
    (2 * kv_lora_rank + qk_rope_head_dim)`` MACs (QK over the latent + rope,
    PV over the latent); the materialized path spends ``kv_lora_rank *
    (qk_nope_head_dim + v_head_dim)`` to materialize K/V once plus ``num_new *
    (qk_nope_head_dim + qk_rope_head_dim + v_head_dim)`` for attention. They
    break even at ``num_new = kv_lora * (nope + v) / (absorbed - materialized
    per-query dims)``: ~171 for DeepSeek-V2/V3 dims, ~398 for GLM-4.7-Flash.
    """
    absorbed_dims = 2 * kv_lora_rank + qk_rope_head_dim
    materialized_dims = qk_nope_head_dim + qk_rope_head_dim + v_head_dim
    if materialized_dims >= absorbed_dims:
        return None
    break_even = (
        kv_lora_rank
        * (qk_nope_head_dim + v_head_dim)
        / (absorbed_dims - materialized_dims)
    )
    step = _MATERIALIZED_THRESHOLD_ROUNDING
    return math.ceil(_MATERIALIZED_CROSSOVER_MARGIN * break_even / step) * step


def is_mla_attention(module: nn.Module) -> bool:
    """Return whether a module exposes the MLA surface used by the wrapper."""
    if not all(
        hasattr(module, name)
        for name in (
            "num_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
        )
    ):
        return False
    has_query = (
        hasattr(module, "q_proj")
        if module.q_lora_rank is None
        else all(
            hasattr(module, name) for name in ("q_a_proj", "q_a_layernorm", "q_b_proj")
        )
    )
    has_attention = (
        hasattr(module, "embed_q") and hasattr(module, "unembed_out")
    ) or hasattr(module, "kv_b_proj")
    has_output = hasattr(module, "o_proj") or (
        hasattr(module, "dense") and hasattr(module, "g_proj")
    )
    has_q_dim = hasattr(module, "q_head_dim") or hasattr(module, "qk_head_dim")
    has_scale = hasattr(module, "scale") or hasattr(module, "softmax_scale")
    has_rope = hasattr(module, "rope") or hasattr(module, "rotary_emb")
    return (
        has_query
        and has_attention
        and has_output
        and has_q_dim
        and has_scale
        and has_rope
    )


class MLAPagedAttentionWrapper(nn.Module):
    """Wraps an MLA attention module to use a paged latent cache.

    MLA (GLM/DeepSeek/MiniCPM3 lineage) compresses KV into a latent before caching:

        latent = [kv_norm || k_pe_roped]  # kv_lora_rank + qk_rope_head_dim dims

    Each call scatter-writes the new tokens' latents into the scheduled cache
    slots, then gather-reads all past latents per request via block tables.

    Some models expose absorbed MLA helpers: embed_q projects q_nope into
    kv_lora_rank space, and unembed_out maps the output back to v_head_dim.
    MiniCPM3 instead keeps kv_b_proj as the public K/V reconstruction path.
    This wrapper handles both layouts while sharing the paged latent cache.

    When no PagedAttentionContext is active the original module is called as-is.
    """

    # Single-pass Metal kernel admission: kv_lora_rank=512, qk_rope_head_dim=64,
    # block_size ∈ {16, 32}, fp16 / bf16. Workloads outside this set fall
    # through to the MLX SDPA slow path.
    _KERNEL_KV_LORA_RANK = 512
    _KERNEL_QK_ROPE_HEAD_DIM = 64

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        latent_cache: MLAPagedLatentCache,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_mla_layer_idx", layer_idx)
        object.__setattr__(self, "_mla_latent_cache", latent_cache)
        is_absorbed = hasattr(inner, "embed_q") and hasattr(inner, "unembed_out")
        object.__setattr__(self, "_is_absorbed", is_absorbed)
        # Smallest chunk (new tokens) over cached context that the
        # materialized-prefill path takes; derived from this layer's attention
        # dims (see ``materialized_min_new_tokens_with_past``).
        dims = {
            name: getattr(inner, name, None)
            for name in (
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
            )
        }
        object.__setattr__(
            self,
            "_materialized_min_new_with_past",
            materialized_min_new_tokens_with_past(**dims)
            if is_absorbed and all(isinstance(d, int) for d in dims.values())
            else None,
        )
        if is_absorbed:
            object.__setattr__(
                self, "_apply_mla_attention", self._apply_absorbed_mla_attention
            )
        else:
            object.__setattr__(
                self, "_apply_mla_attention", self._apply_kv_b_proj_attention
            )

    def rebind_cache(
        self, latent_cache: MLAPagedLatentCache, *, cache_idx: int
    ) -> None:
        """Refresh the latent cache and compact layer index."""
        object.__setattr__(self, "_mla_layer_idx", cache_idx)
        object.__setattr__(self, "_mla_latent_cache", latent_cache)

    def _attention_scale(self) -> float:
        inner = self._inner
        scale = getattr(inner, "scale", None)
        if scale is None:
            scale = inner.softmax_scale
        return scale

    def _q_head_dim(self) -> int:
        inner = self._inner
        q_head_dim = getattr(inner, "q_head_dim", None)
        if q_head_dim is None:
            q_head_dim = inner.qk_head_dim
        return int(q_head_dim)

    def _project_output(self, x: mx.array, output: mx.array) -> mx.array:
        """Apply the model's output gate and projection."""
        inner = self._inner
        if hasattr(inner, "o_proj"):
            return inner.o_proj(output)
        if not hasattr(inner, "dense") or not hasattr(inner, "g_proj"):
            raise RuntimeError(
                f"Unsupported MLA output projection for {type(inner).__name__}"
            )

        batch, length, _ = output.shape
        output = output.reshape(batch, length, inner.num_heads, inner.v_head_dim)
        gate = mx.sigmoid(inner.g_proj(x).astype(mx.float32)).astype(output.dtype)
        if gate.shape[-1] != inner.num_heads:
            raise RuntimeError(
                f"Unsupported MLA gate width {gate.shape[-1]} for "
                f"{type(inner).__name__}; expected {inner.num_heads}"
            )
        output = output * gate[..., None]
        return inner.dense(output.reshape(batch, length, -1))

    @staticmethod
    def _causal_valid_mask(
        *,
        num_new: int,
        ctx_len: int,
        past_len: int,
    ) -> mx.array | None:
        if num_new == 1:
            return None
        rows = mx.arange(num_new).reshape(-1, 1)
        cols = mx.arange(ctx_len).reshape(1, -1)
        return (cols <= (past_len + rows)).reshape(1, 1, num_new, ctx_len)

    def _can_use_kernel(
        self,
        inner: nn.Module,
        latent_cache: MLAPagedLatentCache,
        ctx: Any,
    ) -> bool:
        """Admission check for the single-pass Metal kernel fast path.

        Returns True only when every dimension matches the kernel's
        instantiated specialization and every request is decode-only.
        Workloads outside this set fall through to ``_slow_path_per_request``
        (MLX SDPA) — no silent fallback, no scaffolding for routing
        between kernel variants (this PR ships single-pass only;
        FA / 2pass / pr_mma land in follow-ups once each has its own
        real-model parity proof, per the alignment with reviewers on
        ``Ship one kernel, prove it wins'')."""
        if self._kernel_mismatch(inner, latent_cache) is not None:
            return False
        cu = ctx.cu_seqlens
        for i in range(len(ctx.context_lens)):
            if cu[i + 1] - cu[i] != 1:
                return False
        return True

    def _kernel_mismatch(
        self, inner: nn.Module, latent_cache: MLAPagedLatentCache
    ) -> str | None:
        """Why the single-pass kernel cannot serve this layer, or ``None``.

        The static half of ``_can_use_kernel``: the switch, the absorbed
        layout, the instantiated dims and the cache.  The forward also needs
        a decode-only batch.
        """
        if not envs.VLLM_METAL_MLA_KERNEL:
            return "VLLM_METAL_MLA_KERNEL is off"
        if not self._is_absorbed:
            return "no absorbed embed_q/unembed_out"
        if inner.kv_lora_rank != self._KERNEL_KV_LORA_RANK:
            return (
                f"kv_lora_rank {inner.kv_lora_rank}, "
                f"the kernel takes {self._KERNEL_KV_LORA_RANK}"
            )
        if inner.qk_rope_head_dim != self._KERNEL_QK_ROPE_HEAD_DIM:
            return (
                f"qk_rope_head_dim {inner.qk_rope_head_dim}, "
                f"the kernel takes {self._KERNEL_QK_ROPE_HEAD_DIM}"
            )
        if latent_cache.block_size not in MLA_KERNEL_BLOCK_SIZES:
            sizes = " or ".join(str(s) for s in sorted(MLA_KERNEL_BLOCK_SIZES))
            return f"block size {latent_cache.block_size}, the kernel takes {sizes}"
        if latent_cache.dtype not in (mx.float16, mx.bfloat16):
            dtype = str(latent_cache.dtype).rsplit(".", 1)[-1]
            return f"{dtype} cache, the kernel takes float16 or bfloat16"
        if not latent_cache.has_dense_pages:
            return "the kernel requires dense latent-cache pages"
        return None

    def decode_kernel_mismatch(self) -> str | None:
        """Why decode on this layer skips the single-pass kernel, or ``None``."""
        return self._kernel_mismatch(self._inner, self._mla_latent_cache)

    @staticmethod
    def _pick_heads_per_tg(num_heads: int, batch_size: int) -> int:
        """Pick HEADS_PER_TG (G) for the single-pass kernel. G=2 packs 2
        query heads into one threadgroup so each K/V load is reused for
        2 dot products; G=1 keeps the wider NUM_THREADS=1024 layout for
        cells too small to saturate the GPU. Bench on M5 Max (RFC #360)
        shows G=2 wins once B*H ≳ 30 launched threadgroups; B=1 with
        small H stays on G=1. Falls back to G=1 when num_heads is odd
        (kernel requires num_heads % G == 0)."""
        if num_heads % 2 != 0:
            return 1
        if batch_size == 1 and num_heads < 32:
            return 1
        return 2

    def _kernel_fast_path_single_pass(
        self,
        inner: nn.Module,
        latent_cache: MLAPagedLatentCache,
        layer_idx: int,
        q_nope: mx.array,  # [1, num_heads, seq_len, qk_nope_head_dim]
        q_pe: mx.array,  # [1, num_heads, seq_len, qk_rope_head_dim] (post-RoPE)
        ctx: Any,
        seq_len: int,
    ) -> mx.array:
        """Single-pass MLA decode fast path: project q_nope through
        embed_q, dispatch the kernel for the whole batch in one call,
        recover v_head_dim through unembed_out, and concatenate for
        o_proj. Replaces the per-request Python loop entirely when the
        gate above accepts."""
        from vllm_metal.metal import metal_mla_paged_attention

        # Cast Q to the latent cache dtype so we hit a real kernel
        # specialization. In production this is a no-op (weights are
        # already fp16/bf16); test fixtures with default fp32 Linear
        # weights need the cast.
        target_dtype = latent_cache.dtype
        q_nope_proj = inner.embed_q(q_nope).astype(target_dtype)
        q_pe_t = q_pe.astype(target_dtype)
        q_nope_kernel = q_nope_proj.transpose(0, 2, 1, 3).reshape(
            seq_len, inner.num_heads, inner.kv_lora_rank
        )
        q_pe_kernel = q_pe_t.transpose(0, 2, 1, 3).reshape(
            seq_len, inner.num_heads, inner.qk_rope_head_dim
        )

        kernel_inputs = _kernel_inputs(ctx)

        out_kvr = metal_mla_paged_attention(
            q_nope=q_nope_kernel,
            q_pe=q_pe_kernel,
            latent_cache=latent_cache.latent_caches[layer_idx],
            block_tables=kernel_inputs.block_tables,
            context_lens=kernel_inputs.context_lens,
            cu_seqlens_q=kernel_inputs.cu_seqlens_q,
            scale=self._attention_scale(),
            heads_per_tg=self._pick_heads_per_tg(inner.num_heads, seq_len),
        )

        # Recover v_head_dim and assemble [1, seq_len, num_heads * v_head_dim]
        # for o_proj — matching the slow path's exit shape.
        out_for_unembed = out_kvr.reshape(
            1, seq_len, inner.num_heads, inner.kv_lora_rank
        ).transpose(0, 2, 1, 3)
        out_unembedded = inner.unembed_out(out_for_unembed)
        return out_unembedded.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)

    def _materialized_segments(self, inner: nn.Module, ctx: Any) -> list[bool] | None:
        """Per-segment routing for the materialized-prefill fast path (RFC #360
        Phase 2): an absorbed model with MultiLinear ``embed_q``/``unembed_out``.
        On by default — bitwise-equal to the absorbed kv_lora-space loop
        (absorption identity), just at a cheaper attention dim.

        Returns one flag per segment (True = materialize), or ``None`` when no
        multi-token segment qualifies and the whole batch stays on the absorbed
        loop / kernel paths. Segments are routed independently, so a prefill
        segment packed next to decode rows (the common continuous-batching
        shape) still materializes. Segments with past context (chunked-prefill
        continuations, prefix-cache hits) materialize their cached K/V too, but
        only once the chunk has >= ``_materialized_min_new_with_past`` new
        tokens (derived from the attention dims); decode-shaped rows and small
        continuation chunks take the absorbed path."""
        if not self._is_absorbed:
            return None
        # Materialization reverses embed_q/unembed_out via MultiLinear's transpose
        # flag (per-head + quantization-aware).
        if type(inner.embed_q).__name__ not in ("MultiLinear", "QuantizedMultiLinear"):
            return None
        cu = ctx.cu_seqlens
        min_new = self._materialized_min_new_with_past
        routed: list[bool] = []
        has_prefill = False
        for i, ctx_len in enumerate(ctx.context_lens):
            num_new = cu[i + 1] - cu[i]
            # Decode-shaped rows and small continuation chunks: materializing a
            # whole context for a few queries loses to the absorbed path.
            ok = ctx_len == num_new or (
                min_new is not None and num_new > 1 and num_new >= min_new
            )
            routed.append(ok)
            has_prefill = has_prefill or (ok and num_new > 1)
        return routed if has_prefill else None

    def _absorbed_segment(
        self,
        inner: nn.Module,
        latent_cache: MLAPagedLatentCache,
        layer_idx: int,
        q_nope: mx.array,  # [1, nheads, seq, qk_nope_head_dim]
        q_pe: mx.array,  # [1, nheads, seq, qk_rope_head_dim] (post-RoPE)
        ctx: Any,
        req_idx: int,
    ) -> mx.array:
        """Absorbed / kv_b_proj attention for one segment, reading its whole
        context from the paged cache. Returns [1, num_new, num_heads * v_head_dim]."""
        ctx_len = ctx.context_lens[req_idx]
        req_start = ctx.cu_seqlens[req_idx]
        req_end = ctx.cu_seqlens[req_idx + 1]
        num_new = req_end - req_start
        past_len = ctx_len - num_new  # tokens cached before this step

        # Gather this request's full context from the paged cache.
        # Block indexing: each block holds block_size contiguous token slots.
        n_blocks = math.ceil(ctx_len / latent_cache.block_size)
        blocks = _block_table_rows(ctx)[req_idx][:n_blocks]
        all_latent = latent_cache.latent_caches[layer_idx][blocks].reshape(
            -1, latent_cache.latent_dim
        )[:ctx_len]

        all_kv_norm = all_latent[:, : inner.kv_lora_rank]
        all_k_pe = all_latent[:, inner.kv_lora_rank :]

        rq_nope = q_nope[:, :, req_start:req_end, :]
        rq_pe = q_pe[:, :, req_start:req_end, :]

        k_pe_r = all_k_pe.reshape(1, 1, ctx_len, inner.qk_rope_head_dim)
        causal_mask = self._causal_valid_mask(
            num_new=num_new, ctx_len=ctx_len, past_len=past_len
        )

        out = self._apply_mla_attention(
            rq_nope=rq_nope,
            rq_pe=rq_pe,
            all_kv_norm=all_kv_norm,
            k_pe=k_pe_r,
            causal_mask=causal_mask,
        )
        return out.transpose(0, 2, 1, 3).reshape(1, num_new, -1)

    def _absorbed_decode_batch(
        self,
        inner: nn.Module,
        latent_cache: MLAPagedLatentCache,
        layer_idx: int,
        q_nope: mx.array,  # [1, nheads, seq, qk_nope_head_dim]
        q_pe: mx.array,  # [1, nheads, seq, qk_rope_head_dim] (post-RoPE)
        ctx: Any,
        req_indices: list[int],
    ) -> mx.array:
        """Absorbed attention for single-token decode segments, batched: one
        padded cache gather and one SDPA for the whole group instead of one
        SDPA dispatch per request. Each decode row attends to its own context
        (padded columns masked out). Returns [n, nheads, 1, v_head_dim] in
        ``req_indices`` order.

        Only valid for absorbed models (embed_q/unembed_out) and num_new==1
        segments — multi-token segments need the per-segment causal mask.

        The padded gather materializes ``n * max_ctx`` latent rows, so the
        row list is split into chunks that each stay under
        ``_DECODE_BATCH_MAX_TOKENS`` — large batches chunk instead of
        falling back to the per-segment loop."""
        n = len(req_indices)
        kv_lora_rank = inner.kv_lora_rank
        kernel = _kernel_inputs(ctx)

        meta = _mla_metadata(ctx)
        idx = meta.decode_batch_idx
        if meta.decode_batch_rows == req_indices:
            if idx is None:
                idx = meta.decode_batch_idx = mx.array(req_indices, dtype=mx.int32)
        else:
            # A caller outside _decode_batch_rows never populates the
            # per-forward cache — build a throwaway index for this call.
            idx = mx.array(req_indices, dtype=mx.int32)

        outs = []
        c0 = 0
        while c0 < n:
            # A chunk's padded volume is (row count * its max context); keep
            # a running max so the cap holds for any row order. Rows from
            # _decode_batch_rows are sorted, which keeps that max tight.
            c1 = c0
            chunk_max = 0
            while c1 < n:
                next_max = max(chunk_max, ctx.context_lens[req_indices[c1]])
                if (c1 - c0 + 1) * next_max > _DECODE_BATCH_MAX_TOKENS:
                    break
                chunk_max = next_max
                c1 += 1
            c1 = max(c1, c0 + 1)
            rows = req_indices[c0:c1]
            cidx = idx[c0:c1]
            n_c = len(rows)
            max_ctx_c = max(ctx.context_lens[i] for i in rows)
            # Padded block tables are built once per forward (#821); padding
            # entries point at block 0 and are masked out below. Slice to the
            # chunk's own block span so co-scheduled long-context segments
            # don't inflate the gather.
            n_blocks = math.ceil(max_ctx_c / latent_cache.block_size)
            all_latent = latent_cache.latent_caches[layer_idx][
                kernel.block_tables[cidx][:, :n_blocks]
            ].reshape(n_c, -1, latent_cache.latent_dim)[:, :max_ctx_c]
            all_kv_norm = all_latent[..., :kv_lora_rank]
            all_k_pe = all_latent[..., kv_lora_rank:]

            # cu_seqlens_q is already on-device from _kernel_inputs; gather the
            # decode rows' packed start offsets instead of re-uploading them.
            starts = mx.take(kernel.cu_seqlens_q, cidx)
            rq_nope = mx.take(q_nope[0], starts, axis=1).transpose(1, 0, 2)[
                :, :, None, :
            ]
            rq_pe = mx.take(q_pe[0], starts, axis=1).transpose(1, 0, 2)[:, :, None, :]

            cols = mx.arange(max_ctx_c).reshape(1, -1)
            valid = (
                cols < kernel.context_lens[cidx].astype(mx.int32).reshape(-1, 1)
            ).reshape(n_c, 1, 1, max_ctx_c)
            outs.append(
                self._apply_mla_attention(
                    rq_nope=rq_nope,
                    rq_pe=rq_pe,
                    all_kv_norm=all_kv_norm,
                    k_pe=all_k_pe.reshape(n_c, 1, max_ctx_c, inner.qk_rope_head_dim),
                    causal_mask=valid,
                )
            )  # [n_c, nheads, 1, v_head_dim]
            c0 = c1
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)

    def _decode_batch_rows(self, ctx: Any) -> list[int] | None:
        """Request indices of the single-token decode segments worth one
        batched absorbed pass: rows whose context fits under the padded-gather
        cap, when enough of them batch to amortize the dispatches.  ``None``
        for non-absorbed models (kv_b_proj attention is per-head already) or
        when the batching gates reject.  Rows left out keep the per-segment
        loop — one long context must not inflate the padded gather for the
        whole batch.  Routing never marks decode rows, so every index
        returned here is an unrouted segment wherever it's used.  The result
        is memoized on the per-forward metadata so all layers share it."""
        if not self._is_absorbed:
            return None
        meta = _mla_metadata(ctx)
        if meta.decode_batch_rows is None:
            cu = ctx.cu_seqlens
            idx = [
                i
                for i, ctx_len in enumerate(ctx.context_lens)
                if cu[i + 1] - cu[i] == 1 and ctx_len <= _DECODE_BATCH_MAX_CTX
            ]
            if len(idx) >= _DECODE_BATCH_MIN_ROWS:
                # Ascending context order keeps each chunk's padded span tight.
                idx.sort(key=ctx.context_lens.__getitem__)
            else:
                idx = []
            meta.decode_batch_rows = idx
        return meta.decode_batch_rows or None

    def _materialized_prefill(
        self,
        inner: nn.Module,
        latent_cache: MLAPagedLatentCache,
        layer_idx: int,
        q_nope: mx.array,  # [1, nheads, seq, qk_nope_head_dim]
        q_pe: mx.array,  # [1, nheads, seq, qk_rope_head_dim] (post-RoPE)
        kv_norm: mx.array,  # [1, seq, kv_lora_rank]
        k_pe: mx.array,  # [1, 1, seq, qk_rope_head_dim] (post-RoPE)
        ctx: Any,
        seq_len: int,
        routed: list[bool],
    ) -> mx.array:
        """Materialize full per-head K/V from the absorbed embed_q/unembed_out
        weights (mirroring upstream ``forward_mha``) and run prefill as standard
        MHA via MLX SDPA — much cheaper than the absorbed kv_lora-space path
        (512-wide MQA), with no custom kernel. Returns
        [1, seq, num_heads * v_head_dim] for ``o_proj``.

        ``K_nope = embed_q(kv_norm, transpose=False)``, ``V = unembed_out(kv_norm)``;
        MultiLinear's transpose flag reuses ``quantized_matmul`` so quantized
        weights work unchanged. PE folds into the materialized Q·K (no extra mask).

        Segments with past context (chunked prefill, prefix-cache hits)
        gather their past latents from the paged cache — the scatter in
        ``__call__`` already ran — and materialize their K/V with the same
        recipe; new tokens keep using the in-graph kv_norm/k_pe. Segments not
        flagged in ``routed`` (decode rows, small continuation chunks) run the
        absorbed per-segment attention instead, in packed order."""
        nheads = inner.num_heads
        scale = self._attention_scale()
        # Leading [1, ...] axis so the per-head weights broadcast across heads.
        # MultiLinear's dense `x @ weight` broadcasts a 2-D x against the
        # [nheads, ...] weight, but QuantizedMultiLinear's quantized_matmul does
        # not — a 2-D x collapses the head batch (returns [seq, ...]), which
        # breaks the concat below on quantized checkpoints (GLM-4.7-Flash-4bit).
        # [1, seq, kv_lora] broadcasts correctly for both dense and quantized.
        kvn = kv_norm.reshape(1, seq_len, inner.kv_lora_rank)
        # Batched materialization: one GEMM each over all tokens in the step.
        k_nope = inner.embed_q(kvn, transpose=False)  # [nheads, seq, qk_nope]
        values = inner.unembed_out(kvn)  # [nheads, seq, v_head_dim]
        k_pe_b = mx.broadcast_to(
            k_pe.reshape(1, seq_len, inner.qk_rope_head_dim),
            (nheads, seq_len, inner.qk_rope_head_dim),
        )
        keys = mx.concatenate([k_nope, k_pe_b], axis=-1)  # [nheads, seq, qk]
        queries = mx.concatenate([q_nope[0], q_pe[0]], axis=-1)  # [nheads, seq, qk]

        cu = ctx.cu_seqlens
        outs: list[mx.array | None] = [None] * len(ctx.context_lens)
        # Decode rows packed next to a prefill take one batched absorbed pass
        # instead of one SDPA dispatch each (same math as the per-segment
        # loop), when the batching gates accept them. Unrouted rows outside
        # the gate — decode rows over the context cap, small chunks — fall
        # through to the per-segment absorbed loop below.
        decode_idx = self._decode_batch_rows(ctx)
        if decode_idx is not None:
            dec = self._absorbed_decode_batch(
                inner, latent_cache, layer_idx, q_nope, q_pe, ctx, decode_idx
            )  # [d, nheads, 1, v_head_dim]
            for j, i in enumerate(decode_idx):
                outs[i] = dec[j].reshape(1, nheads, inner.v_head_dim)
        for i, ctx_len in enumerate(ctx.context_lens):  # per-request causal MHA
            s, e = cu[i], cu[i + 1]
            num_new = e - s
            if outs[i] is not None:
                continue
            if not routed[i]:
                out = self._absorbed_segment(
                    inner, latent_cache, layer_idx, q_nope, q_pe, ctx, i
                )  # [1, num_new, nheads * v_head_dim]
                outs[i] = out[0].reshape(num_new, nheads, inner.v_head_dim)
                continue
            past = ctx_len - num_new
            k_i = keys[:, s:e]
            v_i = values[:, s:e]
            if past > 0:
                # Continuation chunk / prefix-cache hit: gather this segment's
                # past latents from the paged cache and materialize their K/V
                # with the same recipe (leading [1, past, kv_lora] axis so
                # QuantizedMultiLinear broadcasts across heads as above).
                rows = _block_table_rows(ctx)
                n_blocks = math.ceil(past / latent_cache.block_size)
                past_latent = (
                    latent_cache.latent_caches[layer_idx][rows[i][:n_blocks]]
                    .reshape(-1, latent_cache.latent_dim)[:past]
                    .astype(kv_norm.dtype)
                )
                kvn_past = past_latent[:, : inner.kv_lora_rank].reshape(
                    1, past, inner.kv_lora_rank
                )
                k_pe_past = mx.broadcast_to(
                    past_latent[:, inner.kv_lora_rank :].reshape(
                        1, past, inner.qk_rope_head_dim
                    ),
                    (nheads, past, inner.qk_rope_head_dim),
                )
                k_past = mx.concatenate(
                    [inner.embed_q(kvn_past, transpose=False), k_pe_past],
                    axis=-1,
                )  # [nheads, past, qk]
                v_past = inner.unembed_out(kvn_past)  # [nheads, past, v_head_dim]
                k_i = mx.concatenate([k_past, k_i], axis=1)
                v_i = mx.concatenate([v_past, v_i], axis=1)
            out = scaled_dot_product_attention(
                queries[:, s:e][None],
                k_i[None],
                v_i[None],
                cache=None,
                scale=scale,
                mask="causal",  # lower-right aligned: covers past + new keys
            )  # [1, nheads, num_new, v_head_dim]
            outs[i] = out[0].transpose(1, 0, 2)  # [num_new, nheads, v_head_dim]
        # Every request index is covered above (batched decode, unrouted
        # absorbed, or materialized); fail loudly if that invariant breaks
        # rather than silently dropping a segment's tokens.
        assert all(o is not None for o in outs)
        filled = cast(list[mx.array], outs)
        final = mx.concatenate(filled, axis=0) if len(filled) > 1 else filled[0]
        return final.reshape(1, seq_len, nheads * inner.v_head_dim)

    def _apply_absorbed_mla_attention(
        self,
        *,
        rq_nope: mx.array,
        rq_pe: mx.array,
        all_kv_norm: mx.array,
        k_pe: mx.array,
        causal_mask: mx.array | None,
    ) -> mx.array:
        inner = self._inner
        scale = self._attention_scale()
        # A matmul broadcast over the query heads rereads K once per head when
        # b > 1, and MLX's SDPA takes that path for head dims past 256. The
        # heads share one latent head, so fold them into the query axis.
        b, h, q_len, _ = rq_pe.shape

        # PE branch: q_pe · k_pe contributes an additive score bias.
        # Passing this as the `mask` to scaled_dot_product_attention adds it
        # to the nope scores before softmax, matching the original model exactly.
        pe_scores = (
            (rq_pe * scale).reshape(b, 1, h * q_len, -1) @ k_pe.swapaxes(-1, -2)
        ).reshape(b, h, q_len, -1)
        if causal_mask is not None:
            fill = mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
            pe_scores = mx.where(causal_mask, pe_scores, fill)

        # Nope branch: embed_q absorbs q_nope into kv_lora_rank space;
        # kv_norm is shared across heads as k=v (single-head broadcast).
        # Leading dims (none, or a request axis from _absorbed_decode_batch)
        # are preserved.
        ctx_len = all_kv_norm.shape[-2]
        rq_nope_proj = inner.embed_q(rq_nope)
        kv = all_kv_norm.reshape(-1, 1, ctx_len, inner.kv_lora_rank)

        out = scaled_dot_product_attention(
            rq_nope_proj.reshape(b, 1, h * q_len, -1),
            kv,
            kv,
            cache=None,
            scale=scale,
            mask=pe_scores.reshape(b, 1, h * q_len, ctx_len),
        ).reshape(b, h, q_len, -1)
        return inner.unembed_out(out)  # recover v_head_dim from kv_lora_rank

    def _apply_kv_b_proj_attention(
        self,
        *,
        rq_nope: mx.array,
        rq_pe: mx.array,
        all_kv_norm: mx.array,
        k_pe: mx.array,
        causal_mask: mx.array | None,
    ) -> mx.array:
        inner = self._inner
        scale = self._attention_scale()
        ctx_len = all_kv_norm.shape[0]

        # MiniCPM3-style MLA keeps a single kv_b_proj instead of pre-split
        # embed_q/unembed_out modules. Rebuild K/V from the cached latent using
        # the model's own projection to preserve quantized Linear behavior and
        # the source model's layout.
        kv = inner.kv_b_proj(all_kv_norm.reshape(1, ctx_len, inner.kv_lora_rank))
        kv = kv.reshape(1, ctx_len, inner.num_heads, -1).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [inner.qk_nope_head_dim], axis=-1)
        k_pe = mx.broadcast_to(
            k_pe,
            (1, inner.num_heads, ctx_len, inner.qk_rope_head_dim),
        )
        queries = mx.concatenate([rq_nope, rq_pe], axis=-1)
        keys = mx.concatenate([k_nope, k_pe], axis=-1)
        attn_mask = None
        if causal_mask is not None:
            fill = mx.array(mx.finfo(queries.dtype).min, queries.dtype)
            attn_mask = mx.where(causal_mask, mx.array(0, queries.dtype), fill)

        return scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=scale, mask=attn_mask
        )

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)
        if not ctx.block_tables:
            raise RuntimeError(
                "MLAPagedAttentionWrapper called with empty block_tables"
            )

        inner = self._inner
        layer_idx: int = self._mla_layer_idx
        latent_cache: MLAPagedLatentCache = self._mla_latent_cache

        _, seq_len, _ = x.shape  # B=1, seq_len = total new tokens across all requests

        # Query path — q_lora_rank is None for models without query compression
        if inner.q_lora_rank is None:
            q = inner.q_proj(x)
        else:
            q = inner.q_b_proj(inner.q_a_layernorm(inner.q_a_proj(x)))
        q = q.reshape(1, seq_len, inner.num_heads, self._q_head_dim()).transpose(
            0, 2, 1, 3
        )
        q_nope, q_pe = mx.split(q, [inner.qk_nope_head_dim], axis=-1)

        # KV path — kv_a_proj produces both the lora latent and the rope key in one shot
        kv_out = inner.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe_raw = mx.split(kv_out, [inner.kv_lora_rank], axis=-1)
        kv_norm = inner.kv_a_layernorm(compressed_kv)  # what ends up in the cache
        k_pe = k_pe_raw.reshape(1, seq_len, 1, inner.qk_rope_head_dim).transpose(
            0, 2, 1, 3
        )

        # RoPE is applied per request segment so each request starts at its own position
        q_pe, k_pe = apply_packed_rope(
            inner,
            q_pe,
            k_pe,
            ctx.cu_seqlens,
            offsets=ctx.offsets or None,
        )

        # Concatenate kv_norm and the roped k_pe into a single per-token latent,
        # then scatter it into the scheduler-assigned cache slots. Shared
        # upstream views use an alias-preserving native write.
        k_pe_seq = k_pe.transpose(0, 2, 1, 3).reshape(
            1, seq_len, inner.qk_rope_head_dim
        )
        latent_new = mx.concatenate([kv_norm, k_pe_seq], axis=-1)
        latent_flat = latent_new.reshape(seq_len, latent_cache.latent_dim).astype(
            latent_cache.dtype
        )

        latent_cache.write_slots(
            layer_idx, _mla_metadata(ctx).slot_mapping, latent_flat
        )

        # Materialized-prefill fast path (opt-in; absorbed model; routed per
        # segment — segments with cached context only when the chunk has >= 512
        # new tokens, decode rows never). Materializes full K/V and runs
        # standard MHA via MLX SDPA instead of the absorbed 512-wide MQA loop
        # (RFC #360 Phase 2); unrouted segments take the absorbed attention.
        routed = self._materialized_segments(inner, ctx)
        if routed is not None:
            final = self._materialized_prefill(
                inner,
                latent_cache,
                layer_idx,
                q_nope,
                q_pe,
                kv_norm,
                k_pe,
                ctx,
                seq_len,
                routed,
            )
            # For all-fresh batches `final` is computed from kv_norm/k_pe
            # directly and does not gather
            # from latent_caches[layer_idx]. The SDPA loop below gathers, so its
            # scatter-write rides the logits graph and is forced by the runner's
            # eval; here that dependency is absent, so the cache write would stay
            # lazy and be deferred into the first decode (inflating its latency
            # and the live graph across all-prefill steps). Schedule it now so it
            # lands during prefill, overlapped with the rest of the forward.
            mx.async_eval(latent_cache.latent_caches[layer_idx])
            return self._project_output(x, final)

        # Env-gated single-pass Metal kernel fast path. Falls through
        # to the per-request MLX SDPA loop below when the gate rejects
        # (VLLM_METAL_MLA_KERNEL unset, wrong inner dims, non-decode,
        # or unsupported block_size / dtype).
        if self._can_use_kernel(inner, latent_cache, ctx):
            final = self._kernel_fast_path_single_pass(
                inner, latent_cache, layer_idx, q_nope, q_pe, ctx, seq_len
            )
            return self._project_output(x, final)

        # Short-context one-token decode rows take one padded batched pass
        # (one gather + one SDPA) instead of one SDPA dispatch each; padded
        # columns are masked. Longer decode rows and multi-token segments
        # keep the per-segment loop below (MiniCPM3-style kv_b_proj always
        # loops — its attention is per-head already, not absorbed).
        n_reqs = len(ctx.context_lens)
        decode_idx = self._decode_batch_rows(ctx)
        batched = (
            self._absorbed_decode_batch(
                inner, latent_cache, layer_idx, q_nope, q_pe, ctx, decode_idx
            )
            if decode_idx is not None
            else None
        )  # [len(decode_idx), nheads, 1, v_head_dim] in decode_idx order
        batched_at = {i: j for j, i in enumerate(decode_idx or [])}

        outputs = []
        for req_idx in range(n_reqs):
            j = batched_at.get(req_idx)
            if j is not None:
                outputs.append(batched[j].reshape(1, 1, -1))
            else:
                outputs.append(
                    self._absorbed_segment(
                        inner, latent_cache, layer_idx, q_nope, q_pe, ctx, req_idx
                    )
                )

        final = mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
        return self._project_output(x, final)
