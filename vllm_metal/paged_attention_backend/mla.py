# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention

from vllm_metal import envs
from vllm_metal.metal import warm_up_kernels
from vllm_metal.metal_kernel_backend.packed_prefill_compat import apply_packed_rope
from vllm_metal.mlx_backend.mla_cache import MLAPagedLatentCache
from vllm_metal.paged_attention_common import find_attn_attr, find_layers, get_context

# Default rope head dim for GLM/DeepSeek-V2 lineage models.
# Used as fallback when qk_rope_head_dim is absent from model config.
MLA_DEFAULT_QK_ROPE_HEAD_DIM = 64


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
    _KERNEL_BLOCK_SIZES = frozenset({16, 32})

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
        if is_absorbed:
            object.__setattr__(
                self, "_apply_mla_attention", self._apply_absorbed_mla_attention
            )
        else:
            object.__setattr__(
                self, "_apply_mla_attention", self._apply_kv_b_proj_attention
            )

    def _attention_scale(self) -> float:
        inner = self._inner
        scale = getattr(inner, "scale", None)
        if scale is None:
            scale = inner.softmax_scale
        return scale

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
        if not envs.VLLM_METAL_MLA_KERNEL:
            return False
        if not self._is_absorbed:
            return False
        if inner.kv_lora_rank != self._KERNEL_KV_LORA_RANK:
            return False
        if inner.qk_rope_head_dim != self._KERNEL_QK_ROPE_HEAD_DIM:
            return False
        if latent_cache.block_size not in self._KERNEL_BLOCK_SIZES:
            return False
        if latent_cache.dtype not in (mx.float16, mx.bfloat16):
            return False
        cu = ctx.cu_seqlens
        for i in range(len(ctx.context_lens)):
            if cu[i + 1] - cu[i] != 1:
                return False
        return True

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

        # Pad block_tables (list[list[int]]) into a 2D [num_seqs, max_blocks]
        # int32 array. The kernel reads block_table_row[0..n_context_blocks-1];
        # padding entries beyond n_context_blocks are never read.
        bts = ctx.block_tables
        max_blocks = max(len(bt) for bt in bts)
        padded = [bt + [0] * (max_blocks - len(bt)) for bt in bts]
        block_tables_mx = mx.array(padded, dtype=mx.int32)

        context_lens_mx = mx.array(list(ctx.context_lens), dtype=mx.uint32)
        cu_seqlens_q_mx = mx.array(list(ctx.cu_seqlens), dtype=mx.int32)

        out_kvr = metal_mla_paged_attention(
            q_nope=q_nope_kernel,
            q_pe=q_pe_kernel,
            latent_cache=latent_cache.latent_caches[layer_idx],
            block_tables=block_tables_mx,
            context_lens=context_lens_mx,
            cu_seqlens_q=cu_seqlens_q_mx,
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

        # PE branch: q_pe · k_pe contributes an additive score bias.
        # Passing this as the `mask` to scaled_dot_product_attention adds it
        # to the nope scores before softmax, matching the original model exactly.
        pe_scores = (rq_pe * scale) @ k_pe.swapaxes(-1, -2)
        if causal_mask is not None:
            fill = mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
            pe_scores = mx.where(causal_mask, pe_scores, fill)

        # Nope branch: embed_q absorbs q_nope into kv_lora_rank space;
        # kv_norm is shared across heads as k=v (single-head broadcast).
        ctx_len = all_kv_norm.shape[0]
        rq_nope_proj = inner.embed_q(rq_nope)
        kv = all_kv_norm.reshape(1, 1, ctx_len, inner.kv_lora_rank)

        out = scaled_dot_product_attention(
            rq_nope_proj, kv, kv, cache=None, scale=scale, mask=pe_scores
        )
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
        q = q.reshape(1, seq_len, inner.num_heads, inner.q_head_dim).transpose(
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
        # then scatter-write it into the cache at the scheduler-assigned slots.
        # MLX arrays are functional, so the indexed update returns a new array
        # that we explicitly reassign back into the cache list.
        k_pe_seq = k_pe.transpose(0, 2, 1, 3).reshape(
            1, seq_len, inner.qk_rope_head_dim
        )
        latent_new = mx.concatenate([kv_norm, k_pe_seq], axis=-1)
        latent_flat = latent_new.reshape(seq_len, latent_cache.latent_dim).astype(
            latent_cache.dtype
        )

        flat = latent_cache.latent_caches[layer_idx].reshape(
            -1, latent_cache.latent_dim
        )
        flat[mx.array(ctx.slot_mapping, dtype=mx.int64)] = latent_flat
        latent_cache.latent_caches[layer_idx] = flat.reshape(
            latent_cache.num_blocks, latent_cache.block_size, latent_cache.latent_dim
        )

        # Env-gated single-pass Metal kernel fast path. Falls through
        # to the per-request MLX SDPA loop below when the gate rejects
        # (VLLM_METAL_MLA_KERNEL unset, wrong inner dims, non-decode,
        # or unsupported block_size / dtype).
        if self._can_use_kernel(inner, latent_cache, ctx):
            final = self._kernel_fast_path_single_pass(
                inner, latent_cache, layer_idx, q_nope, q_pe, ctx, seq_len
            )
            return inner.o_proj(final)

        # Pre-convert block tables once to avoid a new mx.array allocation per request
        block_tables_mx = [mx.array(bt, dtype=mx.int32) for bt in ctx.block_tables]

        outputs = []
        for req_idx, ctx_len in enumerate(ctx.context_lens):
            req_start = ctx.cu_seqlens[req_idx]
            req_end = ctx.cu_seqlens[req_idx + 1]
            num_new = req_end - req_start
            past_len = ctx_len - num_new  # tokens cached before this step

            # Gather this request's full context from the paged cache.
            # Block indexing: each block holds block_size contiguous token slots.
            n_blocks = math.ceil(ctx_len / latent_cache.block_size)
            blocks = block_tables_mx[req_idx][:n_blocks]
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

            out = out.transpose(0, 2, 1, 3).reshape(1, num_new, -1)
            outputs.append(out)

        final = mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
        return inner.o_proj(final)


class MLAPagedAttentionBackend:
    """Paged attention backend for MLA models.

    Implements the PagedAttentionBackend protocol. Uses MLX-native
    scatter/gather (cache I/O only; attention is MLX SDPA by default
    or an opt-in single-pass Metal kernel, RFC #360) because MLA
    latents do not fit the standard (num_heads, head_dim) kernel
    layout.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        latent_dim: int,
        block_size: int,
        dtype: mx.Dtype,
    ) -> None:
        self._num_layers = num_layers
        self._latent_dim = latent_dim
        self._block_size = block_size
        self._dtype = dtype
        self._cache: MLAPagedLatentCache | None = None

    def _require_initialized(self, caller: str) -> MLAPagedLatentCache:
        if self._cache is None:
            raise RuntimeError(f"{caller}() called before initialize()")
        return self._cache

    def initialize(self, num_blocks: int) -> None:
        self._cache = MLAPagedLatentCache(
            num_layers=self._num_layers,
            latent_dim=self._latent_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
        )

    def patch_model(self, model: Any) -> int:
        cache = self._require_initialized("patch_model")
        return self._patch_model(model, cache)

    def _patch_model(self, model: Any, latent_cache: MLAPagedLatentCache) -> int:
        patched = 0

        for layer_idx, layer in enumerate(find_layers(model)):
            attn_attr = find_attn_attr(layer)
            if attn_attr is None:
                continue

            attn = getattr(layer, attn_attr)
            if isinstance(attn, MLAPagedAttentionWrapper):
                # Already patched — refresh cache reference (e.g. after re-initialisation)
                object.__setattr__(attn, "_mla_latent_cache", latent_cache)
                patched += 1
                continue

            setattr(
                layer,
                attn_attr,
                MLAPagedAttentionWrapper(attn, layer_idx, latent_cache),
            )
            patched += 1

        return patched

    def warm_up(self) -> None:
        self._require_initialized("warm_up")
        warm_up_kernels()

    def num_blocks(self) -> int:
        return self._require_initialized("num_blocks").num_blocks
