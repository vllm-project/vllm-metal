# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from vllm.logger import init_logger

from vllm_metal.metal_kernel_backend.packed_prefill_compat import apply_packed_rope
from vllm_metal.mlx_backend.mla_cache import MLAPagedLatentCache
from vllm_metal.paged_attention_common import find_layers_and_attr, get_context

logger = init_logger(__name__)


class MLAPagedAttentionWrapper(nn.Module):
    """Wraps an MLA attention module to use a paged latent cache.

    MLA (GLM/DeepSeek lineage) compresses KV into a latent before caching:

        latent = [kv_norm || k_pe_roped]  # kv_lora_rank + qk_rope_head_dim dims

    Each call scatter-writes the new tokens' latents into the scheduled cache
    slots, then gather-reads all past latents per request via block tables.

    At decode time the original model absorbs embed_q into q_nope (projects it
    into kv_lora_rank space) and sets k=v=kv_norm shared across all heads.
    This wrapper uses the same formulation for prefill too — the decode and
    prefill attention scores are identical by linearity of embed_q. After the
    attention output, unembed_out maps back to v_head_dim.

    When no PagedAttentionContext is active the original module is called as-is.
    """

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
            offsets=ctx.offsets if ctx.offsets else None,
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
        flat[mx.array(ctx.slot_mapping, dtype=mx.uint32)] = latent_flat
        latent_cache.latent_caches[layer_idx] = flat.reshape(
            latent_cache.num_blocks, latent_cache.block_size, latent_cache.latent_dim
        )

        outputs = []
        for req_idx, (block_ids, ctx_len) in enumerate(
            zip(ctx.block_tables, ctx.context_lens, strict=True)
        ):
            req_start = ctx.cu_seqlens[req_idx]
            req_end = ctx.cu_seqlens[req_idx + 1]
            num_new = req_end - req_start
            past_len = ctx_len - num_new  # tokens cached before this step

            # Gather this request's full context from the paged cache.
            # Block indexing: each block holds block_size contiguous token slots.
            n_blocks = math.ceil(ctx_len / latent_cache.block_size)
            blocks = mx.array(block_ids[:n_blocks], dtype=mx.uint32)
            all_latent = latent_cache.latent_caches[layer_idx][blocks].reshape(
                -1, latent_cache.latent_dim
            )[:ctx_len]

            all_kv_norm = all_latent[:, : inner.kv_lora_rank]
            all_k_pe = all_latent[:, inner.kv_lora_rank :]

            rq_nope = q_nope[:, :, req_start:req_end, :]
            rq_pe = q_pe[:, :, req_start:req_end, :]

            # PE branch: q_pe · k_pe contributes an additive score bias.
            # Passing this as the `mask` to scaled_dot_product_attention adds it
            # to the nope scores before softmax, matching the original model exactly.
            k_pe_r = all_k_pe.reshape(1, 1, ctx_len, inner.qk_rope_head_dim)
            pe_scores = (rq_pe * inner.scale) @ k_pe_r.swapaxes(-1, -2)

            # Causal mask for prefill: new token i can attend to positions 0..past_len+i.
            # Decode (num_new==1) needs no mask — the single token attends everywhere.
            if num_new > 1:
                rows = mx.arange(num_new).reshape(-1, 1)
                cols = mx.arange(ctx_len).reshape(1, -1)
                valid = (cols <= (past_len + rows)).reshape(1, 1, num_new, ctx_len)
                fill = mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
                pe_scores = mx.where(valid, pe_scores, fill)

            # Nope branch: embed_q absorbs q_nope into kv_lora_rank space;
            # kv_norm is shared across heads as k=v (single-head broadcast).
            rq_nope_proj = inner.embed_q(rq_nope)
            kv = all_kv_norm.reshape(1, 1, ctx_len, inner.kv_lora_rank)

            out = scaled_dot_product_attention(
                rq_nope_proj, kv, kv, cache=None, scale=inner.scale, mask=pe_scores
            )
            out = inner.unembed_out(out)  # recover v_head_dim from kv_lora_rank
            out = out.transpose(0, 2, 1, 3).reshape(1, num_new, -1)
            outputs.append(out)

        final = mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
        return inner.o_proj(final)


class MLAPagedAttentionBackend:
    """Paged attention backend for MLA models (GLM/DeepSeek lineage).

    Implements the PagedAttentionBackend protocol. Uses MLX-native
    scatter/gather (no vendored C++/Metal kernel) because MLA latents
    do not fit the standard (num_heads, head_dim) kernel layout.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        block_size: int,
        dtype: mx.Dtype,
    ) -> None:
        self._num_layers = num_layers
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
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
            kv_lora_rank=self._kv_lora_rank,
            qk_rope_head_dim=self._qk_rope_head_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
        )

    def patch_model(self, model: Any) -> int:
        cache = self._require_initialized("patch_model")
        return self._patch_model(model, cache)

    def _patch_model(self, model: Any, latent_cache: MLAPagedLatentCache) -> int:
        layer_list, attn_attr = find_layers_and_attr(model)
        patched = 0

        for layer_idx, layer in enumerate(layer_list):
            attn = getattr(layer, attn_attr)
            if isinstance(attn, MLAPagedAttentionWrapper):
                # Already patched — refresh cache reference (e.g. after re-initialisation)
                object.__setattr__(attn, "_mla_latent_cache", latent_cache)
                patched += 1
                continue

            setattr(
                layer, attn_attr, MLAPagedAttentionWrapper(attn, layer_idx, latent_cache)
            )
            patched += 1

        return patched

    def warm_up(self) -> None:
        # MLX ops JIT-compile on first use; no Metal shader warm-up needed.
        self._require_initialized("warm_up")
        logger.info("MLA paged attention (MLX-native): skipping Metal kernel warm-up")

    def num_blocks(self) -> int:
        return self._require_initialized("num_blocks").num_blocks
