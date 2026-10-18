# SPDX-License-Identifier: Apache-2.0
"""GLM-5-Next sparse MLA on vLLM's paged latent and pooled-key caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole

from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.caches.storage import KVCacheStorage

# vLLM's name for this cache, built inline by its GLM-5-Next Indexer.
INDEXER_CACHE_SUFFIX = ".indexer.k_cache"


def is_sparse_mla(model_args: Mapping[str, Any]) -> bool:
    """Whether the model's MLA layers carry GLM-5-Next's pooled indexer."""
    return "index_kpool" in model_args


def indexer_cache_spec(
    model_args: Mapping[str, Any], *, block_size: int, dtype: torch.dtype
) -> MLAAttentionSpec:
    """vLLM's pooled-key spec for one sparse layer.

    It is in the model dtype; vLLM's ``Glm5NextIndexerCache`` builds an fp8 one
    inline.
    """
    if model_args.get("qk_rope_head_dim") != 0:
        raise ValueError(
            "GLM-5-Next sparse MLA: needs qk_rope_head_dim 0 (NoPE), got "
            f"{model_args.get('qk_rope_head_dim')!r}."
        )
    if any(kind != "full" for kind in model_args.get("indexer_types") or ()):
        raise NotImplementedError(
            "GLM-5-Next sparse MLA: layers that reuse another layer's top-k "
            "(indexer_types 'shared') are not supported."
        )
    kpool = int(model_args["index_kpool"])
    head_dim = int(model_args["index_head_dim"])
    # vLLM asserts this in Glm5NextIndexerCache.__init__.
    if block_size % kpool:
        raise ValueError(
            f"GLM-5-Next sparse MLA: KV cache block_size {block_size} must be a "
            f"multiple of index_kpool {kpool}, so pools never straddle blocks."
        )
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_dim,
        dtype=dtype,
        cache_role=SparseCacheRole.INDEXER,
        tokens_per_state=kpool,
    )


def max_latent_blocks(*, block_size: int, latent_dim: int) -> int:
    """The most latent blocks mlx-vlm's kernel can address with 32-bit offsets."""
    return (2**31 - 1) // (block_size * latent_dim)


@dataclass(frozen=True)
class SparseIndexerCaches:
    """The sparse layers' pooled-key pages and their unfinished pools' rows."""

    # [num_blocks, block_size // index_kpool, index_head_dim], vLLM's pages
    pooled: MLAPagedLatentCache
    # [num_blocks, index_kpool, 2 * index_head_dim]: raw key | gate of the
    # unfinished pool in that latent block, at its position % index_kpool
    tail: MLAPagedLatentCache

    @classmethod
    def from_upstream(
        cls,
        storage: KVCacheStorage,
        attention_names: Sequence[str],
        *,
        latent: MLAPagedLatentCache,
    ) -> SparseIndexerCaches:
        """Bind vLLM's pooled keys and allocate the tail rows.

        The pooled keys sit next to the latent in vLLM's layout; the tail rows
        sit beside each latent block, in a table of their own.
        """
        limit = max_latent_blocks(
            block_size=latent.block_size, latent_dim=latent.latent_dim
        )
        if latent.num_blocks > limit:
            raise ValueError(
                f"GLM-5-Next sparse MLA: {latent.num_blocks} KV blocks exceed "
                f"the {limit} that mlx-vlm's indexed attention kernel can "
                "address with its 32-bit offsets; lower "
                "--num-gpu-blocks-override."
            )
        if not latent.has_dense_pages:
            raise ValueError("GLM-5-Next sparse MLA: requires dense latent pages.")
        pooled = MLAPagedLatentCache.from_upstream(
            storage, [name + INDEXER_CACHE_SUFFIX for name in attention_names]
        )
        kpool = latent.block_size // pooled.block_size
        # vLLM's ring is a KV group of its own, for KV transfer and speculative
        # tokens; neither runs here, so the rows sit beside their latent block.
        tail = MLAPagedLatentCache(
            num_layers=len(attention_names),
            latent_dim=2 * pooled.latent_dim,
            num_blocks=latent.num_blocks,
            block_size=kpool,
            dtype=latent.dtype,
        )
        return cls(pooled=pooled, tail=tail)
