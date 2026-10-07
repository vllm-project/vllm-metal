# SPDX-License-Identifier: Apache-2.0
"""Apply upstream RoPE caches to packed Q/K in one precompiled Metal dispatch."""

from functools import cache

import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


@cache
def _kernel(dtype):
    from vllm_metal import envs
    from vllm_metal.metal.build import prepare_metallib

    path = prepare_metallib(
        "rope_kern", build_from_source=envs.VLLM_METAL_BUILD_FROM_SOURCE
    )
    library = torch.mps.load_metallib(path)
    name = "rope_qk_fp16" if dtype == torch.float16 else "rope_qk_bf16"
    return library, getattr(library, name)


def apply_rope(positions, query, key, cos_sin_cache, head_size):
    if positions.numel() == 0:
        return query, key
    positions = positions.flatten().contiguous()
    q = query.reshape(positions.numel(), -1)
    k = key.reshape(positions.numel(), -1)
    out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    out_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _, kernel = _kernel(q.dtype)
    kernel(
        q,
        k,
        cos_sin_cache,
        positions,
        out_q,
        out_k,
        (q.shape[1], k.shape[1], head_size, q.stride(0), k.stride(0)),
        threads=((q.shape[1] + k.shape[1]) // 2, len(q), 1),
        group_size=(256, 1, 1),
    )
    return out_q.view(query.shape), out_k.view(key.shape)


@RotaryEmbedding.register_oot
class MPSRotaryEmbedding(RotaryEmbedding):
    def forward_native(self, positions, query, key=None):
        if (
            key is None
            or not self.is_neox_style
            or self.rotary_dim != self.head_size
            or query.device.type != "mps"
            or query.dtype not in (torch.float16, torch.bfloat16)
            or key.dtype != query.dtype
            or key.device != query.device
            or positions.device != query.device
            or positions.dtype != torch.int64
            or query.stride(-1) != 1
            or key.stride(-1) != 1
        ):
            return super().forward_native(positions, query, key)
        return apply_rope(
            positions,
            query,
            key,
            self._match_cos_sin_cache_dtype(query),
            self.head_size,
        )

    forward_oot = forward_native
