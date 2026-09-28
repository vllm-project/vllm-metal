# SPDX-License-Identifier: Apache-2.0
"""MPS-stream dispatch parity, including interleaved KV and split decode."""

from pathlib import Path

import pytest
import torch
from torch.nn import functional

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Requires Apple Silicon MPS"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("force_tiled", [False, True])
def test_mps_paged_attention(dtype, force_tiled):
    from vllm_metal.pytorch_backend.attention import (
        MPSAttentionImpl,
        MPSAttentionMetadata,
    )

    torch.manual_seed(8)
    # One batch exercises cached prefill and decode spanning two partitions.
    counts, starts = [39, 1], [17, 700]
    total = sum(counts)
    # Deliberately strided K/V inputs like vLLM's packed QKV projection.
    qkv = torch.randn(total, 32, 128, dtype=dtype) * 0.25
    q, k, v = qkv.split([16, 8, 8], dim=1)
    table = [list(range(1 + i * 50, 1 + (i + 1) * 50)) for i in range(len(counts))]
    cache = torch.randn(1 + 50 * len(counts), 16, 8, 256, dtype=dtype) * 0.25
    expected_cache = cache.clone()
    reference = []
    offset = 0
    for blocks, start, count in zip(table, starts, counts, strict=True):
        for j in range(count):
            p = start + j
            expected_cache[blocks[p // 16], p % 16, :, :128] = k[offset + j]
            expected_cache[blocks[p // 16], p % 16, :, 128:] = v[offset + j]
        history = expected_cache[blocks].flatten(0, 1)[: start + count]
        mask = (
            torch.arange(start + count)[None, :]
            <= torch.arange(start, start + count)[:, None]
        )
        ref = functional.scaled_dot_product_attention(
            q[offset : offset + count].transpose(0, 1).float(),
            history[..., :128].repeat_interleave(2, dim=1).transpose(0, 1).float(),
            history[..., 128:].repeat_interleave(2, dim=1).transpose(0, 1).float(),
            attn_mask=mask,
        )
        reference.append(ref.transpose(0, 1))
        offset += count
    cu = [0]
    slots = []
    for blocks, start, count in zip(table, starts, counts, strict=True):
        cu.append(cu[-1] + count)
        slots.extend(blocks[p // 16] * 16 + p % 16 for p in range(start, start + count))
    metadata = MPSAttentionMetadata(
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device="mps"),
        seq_lens=torch.tensor(
            [s + c for s, c in zip(starts, counts, strict=True)],
            dtype=torch.int32,
            device="mps",
        ),
        max_seq_len=max(s + c for s, c in zip(starts, counts, strict=True)),
        block_tables=torch.tensor(table, dtype=torch.int32, device="mps"),
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device="mps"),
    )
    gpu_qkv = qkv.to("mps")
    gpu_q, gpu_k, gpu_v = gpu_qkv.split([16, 8, 8], dim=1)
    gpu_cache = cache.to("mps")
    out = torch.empty_like(gpu_q, memory_format=torch.contiguous_format)
    impl = MPSAttentionImpl(16, 128, 128**-0.5, num_kv_heads=8)
    if force_tiled:
        metal = Path(__file__).resolve().parents[1] / "vllm_metal" / "metal"
        impl.ops = type(impl.ops)(
            str(metal / "paged_attention_v2_kern.metallib"), "", 20
        )
    impl.forward(
        None,
        gpu_q,
        gpu_k,
        gpu_v,
        gpu_cache.transpose(1, 2),  # vLLM layout; forward adapts it for Metal.
        metadata,
        out,
    )
    torch.testing.assert_close(
        out.cpu().float(), torch.cat(reference), atol=0.003, rtol=0.03
    )
    torch.testing.assert_close(gpu_cache.cpu(), expected_cache, atol=0, rtol=0)
