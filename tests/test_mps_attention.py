# SPDX-License-Identifier: Apache-2.0
"""MPS-stream dispatch parity, including KV layout, split decode and verification."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Requires Apple Silicon MPS"
)


def test_mps_prebuilt_loading_without_mlx_or_compiler():
    # Exercise VM fallback in a fresh process without MLX or a compiler.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["mlx"] = None
sys.modules["_paged_ops"] = None
sys.modules["torch.utils.cpp_extension"] = None
from vllm_metal.pytorch_backend.mps_ops import _load_mps_module, get_mps_ops
module = _load_mps_module()
module._override_detected_gpu_core_count_for_test(0)
assert module.detected_gpu_core_count() == 0
get_mps_ops()
assert module.gpu_core_count() == module.gpu_core_count() == 14
assert sys.modules["mlx"] is None
assert sys.modules["_paged_ops"] is None
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr.count("GPU core count is unknown") == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "head_size,block_size,window,softcap",
    [
        (64, 8, None, None),
        (96, 16, 33, 2.0),
        (128, 32, None, 0.0),
        (256, 16, 33, None),
        (512, 8, None, 2.0),
    ],
)
@pytest.mark.parametrize(
    "force_tiled,counts,starts,prefilling",
    [
        (False, [39, 1], [17, 700], [True, False]),
        (True, [39, 1], [17, 700], [True, False]),
        (False, [1, 1], [17, 700], None),
        (False, [1, 1], [17, 40], [False, False]),
        (False, [3, 4], [511, 700], [False, False]),
        (False, [5, 1], [17, 700], [False, False]),
        (False, [2, 2], [17, 40], [False, False]),
        (False, [2, 2], [17, 40], [True, False]),
    ],
)
def test_mps_paged_attention(
    dtype,
    head_size,
    block_size,
    window,
    softcap,
    force_tiled,
    counts,
    starts,
    prefilling,
):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from vllm_metal.pytorch_backend.attention import (
        MPSAttentionImpl,
        MPSAttentionMetadataBuilder,
    )

    torch.manual_seed(8)
    total = sum(counts)
    # Deliberately strided K/V inputs like vLLM's packed QKV projection.
    qkv = torch.randn(total, 32, head_size, dtype=dtype) * 0.25
    q, k, v = qkv.split([16, 8, 8], dim=1)
    blocks_per_seq = (max(starts) + max(counts) + block_size - 1) // block_size
    table = [
        list(range(1 + i * blocks_per_seq, 1 + (i + 1) * blocks_per_seq))
        for i in range(len(counts))
    ]
    cache = (
        torch.randn(
            1 + blocks_per_seq * len(counts), block_size, 8, 2 * head_size, dtype=dtype
        )
        * 0.25
    )
    expected_cache = cache.clone()
    reference = []
    offset = 0
    for blocks, start, count in zip(table, starts, counts, strict=True):
        for j in range(count):
            p = start + j
            expected_cache[blocks[p // block_size], p % block_size, :, :head_size] = k[
                offset + j
            ]
            expected_cache[blocks[p // block_size], p % block_size, :, head_size:] = v[
                offset + j
            ]
        history = expected_cache[blocks].flatten(0, 1)[: start + count]
        mask = (
            torch.arange(start + count)[None, :]
            <= torch.arange(start, start + count)[:, None]
        )
        if window is not None:
            mask &= torch.arange(start + count)[None, :] >= (
                torch.arange(start, start + count)[:, None] + 1 - window
            )
        q_ref = q[offset : offset + count].transpose(0, 1).float()
        k_ref = (
            history[..., :head_size].repeat_interleave(2, dim=1).transpose(0, 1).float()
        )
        v_ref = (
            history[..., head_size:].repeat_interleave(2, dim=1).transpose(0, 1).float()
        )
        scores = (q_ref @ k_ref.transpose(-1, -2)) * head_size**-0.5
        if softcap:
            scores = softcap * (scores / softcap).tanh()
        ref = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v_ref
        reference.append(ref.transpose(0, 1))
        offset += count
    cu = [0]
    slots = []
    for blocks, start, count in zip(table, starts, counts, strict=True):
        cu.append(cu[-1] + count)
        slots.extend(
            blocks[p // block_size] * block_size + p % block_size
            for p in range(start, start + count)
        )
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor(cu, dtype=torch.int32, device="mps"),
        query_start_loc_cpu=torch.tensor(cu, dtype=torch.int32),
        num_reqs=len(counts),
        num_actual_tokens=total,
        max_query_len=max(counts),
        seq_lens=torch.tensor(
            [s + c for s, c in zip(starts, counts, strict=True)],
            dtype=torch.int32,
            device="mps",
        ),
        max_seq_len=max(s + c for s, c in zip(starts, counts, strict=True)),
        block_table_tensor=torch.tensor(table, dtype=torch.int32, device="mps"),
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device="mps"),
        is_prefilling=torch.tensor(prefilling) if prefilling is not None else None,
    )
    metadata = MPSAttentionMetadataBuilder(None, [], None, torch.device("mps")).build(
        0, common
    )
    assert metadata.window_seqlen_q == (
        max(counts) if prefilling == [False, False] else 1
    )
    gpu_qkv = qkv.to("mps")
    gpu_q, gpu_k, gpu_v = gpu_qkv.split([16, 8, 8], dim=1)
    gpu_cache = cache.to("mps")
    out = torch.empty_like(gpu_q, memory_format=torch.contiguous_format)
    impl = MPSAttentionImpl(
        16,
        head_size,
        head_size**-0.5,
        num_kv_heads=8,
        sliding_window=window,
        logits_soft_cap=softcap,
    )
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
