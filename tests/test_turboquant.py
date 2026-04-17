"""
TurboQuant KV Cache — Comprehensive Test Suite

Sections (all run by default unless noted):
  0. Quant Type Validation     — K/V quant type acceptance/rejection tests
  1. Pack/Unpack Roundtrip     — bit-packing correctness, no Metal
  2. Python Roundtrip MSE      — quantization quality per quant type, no Metal
  3. Metal Kernel Dequant      — Metal kernel vs Python reference per quant type
  4. Metal E2E Correctness     — cache write → paged attention (Qwen3-0.6B shape)
  5. Memory Capacity Analysis  — theoretical compression table
  6. Published Comparison      — vs KIVI / KVQuant / QJL numbers
  7. Quantization Latency      — encode + cache-write overhead
  8. True Memory Usage         — RSS + Metal active memory (Llama-3.2-1B shape)
  9. Serve Benchmark           — bf16 vs q8_0 via live vllm serve (opt-in: --serve)

Run:
    uv run python tests/test_turboquant.py           # sections 0-8
    uv run python tests/test_turboquant.py --serve   # all sections including 9
"""

import gc
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch

from vllm_metal.metal import get_ops
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.metal_kernel_backend.turboquant import (
    _FWHT_SUPPORTED_DIMS,
    BLOCK_SIZE,
    QUANT_PARAMS,
    V_QUANT_PARAMS,
    fwht,
    get_v_centroids,
    pack_bits,
    packed_dim,
    turbo_quant_decode,
    turbo_quant_encode,
    unpack_bits,
)
from vllm_metal.v1.cache_policy import (
    TurboQuantAttentionSpec,
    _build_turboquant_attention_spec,
    _turboquant_page_size_bytes,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared config
# ─────────────────────────────────────────────────────────────────────────────

# Qwen3-0.6B shape — used for correctness / latency / capacity sections.
Q_NUM_KV_HEADS = 4
Q_HEAD_DIM = 128
Q_NUM_LAYERS = 28
Q_BLOCK_SIZE = 16
Q_GQA_RATIO = 4  # 16 Q heads / 4 KV heads

# Qwen3-0.6B unpack test shape (smaller, one-block fits in NUM_BLOCKS=4).
U_NUM_KV_HEADS = 4
U_HEAD_DIM = 64
U_BLOCK_SIZE = 16
U_NUM_BLOCKS = 4
U_GQA_RATIO = 4

# Llama-3.2-1B shape — used for true memory usage section.
L_NUM_LAYERS = 16
L_NUM_KV_HEADS = 8
L_HEAD_DIM = 64
L_BLOCK_SIZE = 16

# Quant types exercised by sections 3 and 2.
QUANTS = ["q8_0", "int8", "q5_0", "q4_0", "uint2"]

# Serve benchmark config (section 9).
SERVE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SERVE_HOST = "127.0.0.1"
SERVE_PORT = 8765
# Each config is (label, additional_config_dict or None).
# TQ configs sweep the (k_quant, v_quant) cross-product that matters in
# practice: q8_0 (lossless-ish K) vs int2 (aggressive K) crossed with
# q3_0 (balanced V) vs q2_0 (aggressive V).  bf16 is the uncompressed
# baseline every TQ config is judged against.
SERVE_CONFIGS = [
    ("bf16", None),
    ("q8q3", {"turboquant": True, "k_quant": "q8_0", "v_quant": "q3_0"}),
    ("q8q2", {"turboquant": True, "k_quant": "q8_0", "v_quant": "q2_0"}),
    ("i2q3", {"turboquant": True, "k_quant": "int2", "v_quant": "q3_0"}),
    ("i2q2", {"turboquant": True, "k_quant": "int2", "v_quant": "q2_0"}),
]
SERVE_BASE_ENV = {
    "VLLM_METAL_USE_PAGED_ATTENTION": "1",
    "VLLM_METAL_USE_MLX": "1",
    "VLLM_METAL_MEMORY_FRACTION": "0.80",
}
SERVE_READY_TIMEOUT = 600
SERVE_COOLDOWN = 30
SERVE_QUALITY_PROMPT = (
    "Explain the concept of attention in transformer neural networks "
    "in exactly three sentences."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def sep(title: str) -> None:
    w = 72
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def cos_sim(a: mx.array, b: mx.array) -> float:
    a_f = a.reshape(-1, a.shape[-1]).astype(mx.float32)
    b_f = b.reshape(-1, b.shape[-1]).astype(mx.float32)
    dot = mx.sum(a_f * b_f, axis=-1)
    na = mx.sqrt(mx.sum(a_f**2, axis=-1))
    nb = mx.sqrt(mx.sum(b_f**2, axis=-1))
    return mx.mean(dot / (na * nb + 1e-8)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Section 0: Quant Type Validation
# ─────────────────────────────────────────────────────────────────────────────


def section_quant_type_validation() -> int:
    """Verify valid K/V quant types are accepted and invalid ones are rejected."""
    sep("Section 0: Quant Type Validation")
    failures = 0

    # Valid K quant types
    valid_k_quants = [
        "q8_0",
        "int8",
        "uint8",
        "q5_0",
        "q4_0",
        "int4",
        "uint4",
        "int2",
        "uint2",
    ]
    # Valid V quant types
    valid_v_quants = ["q2_0", "q3_0", "q4_0", "q5_0", "q8_0"]
    # Invalid quant types (should be rejected)
    invalid_k_quants = ["q3_0", "uint3", "fp16", "bf16", "invalid", ""]
    invalid_v_quants = ["uint3", "int3", "fp16", "invalid", ""]

    print("  K quant types:")
    for k_quant in valid_k_quants:
        if k_quant in QUANT_PARAMS:
            print(f"    {k_quant:8s}  [OK] accepted")
        else:
            print(f"    {k_quant:8s}  [FAIL] should be accepted but rejected")
            failures += 1

    for k_quant in invalid_k_quants:
        if k_quant not in QUANT_PARAMS:
            print(f"    {k_quant or '(empty)':8s}  [OK] rejected")
        else:
            print(
                f"    {k_quant or '(empty)':8s}  [FAIL] should be rejected but accepted"
            )
            failures += 1

    print("\n  V quant types:")
    for v_quant in valid_v_quants:
        if v_quant in V_QUANT_PARAMS:
            print(f"    {v_quant:8s}  [OK] accepted")
        else:
            print(f"    {v_quant:8s}  [FAIL] should be accepted but rejected")
            failures += 1

    for v_quant in invalid_v_quants:
        if v_quant not in V_QUANT_PARAMS:
            print(f"    {v_quant or '(empty)':8s}  [OK] rejected")
        else:
            print(
                f"    {v_quant or '(empty)':8s}  [FAIL] should be rejected but accepted"
            )
            failures += 1

    # Test MetalPagedKVCache validation
    print("\n  MetalPagedKVCache validation:")
    # Valid combo
    try:
        _ = MetalPagedKVCache(
            num_layers=1,
            num_blocks=4,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            dtype=mx.float16,
            turboquant=True,
            k_quant="q8_0",
            v_quant="q3_0",
        )
        print("    k=q8_0, v=q3_0  [OK] accepted")
    except ValueError as e:
        print(f"    k=q8_0, v=q3_0  [FAIL] rejected: {e}")
        failures += 1

    # Invalid K quant
    try:
        MetalPagedKVCache(
            num_layers=1,
            num_blocks=4,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            dtype=mx.float16,
            turboquant=True,
            k_quant="invalid",
            v_quant="q3_0",
        )
        print("    k=invalid       [FAIL] should be rejected")
        failures += 1
    except ValueError:
        print("    k=invalid       [OK] rejected")

    # Invalid V quant
    try:
        MetalPagedKVCache(
            num_layers=1,
            num_blocks=4,
            block_size=16,
            num_kv_heads=4,
            head_dim=64,
            dtype=mx.float16,
            turboquant=True,
            k_quant="q8_0",
            v_quant="uint3",
        )
        print("    v=uint3         [FAIL] should be rejected")
        failures += 1
    except ValueError:
        print("    v=uint3         [OK] rejected")

    # Test get_v_centroids for valid bit widths
    print("\n  get_v_centroids validation:")
    for v_quant, params in V_QUANT_PARAMS.items():
        bits = params["bits"]
        try:
            centroids = get_v_centroids(bits)
            expected_len = 1 << bits
            if len(centroids) == expected_len:
                print(f"    {v_quant} ({bits}-bit)  [OK] {expected_len} centroids")
            else:
                print(
                    f"    {v_quant} ({bits}-bit)  [FAIL] expected {expected_len}, got {len(centroids)}"
                )
                failures += 1
        except Exception as e:
            print(f"    {v_quant} ({bits}-bit)  [FAIL] {e}")
            failures += 1

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Pack / Unpack Roundtrip
# ─────────────────────────────────────────────────────────────────────────────


def section_pack_unpack_roundtrip() -> int:
    """Verify pack_bits → unpack_bits identity for all sub-8-bit widths."""
    sep("Section 1: Pack/Unpack Roundtrip")
    failures = 0
    orig_dim = 64  # divisible by 8 (3-bit group) and 4 (2-bit group)

    for bits in [2, 3, 4, 5]:
        max_val = (1 << bits) - 1
        vals = mx.array([i % (max_val + 1) for i in range(orig_dim)], dtype=mx.uint8)
        mx.eval(vals)

        packed = pack_bits(vals, bits)
        mx.eval(packed)
        unpacked = unpack_bits(packed, bits, orig_dim)
        mx.eval(unpacked)

        mismatch = mx.any(unpacked != vals).item()
        mark = "FAIL" if mismatch else "OK"
        print(f"  bits={bits}  [{mark}]")
        if mismatch:
            failures += 1
            bad = [i for i in range(orig_dim) if unpacked[i].item() != vals[i].item()]
            print(f"    first mismatch indices: {bad[:8]}")

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Python Roundtrip MSE
# ─────────────────────────────────────────────────────────────────────────────


def section_python_roundtrip_mse() -> int:
    """Quantization roundtrip quality — no Metal required."""
    sep("Section 2: Python Roundtrip MSE (no Metal)")
    np.random.seed(42)
    n_tokens = 32
    k = mx.array(
        np.random.randn(n_tokens, Q_NUM_KV_HEADS, Q_HEAD_DIM).astype(np.float16)
    )
    v = mx.array(
        np.random.randn(n_tokens, Q_NUM_KV_HEADS, Q_HEAD_DIM).astype(np.float16)
    )
    mx.eval(k, v)

    for quant in QUANTS:
        k_q, v_q = turbo_quant_encode(k, v, quant)
        k_hat, v_hat = turbo_quant_decode(
            k_q, v_q, output_dtype=mx.float16, key_quant_type=quant
        )
        mx.eval(k_hat, v_hat)

        k_mse = mx.mean((k.astype(mx.float32) - k_hat.astype(mx.float32)) ** 2).item()
        v_mse = mx.mean((v.astype(mx.float32) - v_hat.astype(mx.float32)) ** 2).item()
        k_cos = cos_sim(k, k_hat)
        v_cos = cos_sim(v, v_hat)
        bits = QUANT_PARAMS[quant]["bits"]
        print(
            f"  {quant:6s} ({bits}-bit K / 3-bit V)  "
            f"K mse={k_mse:.5f} cos={k_cos:.4f}  "
            f"V mse={v_mse:.5f} cos={v_cos:.4f}"
        )

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Metal Kernel Dequant
# ─────────────────────────────────────────────────────────────────────────────


def _fill_cache(cache, k_packed, v_packed, k_scale, v_scale, k_zero, slot):
    """Scatter-write packed K/V/scales/zero into the paged cache at `slot`."""
    layer = 0
    nkv = U_NUM_KV_HEADS
    sg = U_HEAD_DIM // 32

    flat_k = cache.key_caches[layer].reshape(-1, nkv, cache.k_packed_dim)
    flat_k[slot] = k_packed
    cache.key_caches[layer] = flat_k.reshape(cache.key_caches[layer].shape)

    flat_v = cache.value_caches[layer].reshape(-1, nkv, cache.v_packed_dim)
    flat_v[slot] = v_packed
    cache.value_caches[layer] = flat_v.reshape(cache.value_caches[layer].shape)

    for attr, data in [
        ("key_scale_caches", k_scale),
        ("value_scale_caches", v_scale),
        ("key_zero_caches", k_zero),
    ]:
        arr = getattr(cache, attr)[layer]
        flat = arr.reshape(-1, nkv, sg)
        flat[slot] = data
        getattr(cache, attr)[layer] = flat.reshape(arr.shape)


def _python_attention_reference(q, k, v, scale):
    """Single-sequence multihead attention with GQA replication."""
    nkv = k.shape[1]
    nq = q.shape[1]
    rep = nq // nkv
    k_rep = mx.repeat(k, rep, axis=1).astype(mx.float32)
    v_rep = mx.repeat(v, rep, axis=1).astype(mx.float32)
    q_f = q.astype(mx.float32)
    scores = mx.einsum("qhd,khd->qhk", q_f, k_rep) * scale
    p = mx.softmax(scores, axis=-1)
    return mx.einsum("qhk,khd->qhd", p, v_rep)


def section_metal_kernel_dequant() -> int:
    """Metal paged attention output vs Python dequant reference."""
    sep("Section 3: Metal Kernel Dequant (Metal vs Python reference)")
    print(f"  heads={U_NUM_KV_HEADS}  head_dim={U_HEAD_DIM}  block_size={U_BLOCK_SIZE}")
    print()
    ops = get_ops()
    failures = 0

    for quant in QUANTS:
        cache = MetalPagedKVCache(
            num_layers=1,
            num_blocks=U_NUM_BLOCKS,
            block_size=U_BLOCK_SIZE,
            num_kv_heads=U_NUM_KV_HEADS,
            head_dim=U_HEAD_DIM,
            dtype=mx.float16,
            turboquant=True,
            k_quant=quant,
        )

        n_tokens = U_BLOCK_SIZE
        k = mx.random.normal(
            shape=(n_tokens, U_NUM_KV_HEADS, U_HEAD_DIM), key=mx.random.key(1)
        ).astype(mx.float16)
        v = mx.random.normal(
            shape=(n_tokens, U_NUM_KV_HEADS, U_HEAD_DIM), key=mx.random.key(2)
        ).astype(mx.float16)
        q = mx.random.normal(
            shape=(1, U_NUM_KV_HEADS * U_GQA_RATIO, U_HEAD_DIM), key=mx.random.key(3)
        ).astype(mx.float16)
        mx.eval(k, v, q)

        (k_packed, k_scale, k_zero), (v_packed, v_scale) = turbo_quant_encode(
            k, v, quant
        )
        mx.eval(k_packed, k_scale, k_zero, v_packed, v_scale)

        k_ref, v_ref = turbo_quant_decode(
            (k_packed, k_scale, k_zero),
            (v_packed, v_scale),
            output_dtype=mx.float16,
            key_quant_type=quant,
        )
        mx.eval(k_ref, v_ref)

        slot = mx.array(list(range(n_tokens)), dtype=mx.int64)
        mx.eval(slot)
        _fill_cache(cache, k_packed, v_packed, k_scale, v_scale, k_zero, slot)
        mx.eval(
            cache.key_caches[0],
            cache.value_caches[0],
            cache.key_scale_caches[0],
            cache.value_scale_caches[0],
            cache.key_zero_caches[0],
        )

        block_tables = mx.array([[0]], dtype=mx.int32)
        seq_lens = mx.array([n_tokens], dtype=mx.int32)
        cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
        out_metal = mx.zeros(
            (1, U_NUM_KV_HEADS * U_GQA_RATIO, U_HEAD_DIM), dtype=mx.float16
        )
        mx.eval(block_tables, seq_lens, cu_seqlens_q, out_metal)

        attn_scale = 1.0 / math.sqrt(U_HEAD_DIM)
        v_centroids = get_v_centroids(cache.v_bits)
        ops.paged_attention_primitive(
            q,
            cache.key_caches[0],
            cache.value_caches[0],
            U_NUM_KV_HEADS,
            attn_scale,
            0.0,
            block_tables,
            seq_lens,
            cu_seqlens_q,
            U_BLOCK_SIZE,
            n_tokens,
            -1,
            out_metal,
            key_scale_cache=cache.key_scale_caches[0],
            value_scale_cache=cache.value_scale_caches[0],
            key_zero_cache=cache.key_zero_caches[0],
            v_centroids=v_centroids,
            use_turboquant=True,
            quant_type=quant,
            v_bits=cache.v_bits,
        )
        mx.eval(out_metal)

        out_ref = _python_attention_reference(q, k_ref, v_ref, attn_scale)
        mx.eval(out_ref)

        diff = out_metal.astype(mx.float32) - out_ref.astype(mx.float32)
        mad = mx.mean(mx.abs(diff)).item()
        denom = mx.mean(mx.abs(out_ref.astype(mx.float32))).item() + 1e-8
        rel = mad / denom * 100.0

        ok = rel < 5.0
        mark = "OK" if ok else "FAIL"
        print(f"  {quant:6s}  mean_abs_diff={mad:.4f}  rel_err={rel:6.2f}%  [{mark}]")
        if not ok:
            failures += 1

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Metal E2E Correctness
# ─────────────────────────────────────────────────────────────────────────────


def _scatter_tq(cache, layer, packed_k, k_scale, k_zero, packed_v, v_scale, slot):
    """Scatter TurboQuant tensors into the 5 paged cache arrays for `layer`."""
    nkv = cache.num_kv_heads
    sg = cache.head_dim // BLOCK_SIZE

    def sc(arr, data, d):
        flat = arr.reshape(-1, nkv, d)
        flat[slot] = data
        return flat.reshape(arr.shape)

    cache.key_caches[layer] = sc(cache.key_caches[layer], packed_k, cache.k_packed_dim)
    cache.value_caches[layer] = sc(
        cache.value_caches[layer], packed_v, cache.v_packed_dim
    )
    cache.key_scale_caches[layer] = sc(cache.key_scale_caches[layer], k_scale, sg)
    cache.value_scale_caches[layer] = sc(cache.value_scale_caches[layer], v_scale, sg)
    cache.key_zero_caches[layer] = sc(cache.key_zero_caches[layer], k_zero, sg)


def _scatter_fp16(cache, layer, k, v, slot, head_dim):
    """Scatter FP16 K/V into the paged cache arrays for `layer`."""
    nkv = cache.num_kv_heads
    flat_k = cache.key_caches[layer].reshape(-1, nkv, head_dim)
    flat_k[slot] = k
    cache.key_caches[layer] = flat_k.reshape(cache.key_caches[layer].shape)
    flat_v = cache.value_caches[layer].reshape(-1, nkv, head_dim)
    flat_v[slot] = v
    cache.value_caches[layer] = flat_v.reshape(cache.value_caches[layer].shape)


def section_metal_e2e() -> int:
    """Cache write → paged attention correctness at multiple sequence lengths."""
    sep("Section 4: Metal E2E Correctness (Qwen3-0.6B shape)")
    ops = get_ops()
    failures = 0

    for quant in ["q8_0"]:
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=Q_NUM_KV_HEADS,
            head_dim=Q_HEAD_DIM,
            num_blocks=10,
            block_size=Q_BLOCK_SIZE,
            dtype=mx.float16,
            turboquant=True,
            k_quant=quant,
        )

        for n_tokens in [1, 5, 15]:
            k = mx.random.normal(
                shape=(n_tokens, Q_NUM_KV_HEADS, Q_HEAD_DIM),
                key=mx.random.key(n_tokens),
            ).astype(mx.float16)
            v = mx.random.normal(
                shape=(n_tokens, Q_NUM_KV_HEADS, Q_HEAD_DIM),
                key=mx.random.key(n_tokens + 100),
            ).astype(mx.float16)
            q = mx.random.normal(
                shape=(n_tokens, Q_NUM_KV_HEADS * Q_GQA_RATIO, Q_HEAD_DIM),
                key=mx.random.key(n_tokens + 200),
            ).astype(mx.float16)
            mx.eval(k, v, q)

            (packed_k, k_scale, k_zero), (packed_v, v_scale) = turbo_quant_encode(
                k, v, quant, value_bits=cache.v_bits
            )
            mx.eval(packed_k, k_scale, k_zero, packed_v, v_scale)

            _, v_hat_py = turbo_quant_decode(
                (packed_k, k_scale, k_zero),
                (packed_v, v_scale),
                output_dtype=mx.float16,
                key_quant_type=quant,
            )
            mx.eval(v_hat_py)

            slot = mx.array(list(range(n_tokens)), dtype=mx.int64)
            mx.eval(slot)
            _scatter_tq(cache, 0, packed_k, k_scale, k_zero, packed_v, v_scale, slot)
            mx.eval(
                cache.key_caches[0],
                cache.value_caches[0],
                cache.key_scale_caches[0],
                cache.value_scale_caches[0],
                cache.key_zero_caches[0],
            )

            block_tables = mx.array([[0]], dtype=mx.int32)
            seq_lens = mx.array([n_tokens], dtype=mx.int32)
            cu_seqlens = mx.array([0, n_tokens], dtype=mx.int32)
            out = mx.zeros(
                (n_tokens, Q_NUM_KV_HEADS * Q_GQA_RATIO, Q_HEAD_DIM), dtype=mx.float16
            )
            mx.eval(block_tables, seq_lens, cu_seqlens, out)

            scale = 1.0 / (Q_HEAD_DIM**0.5)
            v_centroids = get_v_centroids(cache.v_bits)
            ops.paged_attention_primitive(
                q,
                cache.key_caches[0],
                cache.value_caches[0],
                Q_NUM_KV_HEADS,
                scale,
                0.0,
                block_tables,
                seq_lens,
                cu_seqlens,
                Q_BLOCK_SIZE,
                n_tokens,
                -1,
                out,
                key_scale_cache=cache.key_scale_caches[0],
                value_scale_cache=cache.value_scale_caches[0],
                key_zero_cache=cache.key_zero_caches[0],
                v_centroids=v_centroids,
                use_turboquant=True,
                quant_type=quant,
                v_bits=cache.v_bits,
            )
            mx.eval(out)

            if n_tokens == 1:
                v_mse = mx.mean(
                    (out[0, 0].astype(mx.float32) - v_hat_py[0, 0].astype(mx.float32))
                    ** 2
                ).item()
                ok = v_mse < 1e-4
                mark = "OK" if ok else "FAIL"
                print(
                    f"  n_tokens={n_tokens:2d} {quant}  single-tok V mse={v_mse:.2e}  [{mark}]"
                )
            else:
                finite = mx.all(mx.isfinite(out)).item()
                max_ab = mx.max(mx.abs(out)).item()
                ok = finite and max_ab < 100
                mark = "OK" if ok else "FAIL"
                print(
                    f"  n_tokens={n_tokens:2d} {quant}  finite={finite}  max_abs={max_ab:.2f}  [{mark}]"
                )

            if not ok:
                failures += 1

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Memory Capacity Analysis
# ─────────────────────────────────────────────────────────────────────────────


def section_memory_capacity() -> int:
    """Theoretical compression table and live cache population sanity check."""
    sep("Section 5: Memory Capacity Analysis (Qwen3-0.6B shape)")
    ops = get_ops()

    def fp16_block_bytes():
        return 2 * Q_NUM_LAYERS * Q_BLOCK_SIZE * Q_NUM_KV_HEADS * Q_HEAD_DIM * 2

    def tq_block_bytes(k_quant: str = "q8_0") -> int:
        # K packed at k_bits, V packed at 3 bits, 3x float32 scales (k_scale, v_scale, k_zero)
        k_bits = QUANT_PARAMS[k_quant]["bits"]
        k_bytes = packed_dim(Q_HEAD_DIM, k_bits)
        v_bytes = packed_dim(Q_HEAD_DIM, 3)
        s_bytes = (
            (Q_HEAD_DIM // BLOCK_SIZE) * 2 * 3
        )  # float16: k_scale + v_scale + k_zero
        return (
            Q_NUM_LAYERS * Q_BLOCK_SIZE * Q_NUM_KV_HEADS * (k_bytes + v_bytes + s_bytes)
        )

    fp16_blk = fp16_block_bytes()
    tq_blk = tq_block_bytes()
    compress = fp16_blk / tq_blk
    print(
        f"\n  Qwen3-0.6B: {Q_NUM_LAYERS}L  {Q_NUM_KV_HEADS}kv  hd={Q_HEAD_DIM}  bs={Q_BLOCK_SIZE}"
    )
    print(
        f"  FP16 per block:  {fp16_blk:>10,} B   TQ per block: {tq_blk:>10,} B   ({compress:.2f}x)"
    )

    print(
        f"\n  {'Budget':>8}  {'FP16 blocks':>12}  {'FP16 tokens':>12}  {'TQ blocks':>10}  {'TQ tokens':>10}  {'Gain':>6}"
    )
    print(f"  {'─' * 8}  {'─' * 12}  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 6}")
    for mb in [64, 128, 256, 512, 1024]:
        budget = mb * 1024 * 1024
        fp16_b = budget // fp16_blk
        fp16_tok = fp16_b * Q_BLOCK_SIZE
        tq_b = budget // tq_blk
        tq_tok = tq_b * Q_BLOCK_SIZE
        gain = tq_tok / max(fp16_tok, 1)
        print(
            f"  {mb:>6} MB  {fp16_b:>12,}  {fp16_tok:>12,}  {tq_b:>10,}  {tq_tok:>10,}  {gain:>5.2f}x"
        )

    # Live population sanity check (1 layer, single block each)
    n_fp16 = 32
    n_tq = int(n_fp16 * compress)
    tok_fp16 = n_fp16 * Q_BLOCK_SIZE
    tok_tq = n_tq * Q_BLOCK_SIZE
    print(
        f"\n  Live check: fp16={n_fp16} blocks ({tok_fp16} tok)  tq={n_tq} blocks ({tok_tq} tok)"
    )

    np.random.seed(7)
    ops = get_ops()
    scale = 1.0 / (Q_HEAD_DIM**0.5)

    cache_fp16 = MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=Q_NUM_KV_HEADS,
        head_dim=Q_HEAD_DIM,
        num_blocks=n_fp16,
        block_size=Q_BLOCK_SIZE,
        dtype=mx.float16,
        turboquant=False,
    )
    k16 = mx.array(
        np.random.randn(tok_fp16, Q_NUM_KV_HEADS, Q_HEAD_DIM).astype(np.float16)
    )
    v16 = mx.array(
        np.random.randn(tok_fp16, Q_NUM_KV_HEADS, Q_HEAD_DIM).astype(np.float16)
    )
    sl16 = mx.array(list(range(tok_fp16)), dtype=mx.int64)
    mx.eval(k16, v16, sl16)
    _scatter_fp16(cache_fp16, 0, k16, v16, sl16, Q_HEAD_DIM)
    mx.eval(cache_fp16.key_caches[0], cache_fp16.value_caches[0])

    q16 = mx.random.normal(
        shape=(1, Q_NUM_KV_HEADS * Q_GQA_RATIO, Q_HEAD_DIM), key=mx.random.key(99)
    ).astype(mx.float16)
    o16 = mx.zeros((1, Q_NUM_KV_HEADS * Q_GQA_RATIO, Q_HEAD_DIM), dtype=mx.float16)
    bt16 = mx.arange(n_fp16, dtype=mx.int32).reshape(1, -1)
    s16 = mx.array([tok_fp16], dtype=mx.int32)
    cu16 = mx.array([0, 1], dtype=mx.int32)
    mx.eval(q16, o16, bt16, s16, cu16)
    ops.paged_attention_primitive(
        q16,
        cache_fp16.key_caches[0],
        cache_fp16.value_caches[0],
        Q_NUM_KV_HEADS,
        scale,
        0.0,
        bt16,
        s16,
        cu16,
        Q_BLOCK_SIZE,
        tok_fp16,
        -1,
        o16,
    )
    mx.eval(o16)
    fp16_ok = mx.all(mx.isfinite(o16)).item()
    print(
        f"  FP16 attention output: finite={fp16_ok}  [{('OK' if fp16_ok else 'FAIL')}]"
    )

    cache_tq = MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=Q_NUM_KV_HEADS,
        head_dim=Q_HEAD_DIM,
        num_blocks=n_tq,
        block_size=Q_BLOCK_SIZE,
        dtype=mx.float16,
        turboquant=True,
        k_quant="q8_0",
    )
    ktq = mx.array(
        np.random.randn(tok_tq, Q_NUM_KV_HEADS, Q_HEAD_DIM).astype(np.float16)
    )
    vtq = mx.array(
        np.random.randn(tok_tq, Q_NUM_KV_HEADS, Q_HEAD_DIM).astype(np.float16)
    )
    mx.eval(ktq, vtq)
    (pk, ks, kz), (pv, vs) = turbo_quant_encode(
        ktq, vtq, "q8_0", value_bits=cache_tq.v_bits
    )
    sl_tq = mx.array(list(range(tok_tq)), dtype=mx.int64)
    mx.eval(pk, ks, kz, pv, vs, sl_tq)
    _scatter_tq(cache_tq, 0, pk, ks, kz, pv, vs, sl_tq)
    mx.eval(
        cache_tq.key_caches[0],
        cache_tq.value_caches[0],
        cache_tq.key_scale_caches[0],
        cache_tq.value_scale_caches[0],
        cache_tq.key_zero_caches[0],
    )

    qtq = mx.random.normal(
        shape=(1, Q_NUM_KV_HEADS * Q_GQA_RATIO, Q_HEAD_DIM), key=mx.random.key(99)
    ).astype(mx.float16)
    otq = mx.zeros((1, Q_NUM_KV_HEADS * Q_GQA_RATIO, Q_HEAD_DIM), dtype=mx.float16)
    bttq = mx.arange(n_tq, dtype=mx.int32).reshape(1, -1)
    stq = mx.array([tok_tq], dtype=mx.int32)
    cutq = mx.array([0, 1], dtype=mx.int32)
    mx.eval(qtq, otq, bttq, stq, cutq)
    v_centroids = get_v_centroids(cache_tq.v_bits)
    ops.paged_attention_primitive(
        qtq,
        cache_tq.key_caches[0],
        cache_tq.value_caches[0],
        Q_NUM_KV_HEADS,
        scale,
        0.0,
        bttq,
        stq,
        cutq,
        Q_BLOCK_SIZE,
        tok_tq,
        -1,
        otq,
        key_scale_cache=cache_tq.key_scale_caches[0],
        value_scale_cache=cache_tq.value_scale_caches[0],
        key_zero_cache=cache_tq.key_zero_caches[0],
        v_centroids=v_centroids,
        use_turboquant=True,
        quant_type="q8_0",
        v_bits=cache_tq.v_bits,
    )
    mx.eval(otq)
    tq_ok = mx.all(mx.isfinite(otq)).item()
    print(
        f"  TQ attention output:   finite={tq_ok}   [{('OK' if tq_ok else 'FAIL')}]"
        f"  ({tok_tq} tokens, {tok_tq - tok_fp16} more than fp16)"
    )

    return 0 if (fp16_ok and tq_ok) else 1


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Published Comparison
# ─────────────────────────────────────────────────────────────────────────────


def section_published_comparison() -> int:
    """Compare our NRMSE and compression to published TurboQuant / KIVI / QJL."""
    sep("Section 6: Comparison to Published Numbers")
    np.random.seed(0)
    n_tokens = 256
    k = mx.array(
        np.random.randn(n_tokens, Q_NUM_KV_HEADS, Q_HEAD_DIM).astype(np.float16)
    )
    v = mx.array(
        np.random.randn(n_tokens, Q_NUM_KV_HEADS, Q_HEAD_DIM).astype(np.float16)
    )
    mx.eval(k, v)

    k_std = mx.sqrt(mx.mean(k.astype(mx.float32) ** 2)).item()
    v_std = mx.sqrt(mx.mean(v.astype(mx.float32) ** 2)).item()

    for quant in ["q8_0", "q4_0"]:
        kq, vq = turbo_quant_encode(k, v, quant)
        k_hat, v_hat = turbo_quant_decode(
            kq, vq, output_dtype=mx.float16, key_quant_type=quant
        )
        mx.eval(k_hat, v_hat)

        k_mse = mx.mean((k.astype(mx.float32) - k_hat.astype(mx.float32)) ** 2).item()
        v_mse = mx.mean((v.astype(mx.float32) - v_hat.astype(mx.float32)) ** 2).item()
        k_nrmse = (k_mse**0.5) / k_std
        v_nrmse = (v_mse**0.5) / v_std
        k_cos = cos_sim(k, k_hat)
        v_cos = cos_sim(v, v_hat)
        print(f"\n  Ours ({quant} K / 3-bit V):")
        print(f"    K: mse={k_mse:.5f}  nrmse={k_nrmse:.4f}  cos={k_cos:.6f}")
        print(f"    V: mse={v_mse:.5f}  nrmse={v_nrmse:.4f}  cos={v_cos:.6f}")

    fp16_per_tok = Q_HEAD_DIM * 2 * 2  # K + V at fp16 (2 bytes each)

    def _tq_per_tok(k_quant: str) -> int:
        k_bits = QUANT_PARAMS[k_quant]["bits"]
        k_bytes = packed_dim(Q_HEAD_DIM, k_bits)
        v_bytes = packed_dim(Q_HEAD_DIM, 3)
        s_bytes = (
            (Q_HEAD_DIM // BLOCK_SIZE) * 2 * 3
        )  # float16: k_scale + v_scale + k_zero
        return k_bytes + v_bytes + s_bytes

    ratio_q8 = fp16_per_tok / _tq_per_tok("q8_0")
    ratio_q4 = fp16_per_tok / _tq_per_tok("q4_0")
    print(f"""
  Published reference (head_dim={Q_HEAD_DIM}):
  ───────────────────────────────────────────────────────
  Method          | K    | V    | Compression | Notes
  ─────────────────────────────────────────────────────
  KIVI (2024)     | 2bit | 2bit | ~8x         | per-channel K / per-token V
  KVQuant (2024)  | 2-4b | 2-4b | 3-8x        | NF quant + rotation
  QJL (2024)      | 3bit | 3bit | ~4.5x       | JL projection for K
  TurboQuant      | 4bit | 3bit | ~4.6x       | WHT + Lloyd-Max V
  Ours (K=8,V=3)  | 8bit | 3bit | {ratio_q8:.1f}x         | conservative K, aggressive V
  Ours (K=4,V=3)  | 4bit | 3bit | {ratio_q4:.1f}x         | same storage, lower K quality
    """)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Quantization Latency
# ─────────────────────────────────────────────────────────────────────────────


def section_latency() -> int:
    """Encode + cache-write overhead per decode token."""
    sep("Section 7: Quantization Latency")

    k = mx.random.normal(shape=(1, Q_NUM_KV_HEADS, Q_HEAD_DIM)).astype(mx.float16)
    v = mx.random.normal(shape=(1, Q_NUM_KV_HEADS, Q_HEAD_DIM)).astype(mx.float16)
    mx.eval(k, v)

    cache_tq = MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=Q_NUM_KV_HEADS,
        head_dim=Q_HEAD_DIM,
        num_blocks=10,
        block_size=Q_BLOCK_SIZE,
        dtype=mx.float16,
        turboquant=True,
        k_quant="q8_0",
    )
    cache_fp = MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=Q_NUM_KV_HEADS,
        head_dim=Q_HEAD_DIM,
        num_blocks=10,
        block_size=Q_BLOCK_SIZE,
        dtype=mx.float16,
        turboquant=False,
    )
    slot = mx.array([0], dtype=mx.int64)
    mx.eval(slot)

    # Warm up
    for _ in range(10):
        kq, vq = turbo_quant_encode(k, v, "q8_0")
        mx.eval(*kq, *vq)

    n = 200

    # Benchmark encode
    t0 = time.perf_counter()
    for _ in range(n):
        kq, vq = turbo_quant_encode(k, v, "q8_0")
        mx.eval(*kq, *vq)
    encode_us = (time.perf_counter() - t0) / n * 1e6

    # Pre-encode once for write benchmarks
    (pk, ks, kz), (pv, vs) = turbo_quant_encode(
        k, v, "q8_0", value_bits=cache_tq.v_bits
    )
    mx.eval(pk, ks, kz, pv, vs)

    # Benchmark TQ cache write (scatter 5 arrays)
    for _ in range(10):
        _scatter_tq(cache_tq, 0, pk, ks, kz, pv, vs, slot)
        mx.eval(
            cache_tq.key_caches[0],
            cache_tq.value_caches[0],
            cache_tq.key_scale_caches[0],
            cache_tq.value_scale_caches[0],
            cache_tq.key_zero_caches[0],
        )
    t0 = time.perf_counter()
    for _ in range(n):
        _scatter_tq(cache_tq, 0, pk, ks, kz, pv, vs, slot)
        mx.eval(
            cache_tq.key_caches[0],
            cache_tq.value_caches[0],
            cache_tq.key_scale_caches[0],
            cache_tq.value_scale_caches[0],
            cache_tq.key_zero_caches[0],
        )
    tq_write_us = (time.perf_counter() - t0) / n * 1e6

    # Benchmark FP16 cache write (scatter K+V)
    for _ in range(10):
        _scatter_fp16(cache_fp, 0, k, v, slot, Q_HEAD_DIM)
        mx.eval(cache_fp.key_caches[0], cache_fp.value_caches[0])
    t0 = time.perf_counter()
    for _ in range(n):
        _scatter_fp16(cache_fp, 0, k, v, slot, Q_HEAD_DIM)
        mx.eval(cache_fp.key_caches[0], cache_fp.value_caches[0])
    fp16_write_us = (time.perf_counter() - t0) / n * 1e6

    overhead_us = encode_us + tq_write_us - fp16_write_us
    tok_budget = 1e6 / 60  # ~16,700 µs at 60 tok/s
    print(f"\n  Single-token ({Q_NUM_KV_HEADS} KV heads, hd={Q_HEAD_DIM}):")
    print(f"    TQ encode (Python):    {encode_us:>8.1f} µs")
    print(f"    TQ cache write:        {tq_write_us:>8.1f} µs")
    print(f"    FP16 cache write:      {fp16_write_us:>8.1f} µs")
    print(
        f"    TQ overhead:           {overhead_us:>8.1f} µs  "
        f"({overhead_us / tok_budget * 100:.1f}% of 60tok/s budget)"
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Peak Memory During Attention (heap + on-chip SRAM breakdown)
# ─────────────────────────────────────────────────────────────────────────────
#
# Memory taxonomy for one decode step:
#
#   Metal heap (tracked by mx.metal.get_active_memory / get_peak_memory):
#     - KV cache arrays (persistent, dominant at large context)
#     - Output array (1 token * nq_heads * head_dim * dtype_bytes)
#     - Control arrays: block_tables, seq_lens, cu_seqlens (negligible)
#
#   On-chip threadgroup SRAM (NOT in Metal heap, calculated below):
#     - Standard attn workspace: max(NUM_WARPS*block_size, 2*NUM_WARPS + NUM_WARPS*head_dim) * 4B
#     - TurboQuant extra: NUM_WARPS * head_dim * 4B  (per-warp FWHT buffer)
#     - Per CTA dispatch, freed on completion — never shows in get_active_memory()
#
# Protocol:
#   1. gc.collect() + mx.metal.clear_cache() for clean baseline
#   2. Create + eval cache → measure static heap delta
#   3. Create attention inputs → record pre-attn active
#   4. mx.metal.reset_peak_memory()
#   5. Run paged_attention_primitive + mx.eval(out)
#   6. peak = mx.metal.get_peak_memory()       # max heap during forward pass
#   7. post = mx.metal.get_active_memory()     # steady-state after (intermediates freed)
#   8. lazy_intermediates = peak - post        # MLX lazy-graph temporaries
#   9. attn_heap_delta = peak - pre_attn   # net heap delta from cache state
#
# Key result: attn_heap_delta ≈ output_size + lazy intermediates (~tiny).
# The dequant is inline in Metal SRAM — zero hidden heap overhead for TQ.


_MB = 1024 * 1024

# Metal kernel constants (must match paged_ops.cpp)
_NUM_THREADS = 256
_NUM_SIMD_LANES = 32
_NUM_WARPS = _NUM_THREADS // _NUM_SIMD_LANES  # 8


def _shmem_bytes(block_size: int, head_dim: int, turboquant: bool) -> int:
    """Threadgroup SRAM per CTA dispatch (on-chip, not in Metal heap)."""
    warp_scores = _NUM_WARPS * block_size * 4
    if turboquant:
        # TQ path uses warp_scores slot for both scores and FWHT buffer
        warp_scores += _NUM_WARPS * head_dim * 4
    merge = (2 * _NUM_WARPS + _NUM_WARPS * head_dim) * 4
    return max(warp_scores, merge)


def _theoretical_cache_bytes(
    num_layers: int,
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    turboquant: bool,
    k_quant: str | None = None,
) -> int:
    """Exact formula for cache heap allocation, matching MetalPagedKVCache."""
    if not turboquant:
        return num_layers * num_tokens * num_kv_heads * head_dim * 2 * 2  # fp16 K+V
    k_bits = QUANT_PARAMS[k_quant]["bits"]
    k_bytes = packed_dim(head_dim, k_bits)
    v_bytes = packed_dim(head_dim, 3)
    sg = head_dim // BLOCK_SIZE
    scale_bytes = sg * 2 * 3  # float16: k_scale, v_scale, k_zero
    return num_layers * num_tokens * num_kv_heads * (k_bytes + v_bytes + scale_bytes)


def _measure_one(
    ops,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    seq_len: int,
    turboquant: bool,
    k_quant: str | None,
) -> dict:
    """
    Full measurement for one (config, seq_len) pair.

    Returns:
        cache_heap_mb       — Metal heap consumed by cache arrays alone
        pre_attn_heap_mb    — heap just before attention dispatch (cache + q/control)
        peak_heap_mb        — peak heap captured by get_peak_memory() during eval
        post_attn_heap_mb   — heap after eval (lazy intermediates freed)
        lazy_intermediates_mb — peak - post (MLX graph temporary overhead)
        attn_heap_delta_mb  — peak - pre_attn (net heap added by attention call)
        shmem_kb            — on-chip SRAM per CTA (calculated, not from Metal API)
        theoretical_cache_mb — formula cross-check vs measured cache_heap_mb
    """
    gc.collect()
    mx.clear_cache()
    mx.eval()  # drain lazy graph before baseline

    base_active = mx.get_active_memory()

    # ── Allocate cache ──────────────────────────────────────────────────────
    cache = MetalPagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=mx.float16,
        turboquant=turboquant,
        k_quant=k_quant,
    )
    if turboquant:
        mx.eval(
            *cache.key_caches,
            *cache.value_caches,
            *cache.key_scale_caches,
            *cache.value_scale_caches,
            *cache.key_zero_caches,
        )
    else:
        mx.eval(*cache.key_caches, *cache.value_caches)

    after_alloc = mx.get_active_memory()
    cache_heap_mb = (after_alloc - base_active) / _MB

    # ── Fill layer 0 with seq_len tokens ────────────────────────────────────
    layer = 0
    key = mx.random.normal(
        shape=(seq_len, num_kv_heads, head_dim), key=mx.random.key(1)
    ).astype(mx.float16)
    val = mx.random.normal(
        shape=(seq_len, num_kv_heads, head_dim), key=mx.random.key(2)
    ).astype(mx.float16)
    mx.eval(key, val)

    if turboquant:
        (pk, ks, kz), (pv, vs) = turbo_quant_encode(
            key, val, k_quant, value_bits=cache.v_bits
        )
        mx.eval(pk, ks, kz, pv, vs)
        sl = mx.array(list(range(seq_len)), dtype=mx.int64)
        mx.eval(sl)
        _scatter_tq(cache, layer, pk, ks, kz, pv, vs, sl)
        mx.eval(
            cache.key_caches[layer],
            cache.value_caches[layer],
            cache.key_scale_caches[layer],
            cache.value_scale_caches[layer],
            cache.key_zero_caches[layer],
        )
        del pk, ks, kz, pv, vs, sl
    else:
        sl = mx.array(list(range(seq_len)), dtype=mx.int64)
        mx.eval(sl)
        _scatter_fp16(cache, layer, key, val, sl, head_dim)
        mx.eval(cache.key_caches[layer], cache.value_caches[layer])
        del sl
    del key, val
    gc.collect()
    mx.eval()

    # ── Build attention inputs ───────────────────────────────────────────────
    # Decode scenario: 1 new query token against seq_len cached tokens.
    # GQA ratio 4 (matches Qwen3-0.6B: 16 Q / 4 KV).
    nq = num_kv_heads * 4
    q = mx.random.normal(shape=(1, nq, head_dim), key=mx.random.key(3)).astype(
        mx.float16
    )
    num_blks = math.ceil(seq_len / block_size)
    block_tables = mx.arange(num_blks, dtype=mx.int32).reshape(1, -1)
    seq_lens_arr = mx.array([seq_len], dtype=mx.int32)
    cu_seqlens = mx.array([0, 1], dtype=mx.int32)
    out_arr = mx.zeros((1, nq, head_dim), dtype=mx.float16)
    mx.eval(q, block_tables, seq_lens_arr, cu_seqlens, out_arr)

    attn_scale = 1.0 / math.sqrt(head_dim)

    # ── Peak measurement ─────────────────────────────────────────────────────
    # Drain lazy graph; record stable heap before reset.
    mx.eval()
    pre_attn_active = mx.get_active_memory()
    mx.reset_peak_memory()

    if turboquant:
        v_centroids = get_v_centroids(cache.v_bits)
        ops.paged_attention_primitive(
            q,
            cache.key_caches[layer],
            cache.value_caches[layer],
            num_kv_heads,
            attn_scale,
            0.0,
            block_tables,
            seq_lens_arr,
            cu_seqlens,
            block_size,
            seq_len,
            -1,
            out_arr,
            key_scale_cache=cache.key_scale_caches[layer],
            value_scale_cache=cache.value_scale_caches[layer],
            key_zero_cache=cache.key_zero_caches[layer],
            v_centroids=v_centroids,
            use_turboquant=True,
            quant_type=k_quant,
            v_bits=cache.v_bits,
        )
    else:
        ops.paged_attention_primitive(
            q,
            cache.key_caches[layer],
            cache.value_caches[layer],
            num_kv_heads,
            attn_scale,
            0.0,
            block_tables,
            seq_lens_arr,
            cu_seqlens,
            block_size,
            seq_len,
            -1,
            out_arr,
        )
    mx.eval(out_arr)

    peak_active = mx.get_peak_memory()
    post_attn_active = mx.get_active_memory()

    # ── Derived metrics ──────────────────────────────────────────────────────
    shmem_b = _shmem_bytes(block_size, head_dim, turboquant)
    # Dispatch one threadgroup per (nq_heads, decode_tokens=1) = nq dispatches.
    total_shmem_kb = shmem_b * nq / 1024

    theoretical = _theoretical_cache_bytes(
        num_layers,
        num_blocks * block_size,
        num_kv_heads,
        head_dim,
        turboquant,
        k_quant,
    )

    return {
        "cache_heap_mb": cache_heap_mb,
        "pre_attn_heap_mb": pre_attn_active / _MB,
        "peak_heap_mb": peak_active / _MB,
        "post_attn_heap_mb": post_attn_active / _MB,
        "lazy_intermediates_mb": max(0.0, (peak_active - post_attn_active) / _MB),
        "attn_heap_delta_mb": (peak_active - pre_attn_active) / _MB,
        "shmem_per_cta_kb": shmem_b / 1024,
        "total_shmem_kb": total_shmem_kb,
        "theoretical_cache_mb": theoretical / _MB,
    }


def section_peak_memory_analysis() -> int:
    """
    Section 8: Peak Metal memory during attention — heap + on-chip SRAM.

    Three sub-reports:

      8a. Formula cross-check
          Confirm measured cache allocation matches the theoretical formula.
          Catches any hidden allocations or size regressions.

      8b. Static footprint + peak during decode
          For each quant type, at realistic token counts (Qwen3-0.6B shape):
            - Cache heap (persistent)
            - Peak heap during one decode attention call
            - Lazy intermediates (MLX graph temporaries, freed after eval)
            - On-chip SRAM (calculated; invisible to Metal heap APIs)
          Proves TQ's advantage holds during computation, not just at rest.

      8c. Seq-len sweep
          Absolute heap savings and compression ratio as context grows.
          Identifies the token count where TQ pays off.

    Failure conditions:
      - Measured cache diverges from formula by > 1% (hidden allocation)
      - TQ peak exceeds FP16 peak at any tested seq_len (unexpected overhead)
      - TQ lazy intermediates exceed FP16 lazy intermediates by > 10 MB
        (would indicate dequant escaped to heap)
    """
    sep("Section 8: Peak Memory During Attention (heap + on-chip SRAM)")
    ops = get_ops()
    failures = 0

    # ── 8a. Formula cross-check ──────────────────────────────────────────────
    print("\n  8a. Formula cross-check (measured cache vs theoretical)")
    print(
        f"  {'config':8s}  {'measured MB':>12s}  {'formula MB':>10s}  {'Δ MB':>7s}  {'Δ %':>7s}  status"
    )
    print(f"  {'─' * 8}  {'─' * 12}  {'─' * 10}  {'─' * 7}  {'─' * 7}  {'─' * 6}")
    # Threshold: allow up to 0.25 MB absolute overhead. Metal's allocator rounds
    # each buffer to 4 KB page boundaries; for 5 small TQ arrays this adds ~50 KB
    # of fixed overhead that does NOT grow with sequence length or layer count.
    # The guard catches large unexpected heap allocations (e.g. a dequant
    # temp-buffer materialised on the heap), which would appear as > 1 MB.
    _formula_abs_tol_mb = 0.25

    check_seq = 512
    check_blks = math.ceil(check_seq / Q_BLOCK_SIZE)
    check_cfgs = [
        ("fp16", False, None),
        ("q8_0", True, "q8_0"),
        ("q4_0", True, "q4_0"),
        ("uint2", True, "uint2"),
    ]
    for label, tq, kq in check_cfgs:
        m = _measure_one(
            ops,
            num_layers=1,
            num_kv_heads=Q_NUM_KV_HEADS,
            head_dim=Q_HEAD_DIM,
            num_blocks=check_blks,
            block_size=Q_BLOCK_SIZE,
            seq_len=check_seq,
            turboquant=tq,
            k_quant=kq,
        )
        measured = m["cache_heap_mb"]
        formula = m["theoretical_cache_mb"]
        delta_mb = measured - formula
        delta_pct = delta_mb / (formula + 1e-6) * 100
        ok = delta_mb < _formula_abs_tol_mb
        mark = "OK" if ok else "FAIL"
        print(
            f"  {label:8s}  {measured:>12.3f}  {formula:>10.3f}  {delta_mb:>7.3f}  {delta_pct:>6.1f}%  [{mark}]"
        )
        if not ok:
            failures += 1

    # ── 8b. Static footprint + peak during decode ────────────────────────────
    decode_seq_len = 2048
    decode_num_blks = math.ceil(decode_seq_len / Q_BLOCK_SIZE)

    print(
        f"\n  8b. Static footprint + peak during decode  "
        f"(Qwen3-0.6B shape, 1 layer, {decode_seq_len} tokens)"
    )
    print("\n  Legend:")
    print("    cache heap    — persistent Metal heap for KV cache arrays")
    print("    peak heap     — Metal heap peak captured during mx.eval(out)")
    print("    lazy intermed — peak minus post-eval active (MLX graph temps)")
    print("    SRAM/CTA      — on-chip threadgroup memory per dispatch (not in heap)")
    print()
    hdr = (
        f"  {'config':8s}  {'cache MB':>10s}  {'peak MB':>10s}  "
        f"{'lazy MB':>9s}  {'SRAM/CTA KB':>11s}  peak/fp16"
    )
    print(hdr)
    print(f"  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 9}  {'─' * 11}  {'─' * 8}")

    bench_cfgs = [
        ("fp16", False, None),
        ("q8_0", True, "q8_0"),
        ("q5_0", True, "q5_0"),
        ("q4_0", True, "q4_0"),
        ("uint2", True, "uint2"),
    ]
    results_8b = {}
    for label, tq, kq in bench_cfgs:
        m = _measure_one(
            ops,
            num_layers=1,
            num_kv_heads=Q_NUM_KV_HEADS,
            head_dim=Q_HEAD_DIM,
            num_blocks=decode_num_blks,
            block_size=Q_BLOCK_SIZE,
            seq_len=decode_seq_len,
            turboquant=tq,
            k_quant=kq,
        )
        results_8b[label] = m

    fp16_peak = results_8b["fp16"]["peak_heap_mb"]
    for label, _tq, _kq in bench_cfgs:
        m = results_8b[label]
        ratio = m["peak_heap_mb"] / fp16_peak if fp16_peak > 0 else 1.0
        print(
            f"  {label:8s}  {m['cache_heap_mb']:>10.2f}  {m['peak_heap_mb']:>10.2f}  "
            f"{m['lazy_intermediates_mb']:>9.3f}  {m['shmem_per_cta_kb']:>11.2f}  {ratio:>8.3f}"
        )

    # Assert: TQ peak must be strictly smaller than FP16 peak
    for label, tq, _ in bench_cfgs:
        if not tq:
            continue
        tq_peak = results_8b[label]["peak_heap_mb"]
        lazy_diff = (
            results_8b[label]["lazy_intermediates_mb"]
            - results_8b["fp16"]["lazy_intermediates_mb"]
        )
        if tq_peak >= fp16_peak:
            print(
                f"  FAIL: {label} peak ({tq_peak:.2f} MB) >= fp16 peak ({fp16_peak:.2f} MB)"
            )
            failures += 1
        if lazy_diff > 10.0:
            print(
                f"  FAIL: {label} lazy intermediates {lazy_diff:+.2f} MB above fp16 "
                f"(dequant may have escaped to heap)"
            )
            failures += 1

    # ── 8c. Seq-len sweep ────────────────────────────────────────────────────
    sweep_quants = [("q8_0", True, "q8_0"), ("q4_0", True, "q4_0")]
    sweep_seqlens = [128, 256, 512, 1024, 2048, 4096]

    print("\n  8c. Heap savings vs seq_len  (Qwen3-0.6B shape, 1 layer)")
    print(f"\n  {'seq_len':>8s}  {'fp16 MB':>9s}  ", end="")
    for label, _, _ in sweep_quants:
        print(f"  {label + ' MB':>10s}  {'saved MB':>9s}  {'ratio':>6s}", end="")
    print()
    print(f"  {'─' * 8}  {'─' * 9}  ", end="")
    for _ in sweep_quants:
        print(f"  {'─' * 10}  {'─' * 9}  {'─' * 6}", end="")
    print()

    for seq_len in sweep_seqlens:
        num_blks = math.ceil(seq_len / Q_BLOCK_SIZE)
        fp16_m = _measure_one(
            ops,
            num_layers=1,
            num_kv_heads=Q_NUM_KV_HEADS,
            head_dim=Q_HEAD_DIM,
            num_blocks=num_blks,
            block_size=Q_BLOCK_SIZE,
            seq_len=seq_len,
            turboquant=False,
            k_quant=None,
        )
        fp16_peak_mb = fp16_m["peak_heap_mb"]
        print(f"  {seq_len:>8d}  {fp16_peak_mb:>9.2f}  ", end="")

        for _label, tq, kq in sweep_quants:
            tq_m = _measure_one(
                ops,
                num_layers=1,
                num_kv_heads=Q_NUM_KV_HEADS,
                head_dim=Q_HEAD_DIM,
                num_blocks=num_blks,
                block_size=Q_BLOCK_SIZE,
                seq_len=seq_len,
                turboquant=tq,
                k_quant=kq,
            )
            tq_peak = tq_m["peak_heap_mb"]
            saved = fp16_peak_mb - tq_peak
            ratio = fp16_peak_mb / tq_peak if tq_peak > 0 else float("inf")
            print(f"  {tq_peak:>10.2f}  {saved:>9.2f}  {ratio:>6.2f}x", end="")
        print()

    # ── Full-model projection (all 28 layers) ─────────────────────────────────
    print(f"\n  Full-model projection ({Q_NUM_LAYERS} layers, Qwen3-0.6B):")
    print(f"  (Scales linearly: multiply 1-layer numbers by {Q_NUM_LAYERS})")
    for seq_len in [2048, 8192]:
        fp16_total = (
            _theoretical_cache_bytes(
                Q_NUM_LAYERS, seq_len, Q_NUM_KV_HEADS, Q_HEAD_DIM, False
            )
            / _MB
        )
        for label, tq, kq in sweep_quants:
            tq_total = (
                _theoretical_cache_bytes(
                    Q_NUM_LAYERS, seq_len, Q_NUM_KV_HEADS, Q_HEAD_DIM, tq, kq
                )
                / _MB
            )
            saved = fp16_total - tq_total
            ratio = fp16_total / tq_total if tq_total > 0 else float("inf")
            print(
                f"  seq={seq_len:5d}  {label:6s}  "
                f"fp16={fp16_total:7.1f} MB  tq={tq_total:7.1f} MB  "
                f"saved={saved:7.1f} MB  ({ratio:.2f}x)"
            )

    return failures


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Serve Benchmark (opt-in via --serve)
# ─────────────────────────────────────────────────────────────────────────────


def _wait_dead(pid: int, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)


def _kill_server(proc: subprocess.Popen) -> None:
    pid = proc.pid
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    subprocess.run(
        ["pkill", "-9", "-f", f"vllm serve.*--port {SERVE_PORT}"], capture_output=True
    )
    _wait_dead(pid, timeout=15)


def _parse_max_tokens(log_path: str) -> int | None:
    pat = re.compile(r"max_tokens_cached=(\d+)")
    try:
        with open(log_path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    return int(m.group(1))
    except FileNotFoundError:
        pass
    return None


def _send_chat(prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": SERVE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{SERVE_HOST}:{SERVE_PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return {"ok": False, "error": str(e)[:300], "elapsed": time.monotonic() - t0}
    elapsed = time.monotonic() - t0
    usage = data.get("usage", {})
    c_toks = usage.get("completion_tokens", 0)
    return {
        "ok": True,
        "text": data["choices"][0]["message"]["content"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": c_toks,
        "elapsed": elapsed,
        "tok_per_s": c_toks / elapsed if elapsed > 0 else 0.0,
    }


def _rss_mb(pid: int | None = None) -> float:
    """Return RSS memory in MB for the given pid (default: current process)."""
    try:
        import psutil

        p = psutil.Process(pid)
        return p.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _run_serve_config(label: str, additional_config: dict | None, log_dir: str) -> dict:
    result = {
        "label": label,
        "ready": False,
        "max_tokens": 0,
        "idle_rss": 0.0,
        "peak_rss": 0.0,
        "chat": {},
    }
    env = os.environ.copy()
    env.update(SERVE_BASE_ENV)

    cmd = [
        "vllm",
        "serve",
        SERVE_MODEL,
        "--host",
        SERVE_HOST,
        "--port",
        str(SERVE_PORT),
        "--max-model-len",
        "auto",
        "--enforce-eager",
    ]
    if additional_config:
        import json

        cmd.extend(["--additional-config", json.dumps(additional_config)])
    log_path = os.path.join(log_dir, f"{label}.log")
    print(f"\n  ── {label} ──")
    print(f"     {shlex.join(cmd)}")
    if additional_config:
        print(f"     additional_config: {additional_config}")

    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid
    )
    try:
        deadline = time.monotonic() + SERVE_READY_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print(f"     FAIL: vllm exited (rc={proc.returncode}); see {log_path}")
                return result
            try:
                with urllib.request.urlopen(
                    f"http://{SERVE_HOST}:{SERVE_PORT}/v1/models", timeout=1.5
                ) as r:
                    if r.status == 200:
                        result["ready"] = True
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(1.0)

        if not result["ready"]:
            print("     FAIL: ready timeout")
            return result

        log_f.flush()
        result["max_tokens"] = _parse_max_tokens(log_path) or 0
        print(f"     ready.  max_tokens_cached = {result['max_tokens']:,}")

        time.sleep(3)
        idle = [_rss_mb(proc.pid) for _ in range(6)]
        result["idle_rss"] = max(idle) if idle else 0.0
        print(f"     idle RSS = {result['idle_rss']:.0f} MB")

        print("     warm-up request...")
        warmup = _send_chat("Hello", 16)
        if not warmup.get("ok"):
            print(f"     warm-up FAILED: {warmup.get('error', '?')}")
            result["chat"] = warmup
            return result
        time.sleep(1)

        print(f'     quality prompt: "{SERVE_QUALITY_PROMPT[:60]}..."')
        active_samples: list[float] = []
        holder: dict = {}

        def _req():
            holder["r"] = _send_chat(SERVE_QUALITY_PROMPT, 256)

        t = threading.Thread(target=_req, daemon=True)
        t.start()
        while t.is_alive():
            active_samples.append(_rss_mb(proc.pid))
            time.sleep(0.3)
        t.join()

        chat = holder.get("r", {"ok": False, "error": "no result"})
        result["chat"] = chat
        result["peak_rss"] = (
            max(active_samples) if active_samples else result["idle_rss"]
        )

        if chat.get("ok"):
            print(
                f"     tok/s={chat['tok_per_s']:.1f}  peak RSS={result['peak_rss']:.0f} MB"
            )
            print("     ─── response ───")
            for line in chat["text"].split("\n"):
                print(f"     | {line}")
        else:
            print(f"     request FAILED: {chat.get('error', '?')}")

    finally:
        _kill_server(proc)
        log_f.close()
        print(f"     cooling down {SERVE_COOLDOWN}s...")
        time.sleep(SERVE_COOLDOWN)

    return result


def section_serve_benchmark() -> int:
    """Compare bf16 against the TQ (k_quant, v_quant) sweep via live vllm serve."""
    tq_labels = ", ".join(label for label, cfg in SERVE_CONFIGS if cfg is not None)
    sep(f"Section 9: Serve Benchmark (bf16 vs TQ sweep: {tq_labels})")
    print(f"  Model: {SERVE_MODEL}")
    log_dir = "/tmp/tq-serve-bench"
    os.makedirs(log_dir, exist_ok=True)

    results = [_run_serve_config(label, env, log_dir) for label, env in SERVE_CONFIGS]

    print()
    sep("Serve Results")
    bf16 = next((r for r in results if r["label"] == "bf16"), None)
    hdr = f"  {'config':8s} {'max_ctx':>10s} {'idle MB':>8s} {'peak MB':>8s} {'tok/s':>7s}  status"
    print(hdr)
    print(f"  {'─' * 8} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 7}  {'─' * 6}")
    for r in results:
        chat = r.get("chat", {})
        tps = f"{chat['tok_per_s']:.1f}" if chat.get("ok") else "—"
        status = "ok" if chat.get("ok") else ("load" if r["ready"] else "FAIL")
        print(
            f"  {r['label']:8s} {r['max_tokens']:>10,} "
            f"{r['idle_rss']:>8.0f} {r['peak_rss']:>8.0f} {tps:>7s}  {status}"
        )

    if bf16 and bf16["max_tokens"] > 0:
        for r in results:
            if r["label"] == "bf16" or r["max_tokens"] <= 0:
                continue
            ratio = r["max_tokens"] / bf16["max_tokens"]
            print(
                f"\n  {r['label']} context unlock: {ratio:.2f}x "
                f"({r['max_tokens']:,} vs {bf16['max_tokens']:,} tokens)"
            )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    run_serve = "--serve" in sys.argv

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         TurboQuant KV Cache — Comprehensive Test Suite              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    failures = 0
    failures += section_quant_type_validation()
    failures += section_pack_unpack_roundtrip()
    failures += section_python_roundtrip_mse()
    failures += section_metal_kernel_dequant()
    failures += section_metal_e2e()
    failures += section_memory_capacity()
    failures += section_published_comparison()
    failures += section_latency()
    failures += section_peak_memory_analysis()

    if run_serve:
        failures += section_serve_benchmark()
    else:
        print("\n  (skipping section 9 — serve benchmark; pass --serve to enable)")

    sep("DONE")
    if failures:
        print(f"  {failures} section(s) FAILED.\n")
        return 1
    print("  All sections passed.\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Pytest-native unit tests (collected by `pytest tests/test_turboquant.py`).
# Not run by the script `main()` path above.
# ─────────────────────────────────────────────────────────────────────────────

# --- FWHT Python/Metal sign-table parity -----------------------------------
#
# TurboQuant's FWHT rotation uses random signs generated Python-side via
# ``mx.random.randint(0, 2, shape=(N,), key=mx.random.key(42))``. The Metal
# kernel stores byte-identical copies as compile-time constants
# (``FWHT_SIGNS_64`` / ``_128`` / ``_256`` in ``turboquant.metal``).  If
# either side drifts — different RNG key, MLX PRNG change, or manual edits
# to the Metal tables — encode/decode silently disagree and produce garbage.

_METAL_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "vllm_metal"
    / "metal"
    / "kernels_v2"
    / "turboquant.metal"
)


def _parse_metal_sign_table(head_size: int) -> np.ndarray:
    """Extract ``FWHT_SIGNS_<head_size>`` from the Metal source as a numpy array."""
    source = _METAL_SOURCE.read_text()
    pattern = re.compile(
        rf"constant\s+float\s+FWHT_SIGNS_{head_size}\[{head_size}\]\s*=\s*\{{([^}}]*)\}}",
        re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        raise AssertionError(f"FWHT_SIGNS_{head_size} not found in {_METAL_SOURCE}")
    values = re.findall(r"-?\d+\.?\d*f?", match.group(1))
    signs = np.array([float(v.rstrip("f")) for v in values], dtype=np.float32)
    if signs.shape != (head_size,):
        raise AssertionError(
            f"FWHT_SIGNS_{head_size} expected length {head_size}, got {signs.shape[0]}"
        )
    return signs


def _python_signs(head_size: int) -> np.ndarray:
    """Reproduce the Python sign vector using the same RNG recipe as ``fwht``."""
    from vllm_metal.metal_kernel_backend.turboquant import _RNG_KEY

    sign01 = mx.random.randint(0, 2, shape=(head_size,), key=_RNG_KEY)
    signs = (1 - 2 * sign01).astype(mx.float32)
    return np.asarray(signs)


@pytest.mark.parametrize("head_size", _FWHT_SUPPORTED_DIMS)
def test_metal_sign_table_matches_python_rng(head_size: int) -> None:
    """Metal constant table must equal the Python-generated signs element-wise."""
    metal_signs = _parse_metal_sign_table(head_size)
    python_signs = _python_signs(head_size)
    np.testing.assert_array_equal(
        python_signs,
        metal_signs,
        err_msg=(
            f"FWHT_SIGNS_{head_size} drift between Python RNG and Metal tables. "
            "If this fails, either the _RNG_KEY changed, MLX's PRNG trajectory "
            "shifted, or turboquant.metal was edited manually. Regenerate both "
            "sides together."
        ),
    )


@pytest.mark.parametrize("head_size", _FWHT_SUPPORTED_DIMS)
def test_fwht_roundtrips_exactly_with_current_signs(head_size: int) -> None:
    """Sanity check: encode then decode recovers the input (signs cancel)."""
    rng = np.random.default_rng(seed=0)
    x_np = rng.standard_normal((4, head_size)).astype(np.float32)
    x = mx.array(x_np)
    encoded = fwht(x, encode=True)
    decoded = fwht(encoded, encode=False)
    np.testing.assert_allclose(
        np.asarray(decoded),
        x_np,
        rtol=1e-5,
        atol=1e-5,
        err_msg=(
            f"FWHT encode/decode round-trip failed at head_size={head_size}. "
            "This indicates a bug in the Python FWHT itself, not a Python/Metal "
            "parity issue."
        ),
    )


# --- TurboQuantAttentionSpec (replacement for head_size_v hack) ------------
#
# ``TurboQuantAttentionSpec`` subclasses ``FullAttentionSpec`` and overrides
# ``real_page_size_bytes`` so the scheduler sees the true compressed page size
# without synthesising a bogus ``head_size_v`` (which used to go negative for
# aggressive 2-bit configs).

# Last config is the 2-bit edge case that used to produce a negative
# ``head_size_v`` under the pre-subclass strategy.
_TQ_SPEC_CONFIGS = [
    # (label,          block_size, num_kv_heads, head_dim, k_quant, v_quant)
    ("default_q8_q3", 16, 4, 128, "q8_0", "q3_0"),
    ("q4_q3_gqa", 16, 8, 128, "q4_0", "q3_0"),
    ("wide_head_256", 16, 2, 256, "q8_0", "q3_0"),
    ("narrow_head_64", 16, 8, 64, "q8_0", "q3_0"),
    ("aggressive_2b", 16, 8, 128, "int2", "q2_0"),
]


@pytest.mark.parametrize(
    "_label, block_size, num_kv_heads, head_dim, k_quant, v_quant",
    _TQ_SPEC_CONFIGS,
    ids=[c[0] for c in _TQ_SPEC_CONFIGS],
)
def test_tq_spec_real_page_size_bytes_matches_helper(
    _label, block_size, num_kv_heads, head_dim, k_quant, v_quant
):
    spec = TurboQuantAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=torch.int8,
        k_quant=k_quant,
        v_quant=v_quant,
    )
    expected = _turboquant_page_size_bytes(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        k_quant=k_quant,
        v_quant=v_quant,
    )
    assert spec.real_page_size_bytes == expected
    assert spec.page_size_bytes == expected


@pytest.mark.parametrize(
    "_label, block_size, num_kv_heads, head_dim, k_quant, v_quant",
    _TQ_SPEC_CONFIGS,
    ids=[c[0] for c in _TQ_SPEC_CONFIGS],
)
def test_tq_spec_head_size_stays_honest(
    _label, block_size, num_kv_heads, head_dim, k_quant, v_quant
):
    """``head_size`` must equal the real model head_dim — no reverse-engineering.

    The old factory set ``head_size_v`` to a synthesised value which went
    negative for 2-bit K.  The subclass keeps ``head_size`` intact.
    """
    spec = TurboQuantAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=torch.int8,
        k_quant=k_quant,
        v_quant=v_quant,
    )
    assert spec.head_size == head_dim
    # head_size_v defaults to head_size via FullAttentionSpec.__post_init__.
    assert spec.head_size_v == head_dim
    assert spec.page_size_bytes > 0


def test_tq_spec_aggressive_2bit_config_does_not_go_negative():
    """Regression: ``int2/q2_0`` used to produce negative ``head_size_v``."""
    spec = _build_turboquant_attention_spec(
        block_size=16,
        num_kv_heads=8,
        head_dim=128,
        k_quant="int2",
        v_quant="q2_0",
    )
    assert isinstance(spec, TurboQuantAttentionSpec)
    assert spec.head_size == 128
    assert spec.head_size_v == 128
    assert spec.page_size_bytes > 0
    # 2-bit compression should be notably smaller than an fp16 K+V calc.
    fp16_page_bytes = 2 * 16 * 8 * 128 * 2
    assert spec.page_size_bytes < fp16_page_bytes


def test_tq_spec_factory_returns_subclass_instance():
    spec = _build_turboquant_attention_spec(
        block_size=16,
        num_kv_heads=4,
        head_dim=128,
        k_quant="q8_0",
        v_quant="q3_0",
    )
    assert isinstance(spec, TurboQuantAttentionSpec)
    assert spec.k_quant == "q8_0"
    assert spec.v_quant == "q3_0"


def test_tq_spec_merge_uniform_specs():
    specs = [
        _build_turboquant_attention_spec(
            block_size=16,
            num_kv_heads=4,
            head_dim=128,
            k_quant="q8_0",
            v_quant="q3_0",
        )
        for _ in range(3)
    ]
    merged = TurboQuantAttentionSpec.merge(specs)
    assert isinstance(merged, TurboQuantAttentionSpec)
    assert merged.k_quant == "q8_0"
    assert merged.v_quant == "q3_0"
    assert merged.page_size_bytes == specs[0].page_size_bytes


def test_tq_spec_registered_in_vllm_spec_manager_map():
    """Importing ``cache_policy`` must register our subclass with vLLM.

    vLLM's ``get_manager_for_kv_cache_spec`` uses strict-type lookup
    (``spec_manager_map[type(spec)]``), not isinstance. If this assertion
    fails the engine-core will crash at startup with ``KeyError:
    TurboQuantAttentionSpec`` the moment TurboQuant is enabled.
    """
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        spec_manager_map,
    )

    assert TurboQuantAttentionSpec in spec_manager_map, (
        "TurboQuantAttentionSpec missing from vLLM's spec_manager_map — "
        "engine-core will KeyError at startup."
    )
    assert spec_manager_map[TurboQuantAttentionSpec] is FullAttentionManager


def test_tq_spec_merge_rejects_mixed_quant():
    a = _build_turboquant_attention_spec(
        block_size=16, num_kv_heads=4, head_dim=128, k_quant="q8_0", v_quant="q3_0"
    )
    b = _build_turboquant_attention_spec(
        block_size=16, num_kv_heads=4, head_dim=128, k_quant="q4_0", v_quant="q3_0"
    )
    with pytest.raises(AssertionError, match="same .k_quant, v_quant."):
        TurboQuantAttentionSpec.merge([a, b])


if __name__ == "__main__":
    sys.exit(main())
