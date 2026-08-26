# SPDX-License-Identifier: Apache-2.0
"""TurboQuant unit and kernel-contract tests."""

import math
import re
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from tests.stub_runner import make_stub_runner
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.turboquant import (
    _RNG_KEY,
    BLOCK_SIZE,
    FWHT_SUPPORTED_HEAD_DIMS,
    QUANT_PARAMS,
    V_QUANT_PARAMS,
    fwht,
    get_v_centroids,
    packed_dim,
    turbo_quant_decode,
    turbo_quant_encode,
)
from vllm_metal.config import get_config, reset_config
from vllm_metal.metal import get_ops
from vllm_metal.v1.cache_policy import (
    TurboQuantAttentionSpec,
    _build_turboquant_attention_spec,
    turboquant_page_size_bytes,
)


def mean_cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_f = a.reshape(-1, a.shape[-1]).astype(mx.float32)
    b_f = b.reshape(-1, b.shape[-1]).astype(mx.float32)
    dot = mx.sum(a_f * b_f, axis=-1)
    a_norm = mx.linalg.norm(a_f, axis=-1)
    b_norm = mx.linalg.norm(b_f, axis=-1)
    return mx.mean(dot / (a_norm * b_norm + 1e-8)).item()


def _fill_cache(
    cache: MetalPagedKVCache,
    k_packed: mx.array,
    v_packed: mx.array,
    k_scale: mx.array,
    v_scale: mx.array,
    k_zero: mx.array,
    slot: mx.array,
) -> None:
    """Scatter packed values into one layer of the paged cache."""
    num_kv_heads = k_packed.shape[1]
    scale_group_count = k_scale.shape[-1]

    flat_k = cache.key_caches[0].reshape(-1, num_kv_heads, cache.k_packed_dim)
    flat_k[slot] = k_packed
    cache.key_caches[0] = flat_k.reshape(cache.key_caches[0].shape)

    flat_v = cache.value_caches[0].reshape(-1, num_kv_heads, cache.v_packed_dim)
    flat_v[slot] = v_packed
    cache.value_caches[0] = flat_v.reshape(cache.value_caches[0].shape)

    for arrays, data in (
        (cache.key_scale_caches, k_scale),
        (cache.value_scale_caches, v_scale),
        (cache.key_zero_caches, k_zero),
    ):
        flat = arrays[0].reshape(-1, num_kv_heads, scale_group_count)
        flat[slot] = data
        arrays[0] = flat.reshape(arrays[0].shape)


def _python_attention_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
) -> mx.array:
    """Return a single-sequence grouped-query attention reference."""
    repeats = q.shape[1] // k.shape[1]
    k_rep = mx.repeat(k, repeats, axis=1).astype(mx.float32)
    v_rep = mx.repeat(v, repeats, axis=1).astype(mx.float32)
    scores = mx.einsum("qhd,khd->qhk", q.astype(mx.float32), k_rep) * scale
    return mx.einsum("qhk,khd->qhd", mx.softmax(scores, axis=-1), v_rep)


# --- FWHT Python/Metal sign-table parity -----------------------------------
#
# TurboQuant's FWHT rotation uses random signs generated Python-side via
# ``mx.random.randint(0, 2, shape=(N,), key=mx.random.key(42))``. The Metal
# kernel stores byte-identical copies as compile-time constants
# (``FWHT_SIGNS_64`` / ``_128`` / ``_256`` / ``_512`` in ``turboquant.metal``).  If
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
    sign01 = mx.random.randint(0, 2, shape=(head_size,), key=_RNG_KEY)
    signs = (1 - 2 * sign01).astype(mx.float32)
    return np.asarray(signs)


@pytest.mark.parametrize("head_size", FWHT_SUPPORTED_HEAD_DIMS)
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


@pytest.mark.parametrize("head_size", FWHT_SUPPORTED_HEAD_DIMS)
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


def test_turboquant_cache_accepts_head_dim_512() -> None:
    """Regression: TurboQuant cache allocation should accept 512-dim heads."""
    num_layers = 1
    num_blocks = 2
    block_size = 16
    num_kv_heads = 2
    head_dim = 512
    k_quant = "q8_0"
    v_quant = "q3_0"
    key_bits = QUANT_PARAMS[k_quant]["bits"]
    value_bits = V_QUANT_PARAMS[v_quant]["bits"]
    scale_group_count = head_dim // BLOCK_SIZE
    cache = MetalPagedKVCache(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=mx.float16,
        turboquant=True,
        k_quant=k_quant,
        v_quant=v_quant,
    )

    assert cache.k_packed_dim == packed_dim(head_dim, key_bits)
    assert cache.v_packed_dim == packed_dim(head_dim, value_bits)
    assert cache.key_caches[0].shape == (
        num_blocks,
        block_size,
        num_kv_heads,
        cache.k_packed_dim,
    )
    assert cache.value_caches[0].shape == (
        num_blocks,
        block_size,
        num_kv_heads,
        cache.v_packed_dim,
    )
    assert cache.key_scale_caches[0].shape == (
        num_blocks,
        block_size,
        num_kv_heads,
        scale_group_count,
    )
    assert cache.value_scale_caches[0].shape == (
        num_blocks,
        block_size,
        num_kv_heads,
        scale_group_count,
    )
    assert cache.key_zero_caches[0].shape == (
        num_blocks,
        block_size,
        num_kv_heads,
        scale_group_count,
    )


def test_turboquant_512_head_dim_matches_python_reference() -> None:
    """The 512-dim TurboQuant kernel path should match Python dequant attention."""
    num_blocks = 4
    block_size = 16
    num_tokens = block_size
    num_kv_heads = 2
    num_query_heads = 4
    head_dim = 512
    cache = MetalPagedKVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=mx.float16,
        turboquant=True,
        k_quant="q8_0",
        v_quant="q3_0",
    )
    k = mx.random.normal(
        shape=(num_tokens, num_kv_heads, head_dim), key=mx.random.key(11)
    ).astype(mx.float16)
    v = mx.random.normal(
        shape=(num_tokens, num_kv_heads, head_dim), key=mx.random.key(12)
    ).astype(mx.float16)
    q = mx.random.normal(
        shape=(1, num_query_heads, head_dim), key=mx.random.key(13)
    ).astype(mx.float16)
    mx.eval(k, v, q)

    (k_packed, k_scale, k_zero), (v_packed, v_scale) = turbo_quant_encode(k, v, "q8_0")
    mx.eval(k_packed, k_scale, k_zero, v_packed, v_scale)

    k_ref, v_ref = turbo_quant_decode(
        (k_packed, k_scale, k_zero),
        (v_packed, v_scale),
        output_dtype=mx.float16,
        key_quant_type="q8_0",
    )
    mx.eval(k_ref, v_ref)

    slot = mx.arange(num_tokens, dtype=mx.int64)
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
    seq_lens = mx.array([num_tokens], dtype=mx.int32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    out_metal = mx.zeros((1, num_query_heads, head_dim), dtype=mx.float16)
    mx.eval(block_tables, seq_lens, cu_seqlens_q, out_metal)

    attn_scale = 1.0 / math.sqrt(head_dim)
    v_centroids = get_v_centroids(cache.v_bits)
    ops = get_ops()
    ops.paged_attention_primitive(
        q,
        cache.key_caches[0],
        cache.value_caches[0],
        num_kv_heads,
        attn_scale,
        0.0,
        block_tables,
        seq_lens,
        cu_seqlens_q,
        block_size,
        num_tokens,
        -1,
        out_metal,
        key_scale_cache=cache.key_scale_caches[0],
        value_scale_cache=cache.value_scale_caches[0],
        key_zero_cache=cache.key_zero_caches[0],
        v_centroids=v_centroids,
        use_turboquant=True,
        quant_type="q8_0",
        v_bits=cache.v_bits,
    )
    mx.eval(out_metal)

    out_ref = _python_attention_reference(q, k_ref, v_ref, attn_scale)
    mx.eval(out_ref)

    diff = out_metal.astype(mx.float32) - out_ref.astype(mx.float32)
    mean_abs_diff = mx.mean(mx.abs(diff)).item()
    ref_mean_abs = mx.mean(mx.abs(out_ref.astype(mx.float32))).item() + 1e-8
    relative_error_percent = mean_abs_diff / ref_mean_abs * 100.0

    assert relative_error_percent < 5.0


def test_tq_encode_kernel_supports_head_dim_512() -> None:
    """The fused ``ops.tq_encode`` Metal kernel must accept head_dim=512.

    Regression for the kernel-side 512 gap: ``MetalPagedKVCache`` and the
    decode kernel already supported 512-dim, but the fused encode primitive
    only had instantiations for {64, 128, 256} and rejected 512 with a
    runtime guard — breaking Gemma-style models with full-attn head_dim=512
    on the first forward pass after the Python encode fallback was removed.

    This test goes through ``ops.tq_encode`` directly (not the Python
    ``_fill_cache`` path used by ``test_turboquant_512_head_dim_matches_
    python_reference``) and verifies its outputs match the Python encode.
    """
    head_dim = 512
    num_tokens = 16
    num_kv_heads = 2
    num_blocks = 4
    block_size = 16
    k_quant, v_quant = "q8_0", "q3_0"
    k_bits = QUANT_PARAMS[k_quant]["bits"]
    v_bits = V_QUANT_PARAMS[v_quant]["bits"]
    k_signed = bool(QUANT_PARAMS[k_quant]["signed"])

    cache = MetalPagedKVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=mx.float16,
        turboquant=True,
        k_quant=k_quant,
        v_quant=v_quant,
    )

    rng = np.random.default_rng(seed=512)
    k = mx.array(
        rng.standard_normal((num_tokens, num_kv_heads, head_dim)).astype(np.float16)
    )
    v = mx.array(
        rng.standard_normal((num_tokens, num_kv_heads, head_dim)).astype(np.float16)
    )
    slot_mapping = mx.arange(num_tokens, dtype=mx.int64)
    mx.eval(k, v, slot_mapping)

    ops = get_ops()
    v_centroids = get_v_centroids(v_bits)
    new_k, new_v, new_ks, new_vs, new_kz = ops.tq_encode(
        k,
        v,
        cache.key_caches[0],
        cache.value_caches[0],
        cache.key_scale_caches[0],
        cache.value_scale_caches[0],
        cache.key_zero_caches[0],
        slot_mapping,
        v_centroids,
        v_bits,
        k_bits,
        k_signed,
    )
    mx.eval(new_k, new_v, new_ks, new_vs, new_kz)

    (
        (packed_k_ref, k_scale_ref, k_zero_ref),
        (
            packed_v_ref,
            v_scale_ref,
        ),
    ) = turbo_quant_encode(k, v, k_quant, value_bits=v_bits)
    mx.eval(packed_k_ref, k_scale_ref, k_zero_ref, packed_v_ref, v_scale_ref)

    # K indices: ±2 on the int8 grid, ≥95% exact (same tolerance the
    # encode-parity sweep uses for the 128-dim case).
    flat_k_kernel = new_k.reshape(-1, num_kv_heads, head_dim)[:num_tokens]
    k_kernel = np.asarray(flat_k_kernel.astype(mx.int32))
    k_ref = np.asarray(packed_k_ref.astype(mx.int32))
    k_diff = np.abs(k_kernel - k_ref)
    assert (k_diff <= 2).all(), f"K indices drift > 2: max={int(k_diff.max())}"
    assert (k_diff == 0).mean() >= 0.99, (
        f"K exact-match rate {(k_diff == 0).mean():.4f} below 0.99"
    )

    # K scales / zero-points (fp16).
    scale_groups = head_dim // BLOCK_SIZE
    flat_ks = new_ks.reshape(-1, num_kv_heads, scale_groups)[:num_tokens]
    flat_kz = new_kz.reshape(-1, num_kv_heads, scale_groups)[:num_tokens]
    assert np.allclose(
        np.asarray(flat_ks.astype(mx.float32)),
        np.asarray(k_scale_ref.astype(mx.float32)),
        rtol=1e-3,
        atol=1e-3,
    )
    kz_diff = np.abs(
        np.asarray(flat_kz.astype(mx.float32))
        - np.asarray(k_zero_ref.astype(mx.float32))
    )
    assert (kz_diff <= 1.0).all(), f"K zero-point drift > 1: max={kz_diff.max():.2e}"

    # V scales (fp16).
    flat_vs = new_vs.reshape(-1, num_kv_heads, scale_groups)[:num_tokens]
    assert np.allclose(
        np.asarray(flat_vs.astype(mx.float32)),
        np.asarray(v_scale_ref.astype(mx.float32)),
        rtol=1e-2,
        atol=1e-3,
    )


def test_turboquant_per_layer_shapes_raise_early() -> None:
    """TurboQuant must keep rejecting per-layer KV shapes until PR2 lands."""
    reset_config()
    config = get_config()
    config.turboquant = True
    config.k_quant = "q8_0"
    config.v_quant = "q3_0"

    try:
        runner = make_stub_runner(
            num_layers=2,
            num_kv_cache_layers=2,
            num_kv_heads=16,
            head_dim=256,
            kv_cache_dtype=mx.bfloat16,
            cache_config=SimpleNamespace(block_size=16),
            kv_heads_per_layer=[16, 4],
            head_dim_per_layer=[256, 512],
        )

        with pytest.raises(
            NotImplementedError, match="TurboQuant with per-layer KV shapes"
        ):
            runner.get_kv_cache_spec()

        with pytest.raises(
            NotImplementedError, match="TurboQuant with per-layer KV shapes"
        ):
            runner.build_paged_attention_runtime(block_size=16)
    finally:
        reset_config()


# --- TurboQuantAttentionSpec (replacement for head_size_v hack) ------------
#
# ``TurboQuantAttentionSpec`` subclasses ``FullAttentionSpec`` and derives the
# packed ``state_content_bytes`` in ``__post_init__`` so the scheduler sees the
# true compressed page size (vLLM 0.28.0 computes ``page_size_bytes`` from that
# field; ``real_page_size_bytes`` is now just upstream's alias) without
# synthesising a bogus ``head_size_v`` (which used to go negative for
# aggressive 2-bit configs).

# Last config is the 2-bit edge case that used to produce a negative
# ``head_size_v`` under the pre-subclass strategy.
_TQ_SPEC_CONFIGS = [
    # (block_size, num_kv_heads, head_dim, k_quant, v_quant)
    pytest.param(16, 4, 128, "q8_0", "q3_0", id="default_q8_q3"),
    pytest.param(16, 8, 128, "q4_0", "q3_0", id="q4_q3_gqa"),
    pytest.param(16, 2, 256, "q8_0", "q3_0", id="wide_head_256"),
    pytest.param(16, 2, 512, "q8_0", "q3_0", id="wide_head_512"),
    pytest.param(16, 8, 64, "q8_0", "q3_0", id="narrow_head_64"),
    pytest.param(16, 8, 128, "int2", "q2_0", id="aggressive_2b"),
]


@pytest.mark.parametrize(
    "block_size, num_kv_heads, head_dim, k_quant, v_quant",
    _TQ_SPEC_CONFIGS,
)
def test_tq_spec_page_size_bytes_matches_helper(
    block_size, num_kv_heads, head_dim, k_quant, v_quant
):
    """Even a bare construction bills the scheduler the packed page size.

    vLLM 0.28.0 computes ``page_size_bytes`` from the ``state_content_bytes``
    field (``real_page_size_bytes`` is a dead alias); ``__post_init__``
    derives the field, so no construction path can fall back to the dense
    int8 formula.
    """
    spec = TurboQuantAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=torch.int8,
        k_quant=k_quant,
        v_quant=v_quant,
    )
    expected = turboquant_page_size_bytes(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        k_quant=k_quant,
        v_quant=v_quant,
    )
    assert spec.page_size_bytes == expected
    assert spec.real_page_size_bytes == expected


@pytest.mark.parametrize(
    "block_size, num_kv_heads, head_dim, k_quant, v_quant",
    _TQ_SPEC_CONFIGS,
)
def test_tq_spec_head_size_stays_honest(
    block_size, num_kv_heads, head_dim, k_quant, v_quant
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


def test_tq_spec_resolves_to_full_attention_manager():
    """TurboQuant specs must resolve to vLLM's full-attention manager.

    vLLM 0.23 resolves ``FullAttentionSpec`` subclasses through
    ``KVCacheSpecRegistry`` MRO fallback.
    """
    spec = _build_turboquant_attention_spec(
        block_size=16,
        num_kv_heads=4,
        head_dim=128,
        k_quant="q8_0",
        v_quant="q3_0",
    )

    assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
    assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is FullAttentionSpec


def test_tq_spec_merge_rejects_mixed_quant():
    a = _build_turboquant_attention_spec(
        block_size=16, num_kv_heads=4, head_dim=128, k_quant="q8_0", v_quant="q3_0"
    )
    b = _build_turboquant_attention_spec(
        block_size=16, num_kv_heads=4, head_dim=128, k_quant="q4_0", v_quant="q3_0"
    )
    with pytest.raises(ValueError, match=r"same \(k_quant, v_quant\)"):
        TurboQuantAttentionSpec.merge([a, b])


@pytest.mark.parametrize(
    "head_dim,k_quant,v_bits",
    [
        (64, "q8_0", 3),
        (128, "q8_0", 3),
        (128, "q4_0", 3),
        (128, "q8_0", 4),
        (256, "q8_0", 3),
        (512, "q8_0", 3),
    ],
    ids=[
        "hs64-q80-v3",
        "hs128-q80-v3",
        "hs128-q40-v3",
        "hs128-q80-v4",
        "hs256-q80-v3",
        "hs512-q80-v3",
    ],
)
def test_metal_encode_python_decode_roundtrip(
    head_dim: int, k_quant: str, v_bits: int
) -> None:
    """End-to-end: Metal `ops.tq_encode` → Python `turbo_quant_decode` → fp16 K/V.

    The encode-parity test (`test_tq_encode_kernel_supports_head_dim_512` and
    `section_metal_encode_parity`) only checks that the Metal kernel produces
    the same packed bytes as Python `turbo_quant_encode`.  That's necessary
    but not sufficient: a bug in the bit-packing layout, scale-group stride,
    or signed-vs-unsigned interpretation could produce parity-matching but
    semantically wrong cache contents that decode to garbage.

    This test goes the other direction: feed random fp16 K/V through the
    Metal encode kernel, then read the packed cache bytes and dequantise
    via the Python decode helpers (which the production decode kernel
    must agree with by parity guarantees).  If the round-trip K/V have
    abnormally high error vs the original, the bug is in the Metal encode
    layout regardless of which path consumes it.

    Tolerances are calibrated from the published quantisation MSE numbers:
    q8_0 K → MSE ≲ 4e-4, 3-bit V → cos_sim ≥ 0.97.  4-bit V is tighter
    (≥ 0.99); we use the same threshold across V widths since the centroid
    table compensates.
    """
    num_tokens = 16
    num_kv_heads = 2
    num_blocks = 4
    block_size = 16
    k_bits = QUANT_PARAMS[k_quant]["bits"]
    k_signed = bool(QUANT_PARAMS[k_quant]["signed"])

    # The cache infra requires v_quant to be a registered name; pick the one
    # that matches v_bits.  We only run the kernel through `ops.tq_encode`
    # which takes v_bits directly, so the cache's v_quant is just for shape.
    v_quant = {3: "q3_0", 4: "q4_0", 8: "uint8"}.get(v_bits, "q3_0")

    cache = MetalPagedKVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=mx.float16,
        turboquant=True,
        k_quant=k_quant,
        v_quant=v_quant,
    )

    seed = head_dim * 100 + QUANT_PARAMS[k_quant]["bits"] * 10 + v_bits
    rng = np.random.default_rng(seed=seed)
    k = mx.array(
        rng.standard_normal((num_tokens, num_kv_heads, head_dim)).astype(np.float16)
    )
    v = mx.array(
        rng.standard_normal((num_tokens, num_kv_heads, head_dim)).astype(np.float16)
    )
    slot_mapping = mx.arange(num_tokens, dtype=mx.int64)
    mx.eval(k, v, slot_mapping)

    # ---- Metal encode ----
    ops = get_ops()
    v_centroids = get_v_centroids(v_bits)
    new_k, new_v, new_ks, new_vs, new_kz = ops.tq_encode(
        k,
        v,
        cache.key_caches[0],
        cache.value_caches[0],
        cache.key_scale_caches[0],
        cache.value_scale_caches[0],
        cache.key_zero_caches[0],
        slot_mapping,
        v_centroids,
        v_bits,
        k_bits,
        k_signed,
    )
    mx.eval(new_k, new_v, new_ks, new_vs, new_kz)

    # ---- Slice the populated rows out of the paged cache ----
    scale_groups = head_dim // BLOCK_SIZE
    k_packed_dim = cache.k_packed_dim
    v_packed_dim = cache.v_packed_dim

    k_pkt = new_k.reshape(-1, num_kv_heads, k_packed_dim)[:num_tokens]
    v_pkt = new_v.reshape(-1, num_kv_heads, v_packed_dim)[:num_tokens]
    ks_pkt = new_ks.reshape(-1, num_kv_heads, scale_groups)[:num_tokens]
    vs_pkt = new_vs.reshape(-1, num_kv_heads, scale_groups)[:num_tokens]
    kz_pkt = new_kz.reshape(-1, num_kv_heads, scale_groups)[:num_tokens]
    mx.eval(k_pkt, v_pkt, ks_pkt, vs_pkt, kz_pkt)

    # ---- Python decode on Metal-encoded bytes ----
    k_hat, v_hat = turbo_quant_decode(
        (k_pkt, ks_pkt, kz_pkt),
        (v_pkt, vs_pkt),
        output_dtype=mx.float16,
        key_quant_type=k_quant,
        value_bits=v_bits,
    )
    mx.eval(k_hat, v_hat)

    # ---- Compare to original ----
    k_mse = mx.mean((k.astype(mx.float32) - k_hat.astype(mx.float32)) ** 2).item()
    v_mse = mx.mean((v.astype(mx.float32) - v_hat.astype(mx.float32)) ** 2).item()
    k_cos = mean_cosine_similarity(k, k_hat)
    v_cos = mean_cosine_similarity(v, v_hat)

    # K tolerance: q8_0 random-input MSE ≈ 1e-4, q4_0 ≈ 5e-3 to 1e-2.  We
    # set thresholds at ~5x the typical observed value so a bit-layout bug
    # (which would give MSE ≥ 0.1, often ≥ 0.5) trips the assertion while
    # benign quant noise doesn't.
    k_mse_threshold = 1e-3 if k_bits >= 8 else (3e-2 if k_bits >= 4 else 1e-1)
    assert k_mse < k_mse_threshold, (
        f"K roundtrip MSE {k_mse:.6f} ≥ {k_mse_threshold} "
        f"(k_quant={k_quant}, head_dim={head_dim}) — Metal encode bytes do "
        f"not decode back to the input via Python; suspect bit-packing or "
        f"scale-group layout mismatch."
    )
    assert k_cos >= 0.99, (
        f"K roundtrip cos_sim {k_cos:.4f} < 0.99 "
        f"(k_quant={k_quant}, head_dim={head_dim})"
    )

    # V tolerance: 3-bit FWHT+Lloyd-Max gives cos_sim ≥ 0.97 typically.
    v_cos_threshold = 0.95 if v_bits <= 3 else 0.98
    assert v_cos >= v_cos_threshold, (
        f"V roundtrip cos_sim {v_cos:.4f} < {v_cos_threshold} "
        f"(v_bits={v_bits}, head_dim={head_dim}) — Metal V encode bytes do "
        f"not decode back to the input via Python; suspect FWHT sign-table "
        f"mismatch, centroid lookup, or sub-8-bit packing layout."
    )
    # V MSE is bounded loosely by 1.0 (3-bit is intrinsically lossy);
    # finite + reasonable is the real check.
    assert math.isfinite(v_mse) and v_mse < 2.0, (
        f"V roundtrip MSE {v_mse} not finite/reasonable"
    )
