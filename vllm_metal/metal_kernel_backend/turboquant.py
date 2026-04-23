# SPDX-License-Identifier: Apache-2.0
"""TurboQuant quantization helpers for Metal paged attention.

Provides the Python-side encode/decode path for TurboQuant KV compression,
including key/value quantization metadata, bit packing helpers, and the FWHT
rotation/sign tables used by the Metal dequantization kernels.
"""

import mlx.core as mx
from vllm.logger import init_logger

logger = init_logger(__name__)

_RNG_KEY = mx.random.key(42)

# === Pre-computed Lloyd-Max tables for 3-bit (fast path for Metal V dequant) ===
# These match the Metal kernel's CENTROIDS_3BIT table and are used directly
# without going through the dynamic Lloyd-Max iteration.
CENTROIDS_3BIT = mx.array(
    [
        -2.15195,
        -1.34391,
        -0.75601,
        -0.24509,
        0.24509,
        0.75601,
        1.34391,
        2.15195,
    ],
    dtype=mx.float32,
)

BOUNDARIES_3BIT = mx.array(
    [
        -1.74793,
        -1.04996,
        -0.50055,
        0.0,
        0.50055,
        1.04996,
        1.74793,
    ],
    dtype=mx.float32,
)

# Quantization block size: number of elements per scale group.
# Must match the Metal kernel's compile-time SCALE_GROUP_SIZE constant (also 32)
# and the scale cache allocation in cache.py (head_dim // 32 groups).
BLOCK_SIZE = 32

# === Quantization parameters ===
# - "signed":  True  → stored as int8 (char in Metal), idx in [-max, max]
#              False → stored as uint8 (uchar in Metal), idx in [0, max]
# - "bits":    storage bit width (sub-8-bit types are bit-packed into uint8)
# - Sub-8-bit types are always unsigned: bit packing only works on unsigned bytes
#   and the Metal kernel unpacks them as unsigned raw values.
# Key quantization params (used with --turboquant-k-quant)
QUANT_PARAMS = {
    # Signed 8-bit (no packing)
    "q8_0": {"signed": True, "bits": 8, "dtype": mx.int8, "block_size": 32},
    "int8": {"signed": True, "bits": 8, "dtype": mx.int8, "block_size": 32},
    # Unsigned 8-bit (no packing)
    "uint8": {"signed": False, "bits": 8, "dtype": mx.uint8, "block_size": 32},
    # Sub-8-bit unsigned (bit-packed into uint8 bytes)
    "q5_0": {"signed": False, "bits": 5, "dtype": mx.uint8, "block_size": 32},
    "q4_0": {"signed": False, "bits": 4, "dtype": mx.uint8, "block_size": 32},
    "int4": {"signed": False, "bits": 4, "dtype": mx.uint8, "block_size": 32},
    "uint4": {"signed": False, "bits": 4, "dtype": mx.uint8, "block_size": 32},
    "int2": {"signed": False, "bits": 2, "dtype": mx.uint8, "block_size": 32},
    "uint2": {"signed": False, "bits": 2, "dtype": mx.uint8, "block_size": 32},
}

# Value quantization params (used with --turboquant-v-quant)
# V uses Lloyd-Max quantization with FWHT rotation.
# Centroids are lazily computed for arbitrary bit widths.
V_QUANT_PARAMS = {
    "q2_0": {"signed": False, "bits": 2, "dtype": mx.uint8, "block_size": 32},
    "q3_0": {"signed": False, "bits": 3, "dtype": mx.uint8, "block_size": 32},
    "q4_0": {"signed": False, "bits": 4, "dtype": mx.uint8, "block_size": 32},
    "q5_0": {"signed": False, "bits": 5, "dtype": mx.uint8, "block_size": 32},
    "q8_0": {"signed": False, "bits": 8, "dtype": mx.uint8, "block_size": 32},
}


def searchsorted(boundaries, x):
    return (x[..., None] > boundaries).sum(axis=-1)


FWHT_SUPPORTED_HEAD_DIMS = (64, 128, 256, 512)


def fwht(x: mx.array, encode: bool) -> mx.array:
    dim = x.shape[-1]
    if dim not in FWHT_SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"FWHT only supports head_dim in {FWHT_SUPPORTED_HEAD_DIMS}, got {dim}. "
            "The Metal kernel has hardcoded sign tables only for these sizes."
        )
    sign01 = mx.random.randint(0, 2, shape=(dim,), key=_RNG_KEY)
    signs = 1 - 2 * sign01
    if encode:
        x = x * signs
        x = mx.hadamard_transform(x)
    else:
        x = mx.hadamard_transform(x)
        x *= signs
    return x


# === Lloyd-Max dynamic centroid computation (pure MLX) ===
# For bits != 3, centroids/boundaries are computed via iterative Lloyd-Max
# on unit-normal samples (FWHT output is approximately unit normal).
# Results are cached per bit width; the first call pays the iteration cost.
_LLOYD_MAX_CACHE: dict[int, tuple[mx.array, mx.array]] = {}


def _compute_lloyd_max_normal(bits: int) -> tuple[mx.array, mx.array]:
    """Iterative Lloyd-Max for unit-normal source at given bit width.

    Runs entirely on MLX (no numpy). Uses one-hot masking to compute
    per-cluster means without Python loops over clusters.
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    n = 1 << bits
    samples = mx.random.normal(shape=(200_000,), key=mx.random.key(0)).astype(
        mx.float32
    )
    # Uniform init over the observed range.
    lo = mx.min(samples)
    hi = mx.max(samples)
    centroids = mx.linspace(lo.item(), hi.item(), num=n).astype(mx.float32)

    cluster_ids = mx.arange(n, dtype=mx.int32)
    converged = False
    max_diff = 0.0
    # Relax threshold for higher bit widths where centroids are densely packed
    threshold = 1e-4 if bits >= 5 else 1e-6
    for _ in range(500):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        # assign[i] = cluster index for sample[i] (in [0, n))
        assign = searchsorted(boundaries, samples).astype(mx.int32)
        # one_hot: (N, n)
        one_hot = (assign[:, None] == cluster_ids[None, :]).astype(mx.float32)
        counts = one_hot.sum(axis=0)  # (n,)
        sums = (one_hot * samples[:, None]).sum(axis=0)  # (n,)
        # Empty clusters keep their old centroid to avoid divide-by-zero.
        new_centroids = mx.where(counts > 0, sums / mx.maximum(counts, 1.0), centroids)
        diff = mx.abs(new_centroids - centroids).max().item()
        if diff < threshold:
            centroids = new_centroids
            converged = True
            break
        max_diff = diff
        centroids = new_centroids
    if not converged:
        logger.warning(
            "Lloyd-Max did not converge in 500 iterations for bits=%d; "
            "quantization quality may be suboptimal. Current convergence=%s",
            bits,
            f"{max_diff:.20f}",
        )
    boundaries = (centroids[:-1] + centroids[1:]) * 0.5
    return centroids, boundaries


def lloyd_max_centroids(bits: int) -> tuple[mx.array, mx.array]:
    """Return (centroids, boundaries) for Lloyd-Max quantization of unit normal.

    For bits == 3, returns the precomputed lookup used by the Metal kernel.
    For other bit widths, computes via iterative Lloyd-Max (cached per width).
    """
    if bits == 3:
        return CENTROIDS_3BIT, BOUNDARIES_3BIT
    if bits not in _LLOYD_MAX_CACHE:
        _LLOYD_MAX_CACHE[bits] = _compute_lloyd_max_normal(bits)
    return _LLOYD_MAX_CACHE[bits]


def get_v_centroids(bits: int) -> mx.array:
    """Get Lloyd-Max centroids for V quantization at given bit width.

    Returns a float32 array of 2^bits centroids, suitable for passing to
    the Metal kernel as a buffer parameter. Results are cached.

    Args:
        bits: Quantization bit width (1-8).

    Returns:
        mx.array of shape (2^bits,) with float32 centroids.
    """
    centroids, _ = lloyd_max_centroids(bits)
    return centroids.astype(mx.float32)


def lm_quant(x: mx.array, bits: int = 3) -> tuple[mx.array, mx.array]:
    """Lloyd-Max quantize ``x`` along its last dim.

    Args:
        x:    Input tensor, last dim must be divisible by BLOCK_SIZE.
        bits: Target bit width. 3 uses the precomputed lookup; other
              widths compute centroids dynamically via Lloyd-Max.

    Returns:
        (indices, scale) where ``indices`` are uint8 values in [0, 2^bits - 1]
        and ``scale`` is the per-block RMS normalizer.
    """
    _, boundaries = lloyd_max_centroids(bits)
    shape = x.shape
    x = x.reshape(*shape[:-1], -1, BLOCK_SIZE)
    scale = mx.sqrt(mx.mean(x**2, axis=-1, keepdims=True))
    x_norm = x / (scale + 1e-8)
    indices = searchsorted(boundaries, x_norm).reshape(shape)
    return indices.astype(mx.uint8), scale.squeeze(-1).astype(mx.float16)


def lm_de_quant(
    indices: mx.array,
    scale: mx.array,
    bits: int = 3,
    block_size: int = 32,
) -> mx.array:
    """Lloyd-Max dequantize (centroid lookup + block-wise rescale)."""
    centroids, _ = lloyd_max_centroids(bits)
    shape = indices.shape
    idx_r = indices.reshape(*shape[:-1], -1, block_size)
    x_norm = centroids[idx_r.astype(mx.int32)]
    x = x_norm * scale[..., None]
    return x.reshape(shape)


def packed_dim(head_dim: int, bits: int) -> int:
    """Packed byte count for head_dim elements at given bit width."""
    if (head_dim * bits) % 8 != 0:
        raise ValueError(
            f"head_dim={head_dim} * bits={bits} = {head_dim * bits} is not divisible by 8; "
            "choose a head_dim and bit width whose product is byte-aligned"
        )
    return head_dim * bits // 8


def _pack_2bit(vals: mx.array) -> mx.array:
    shape = vals.shape
    g = vals.reshape(*shape[:-1], -1, 4).astype(mx.uint8)
    packed = (
        (g[..., 0] & 0x3)
        | ((g[..., 1] & 0x3) << 2)
        | ((g[..., 2] & 0x3) << 4)
        | ((g[..., 3] & 0x3) << 6)
    )
    return packed.reshape(*shape[:-1], -1)


def _unpack_2bit(packed: mx.array, orig_dim: int) -> mx.array:
    shape = packed.shape
    g = packed.reshape(*shape[:-1], -1, 1)
    v0 = g & 0x3
    v1 = (g >> 2) & 0x3
    v2 = (g >> 4) & 0x3
    v3 = (g >> 6) & 0x3
    unpacked = mx.concatenate([v0, v1, v2, v3], axis=-1)
    return unpacked.reshape(*shape[:-1], orig_dim).astype(mx.uint8)


def _pack_3bit(vals: mx.array) -> mx.array:
    shape = vals.shape
    g = vals.reshape(*shape[:-1], -1, 8).astype(mx.uint32)
    v = [g[..., i] for i in range(8)]
    b0 = (v[0] & 0x7) | ((v[1] & 0x7) << 3) | ((v[2] & 0x7) << 6)
    b1 = (
        ((v[2] & 0x7) >> 2)
        | ((v[3] & 0x7) << 1)
        | ((v[4] & 0x7) << 4)
        | ((v[5] & 0x7) << 7)
    )
    b2 = ((v[5] & 0x7) >> 1) | ((v[6] & 0x7) << 2) | ((v[7] & 0x7) << 5)
    packed = mx.stack([b0, b1, b2], axis=-1).astype(mx.uint8)
    return packed.reshape(*shape[:-1], -1)


def _unpack_3bit(packed: mx.array, orig_dim: int) -> mx.array:
    shape = packed.shape
    g = packed.reshape(*shape[:-1], -1, 3).astype(mx.uint32)
    b0, b1, b2 = g[..., 0], g[..., 1], g[..., 2]
    combined = b0 | (b1 << 8) | (b2 << 16)
    vals = []
    for i in range(8):
        vals.append((combined >> (i * 3)) & 0x7)
    unpacked = mx.stack(vals, axis=-1).astype(mx.uint8)
    return unpacked.reshape(*shape[:-1], orig_dim)


def _pack_4bit(vals: mx.array) -> mx.array:
    shape = vals.shape
    g = vals.reshape(*shape[:-1], -1, 2).astype(mx.uint8)
    packed = (g[..., 0] & 0xF) | ((g[..., 1] & 0xF) << 4)
    return packed.reshape(*shape[:-1], -1)


def _unpack_4bit(packed: mx.array, orig_dim: int) -> mx.array:
    shape = packed.shape
    g = packed.reshape(*shape[:-1], -1, 1)
    lo = g & 0xF
    hi = (g >> 4) & 0xF
    unpacked = mx.concatenate([lo, hi], axis=-1)
    return unpacked.reshape(*shape[:-1], orig_dim).astype(mx.uint8)


def _pack_5bit(vals: mx.array) -> mx.array:
    shape = vals.shape
    g = vals.reshape(*shape[:-1], -1, 8).astype(mx.uint64)
    combined = g[..., 0] & 0x1F
    for i in range(1, 8):
        combined = combined | ((g[..., i] & 0x1F) << (i * 5))
    # Extract 5 bytes from the 40-bit combined value
    b = [(combined >> (i * 8)) & 0xFF for i in range(5)]
    packed = mx.stack(b, axis=-1).astype(mx.uint8)
    return packed.reshape(*shape[:-1], -1)


def _unpack_5bit(packed: mx.array, orig_dim: int) -> mx.array:
    shape = packed.shape
    g = packed.reshape(*shape[:-1], -1, 5).astype(mx.uint64)
    combined = (
        g[..., 0]
        | (g[..., 1] << 8)
        | (g[..., 2] << 16)
        | (g[..., 3] << 24)
        | (g[..., 4] << 32)
    )
    vals = []
    for i in range(8):
        vals.append((combined >> (i * 5)) & 0x1F)
    unpacked = mx.stack(vals, axis=-1).astype(mx.uint8)
    return unpacked.reshape(*shape[:-1], orig_dim)


_PACK_FNS = {2: _pack_2bit, 3: _pack_3bit, 4: _pack_4bit, 5: _pack_5bit}
_UNPACK_FNS = {2: _unpack_2bit, 3: _unpack_3bit, 4: _unpack_4bit, 5: _unpack_5bit}


def pack_bits(values: mx.array, bits: int) -> mx.array:
    """Pack sub-8-bit values into bytes.

    values: [..., dim] uint8 with each element using only `bits` bits.
    Returns: [..., dim * bits // 8] uint8.
    """
    if bits == 8:
        return values
    if bits not in _PACK_FNS:
        raise ValueError(f"Unsupported bit width for packing: {bits}")
    return _PACK_FNS[bits](values)


def unpack_bits(packed: mx.array, bits: int, orig_dim: int) -> mx.array:
    """Unpack packed bytes back to per-element uint8.

    packed: [..., packed_dim] uint8.
    Returns: [..., orig_dim] uint8.
    """
    if bits == 8:
        return packed
    if bits not in _UNPACK_FNS:
        raise ValueError(f"Unsupported bit width for unpacking: {bits}")
    return _UNPACK_FNS[bits](packed, orig_dim)


def quantize(
    x: mx.array, output_type: str = "int8"
) -> tuple[mx.array, mx.array, mx.array]:
    """Asymmetric block-wise quantization.

    Handles both signed (int8) and unsigned (uint8 / sub-8-bit) types.
    All variants share the same dequantization formula:

        ``x_hat = (indices + zero_point) * scale``

    Args:
        x:           Input tensor, last dim must be divisible by block_size.
        output_type: Key from QUANT_PARAMS.

    Returns:
        (indices, scale, zero_point).  ``scale`` and ``zero_point`` have shape
        ``(..., head_dim // block_size)`` (one per block). Indices are stored
        in the dtype declared by QUANT_PARAMS — unsigned for sub-8-bit so they
        can be bit-packed directly.
    """
    if output_type not in QUANT_PARAMS:
        available = ", ".join(sorted(QUANT_PARAMS.keys()))
        raise ValueError(
            f"Unsupported quantization type: {output_type}. Available: {available}"
        )

    params = QUANT_PARAMS[output_type]
    bits: int = params["bits"]
    signed = params["signed"]
    dtype = params["dtype"]
    block_size = params["block_size"]

    shape = x.shape
    x = x.reshape(*shape[:-1], -1, block_size)

    x_min = mx.min(x, axis=-1, keepdims=True)
    x_max = mx.max(x, axis=-1, keepdims=True)

    if signed:
        # Signed asymmetric: idx in [-max_val, max_val], max_val = 2^(bits-1) - 1.
        # scale * (max_val - (-max_val)) = x_max - x_min  →  scale = (x_max-x_min)/(2*max_val)
        # Zero-point centers the quantization grid on the block midpoint.
        max_val = (1 << (bits - 1)) - 1
        scale = (x_max - x_min) / (2.0 * max_val)
        zero_point = mx.round((x_max + x_min) / (2.0 * (scale + 1e-8)))
        indices = mx.clip(
            mx.round(x / (scale + 1e-8) - zero_point),
            -max_val,
            max_val,
        ).astype(dtype)
    else:
        # Unsigned asymmetric: idx in [0, max_val], max_val = 2^bits - 1.
        # x_min corresponds to idx=0 (via zero_point offset). This keeps all
        # indices non-negative, which is required for correct bit packing.
        max_val = (1 << bits) - 1
        scale = (x_max - x_min) / max_val
        zero_point = mx.round(x_min / (scale + 1e-8))
        indices = mx.clip(
            mx.round(x / (scale + 1e-8) - zero_point),
            0,
            max_val,
        ).astype(dtype)

    indices = indices.reshape(shape)
    return (
        indices,
        scale.squeeze(-1).astype(mx.float16),
        zero_point.squeeze(-1).astype(mx.float16),
    )


def dequantize(
    indices: mx.array, scale: mx.array, zero_point: mx.array, block_size: int = 32
) -> mx.array:
    """Asymmetric uniform dequantization: ``x = (indices + zero_point) * scale``.

    Args:
        indices: Quantized indices (flattened).
        scale: Scale factors per block.
        zero_point: Zero-point offsets per block.
        block_size: Number of elements per scale block.
    Returns:
        Dequantized tensor.
    """
    shape = indices.shape
    indices_reshaped = indices.reshape(*shape[:-1], -1, block_size)
    x = (indices_reshaped.astype(mx.float32) + zero_point[..., None]) * scale[..., None]
    return x.reshape(shape)


def turbo_quant_encode_value(x: mx.array, bits: int = 3) -> tuple[mx.array, mx.array]:
    """Encode V with FWHT and Lloyd-Max quantization at the given bit width.

    Args:
        x:    Input tensor. Last dim must be a power of two (FWHT) and divisible
              by BLOCK_SIZE (Lloyd-Max scale grouping).
        bits: Target bit width. 3 uses the precomputed lookup (fast path, also
              what the Metal kernel consumes); other widths compute centroids
              dynamically via Lloyd-Max.

    Returns:
        (indices, scale). ``indices`` are uint8 values in [0, 2^bits - 1]
        (not yet bit-packed).
    """
    x = fwht(x, True)
    return lm_quant(x, bits)


def turbo_quant_encode_key(
    x: mx.array, quant_type: str = "q8_0"
) -> tuple[mx.array, mx.array, mx.array]:
    """Key quantization — asymmetric uniform (no FWHT)."""
    return quantize(x, quant_type)


def turbo_quant_decode_value(
    indices: mx.array,
    scale: mx.array,
    output_dtype: mx.Dtype = mx.float32,
    block_size: int = 32,
    bits: int = 3,
) -> mx.array:
    """Decode V with Lloyd-Max dequantization and inverse FWHT."""
    x = lm_de_quant(indices, scale, bits=bits, block_size=block_size)
    x = fwht(x, False)
    return x.astype(output_dtype)


def turbo_quant_decode_key(
    indices: mx.array,
    scale: mx.array,
    zero_point: mx.array,
    output_dtype: mx.Dtype = mx.float32,
    block_size: int = 32,
) -> mx.array:
    """Decode K with asymmetric dequantization: ``x = (indices + zero_point) * scale``."""
    return dequantize(indices, scale, zero_point, block_size).astype(output_dtype)


def turbo_quant_encode(
    key: mx.array,
    value: mx.array,
    key_quant: str = "q8_0",
    value_bits: int = 3,
) -> tuple:
    """Encode key and value arrays for TurboQuant storage.

    Returns:
        ``(k_quant, v_quant)`` where
        ``k_quant = (packed_indices, scale, zero_point)`` and
        ``v_quant = (packed_indices, scale)``. Sub-8-bit indices are bit-packed.
    """
    indices_k, scale_k, zero_k = turbo_quant_encode_key(key, key_quant)
    indices_v, scale_v = turbo_quant_encode_value(value, bits=value_bits)

    k_bits: int = QUANT_PARAMS[key_quant]["bits"]
    if k_bits < 8:
        indices_k = pack_bits(indices_k.astype(mx.uint8), k_bits)
    if value_bits < 8:
        indices_v = pack_bits(indices_v, value_bits)

    return (indices_k, scale_k, zero_k), (indices_v, scale_v)


def turbo_quant_decode(
    k_quant: tuple,
    v_quant: tuple,
    output_dtype: mx.Dtype = mx.float32,
    block_size: int = 32,
    key_quant_type: str = "q8_0",
    value_bits: int = 3,
) -> tuple[mx.array, mx.array]:
    """Decode TurboQuant-encoded key/value pairs."""
    k_indices, k_scale, k_zero = k_quant
    v_indices, v_scale = v_quant

    k_bits: int = QUANT_PARAMS[key_quant_type]["bits"]
    orig_k_dim = k_scale.shape[-1] * block_size
    orig_v_dim = v_scale.shape[-1] * block_size

    if k_bits < 8:
        k_indices = unpack_bits(k_indices, k_bits, orig_k_dim)
    if value_bits < 8:
        v_indices = unpack_bits(v_indices, value_bits, orig_v_dim)

    k = turbo_quant_decode_key(k_indices, k_scale, k_zero, output_dtype, block_size)
    v = turbo_quant_decode_value(
        v_indices, v_scale, output_dtype, block_size, bits=value_bits
    )
    return k, v
