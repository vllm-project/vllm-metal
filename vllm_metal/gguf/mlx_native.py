# SPDX-License-Identifier: Apache-2.0
"""MLX-native quantized GGUF tensors that compute with their packed weights.

``GGUFMLXQuantizedTensor`` repacks a GGUF tensor's raw block bytes into the
affine, group-32 representation that ``mx.quantized_matmul`` already consumes:
a ``uint32`` packed ``qweight`` alongside ``scales`` and ``biases``. It wraps
that triple in an explicit, validated contract and exposes
:meth:`~GGUFMLXQuantizedTensor.matmul` /
:meth:`~GGUFMLXQuantizedTensor.embedding`, which run on the packed weights so
supported weights never get expanded into a dense copy.

Q8_0, Q4_0, and Q4_1 blocks carry one fp16 scale (and Q4_1 one fp16 min) per
32 weights, so their triple stores ``float16`` scales/biases and is byte for
byte what MLX's own GGUF repack in ``mx.load`` produces. Q4_K and Q5_K
sub-blocks are also 32 wide, but their scales are the fp32 products
``d*sc`` / ``-dmin*m``, so those triples store ``float32`` to reproduce
``gguf.quants.dequantize`` bit for bit (#761). K-quants with 16-element
sub-groups (Q6_K/Q3_K/Q2_K) cannot repack, since MLX has no group_size=16
kernels, and stay out of scope for this path.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

try:
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "GGUF support requires the optional 'gguf' dependency. "
        "Install it with: pip install 'vllm-metal[gguf]'"
    ) from exc

# qtypes that repack into MLX's affine representation, with their bit widths.
_BITS: dict[GGMLQuantizationType, int] = {
    GGMLQuantizationType.Q8_0: 8,
    GGMLQuantizationType.Q4_0: 4,
    GGMLQuantizationType.Q4_1: 4,
    GGMLQuantizationType.Q4_K: 4,
    GGMLQuantizationType.Q5_K: 5,
}
AFFINE_GGUF_TYPES = frozenset(_BITS)

# K-quant scales are fp32 products; the other qtypes store the file's fp16 scale.
_FP32_SCALE_TYPES = frozenset({GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K})

# Every affine qtype groups by 32 in MLX's affine quant mode.
_GROUP_SIZE = 32
_QUANT_MODE = "affine"

# Rows repack in chunks of about this many weights so the repack temporaries
# stay bounded on large embedding tables (#773).
_REPACK_CHUNK_ELEMENTS = 1 << 24


@dataclass(frozen=True, eq=False)
class GGUFMLXQuantizedTensor:
    """An MLX-native quantized GGUF weight that computes with its packed data.

    The contract consumers can rely on:

    * ``qweight`` — ``uint32`` packed weights, 2-D, shape :attr:`packed_shape`.
    * ``scales`` / ``biases`` — shape ``(out_features, in_features //
      group_size)``; ``float32`` for Q4_K/Q5_K (required for a bit-exact
      K-quant repack, #761) and ``float16`` for Q8_0/Q4_0/Q4_1.
    * ``qweight_type`` — a ``gguf.GGMLQuantizationType`` in
      :data:`AFFINE_GGUF_TYPES`.
    * logical weight is ``(out_features, in_features)``; :attr:`group_size` is 32;
      :attr:`bits` is 8 for Q8_0, 5 for Q5_K, 4 for Q4_0/Q4_1/Q4_K.
    * activations: :meth:`matmul` accepts float16/bfloat16/float32 ``x`` and
      returns ``x``'s dtype; :meth:`embedding` returns an explicit ``output_dtype``.

    Construct directly when you already hold the affine arrays (e.g. a row slice
    of another tensor), or via :meth:`from_raw_blocks` from a tensor's raw GGUF
    block bytes.
    """

    qweight: mx.array
    scales: mx.array
    biases: mx.array
    qweight_type: GGMLQuantizationType

    def __post_init__(self) -> None:
        # Coerce ints to the enum (rejecting unsupported qtypes), then validate
        # the arrays, since this is built straight from GGUF file data.
        object.__setattr__(
            self, "qweight_type", self._normalize_qtype(self.qweight_type)
        )
        self._validate_contract()

    @staticmethod
    def _normalize_qtype(value: GGMLQuantizationType | int) -> GGMLQuantizationType:
        """Coerce to a ``GGMLQuantizationType`` and require a supported qtype."""
        try:
            qweight_type = GGMLQuantizationType(value)
        except ValueError as exc:
            raise ValueError(f"Unknown GGUF quantization type: {value!r}") from exc
        if qweight_type in _BITS:
            return qweight_type
        supported = ", ".join(t.name for t in _BITS)
        raise ValueError(
            f"Unsupported GGUF quantization type for the MLX-native path: "
            f"{qweight_type.name}. Supported qtypes: {supported}."
        )

    def _validate_contract(self) -> None:
        """Validate dtypes and the affine packing shapes against the contract."""
        bits = _BITS[self.qweight_type]
        if self.qweight.dtype != mx.uint32:
            raise ValueError(
                f"{self.qweight_type.name} qweight must be uint32, "
                f"got {self.qweight.dtype}"
            )
        if self.qweight_type in _FP32_SCALE_TYPES:
            expected, expected_name = mx.float32, "float32"
        else:
            expected, expected_name = mx.float16, "float16"
        for name, arr in (("scales", self.scales), ("biases", self.biases)):
            if arr.dtype != expected:
                raise ValueError(
                    f"{self.qweight_type.name} {name} must be {expected_name}, "
                    f"got {arr.dtype}"
                )
        if self.qweight.ndim != 2 or self.scales.ndim != 2 or self.biases.ndim != 2:
            raise ValueError(
                f"{self.qweight_type.name} qweight/scales/biases must be 2-D, got "
                f"{self.qweight.shape}, {self.scales.shape}, {self.biases.shape}"
            )
        if self.scales.shape != self.biases.shape:
            raise ValueError(
                f"{self.qweight_type.name} scales {self.scales.shape} and biases "
                f"{self.biases.shape} must have the same shape"
            )

        out_features, packed_in = self.qweight.shape
        scale_rows, num_groups = self.scales.shape
        if scale_rows != out_features:
            raise ValueError(
                f"{self.qweight_type.name} scales rows {scale_rows} must match "
                f"qweight rows {out_features}"
            )
        # affine packing: a group of 32 weights is 32 * bits bits, i.e. `bits`
        # uint32 words (32 bits each), so the packed inner dim is num_groups * bits.
        if packed_in != num_groups * bits:
            raise ValueError(
                f"{self.qweight_type.name} packed inner dim {packed_in} is "
                f"inconsistent with {num_groups} groups at {bits} bits "
                f"(expected {num_groups * bits})"
            )
        if num_groups < 1:
            raise ValueError(
                f"{self.qweight_type.name} must have at least one group, "
                f"got scales shape {self.scales.shape}"
            )

    @classmethod
    def from_raw_blocks(
        cls,
        block_data: np.ndarray,
        logical_shape: tuple[int, int],
        qweight_type: GGMLQuantizationType,
    ) -> GGUFMLXQuantizedTensor:
        """Repack a tensor's raw GGUF block bytes into the affine triple.

        ``block_data`` is the tensor's byte payload as stored in the file
        (``uint8``, ``gguf.GGUFReader`` order); ``logical_shape`` is
        ``(out_features, in_features)``. The raw source is validated before
        any transform, so a malformed payload fails with a source-aware error
        instead of being reshaped into the expected layout.
        """
        qweight_type = cls._normalize_qtype(qweight_type)
        if len(logical_shape) != 2 or any(dim < 1 for dim in logical_shape):
            raise ValueError(
                f"{qweight_type.name} logical_shape must be 2-D and positive, "
                f"got {logical_shape}"
            )
        out_features, in_features = logical_shape
        block_elems, block_bytes = GGML_QUANT_SIZES[qweight_type]
        if in_features % block_elems:
            raise ValueError(
                f"{qweight_type.name} in_features {in_features} is not a "
                f"multiple of the {block_elems}-element superblock"
            )
        if block_data.dtype != np.uint8:
            raise ValueError(
                f"{qweight_type.name} raw block data must be uint8, "
                f"got {block_data.dtype}"
            )
        expected_bytes = out_features * (in_features // block_elems) * block_bytes
        if block_data.size != expected_bytes:
            raise ValueError(
                f"{qweight_type.name} raw block data has {block_data.size} "
                f"bytes; logical shape {logical_shape} needs {expected_bytes}"
            )

        parse = {
            GGMLQuantizationType.Q8_0: cls._parse_q8_0,
            GGMLQuantizationType.Q4_0: cls._parse_q4_0,
            GGMLQuantizationType.Q4_1: cls._parse_q4_1,
            GGMLQuantizationType.Q4_K: cls._parse_q4_k,
            GGMLQuantizationType.Q5_K: cls._parse_q5_k,
        }[qweight_type]
        bits = _BITS[qweight_type]
        num_groups = in_features // _GROUP_SIZE
        scale_dtype = mx.float32 if qweight_type in _FP32_SCALE_TYPES else mx.float16
        rows = np.ascontiguousarray(block_data).reshape(out_features, -1)
        qweight = mx.zeros((out_features, num_groups * bits), mx.uint32)
        scales = mx.zeros((out_features, num_groups), scale_dtype)
        biases = mx.zeros((out_features, num_groups), scale_dtype)
        chunk_rows = max(1, _REPACK_CHUNK_ELEMENTS // in_features)
        for start in range(0, out_features, chunk_rows):
            stop = min(start + chunk_rows, out_features)
            codes, chunk_scales, chunk_biases = parse(
                mx.array(rows[start:stop]).reshape(-1, block_bytes)
            )
            qweight[start:stop] = cls._pack_codes_le(
                codes.reshape(stop - start, in_features), bits
            )
            scales[start:stop] = chunk_scales.reshape(stop - start, num_groups)
            biases[start:stop] = chunk_biases.reshape(stop - start, num_groups)
            # Evaluating per chunk writes it in place and frees its temporaries.
            mx.eval(qweight, scales, biases)
        return cls(
            qweight=qweight,
            scales=scales,
            biases=biases,
            qweight_type=qweight_type,
        )

    @staticmethod
    def _parse_q8_0(blocks: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Split 34-byte Q8_0 blocks (fp16 ``d``, 32 int8 ``q``; ``w = d*q``).

        Flipping the sign bit turns each int8 into the unsigned code ``q+128``,
        so ``scale = d`` and ``bias = -128*d``.
        """
        d = blocks[:, 0:2].view(mx.float16)
        codes = blocks[:, 2:34] ^ 0x80
        return codes, d, d * -128

    @staticmethod
    def _parse_q4_0(blocks: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Split 18-byte Q4_0 blocks (fp16 ``d``, 16 nibble bytes).

        Byte ``j`` holds element ``j`` in its low nibble and element ``j+16``
        in its high nibble; ``w = d*(q-8)``, so ``scale = d`` and ``bias = -8*d``.
        """
        d = blocks[:, 0:2].view(mx.float16)
        nibbles = blocks[:, 2:18]
        codes = mx.concatenate([nibbles & 0x0F, nibbles >> 4], axis=1)
        return codes, d, d * -8

    @staticmethod
    def _parse_q4_1(blocks: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Split 20-byte Q4_1 blocks (fp16 ``d``, fp16 ``m``, 16 nibble bytes).

        The nibble order matches Q4_0 and ``w = d*q + m``, so ``scale = d`` and
        ``bias = m``.
        """
        d = blocks[:, 0:2].view(mx.float16)
        m = blocks[:, 2:4].view(mx.float16)
        nibbles = blocks[:, 4:20]
        codes = mx.concatenate([nibbles & 0x0F, nibbles >> 4], axis=1)
        return codes, d, m

    @staticmethod
    def _parse_q4_k(blocks: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Split Q4_K superblocks into 4-bit codes and fp32 group-32 affine.

        Each 144-byte superblock holds 256 weights: fp16 ``d`` and ``dmin``,
        twelve bytes packing eight 6-bit sub-scales/mins, then the 128 nibble
        bytes. A weight decodes as ``d*sc*q - dmin*m``, so per group
        ``scale = d*sc`` and ``bias = -dmin*m``.
        """
        scales, biases = GGUFMLXQuantizedTensor._parse_kquant_scale_mins(blocks)
        codes = GGUFMLXQuantizedTensor._split_nibble_chunks(blocks[:, 16:144])
        return codes, scales, biases

    @staticmethod
    def _parse_q5_k(blocks: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Split Q5_K superblocks into 5-bit codes and fp32 group-32 affine.

        A 176-byte superblock is the Q4_K header (``d``, ``dmin``, the 6-bit
        sub-scales/mins) plus 32 high-bit bytes ``qh`` and the same 128
        nibble bytes: element ``e`` of group ``g`` takes bit ``g`` of
        ``qh[e]`` as its fifth bit.
        """
        scales, biases = GGUFMLXQuantizedTensor._parse_kquant_scale_mins(blocks)
        low = GGUFMLXQuantizedTensor._split_nibble_chunks(blocks[:, 48:176])
        qh = blocks[:, 16:48]
        group_bits = mx.arange(8, dtype=mx.uint8)[None, :, None]
        high = ((qh[:, None, :] >> group_bits) & 1) << 4
        return low | high, scales, biases

    @staticmethod
    def _parse_kquant_scale_mins(blocks: mx.array) -> tuple[mx.array, mx.array]:
        """Decode the shared K-quant header into fp32 group scales/biases."""
        d = blocks[:, 0:2].view(mx.float16).astype(mx.float32)
        dmin = blocks[:, 2:4].view(mx.float16).astype(mx.float32)
        packed = blocks[:, 4:16]
        sub_scales = mx.concatenate(
            [
                packed[:, 0:4] & 0x3F,
                (packed[:, 8:12] & 0x0F) | ((packed[:, 0:4] >> 6) << 4),
            ],
            axis=1,
        ).astype(mx.float32)
        sub_mins = mx.concatenate(
            [
                packed[:, 4:8] & 0x3F,
                (packed[:, 8:12] >> 4) | ((packed[:, 4:8] >> 6) << 4),
            ],
            axis=1,
        ).astype(mx.float32)
        return d * sub_scales, -(dmin * sub_mins)

    @staticmethod
    def _split_nibble_chunks(nibble_bytes: mx.array) -> mx.array:
        """Split 128 nibble bytes into (n, 8, 32) codes; chunk ``j`` of 32
        bytes carries group ``2j`` low and group ``2j+1`` high."""
        n_blocks = nibble_bytes.shape[0]
        nibbles = nibble_bytes.reshape(n_blocks, 4, 1, 32)
        halves = mx.concatenate([nibbles & 0x0F, nibbles >> 4], axis=2)
        return halves.reshape(n_blocks, 8, 32)

    @staticmethod
    def _pack_codes_le(codes: mx.array, bits: int) -> mx.array:
        """Pack per-row integer codes into MLX's packed ``uint32`` layout.

        Element ``i`` occupies flat bits ``[i*bits, (i+1)*bits)``,
        little-endian. For 8- and 4-bit codes that layout is just the packed
        bytes read as ``uint32``; otherwise 32 codes fill exactly ``bits``
        words and codes straddle word boundaries.
        """
        rows = codes.shape[0]
        if bits == 8:
            return codes.view(mx.uint32)
        if bits == 4:
            return (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(mx.uint32)
        # uint32 shifts drop the overflowing high bits, which are OR'd into the
        # next word separately.
        grouped = codes.reshape(rows, -1, 32).astype(mx.uint32)
        words = [mx.zeros(grouped.shape[:2], mx.uint32) for _ in range(bits)]
        for position in range(32):
            word, shift = divmod(position * bits, 32)
            code = grouped[:, :, position]
            words[word] = words[word] | (code << shift)
            if shift + bits > 32:
                words[word + 1] = words[word + 1] | (code >> (32 - shift))
        return mx.stack(words, axis=-1).reshape(rows, -1)

    @property
    def bits(self) -> int:
        return _BITS[self.qweight_type]

    @property
    def group_size(self) -> int:
        return _GROUP_SIZE

    @property
    def out_features(self) -> int:
        return self.qweight.shape[0]

    @property
    def in_features(self) -> int:
        return self.scales.shape[1] * _GROUP_SIZE

    @property
    def logical_shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features)

    @property
    def packed_shape(self) -> tuple[int, ...]:
        return tuple(self.qweight.shape)

    def matmul(self, x: mx.array) -> mx.array:
        """Compute ``x @ dequantize(self).T`` without materializing the weight.

        ``x`` may be float16, bfloat16, or float32, and its last dim must be
        :attr:`in_features`; any leading shape is preserved. The result has the
        same dtype as ``x`` (``mx.quantized_matmul`` may promote ``x`` against
        the stored scales to float32, so it is cast back).
        """
        out = mx.quantized_matmul(
            x,
            self.qweight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=_GROUP_SIZE,
            bits=self.bits,
            mode=_QUANT_MODE,
        )
        return out.astype(x.dtype)

    def embedding(self, ids: mx.array, output_dtype: mx.Dtype) -> mx.array:
        """Gather and dequantize embedding rows for ``ids``.

        Only the selected rows are dequantized; the table stays quantized. ``ids``
        of any rank >= 1 are accepted and a trailing :attr:`in_features` axis is
        appended. Ids must be in range; out-of-range ids gather garbage rows (MLX
        gather has no bounds check), so callers own that validation.
        """
        rows = mx.dequantize(
            self.qweight[ids],
            self.scales[ids],
            self.biases[ids],
            group_size=_GROUP_SIZE,
            bits=self.bits,
            mode=_QUANT_MODE,
        )
        return rows.astype(output_dtype)

    def permute_rows(self, index: mx.array) -> GGUFMLXQuantizedTensor:
        """Return a copy with output rows reordered by ``index`` (load-time).

        ``index`` must be a permutation of ``range(out_features)``. The packed
        ``qweight`` and its affine ``scales``/``biases`` all carry output rows
        on axis 0 and quantize along the (untouched) input axis, so gathering
        the three together stays bit-exact to a dequantize-permute-requantize
        and keeps the tensor MLX-native quantized. Used to undo llama.cpp's
        RoPE weight layout on q/k projections at load time.
        """
        out_features = self.out_features
        if index.ndim != 1 or index.shape[0] != out_features:
            raise ValueError(
                f"permute_rows index must be 1-D of length {out_features} "
                f"(out_features), got shape {tuple(index.shape)}"
            )
        ordered = mx.sort(index)
        if not bool(mx.all(ordered == mx.arange(out_features)).item()):
            raise ValueError(
                "permute_rows index must be a permutation of "
                f"range({out_features}); got duplicate or out-of-range values"
            )
        return GGUFMLXQuantizedTensor(
            qweight=self.qweight[index],
            scales=self.scales[index],
            biases=self.biases[index],
            qweight_type=self.qweight_type,
        )

    def eval_arrays(self) -> None:
        """Materialize the packed arrays backing this quantized tensor."""
        mx.eval(self.qweight, self.scales, self.biases)
