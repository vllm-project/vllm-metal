# SPDX-License-Identifier: Apache-2.0
"""MLX-native quantized GGUF tensors that compute with their packed weights.

MLX's GGUF loader (``mx.load``) repacks Q8_0 and Q4_0 weights into the affine,
group-32 representation that ``mx.quantized_matmul`` already consumes: a
``uint32`` packed ``qweight`` alongside ``float16`` ``scales`` and ``biases``.
``GGUFMLXQuantizedTensor`` wraps that triple in an explicit, validated contract
and exposes :meth:`~GGUFMLXQuantizedTensor.matmul` /
:meth:`~GGUFMLXQuantizedTensor.embedding`, which run on the packed weights so
supported weights never get expanded into a dense copy.

Q4_1 is deliberately not supported here. MLX's repack mis-decodes Q4_1 blocks
(it skips a 2-byte block header where Q4_1 uses 4), so ``mx.load`` returns
corrupted Q4_1 weights. Fix tracked upstream at ml-explore/mlx#3664; Q4_1 can
join once a fixed MLX release is in our supported range. K-quants and other
qtypes MLX does not repack natively are out of scope for this path.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

try:
    import gguf
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "GGUF support requires the optional 'gguf' dependency. "
        "Install it with: pip install 'vllm-metal[gguf]'"
    ) from exc

GGMLQuantizationType = gguf.GGMLQuantizationType

# qtypes MLX repacks into its affine representation (see module docstring for
# why Q4_1 is excluded despite being a 4-bit standard qtype).
_BITS: dict[GGMLQuantizationType, int] = {
    GGMLQuantizationType.Q8_0: 8,
    GGMLQuantizationType.Q4_0: 4,
}
MLX_NATIVE_GGUF_TYPES = frozenset(_BITS)

# MLX's GGUF repack always groups these qtypes by 32 in the affine quant mode.
_GROUP_SIZE = 32
_QUANT_MODE = "affine"

_WEIGHT_SUFFIX = ".weight"


@dataclass(frozen=True, eq=False)
class GGUFMLXQuantizedTensor:
    """An MLX-native quantized GGUF weight that computes with its packed data.

    The contract consumers can rely on:

    * ``qweight`` — ``uint32`` packed weights, 2-D, shape :attr:`packed_shape`.
    * ``scales`` / ``biases`` — ``float16``, shape
      ``(out_features, in_features // group_size)``.
    * ``qweight_type`` — a ``gguf.GGMLQuantizationType`` in
      :data:`MLX_NATIVE_GGUF_TYPES`.
    * logical weight is ``(out_features, in_features)``; :attr:`group_size` is 32;
      :attr:`bits` is 8 for Q8_0, 4 for Q4_0.
    * activations: :meth:`matmul` accepts float16/bfloat16/float32 ``x`` and
      returns ``x``'s dtype; :meth:`embedding` returns an explicit ``output_dtype``.

    Construct directly when you already hold the affine arrays (e.g. a row slice
    of another tensor), or via :meth:`from_mx_load` from an ``mx.load`` result.
    """

    qweight: mx.array
    scales: mx.array
    biases: mx.array
    qweight_type: GGMLQuantizationType

    def __post_init__(self) -> None:
        # Coerce ints to the enum (rejecting non-native qtypes), then validate the
        # arrays, since this is built straight from mx.load() / GGUF file data.
        object.__setattr__(
            self, "qweight_type", self._normalize_qtype(self.qweight_type)
        )
        self._validate_contract()

    @staticmethod
    def _normalize_qtype(value: GGMLQuantizationType | int) -> GGMLQuantizationType:
        """Coerce to a ``GGMLQuantizationType`` and require it to be MLX-native."""
        try:
            qweight_type = GGMLQuantizationType(value)
        except ValueError as exc:
            raise ValueError(f"Unknown GGUF quantization type: {value!r}") from exc
        if qweight_type in MLX_NATIVE_GGUF_TYPES:
            return qweight_type
        supported = ", ".join(t.name for t in _BITS)
        if qweight_type == GGMLQuantizationType.Q4_1:
            # Remove this special case once a fixed MLX release (ml-explore/mlx#3664)
            # is our lower bound and Q4_1 can be added to _BITS.
            raise ValueError(
                "GGUF Q4_1 is not supported on the MLX-native path: MLX's GGUF "
                "repack mis-decodes Q4_1 blocks and returns corrupted weights "
                "(ml-explore/mlx#3664). "
                f"Supported MLX-native qtypes: {supported}."
            )
        raise ValueError(
            f"Unsupported GGUF quantization type for the MLX-native path: "
            f"{qweight_type.name}. Supported MLX-native qtypes: {supported}."
        )

    def _validate_contract(self) -> None:
        """Validate dtypes and the affine packing shapes against the contract."""
        bits = _BITS[self.qweight_type]
        if self.qweight.dtype != mx.uint32:
            raise ValueError(
                f"{self.qweight_type.name} qweight must be uint32, "
                f"got {self.qweight.dtype}"
            )
        for name, arr in (("scales", self.scales), ("biases", self.biases)):
            if arr.dtype != mx.float16:
                raise ValueError(
                    f"{self.qweight_type.name} {name} must be float16, got {arr.dtype}"
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
    def from_mx_load(
        cls,
        arrays: dict[str, mx.array],
        weight_name: str,
        qweight_type: GGMLQuantizationType,
    ) -> GGUFMLXQuantizedTensor:
        """Build from an ``mx.load`` result.

        ``weight_name`` is the original GGUF tensor name (``....weight``); MLX
        stores the companion scales/biases under the same prefix. ``qweight_type``
        is supplied by the caller because ``mx.load`` does not expose per-tensor
        quant types — a loader reads it from ``gguf.GGUFReader``.
        """
        qweight_type = cls._normalize_qtype(qweight_type)
        if not weight_name.endswith(_WEIGHT_SUFFIX):
            raise ValueError(
                f"GGUF weight name must end with '{_WEIGHT_SUFFIX}', got "
                f"{weight_name!r}"
            )
        prefix = weight_name[: -len(_WEIGHT_SUFFIX)]
        scales_name = f"{prefix}.scales"
        biases_name = f"{prefix}.biases"
        missing = [
            n for n in (weight_name, scales_name, biases_name) if n not in arrays
        ]
        if missing:
            raise ValueError(
                f"{qweight_type.name} tensor {weight_name!r} is missing MLX repack "
                f"arrays {missing}; MLX did not natively repack this tensor with the "
                f"current MLX version, so this qtype is not supported by the "
                f"MLX-native GGUF path"
            )
        return cls(
            qweight=arrays[weight_name],
            scales=arrays[scales_name],
            biases=arrays[biases_name],
            qweight_type=qweight_type,
        )

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
        same dtype as ``x`` (``mx.quantized_matmul`` promotes a bfloat16 ``x``
        against the float16 scales to float32, so it is cast back).
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

    def eval_arrays(self) -> None:
        """Materialize the packed arrays backing this quantized tensor."""
        mx.eval(self.qweight, self.scales, self.biases)
