# SPDX-License-Identifier: Apache-2.0
"""Raw GGUF quantized tensor representation.

This module intentionally does not load GGUF files. It only defines the
in-memory contract consumed by native Metal primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

GGUF_QTYPE_Q8_0 = 8


@dataclass(frozen=True)
class GGUFQTypeSpec:
    """Layout for one GGUF quantized block type."""

    qtype: int
    name: str
    block_size: int
    type_size: int


_QTYPE_SPECS = {
    GGUF_QTYPE_Q8_0: GGUFQTypeSpec(
        qtype=GGUF_QTYPE_Q8_0,
        name="Q8_0",
        block_size=32,
        type_size=34,
    ),
}


def qtype_spec(qweight_type: int) -> GGUFQTypeSpec:
    """Return the supported GGUF qtype spec for *qweight_type*."""
    qtype = int(qweight_type)
    try:
        return _QTYPE_SPECS[qtype]
    except KeyError as exc:
        supported = ", ".join(spec.name for spec in _QTYPE_SPECS.values())
        raise ValueError(
            f"Unsupported native GGUF qtype id: {qtype}. Supported qtypes: {supported}"
        ) from exc


@dataclass(frozen=True)
class GGUFQuantizedTensor:
    """Raw GGUF qweight plus explicit qtype metadata.

    Contract for Q8_0 in this PR:
    - ``qweight`` is raw GGUF block bytes with shape
      ``(output_dims, blocks_per_row * 34)`` and dtype ``uint8``.
    - Raw block bytes are interpreted as contiguous row-major storage.
    - ``qweight_type`` is an explicit integer tag, currently only Q8_0.
    - ``logical_shape`` is ``(output_dims, input_dims)``.
    - Matmul computes ``x @ weight.T``; the raw weight rows are output rows.
    """

    qweight: mx.array
    qweight_type: int
    logical_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.qweight.dtype != mx.uint8:
            raise ValueError(
                f"GGUF qweight must be raw uint8 block bytes, got {self.qweight.dtype}"
            )
        if len(self.qweight.shape) != 2:
            raise ValueError(
                f"GGUF qweight must be 2D [output_dims, raw_bytes], "
                f"got {self.qweight.shape}"
            )

        spec = qtype_spec(self.qweight_type)
        output_dims, input_dims = _validate_logical_shape(self.logical_shape)
        if input_dims % spec.block_size != 0:
            raise ValueError(
                f"{spec.name} input dimension must be divisible by "
                f"{spec.block_size}, got {input_dims}"
            )

        expected_raw_bytes = (input_dims // spec.block_size) * spec.type_size
        if self.qweight.shape != (output_dims, expected_raw_bytes):
            raise ValueError(
                f"{spec.name} raw qweight shape {self.qweight.shape} does not "
                f"match logical shape {self.logical_shape}; expected "
                f"({output_dims}, {expected_raw_bytes})"
            )

    @property
    def qtype_id(self) -> int:
        return qtype_spec(self.qweight_type).qtype

    @property
    def qtype_name(self) -> str:
        return qtype_spec(self.qweight_type).name

    @property
    def output_dims(self) -> int:
        return self.logical_shape[0]

    @property
    def input_dims(self) -> int:
        return self.logical_shape[1]

    @property
    def raw_block_shape(self) -> tuple[int, int, int]:
        spec = qtype_spec(self.qweight_type)
        return (
            self.output_dims,
            self.input_dims // spec.block_size,
            spec.type_size,
        )

    @property
    def raw_bytes_per_row(self) -> int:
        return self.qweight.shape[1]

    @property
    def matmul_transpose(self) -> bool:
        """Whether matmul uses the logical weight transposed."""
        return True


def _validate_logical_shape(shape: tuple[int, int] | Any) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"GGUF logical shape must be 2D, got {shape}")
    output_dims = int(shape[0])
    input_dims = int(shape[1])
    if output_dims <= 0 or input_dims <= 0:
        raise ValueError(f"GGUF logical shape must be positive, got {shape}")
    return output_dims, input_dims
