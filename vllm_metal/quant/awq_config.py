# SPDX-License-Identifier: Apache-2.0
"""AWQ quantization-config normalization for vllm-metal.

mlx_lm 0.31.3+ ships an AWQ -> MLX-affine repack inside
``_transform_awq_weights``. vllm-metal's ``model_lifecycle`` calls
``mlx_lm.load`` for non-VLM checkpoints, so AWQ checkpoints already
load. This module is the *entry-point preflight*: it normalizes alias
fields the upstream transform does not know about, and rejects
configurations that vllm-metal does not validate, before any model state is
constructed.

Supported configuration:

  quant_method: "awq"
  bits        : 4
  group_size  : 128
  zero_point  : true
  version     : "gemm"   (case-insensitive on input)

Rejected:

  version: "gemv"        (different intra-int32 packing)
  bits != 4              (mlx_lm rejects too, but we want a clearer error
                          and to fail before mlx_lm.load constructs the model)
  group_size != 128
  zero_point: false      (symmetric quantization is not in v1 scope)

GPTQ checkpoints are rejected upstream of this normalizer by
``AWQQuantLoader.for_model``: GPTQ is not part of the v1 support claim
and is deferred to a follow-up PR once a real GPTQ checkpoint is
validated end-to-end.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class UnsupportedQuantizationConfigError(ValueError):
    """Raised before model load when an AWQ config is outside v1 scope.

    The message names the offending field and value so users know exactly
    what to change (re-quantize, switch checkpoint, etc.).
    """


# Aliases produced by raw AutoAWQ tooling. Modern HF releases already
# normalize these to canonical names, but third-party producers do not
# always.
_BITS_ALIASES = ("bits", "w_bit")
_GROUP_SIZE_ALIASES = ("group_size", "q_group_size")


def _pick_alias(raw: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    """Return the first present alias's value, or None if none are set."""
    for key in aliases:
        if key in raw:
            return raw[key]
    return None


def normalize_quant_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an HF ``quantization_config`` dict for v1 AWQ support.

    Returns a new dict with canonical keys (``quant_method``, ``bits``,
    ``group_size``, ``zero_point``, ``version``) suitable for handing to
    ``mlx_lm.load`` via ``model_config={"quantization_config": ...}``.

    Raises:
        UnsupportedQuantizationConfigError: if any field is outside v1 scope.
    """
    quant_method = raw.get("quant_method")
    if quant_method != "awq":
        raise UnsupportedQuantizationConfigError(
            f"quant_method={quant_method!r} is not handled by the AWQ "
            "preflight; expected 'awq'."
        )

    bits = _pick_alias(raw, _BITS_ALIASES)
    if bits is None:
        raise UnsupportedQuantizationConfigError(
            "missing 'bits' (or alias 'w_bit') in quantization_config"
        )
    if bits != 4:
        raise UnsupportedQuantizationConfigError(
            f"bits={bits!r} is not supported; v1 only supports bits=4"
        )

    group_size = _pick_alias(raw, _GROUP_SIZE_ALIASES)
    if group_size is None:
        raise UnsupportedQuantizationConfigError(
            "missing 'group_size' (or alias 'q_group_size') in quantization_config"
        )
    if group_size != 128:
        raise UnsupportedQuantizationConfigError(
            f"group_size={group_size!r} is not supported; v1 only supports "
            "group_size=128"
        )

    zero_point = raw.get("zero_point", True)
    if zero_point is not True:
        raise UnsupportedQuantizationConfigError(
            "symmetric quantization is not supported; v1 requires "
            f"zero_point=true. Got zero_point={zero_point!r}."
        )

    version_raw = raw.get("version", "gemm")
    if not isinstance(version_raw, str):
        raise UnsupportedQuantizationConfigError(
            f"version={version_raw!r} must be a string"
        )
    version = version_raw.lower()
    if version == "gemv":
        raise UnsupportedQuantizationConfigError(
            "version='gemv' is not supported; v1 only supports the 'gemm' "
            "packing variant"
        )
    if version != "gemm":
        raise UnsupportedQuantizationConfigError(
            f"version={version_raw!r} is not supported; v1 only supports "
            "'gemm' (case-insensitive)"
        )

    return {
        "quant_method": "awq",
        "bits": bits,
        "group_size": group_size,
        "zero_point": True,
        "version": "gemm",
    }
