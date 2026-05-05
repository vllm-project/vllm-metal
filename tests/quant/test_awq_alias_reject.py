# SPDX-License-Identifier: Apache-2.0
"""Alias-normalize and reject paths for `normalize_quant_config`.

Synthetic config dicts; no model load, no downloads.
"""

from __future__ import annotations

import pytest

from vllm_metal.quant.awq_config import (
    UnsupportedQuantizationConfigError,
    normalize_quant_config,
)

_CANONICAL = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "version": "gemm",
}


# ---- normalize-success paths ------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        # Canonical AWQ config (every official Qwen / Llama / Mistral release).
        {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "version": "gemm",
        },
        # AutoAWQ alias: w_bit instead of bits.
        {
            "quant_method": "awq",
            "w_bit": 4,
            "group_size": 128,
            "zero_point": True,
            "version": "gemm",
        },
        # AutoAWQ alias: q_group_size instead of group_size.
        {
            "quant_method": "awq",
            "bits": 4,
            "q_group_size": 128,
            "zero_point": True,
            "version": "gemm",
        },
        # Uppercase version field (raw AutoAWQ output).
        {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "version": "GEMM",
        },
        # All three aliases at once.
        {
            "quant_method": "awq",
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "version": "GEMM",
        },
        # GPTQ canonical (same v1 scope as AWQ).
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "version": "gemm",
        },
        # `version` defaults to "gemm" when omitted.
        {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
        },
        # `zero_point` defaults to True when omitted.
        {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "version": "gemm",
        },
    ],
    ids=[
        "canonical-awq",
        "alias-w_bit",
        "alias-q_group_size",
        "alias-uppercase-GEMM",
        "all-three-aliases",
        "canonical-gptq",
        "default-version",
        "default-zero_point",
    ],
)
def test_normalize_returns_canonical(raw):
    out = normalize_quant_config(raw)
    # Result mirrors the original quant_method (awq vs gptq) but every
    # other field is canonicalized.
    assert out["quant_method"] == raw["quant_method"]
    assert out["bits"] == 4
    assert out["group_size"] == 128
    assert out["zero_point"] is True
    assert out["version"] == "gemm"


# ---- reject paths -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "needle"),
    [
        # Wrong/missing quant_method
        ({"quant_method": "fp8", "bits": 4, "group_size": 128}, "quant_method"),
        ({"bits": 4, "group_size": 128}, "quant_method"),
        # Missing bits / group_size
        ({"quant_method": "awq", "group_size": 128}, "bits"),
        ({"quant_method": "awq", "bits": 4}, "group_size"),
        # Unsupported bits / group_size values
        ({"quant_method": "awq", "bits": 8, "group_size": 128}, "bits=8"),
        ({"quant_method": "awq", "bits": 3, "group_size": 128}, "bits=3"),
        ({"quant_method": "awq", "bits": 4, "group_size": 64}, "group_size=64"),
        # zero_point=False (symmetric quantization is not in v1 scope)
        (
            {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": False,
            },
            "zero_point",
        ),
        # version=gemv
        (
            {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "version": "gemv",
            },
            "gemv",
        ),
        # version=GEMV (uppercase still rejected after normalize)
        (
            {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "version": "GEMV",
            },
            "gemv",
        ),
        # GPTQ desc_act=true
        (
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "desc_act": True,
            },
            "desc_act",
        ),
    ],
    ids=[
        "wrong-quant_method-fp8",
        "missing-quant_method",
        "missing-bits",
        "missing-group_size",
        "bits-8",
        "bits-3",
        "group-size-64",
        "zero_point-false",
        "version-gemv",
        "version-uppercase-GEMV",
        "gptq-desc_act-true",
    ],
)
def test_reject_paths(raw, needle):
    with pytest.raises(UnsupportedQuantizationConfigError) as excinfo:
        normalize_quant_config(raw)
    # Error message names the offending field/value.
    assert needle in str(excinfo.value)


def test_reject_path_mentions_field_value():
    """Reject errors must name the offending value, not just the field."""
    with pytest.raises(UnsupportedQuantizationConfigError) as excinfo:
        normalize_quant_config({"quant_method": "awq", "bits": 8, "group_size": 128})
    msg = str(excinfo.value)
    assert "bits" in msg and "8" in msg


def test_canonical_round_trips():
    """Idempotent on already-canonical input."""
    out = normalize_quant_config(_CANONICAL)
    assert normalize_quant_config(out) == out
