# SPDX-License-Identifier: Apache-2.0
"""Parsing and the startup check of the vllm-metal environment variables."""

from __future__ import annotations

import re

import pytest

import vllm_metal.envs as envs


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "VLLM_METAL_SPEC_INGEST_CHUNK",
            "1k",
            "VLLM_METAL_SPEC_INGEST_CHUNK must be an integer, got '1k'",
        ),
        (
            "VLLM_METAL_SPEC_INGEST_CHUNK",
            "-1",
            "VLLM_METAL_SPEC_INGEST_CHUNK must be at least 0 "
            "(0 means single-forward ingest), got '-1'",
        ),
        (
            "VLLM_METAL_RING_BASE_PORT",
            "port",
            "VLLM_METAL_RING_BASE_PORT must be an integer, got 'port'",
        ),
        (
            "VLLM_METAL_RING_BASE_PORT",
            "1023",
            "VLLM_METAL_RING_BASE_PORT must be in [1024, 65535] "
            "(the user-port range), got '1023'",
        ),
        (
            "VLLM_METAL_RING_BASE_PORT",
            "65536",
            "VLLM_METAL_RING_BASE_PORT must be in [1024, 65535] "
            "(the user-port range), got '65536'",
        ),
        (
            "VLLM_METAL_MM_PREFIX_PATH",
            "kernle",
            "VLLM_METAL_MM_PREFIX_PATH must be one of kernel, recompute, got 'kernle'",
        ),
        (
            "VLLM_METAL_MULTIMODAL_MODE",
            "vlm",
            "VLLM_METAL_MULTIMODAL_MODE must be one of auto, multimodal-native, "
            "text-only, got 'vlm'",
        ),
        (
            "VLLM_MLX_DEVICE",
            "GPU",
            "VLLM_MLX_DEVICE must be one of gpu, cpu, got 'GPU'",
        ),
        (
            "VLLM_METAL_TQ_PREFILL",
            "AUTO",
            "VLLM_METAL_TQ_PREFILL must be one of auto, 0, 1, got 'AUTO'",
        ),
        (
            "VLLM_METAL_TQ_PREFILL_MAX_MIB",
            "64MiB",
            "VLLM_METAL_TQ_PREFILL_MAX_MIB must be auto or a nonnegative integer "
            "in MiB, got '64MiB'",
        ),
        (
            "VLLM_METAL_TQ_PREFILL_MAX_MIB",
            "-1",
            "VLLM_METAL_TQ_PREFILL_MAX_MIB must be auto or a nonnegative integer "
            "in MiB, got '-1'",
        ),
    ],
)
def test_a_bad_value_names_the_variable_and_the_value(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=re.escape(message)):
        getattr(envs, name)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("VLLM_METAL_SPEC_INGEST_CHUNK", "0", 0),
        ("VLLM_METAL_SPEC_INGEST_CHUNK", "16", 16),
        ("VLLM_METAL_RING_BASE_PORT", "40000", 40000),
        ("VLLM_METAL_RING_BASE_PORT", "1024", 1024),
        ("VLLM_METAL_RING_BASE_PORT", "65535", 65535),
        ("VLLM_METAL_MM_PREFIX_PATH", "recompute", "recompute"),
        ("VLLM_METAL_MULTIMODAL_MODE", "text-only", "text-only"),
        ("VLLM_MLX_DEVICE", "cpu", "cpu"),
        ("VLLM_METAL_TQ_PREFILL", "1", "1"),
        ("VLLM_METAL_TQ_PREFILL_MAX_MIB", "auto", "auto"),
        ("VLLM_METAL_TQ_PREFILL_MAX_MIB", "0", 0),
        ("VLLM_METAL_TQ_PREFILL_MAX_MIB", "256", 256),
        # Boolean switches: "1" on, anything else off (documented contract).
        ("VLLM_METAL_DECODE_PIPELINE", "1", True),
        ("VLLM_METAL_DECODE_PIPELINE", "0", False),
        ("VLLM_METAL_MLA_KERNEL", "1", True),
        ("VLLM_METAL_NATIVE_SAMPLING", "0", False),
        ("VLLM_METAL_NAX_GQA_PREFILL", "1", True),
        ("VLLM_METAL_NAX_GQA_PREFILL", "0", False),
        ("VLLM_METAL_NAX_GQA_PREFILL", "yes", False),
        ("VLLM_METAL_BUILD_FROM_SOURCE", "yes", False),
    ],
)
def test_a_good_value_parses(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, expected: object
) -> None:
    monkeypatch.setenv(name, value)

    assert getattr(envs, name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("VLLM_METAL_GDN_LAZY_KERNELS", True),
        ("VLLM_METAL_DECODE_PIPELINE", True),
        ("VLLM_METAL_COMPILED_MLP", False),
        ("VLLM_METAL_NATIVE_SAMPLING", True),
        ("VLLM_METAL_MLA_KERNEL", False),
        ("VLLM_METAL_DISABLE_NAX", False),
        ("VLLM_METAL_NAX_GQA_PREFILL", True),
        ("VLLM_METAL_SPEC_VERIFY_WINDOW", False),
        ("VLLM_METAL_BUILD_FROM_SOURCE", False),
    ],
)
def test_boolean_defaults(
    monkeypatch: pytest.MonkeyPatch, name: str, expected: bool
) -> None:
    monkeypatch.delenv(name, raising=False)

    assert getattr(envs, name) is expected


def test_validate_environment_accepts_the_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in envs.environment_variables:
        monkeypatch.delenv(name, raising=False)

    envs.validate_environment()


def test_validate_environment_reports_every_bad_value_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_METAL_SPEC_INGEST_CHUNK", "1k")
    monkeypatch.setenv("VLLM_MLX_DEVICE", "GPU")

    with pytest.raises(ValueError) as excinfo:
        envs.validate_environment()

    message = str(excinfo.value)
    assert "VLLM_METAL_SPEC_INGEST_CHUNK must be an integer, got '1k'" in message
    assert "VLLM_MLX_DEVICE must be one of gpu, cpu, got 'GPU'" in message
