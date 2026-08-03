# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys

import pytest

import vllm_metal as vm


def test_apply_macos_defaults_sets_spawn(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    vm._apply_macos_defaults()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_apply_macos_defaults_respects_user_value(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    monkeypatch.setattr(sys, "platform", "darwin")

    vm._apply_macos_defaults()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


def test_apply_macos_defaults_noop_on_non_macos(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    vm._apply_macos_defaults()
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ


def test_apply_mlx_buffer_defaults_sets_both_limits(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)

    vm._apply_mlx_buffer_defaults()

    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "2000"
    assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "2000"


def test_apply_mlx_buffer_defaults_respects_user_values(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("MLX_MAX_OPS_PER_BUFFER", "50")
    monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "128")

    vm._apply_mlx_buffer_defaults()

    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "50"
    assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "128"


@pytest.mark.parametrize(
    ("kept", "defaulted"),
    [
        ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER"),
        ("MLX_MAX_MB_PER_BUFFER", "MLX_MAX_OPS_PER_BUFFER"),
    ],
)
def test_apply_mlx_buffer_defaults_defaults_each_var_independently(
    monkeypatch, kept: str, defaulted: str
) -> None:
    # The two limits are independent split thresholds: a user tuning one
    # keeps it, and the other still gets the plugin default.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv(kept, "50")
    monkeypatch.delenv(defaulted, raising=False)

    vm._apply_mlx_buffer_defaults()

    assert os.environ[kept] == "50"
    assert os.environ[defaulted] == "2000"


def test_apply_mlx_buffer_defaults_noop_on_non_macos(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)

    vm._apply_mlx_buffer_defaults()

    assert "MLX_MAX_OPS_PER_BUFFER" not in os.environ
    assert "MLX_MAX_MB_PER_BUFFER" not in os.environ


def test_apply_macos_defaults_logs_when_setting(monkeypatch, caplog) -> None:
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    metal_logger = logging.getLogger("vllm_metal")
    original_level = metal_logger.level
    metal_logger.addHandler(caplog.handler)
    metal_logger.setLevel(logging.DEBUG)
    try:
        vm._apply_macos_defaults()
    finally:
        metal_logger.removeHandler(caplog.handler)
        metal_logger.setLevel(original_level)

    assert "defaulting VLLM_WORKER_MULTIPROC_METHOD" in caplog.text
