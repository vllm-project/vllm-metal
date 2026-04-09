# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys

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
