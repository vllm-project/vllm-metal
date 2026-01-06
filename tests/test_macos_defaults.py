# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import vllm_metal as vm


def test_apply_macos_defaults_sets_spawn(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr(vm, "_is_macos", lambda: True)

    vm._apply_macos_defaults()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_apply_macos_defaults_respects_user_value(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    monkeypatch.setattr(vm, "_is_macos", lambda: True)

    vm._apply_macos_defaults()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"
