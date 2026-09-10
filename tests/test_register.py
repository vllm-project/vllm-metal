# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import vllm.envs

import vllm_metal as vm
from vllm_metal.envs import environment_variables as metal_env_vars
from vllm_metal.platform import MetalPlatform


def test_register_merges_metal_env_vars_into_vllm() -> None:
    vm._register()

    missing = [k for k in metal_env_vars if k not in vllm.envs.environment_variables]
    assert not missing, f"metal env vars not registered with vllm: {missing}"


def test_register_pins_v1_model_runner_when_metal_is_selected(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(MetalPlatform, "is_available", classmethod(lambda cls: True))

    assert vm._register() == "vllm_metal.platform.MetalPlatform"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_register_leaves_model_runner_alone_when_metal_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(MetalPlatform, "is_available", classmethod(lambda cls: False))

    assert vm._register() is None
    assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ
