# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import vllm.envs

import vllm_metal as vm
from vllm_metal.envs import environment_variables as metal_env_vars


def test_register_merges_metal_env_vars_into_vllm() -> None:
    vm._register()

    missing = [k for k in metal_env_vars if k not in vllm.envs.environment_variables]
    assert not missing, f"metal env vars not registered with vllm: {missing}"


def test_unset_model_backend_retains_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_metal import compat
    from vllm_metal.platform import MetalPlatform

    applied = 0

    def apply_compat_patches() -> None:
        nonlocal applied
        applied += 1

    monkeypatch.delenv("VLLM_METAL_MODEL_BACKEND", raising=False)
    monkeypatch.setattr(compat, "apply_compat_patches", apply_compat_patches)
    monkeypatch.setattr(MetalPlatform, "is_available", classmethod(lambda cls: True))

    assert vm._register() == "vllm_metal.platform.MetalPlatform"
    assert applied == 1


def test_torch_backend_skips_mlx_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_metal import compat
    from vllm_metal.pytorch_backend.platform import TorchPlatform

    applied = 0

    def apply_compat_patches() -> None:
        nonlocal applied
        applied += 1

    monkeypatch.setenv("VLLM_METAL_MODEL_BACKEND", "torch")
    monkeypatch.setattr(compat, "apply_compat_patches", apply_compat_patches)
    monkeypatch.setattr(TorchPlatform, "is_available", classmethod(lambda cls: True))

    assert vm._register() == "vllm_metal.pytorch_backend.platform.TorchPlatform"
    assert applied == 0


def test_invalid_model_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.platforms import resolve_current_platform_cls_qualname
    from vllm.utils.import_utils import resolve_obj_by_qualname

    monkeypatch.setenv("VLLM_METAL_MODEL_BACKEND", "cuda")
    monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)

    platform_cls = resolve_obj_by_qualname(resolve_current_platform_cls_qualname())
    with pytest.raises(ValueError, match="must be 'mlx' or 'torch'"):
        platform_cls()


def test_unavailable_torch_backend_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.platforms import resolve_current_platform_cls_qualname
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from vllm_metal.pytorch_backend.platform import TorchPlatform

    monkeypatch.setenv("VLLM_METAL_MODEL_BACKEND", "torch")
    monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)
    monkeypatch.setattr(TorchPlatform, "is_available", classmethod(lambda cls: False))

    platform_cls = resolve_obj_by_qualname(resolve_current_platform_cls_qualname())
    assert platform_cls is TorchPlatform
    with pytest.raises(RuntimeError, match="requires PyTorch MPS"):
        platform_cls()
