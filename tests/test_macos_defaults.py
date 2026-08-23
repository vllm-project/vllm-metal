# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import pytest
from vllm.config import SchedulerConfig

import vllm_metal as vm
import vllm_metal.config as vm_config
import vllm_metal.platform as mp
from vllm_metal.platform import MetalPlatform


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


def test_apply_mlx_buffer_defaults_sets_ops_limit_only(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)

    vm._apply_mlx_buffer_defaults()

    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "2000"
    assert "MLX_MAX_MB_PER_BUFFER" not in os.environ


def test_apply_mlx_buffer_defaults_respects_user_values(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("MLX_MAX_OPS_PER_BUFFER", "50")
    monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "128")

    vm._apply_mlx_buffer_defaults()

    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "50"
    assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "128"


def test_apply_mlx_buffer_defaults_preserves_user_mb_limit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("MLX_MAX_OPS_PER_BUFFER", raising=False)
    monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "128")

    vm._apply_mlx_buffer_defaults()

    assert os.environ["MLX_MAX_OPS_PER_BUFFER"] == "2000"
    assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "128"


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


class _Memory:
    def __init__(self, total_gib: float) -> None:
        self.total = int(total_gib * (1 << 30))


def _vllm_config(
    max_num_batched_tokens: int,
    *,
    gpu_memory_utilization: float = 0.9,
    backend: str = "uni",
) -> SimpleNamespace:
    # Stub sets every field the picker reads explicitly (house stub rule);
    # scheduler_config stays the real upstream object.
    return SimpleNamespace(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=8,
            max_model_len=4096,
            is_encoder_decoder=False,
        ),
        cache_config=SimpleNamespace(gpu_memory_utilization=gpu_memory_utilization),
        parallel_config=SimpleNamespace(distributed_executor_backend=backend),
    )


class TestDefaultMbPerBuffer:
    """Memory- and config-conditional MLX_MAX_MB_PER_BUFFER default."""

    @pytest.fixture(autouse=True)
    def _clean_state(self, monkeypatch):
        saved = os.environ.get("MLX_MAX_MB_PER_BUFFER")
        monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER", raising=False)
        monkeypatch.delenv("VLLM_METAL_MEMORY_FRACTION", raising=False)
        monkeypatch.setattr(MetalPlatform, "_mb_default_installed", None)
        vm_config.reset_config()
        yield
        # The production code SETS this var, which monkeypatch does not
        # track — restore the pre-test state explicitly.
        if saved is None:
            os.environ.pop("MLX_MAX_MB_PER_BUFFER", None)
        else:
            os.environ["MLX_MAX_MB_PER_BUFFER"] = saved
        vm_config.reset_config()

    def _apply(
        self,
        monkeypatch,
        total_gib: float,
        batched: int,
        *,
        gpu_memory_utilization: float = 0.9,
        backend: str = "uni",
    ) -> None:
        monkeypatch.setattr(mp.psutil, "virtual_memory", lambda: _Memory(total_gib))
        MetalPlatform._default_mb_per_buffer(
            _vllm_config(
                batched,
                gpu_memory_utilization=gpu_memory_utilization,
                backend=backend,
            )
        )

    def test_large_usable_memory_gets_full_default(self, monkeypatch) -> None:
        self._apply(monkeypatch, 128, 2048)

        assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "2000"

    def test_low_utilization_shrinks_the_tier(self, monkeypatch) -> None:
        """128 GiB at 50% utilization is 64 GiB usable -> the 512 tier."""
        self._apply(monkeypatch, 128, 2048, gpu_memory_utilization=0.5)

        assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "512"

    def test_explicit_memory_fraction_is_honored(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_MEMORY_FRACTION", "0.35")
        vm_config.reset_config()

        self._apply(monkeypatch, 128, 2048)

        assert "MLX_MAX_MB_PER_BUFFER" not in os.environ

    def test_small_memory_gets_no_default(self, monkeypatch) -> None:
        self._apply(monkeypatch, 48, 2048)

        assert "MLX_MAX_MB_PER_BUFFER" not in os.environ

    def test_large_batched_tokens_gets_no_default(self, monkeypatch) -> None:
        """Regression pin for the #585 shape: 8192 batched tokens never
        defaults, regardless of memory."""
        self._apply(monkeypatch, 128, 8192)

        assert "MLX_MAX_MB_PER_BUFFER" not in os.environ

    @pytest.mark.parametrize(
        ("batched", "expected"),
        [(4096, "2000"), (4097, None)],
    )
    def test_batched_tokens_boundary_is_inclusive(
        self, monkeypatch, batched, expected
    ) -> None:
        self._apply(monkeypatch, 128, batched)

        assert os.environ.get("MLX_MAX_MB_PER_BUFFER") == expected

    @pytest.mark.parametrize(
        ("total_gib", "utilization", "expected"),
        [
            (100, 0.9, "2000"),  # 90.0 usable — full-tier boundary
            (99, 0.9, "512"),  # 89.1 usable — just below
            (56, 0.9, "512"),  # 50.4 usable — reduced-tier boundary
            (55, 0.9, None),  # 49.5 usable — below every tier
        ],
    )
    def test_usable_memory_tier_boundaries(
        self, monkeypatch, total_gib, utilization, expected
    ) -> None:
        self._apply(monkeypatch, total_gib, 2048, gpu_memory_utilization=utilization)

        assert os.environ.get("MLX_MAX_MB_PER_BUFFER") == expected

    def test_ray_backend_gets_no_default(self, monkeypatch) -> None:
        """Ray workers do not share the driver's memory or environment."""
        self._apply(monkeypatch, 128, 2048, backend="ray")

        assert "MLX_MAX_MB_PER_BUFFER" not in os.environ

    def test_reconfiguration_removes_the_stale_default(self, monkeypatch) -> None:
        """A later engine with a disqualifying shape must not inherit the
        earlier engine's default (#585 shape via a second engine)."""
        self._apply(monkeypatch, 128, 2048)
        assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "2000"

        self._apply(monkeypatch, 128, 8192)

        assert "MLX_MAX_MB_PER_BUFFER" not in os.environ

    def test_reconfiguration_retiers_the_default(self, monkeypatch) -> None:
        self._apply(monkeypatch, 128, 2048)
        self._apply(monkeypatch, 128, 2048, gpu_memory_utilization=0.5)

        assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "512"

    def test_manual_export_wins_and_survives_reconfiguration(self, monkeypatch) -> None:
        monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "64")

        self._apply(monkeypatch, 128, 2048)
        after_qualifying = os.environ["MLX_MAX_MB_PER_BUFFER"]
        self._apply(monkeypatch, 128, 8192)
        after_disqualifying = os.environ["MLX_MAX_MB_PER_BUFFER"]

        assert after_qualifying == "64"
        assert after_disqualifying == "64"

    def test_reapplication_is_idempotent(self, monkeypatch) -> None:
        self._apply(monkeypatch, 128, 2048)
        self._apply(monkeypatch, 128, 2048)

        assert os.environ["MLX_MAX_MB_PER_BUFFER"] == "2000"
