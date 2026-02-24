# SPDX-License-Identifier: Apache-2.0
"""Tests for kernel_loader: OS-aware revision pinning for Metal compatibility.

Verifies that:
- macOS 16+ uses the latest HF kernel (default revision)
- macOS 15 and earlier pins to the Nov 2025 compat revision (Metal 3.2)
- Both revisions actually load and expose the expected ops

Run with:
    python -m pytest tests/test_kernel_loader.py -v -s
"""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("kernels")

# ---------------------------------------------------------------------------
# Unit tests (no network, no GPU)
# ---------------------------------------------------------------------------


class TestNeedsCompatRevision:
    """Test _needs_compat_revision() with mocked macOS versions."""

    @pytest.mark.parametrize(
        "ver, expected",
        [
            ("15.7.4", True),  # macOS 15 — needs compat
            ("14.5", True),  # macOS 14 — needs compat
            ("26.3", False),  # macOS 26 — modern
            ("", False),  # empty — safe default
        ],
    )
    def test_version_check(self, ver, expected):
        from vllm_metal.metal_kernel_backend.kernel_loader import _needs_compat_revision

        with mock.patch("platform.mac_ver", return_value=(ver, ("", "", ""), "")):
            assert _needs_compat_revision() is expected


class TestGetKernelRevisionSelection:
    """Test that get_paged_attention_ops passes the right revision to get_kernel."""

    def _reset_kernel_cache(self):
        import vllm_metal.metal_kernel_backend.kernel_loader as kl

        kl._kernel = None

    def test_macos_15_uses_compat_revision(self):
        self._reset_kernel_cache()
        with (
            mock.patch("platform.mac_ver", return_value=("15.7.4", ("", "", ""), "")),
            mock.patch("kernels.get_kernel", return_value=mock.MagicMock()) as mk,
        ):
            from vllm_metal.metal_kernel_backend.kernel_loader import (
                _MACOS15_COMPAT_REVISION,
                get_paged_attention_ops,
            )

            get_paged_attention_ops()
            mk.assert_called_once_with(
                "kernels-community/paged-attention",
                revision=_MACOS15_COMPAT_REVISION,
            )
        self._reset_kernel_cache()

    def test_macos_26_uses_latest(self):
        self._reset_kernel_cache()
        with (
            mock.patch("platform.mac_ver", return_value=("26.3", ("", "", ""), "")),
            mock.patch("kernels.get_kernel", return_value=mock.MagicMock()) as mk,
        ):
            from vllm_metal.metal_kernel_backend.kernel_loader import (
                get_paged_attention_ops,
            )

            get_paged_attention_ops()
            mk.assert_called_once_with(
                "kernels-community/paged-attention",
                revision=None,
            )
        self._reset_kernel_cache()


# ---------------------------------------------------------------------------
# Integration tests (require network + MPS)
# ---------------------------------------------------------------------------


def _mps_available() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
class TestKernelLoadsForReal:
    """Actually load the kernel from HuggingFace and verify ops exist."""

    _EXPECTED_OPS = {"reshape_and_cache", "paged_attention_v1"}

    def test_latest_revision_loads(self):
        from kernels import get_kernel

        kernel = get_kernel("kernels-community/paged-attention")
        ops = set(dir(kernel))
        assert self._EXPECTED_OPS <= ops, f"Missing ops: {self._EXPECTED_OPS - ops}"

    def test_compat_revision_loads(self):
        from kernels import get_kernel

        from vllm_metal.metal_kernel_backend.kernel_loader import (
            _MACOS15_COMPAT_REVISION,
        )

        kernel = get_kernel(
            "kernels-community/paged-attention",
            revision=_MACOS15_COMPAT_REVISION,
        )
        ops = set(dir(kernel))
        assert self._EXPECTED_OPS <= ops, f"Missing ops: {self._EXPECTED_OPS - ops}"
