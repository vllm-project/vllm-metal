"""Tests for the single-process loopback rendezvous patch (vllm-metal#625)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_metal.compat import (
    _install_uniproc_loopback_rendezvous,
    ensure_vllm_uniproc_loopback_rendezvous_patch,
)

uniproc_executor = pytest.importorskip("vllm.v1.executor.uniproc_executor")


ORIGINAL_URI = "tcp://192.168.12.165:65334"


def _make_executor(
    cls: type,
    *,
    world_size: int = 1,
    backend: str | None = "uni",
) -> Any:
    """A stand-in carrying only the attributes the patch inspects."""
    obj = object.__new__(cls)
    obj.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=world_size,
            distributed_executor_backend=backend,
        )
    )
    return obj


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the patch over a stub ``_distributed_args`` returning a routable IP."""
    monkeypatch.setattr(
        uniproc_executor.UniProcExecutor,
        "_distributed_args",
        lambda self: (ORIGINAL_URI, 0, 0),
        raising=False,
    )
    monkeypatch.setattr(
        uniproc_executor, "_vllm_metal_uniproc_loopback_patched", False, raising=False
    )
    monkeypatch.delenv("VLLM_HOST_IP", raising=False)
    _install_uniproc_loopback_rendezvous(uniproc_executor)


def _call(obj: Any) -> tuple[str, int, int]:
    return uniproc_executor.UniProcExecutor._distributed_args(obj)


def test_rewrites_to_loopback_preserving_port(patched: None) -> None:
    method, rank, local_rank = _call(_make_executor(uniproc_executor.UniProcExecutor))
    assert method == "tcp://127.0.0.1:65334"
    assert (rank, local_rank) == (0, 0)


def test_explicit_vllm_host_ip_is_preserved(
    patched: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.5")
    method, _, _ = _call(_make_executor(uniproc_executor.UniProcExecutor))
    assert method == ORIGINAL_URI


@pytest.mark.parametrize(
    ("world_size", "backend"),
    [(2, "uni"), (1, "ray"), (1, "mp"), (1, "external_launcher")],
)
def test_untouched_outside_single_process_uni(
    patched: None, world_size: int, backend: str
) -> None:
    obj = _make_executor(
        uniproc_executor.UniProcExecutor, world_size=world_size, backend=backend
    )
    assert _call(obj)[0] == ORIGINAL_URI


def test_external_launcher_subclass_untouched(patched: None) -> None:
    """``ExecutorWithExternalLauncher`` must keep its own ``env://`` rendezvous."""
    obj = _make_executor(uniproc_executor.ExecutorWithExternalLauncher)
    assert _call(obj)[0] == ORIGINAL_URI


def test_patch_is_idempotent(patched: None) -> None:
    _install_uniproc_loopback_rendezvous(uniproc_executor)
    _install_uniproc_loopback_rendezvous(uniproc_executor)
    assert _call(_make_executor(uniproc_executor.UniProcExecutor))[0] == (
        "tcp://127.0.0.1:65334"
    )


def test_hook_installs_when_module_already_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the module already in sys.modules, patch it directly rather than
    installing a redundant meta_path finder."""
    import sys

    monkeypatch.setattr(
        uniproc_executor.UniProcExecutor,
        "_distributed_args",
        lambda self: (ORIGINAL_URI, 0, 0),
        raising=False,
    )
    monkeypatch.setattr(
        uniproc_executor, "_vllm_metal_uniproc_loopback_patched", False, raising=False
    )
    monkeypatch.delenv("VLLM_HOST_IP", raising=False)
    before = len(sys.meta_path)

    ensure_vllm_uniproc_loopback_rendezvous_patch()

    assert len(sys.meta_path) == before
    assert _call(_make_executor(uniproc_executor.UniProcExecutor))[0] == (
        "tcp://127.0.0.1:65334"
    )
