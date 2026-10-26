"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import signal
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from tests.kv_connector_spy import SpyTransferGroup

os.environ["MLX_ENABLE_TF32"] = "0"  # Keep FP32 parity checks strict on M5.


@pytest.fixture(scope="module")
def inspect_vllm_models_in_process():
    """Opt-in, per module: skip vLLM's per-architecture registry subprocess.

    ``ModelRegistry`` inspects each architecture in a fresh interpreter (to
    keep CUDA out of the parent) and caches the result under
    ``VLLM_CACHE_ROOT``. CI starts cold, so every config-building test that
    names a new architecture pays a full vLLM import. There is no CUDA here,
    and the test process has vLLM imported already.

    This bypasses the real subprocess path, so it is not autouse and is
    module-scoped: the original ``_run_in_subprocess`` is restored when the
    requesting module finishes (``test_registry_subprocess_restored.py``
    checks this). ``test_entry_point_mounts_without_manual_register`` keeps
    the real path covered with an empty cache root.
    """

    from vllm.model_executor.models import registry

    with pytest.MonkeyPatch.context() as mp_patch:
        mp_patch.setattr(registry, "_run_in_subprocess", lambda fn: fn())
        yield


@pytest.fixture(scope="session")
def _whisper_hub_tokenizers() -> dict:
    """Session-lifetime cache behind ``whisper_hub_tokenizer_cache``."""
    return {}


@pytest.fixture
def whisper_hub_tokenizer_cache(
    _whisper_hub_tokenizers: dict, monkeypatch: pytest.MonkeyPatch
):
    """Serve hub Whisper tokenizers from a session cache of deep copies.

    ``WhisperTranscriber.load_tokenizer`` resolves a tokenizer-free snapshot
    by loading one of three ``openai/whisper-*`` checkpoints from the hub —
    a ~1.5s metadata round-trip per call even with a warm local cache. Every
    fallback test constructs its own transcriber, so a plain run pays that
    per parameter. Copies (not a shared instance) are handed out even
    though nothing mutates a tokenizer today: ``len`` is read-only, and a
    future mutation would otherwise leak across tests. Only the exact
    no-argument form hits the cache; load-shaping arguments such as
    ``revision=`` must reach the real loader.

    Non-hub paths (the local-snapshot attempt) still hit the real loader.
    """
    import copy

    from transformers import WhisperTokenizer

    real = WhisperTokenizer.from_pretrained

    def cached_from_pretrained(name_or_path, *args, **kwargs):
        if (
            isinstance(name_or_path, str)
            and name_or_path.startswith("openai/whisper-")
            and not args
            and not kwargs
        ):
            if name_or_path not in _whisper_hub_tokenizers:
                _whisper_hub_tokenizers[name_or_path] = real(name_or_path)
            return copy.deepcopy(_whisper_hub_tokenizers[name_or_path])
        return real(name_or_path, *args, **kwargs)

    monkeypatch.setattr(
        WhisperTokenizer, "from_pretrained", staticmethod(cached_from_pretrained)
    )


@pytest.fixture
def run_in_spawn_process(request):
    """Isolate a real engine: MLX is not fork-safe, and engines retain state.

    ``label`` names the spawn in failure messages — it defaults to the test
    name, so single-spawn tests need nothing, while tests that run several
    processes pass one per call to tell which child failed.
    """

    def run(target, *args, timeout=300, label=None):
        label = request.node.name if label is None else label
        process = mp.get_context("spawn").Process(target=target, args=args)
        process.start()
        try:
            process.join(timeout=timeout)
            assert not process.is_alive(), f"{label}: serving test timed out"
            exitcode = process.exitcode
            if exitcode is not None and exitcode < 0:
                # A negative code is the actual terminating OS signal.
                signum = -exitcode
                detail = f"signal {signum} ({signal.Signals(signum).name})"
            else:
                detail = f"exit {exitcode}"
            assert exitcode == 0, f"{label}: child process failed ({detail})"
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)

    return run


def _get_test_seed() -> int:
    """Return the deterministic seed used across tests.

    Override via `VLLM_METAL_TEST_SEED` for debugging.
    """

    raw_seed = os.environ.get("VLLM_METAL_TEST_SEED", "0")
    try:
        return int(raw_seed)
    except ValueError as exc:  # pragma: no cover
        raise ValueError("VLLM_METAL_TEST_SEED must be an integer") from exc


@pytest.fixture
def force_tiled_prefill():
    """Keep prefill batches on the tiled kernel for the duration of a test.

    On an M5 the NAX kernel intercepts eligible prefill batches before
    select_tile_config, so a test that means to exercise the tiled kernel
    silently exercises NAX instead -- and because CI runners are not M5,
    nothing would catch the swap. Tests that assert tiled dispatch request this
    fixture; NAX itself is covered by tools/nax_prefill_parity.py (manual, M5).
    """

    from vllm_metal.metal import get_ops

    ops = get_ops()
    ops.set_nax_enabled(False)
    try:
        yield
    finally:
        ops.set_nax_enabled(True)


@pytest.fixture
def nax_prefill():
    """Route eligible prefill to NAX and record the dispatch family.

    Skips off M5: the NAX kernel cannot run there, and the tiled twin of this
    path is covered by test_mixed_batch_splits_decode_by_work.
    """
    import mlx.core as mx

    from vllm_metal.metal import get_ops

    ops = get_ops()
    if not (ops.nax_supported() and ops.nax_ready()):
        pytest.skip("NAX prefill needs an M5 GPU and the NAX metallib")
    ops.set_nax_enabled(True)
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        yield ops
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


@pytest.fixture(autouse=True)
def _seed_random_generators() -> None:
    """Seed common RNGs to keep tests deterministic."""

    seed = _get_test_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        import mlx.core as mx
    except ImportError:
        return

    mlx_seed = getattr(mx.random, "seed", None)
    if mlx_seed is None:
        return
    mlx_seed(seed)


@pytest.fixture(autouse=True)
def _no_kv_commit_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests off the host's memory.

    The planner's commit probe (``VLLM_METAL_KV_COMMIT_PROBE``) sizes the KV
    pool against what the machine has free right now, so leaving it on would
    make budget assertions depend on whatever else is running on the test box.
    Tests that mean to exercise it set the variable themselves and stub
    ``vllm_metal.v1.cache_policy.probe_commit``.
    """

    monkeypatch.setenv("VLLM_METAL_KV_COMMIT_PROBE", "0")


@pytest.fixture
def spy_group(monkeypatch) -> SpyTransferGroup:
    """Install a spy as the KV transfer group the runner's connector uses."""
    import vllm.v1.worker.gpu.kv_connector as upstream

    import vllm_metal.v1.model_runner as mr
    from tests.kv_connector_spy import SpyTransferGroup

    group = SpyTransferGroup()
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(upstream, "get_kv_transfer_group", lambda: group)
    # The driver opens a forward context for loads; a stub runner has no real
    # VllmConfig, and the spy ignores the context.
    monkeypatch.setattr(upstream, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(upstream, "get_forward_context", lambda: None)
    # Stands in for register_kv_connector_caches, which stub runners skip.
    monkeypatch.setattr(
        mr.MetalModelRunner,
        "_kv_connector_driver",
        upstream.ActiveKVConnector(SimpleNamespace(), {}),
    )
    return group
