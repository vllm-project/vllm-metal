# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MetalKVConnectorBridge: register-once, pointer-stability
fail-fast, and capability detection. No model/server required; uses a fake
runner + fake MLX KV arrays and a fake vLLM KV-transfer group.
"""
import types

import mlx.core as mx
import pytest


def _install_fake_kv_transfer(monkeypatch, connector):
    """Point vllm.distributed.kv_transfer.{has,get}_kv_transfer_group at a fake."""
    import vllm.distributed.kv_transfer as kvt

    monkeypatch.setattr(kvt, "has_kv_transfer_group", lambda: connector is not None)
    monkeypatch.setattr(kvt, "get_kv_transfer_group", lambda: connector)
    # the bridge imports these names into its own module namespace
    import vllm_metal.v1.kv_connector as bridge_mod

    monkeypatch.setattr(bridge_mod, "has_kv_transfer_group",
                        lambda: connector is not None)
    monkeypatch.setattr(bridge_mod, "get_kv_transfer_group", lambda: connector)


class _FakeConnector:
    def __init__(self):
        self.register_count = 0
        self.registered = None

    def register_kv_caches(self, kv_caches):
        self.register_count += 1
        self.registered = kv_caches


class _FakeKVCache:
    def __init__(self, key_caches, value_caches):
        self.key_caches = key_caches
        self.value_caches = value_caches


class _FakeRuntime:
    def __init__(self, kv_cache):
        self.kv_cache = kv_cache


class _FakeCfgGroup:
    def __init__(self, layer_names):
        self.layer_names = layer_names


class _FakeKVCacheConfig:
    def __init__(self, layer_names):
        self.kv_cache_groups = [_FakeCfgGroup(layer_names)]


class _FakeRunner:
    def __init__(self, num_layers=2, nb=4, bs=8, nh=2, hs=16):
        keys = [mx.zeros((nb, bs, nh, hs), dtype=mx.float16) for _ in range(num_layers)]
        vals = [mx.zeros((nb, bs, nh, hs), dtype=mx.float16) for _ in range(num_layers)]
        mx.eval(*keys, *vals)
        self._paged_attention_runtime = _FakeRuntime(_FakeKVCache(keys, vals))
        self.vllm_config = types.SimpleNamespace()


def _bridge(runner):
    from vllm_metal.v1.kv_connector import MetalKVConnectorBridge

    return MetalKVConnectorBridge(runner)


def _cfg(n=2):
    return _FakeKVCacheConfig([f"layer_{i}" for i in range(n)])


def test_register_once_no_second_registration(monkeypatch):
    conn = _FakeConnector()
    _install_fake_kv_transfer(monkeypatch, conn)
    runner = _FakeRunner(num_layers=2)
    b = _bridge(runner)
    b.on_initialize_kv_cache(_cfg(2))
    assert conn.register_count == 1
    # split payload: {name: (k, v)}
    assert set(conn.registered.keys()) == {"layer_0", "layer_1"}
    k, v = conn.registered["layer_0"]
    assert k.data_ptr() != v.data_ptr()

    # Several begin/finish cycles: NO additional registration (pointers stable).
    from vllm.v1.worker.kv_connector_model_runner_mixin import (
        KVConnectorModelRunnerMixin,
    )

    monkeypatch.setattr(
        KVConnectorModelRunnerMixin, "begin_kv_connector_step",
        staticmethod(lambda so: object()),
    )
    for _ in range(3):
        b._materialize_live_kv()
        b._verify_pointer_stability()
    assert conn.register_count == 1


def test_pointer_movement_fails_fast(monkeypatch):
    conn = _FakeConnector()
    _install_fake_kv_transfer(monkeypatch, conn)
    runner = _FakeRunner(num_layers=2)
    b = _bridge(runner)
    b.on_initialize_kv_cache(_cfg(2))
    assert conn.register_count == 1

    # Simulate an out-of-place write: rebind key_caches[0] to a NEW buffer.
    kv = runner._paged_attention_runtime.kv_cache
    new_k = mx.ones((4, 8, 2, 16), dtype=mx.float16)
    mx.eval(new_k)
    kv.key_caches[0] = new_k

    with pytest.raises(RuntimeError, match="Metal KV storage moved"):
        b._verify_pointer_stability()
    # still exactly one registration — no silent second context
    assert conn.register_count == 1


def test_pointer_check_does_not_allocate(monkeypatch):
    conn = _FakeConnector()
    _install_fake_kv_transfer(monkeypatch, conn)
    runner = _FakeRunner(num_layers=3)
    b = _bridge(runner)
    b.on_initialize_kv_cache(_cfg(3))
    before = [k.data_ptr() for k in
              [__import__("vllm_metal.pytorch_backend.tensor_bridge",
                          fromlist=["mlx_to_torch"]).mlx_to_torch(a, device="cpu")
               for a in runner._paged_attention_runtime.kv_cache.key_caches]]
    b._verify_pointer_stability()  # must not raise, must not move buffers
    after = [k.data_ptr() for k in
             [__import__("vllm_metal.pytorch_backend.tensor_bridge",
                         fromlist=["mlx_to_torch"]).mlx_to_torch(a, device="cpu")
              for a in runner._paged_attention_runtime.kv_cache.key_caches]]
    assert before == after


def test_layer_count_mismatch_raises(monkeypatch):
    conn = _FakeConnector()
    _install_fake_kv_transfer(monkeypatch, conn)
    runner = _FakeRunner(num_layers=2)
    b = _bridge(runner)
    # config claims 3 layers but cache has 2
    with pytest.raises(ValueError, match="layer count does not match"):
        b.on_initialize_kv_cache(_cfg(3))


def test_capability_error_when_lifecycle_absent(monkeypatch):
    conn = _FakeConnector()
    _install_fake_kv_transfer(monkeypatch, conn)
    runner = _FakeRunner(num_layers=2)
    b = _bridge(runner)
    # Simulate an older vLLM missing the lifecycle methods.
    from vllm.v1.worker.kv_connector_model_runner_mixin import (
        KVConnectorModelRunnerMixin,
    )

    from vllm_metal.v1.kv_connector import MetalKVConnectorCapabilityError

    for m in ("begin_kv_connector_step", "finish_kv_connector_step",
              "abort_kv_connector_step"):
        monkeypatch.delattr(KVConnectorModelRunnerMixin, m, raising=False)
    with pytest.raises(MetalKVConnectorCapabilityError, match="imperative KVConnector step lifecycle"):
        b.on_initialize_kv_cache(_cfg(2))


def test_no_connector_is_noop(monkeypatch):
    _install_fake_kv_transfer(monkeypatch, None)  # has_kv_transfer_group -> False
    runner = _FakeRunner(num_layers=2)
    b = _bridge(runner)
    b.on_initialize_kv_cache(_cfg(2))  # must not raise, must not register
    assert b._registered_ptrs is None
