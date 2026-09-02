# SPDX-License-Identifier: Apache-2.0
"""Config-time tests for the KV offloading platform wiring.

MetalPlatform.check_and_update_config performs the --kv-offloading-size ->
kv_transfer_config translation itself (it runs before vLLM's own translation,
which would force-set a connector this platform cannot serve) and routes the
connector/spec lookups to the Metal classes.
"""

from types import SimpleNamespace

import pytest

from vllm_metal.config import reset_config
from vllm_metal.platform import MetalPlatform


def _base_config(**cache_overrides) -> SimpleNamespace:
    cache_config = SimpleNamespace(
        block_size=None,
        enable_prefix_caching=False,
        kv_offloading_size=None,
        kv_offloading_backend="native",
        # Read by _pick_mb_buffer_default (#637).
        gpu_memory_utilization=0.9,
        # Populated by --kv-cache-dtype turboquant_*; Metal rejects it up
        # front (TurboQuant runs off --additional-config).
        kv_cache_dtype_skip_layers=[],
    )
    for key, value in cache_overrides.items():
        setattr(cache_config, key, value)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls="auto",
            distributed_executor_backend="auto",
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            disable_custom_all_reduce=False,
        ),
        cache_config=cache_config,
        speculative_config=None,
        model_config=SimpleNamespace(
            model="test-model",
            disable_cascade_attn=False,
            tokenizer=None,
            max_model_len=4096,
            multimodal_config=None,
            hf_config=SimpleNamespace(model_type="qwen3"),
            is_hybrid=False,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=2048,
            max_num_scheduled_tokens=None,
        ),
        kv_transfer_config=None,
        # Real VllmConfig always carries this; the events path reads it.
        kv_events_config=None,
        # The platform hook reads this directly (vllm-metal #604 dropped the
        # getattr fallback in favour of trusting vLLM's typed config).
        additional_config=None,
    )


@pytest.fixture(autouse=True)
def _paged_attention(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    monkeypatch.setattr(
        "vllm_metal.platform.MetalPlatform._is_stt_model",
        staticmethod(lambda *_: False),
        raising=False,
    )
    reset_config()
    yield
    reset_config()


def test_offloading_size_translates_to_metal_connector() -> None:
    vllm_config = _base_config(kv_offloading_size=2.0)
    MetalPlatform.check_and_update_config(vllm_config)

    ktc = vllm_config.kv_transfer_config
    assert ktc is not None
    assert ktc.kv_connector == "MetalOffloadingConnector"
    assert ktc.kv_connector_module_path == "vllm_metal.v1.kv_offload.connector"
    assert ktc.kv_role == "kv_both"
    extra = ktc.kv_connector_extra_config
    assert extra["cpu_bytes_to_use"] == 2 * (1 << 30)
    # One spec serves both the plain host pool and secondary tiers.
    assert extra["spec_name"] == "MetalTieringOffloadingSpec"
    assert extra["spec_module_path"] == "vllm_metal.v1.kv_offload.spec"
    # Upstream translation must be disarmed (it would force-set a connector
    # name after this hook has run).
    assert vllm_config.cache_config.kv_offloading_size is None


def test_secondary_tiers_select_tiering_spec() -> None:
    vllm_config = _base_config(kv_offloading_size=1.0)
    vllm_config.kv_transfer_config = SimpleNamespace(
        kv_connector=None,
        kv_connector_module_path=None,
        kv_role=None,
        kv_connector_extra_config={
            "secondary_tiers": [{"type": "fs", "root_dir": "/tmp/kv"}]
        },
    )
    MetalPlatform.check_and_update_config(vllm_config)

    extra = vllm_config.kv_transfer_config.kv_connector_extra_config
    assert extra["spec_name"] == "MetalTieringOffloadingSpec"


def test_offloading_requires_uni_executor() -> None:
    """The host pool is anonymous RAM shared within one process, so a
    multi-process executor would give the worker a disjoint pool."""
    vllm_config = _base_config(kv_offloading_size=1.0)
    vllm_config.parallel_config.distributed_executor_backend = "mp"
    with pytest.raises(NotImplementedError, match="single-process executor"):
        MetalPlatform.check_and_update_config(vllm_config)


def test_inert_kv_transfer_config_passes_through() -> None:
    """A kv_transfer_config with no connector and no offloading request was
    inert upstream and must stay untouched (no guards, no spec injection)."""
    vllm_config = _base_config()
    inert = SimpleNamespace(
        kv_connector=None,
        kv_connector_module_path=None,
        kv_role="kv_both",
        kv_connector_extra_config={},
    )
    vllm_config.kv_transfer_config = inert
    MetalPlatform.check_and_update_config(vllm_config)

    assert vllm_config.kv_transfer_config is inert
    assert inert.kv_connector is None
    assert inert.kv_connector_module_path is None
    assert inert.kv_connector_extra_config == {}


def test_unsupported_connector_rejected() -> None:
    vllm_config = _base_config()
    vllm_config.kv_transfer_config = SimpleNamespace(
        kv_connector="NixlConnector",
        kv_connector_module_path=None,
        kv_role="kv_both",
        kv_connector_extra_config={},
    )
    with pytest.raises(NotImplementedError, match="NixlConnector"):
        MetalPlatform.check_and_update_config(vllm_config)


def test_explicit_connector_without_size_rejected() -> None:
    vllm_config = _base_config()
    vllm_config.kv_transfer_config = SimpleNamespace(
        kv_connector="OffloadingConnector",
        kv_connector_module_path=None,
        kv_role="kv_both",
        kv_connector_extra_config={},
    )
    with pytest.raises(NotImplementedError, match="--kv-offloading-size"):
        MetalPlatform.check_and_update_config(vllm_config)


def test_unknown_spec_name_rejected() -> None:
    vllm_config = _base_config(kv_offloading_size=1.0)
    vllm_config.kv_transfer_config = SimpleNamespace(
        kv_connector=None,
        kv_connector_module_path=None,
        kv_role=None,
        kv_connector_extra_config={"spec_name": "ARCOffloadingSpec"},
    )
    with pytest.raises(NotImplementedError, match="ARCOffloadingSpec"):
        MetalPlatform.check_and_update_config(vllm_config)


def test_simple_kv_offload_env_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vllm.envs.VLLM_USE_SIMPLE_KV_OFFLOAD", True, raising=False)
    vllm_config = _base_config(kv_offloading_size=1.0)
    with pytest.raises(NotImplementedError, match="VLLM_USE_SIMPLE_KV_OFFLOAD"):
        MetalPlatform.check_and_update_config(vllm_config)


def test_offloading_requires_paged_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "0")
    reset_config()
    vllm_config = _base_config(kv_offloading_size=1.0)
    with pytest.raises(NotImplementedError, match="paged attention"):
        MetalPlatform.check_and_update_config(vllm_config)


def test_lmcache_backend_rejected() -> None:
    vllm_config = _base_config(kv_offloading_size=1.0, kv_offloading_backend="lmcache")
    with pytest.raises(NotImplementedError, match="lmcache"):
        MetalPlatform.check_and_update_config(vllm_config)


def test_validate_metal_support_guards() -> None:
    """The single-group / uniform-attention guard that keeps hybrid and
    non-attention KV layouts off the offload path (e2e-validated on gemma-4;
    unit-pinned here)."""
    from types import SimpleNamespace

    import pytest
    from vllm.v1.kv_cache_interface import AttentionSpec

    from vllm_metal.v1.kv_offload.spec import validate_metal_support

    def group(spec, layer_names=("layer0",)):
        return SimpleNamespace(kv_cache_spec=spec, layer_names=layer_names)

    with pytest.raises(NotImplementedError, match="single KV cache group"):
        validate_metal_support(
            SimpleNamespace(kv_cache_groups=[group(None), group(None)])
        )
    with pytest.raises(NotImplementedError, match="hybrid/sliding-window"):
        validate_metal_support(SimpleNamespace(kv_cache_groups=[group(object())]))

    # vllm-metal #630 puts the draft model's KV in this same group, so a
    # non-uniform spec here usually means spec decode, not a hybrid model.
    # Saying "hybrid/sliding-window" sends the user after the wrong thing.
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        validate_metal_support(
            SimpleNamespace(
                kv_cache_groups=[
                    group(object(), ("layer0", "draft_layers.0.self_attn"))
                ]
            )
        )

    class _FakeAttention(AttentionSpec):
        pass

    ok = _FakeAttention.__new__(_FakeAttention)
    validate_metal_support(SimpleNamespace(kv_cache_groups=[group(ok)]))  # no raise


def test_secondary_tier_gate_is_idempotent() -> None:
    """check_and_update_config runs again in the engine core process, on a
    config MetalTieringOffloadingSpec has already rewritten to name the Metal
    tier class. The gate has to accept its own rewrite or the engine dies at
    startup with 'tier type MetalFileSystemTierManager is not supported'."""
    from vllm_metal.v1.kv_offload.spec import route_fs_tiers_to_metal

    tiers = [{"type": "fs", "root_dir": "/tmp/unused"}]
    config = _base_config(
        kv_offloading_size=8,
    )
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="OffloadingConnector",
        kv_role=None,
        kv_connector_module_path=None,
        kv_connector_extra_config={"secondary_tiers": tiers},
    )
    MetalPlatform.check_and_update_config(config)

    # The spec rewrites the tier in place, exactly as it does at runtime.
    route_fs_tiers_to_metal(config.kv_transfer_config.kv_connector_extra_config)
    assert tiers[0]["type"] == "MetalFileSystemTierManager"

    # Second pass over the rewritten config must not raise.
    MetalPlatform.check_and_update_config(config)


def test_kv_events_enable_the_tier_switch() -> None:
    """A KV-aware router consumes the BlockStored events the tier emits, and
    the tier only emits them when its own enable_kv_events is set AND events
    are on globally. Set them up for the user rather than making them line up
    two switches by hand."""
    vllm_config = _base_config(kv_offloading_size=1.0)
    vllm_config.kv_events_config = SimpleNamespace(enable_kv_cache_events=True)
    tiers = [{"type": "fs", "root_dir": "/tmp/kv"}]
    vllm_config.kv_transfer_config = SimpleNamespace(
        kv_connector=None,
        kv_connector_module_path=None,
        kv_role=None,
        kv_connector_extra_config={"secondary_tiers": tiers},
    )
    MetalPlatform.check_and_update_config(vllm_config)
    assert tiers[0]["enable_kv_events"] is True


def test_tier_events_without_global_events_rejected() -> None:
    """The reverse combination emits nothing and only warns mid-startup,
    which looks exactly like a broken router. Refuse it at config time."""
    vllm_config = _base_config(kv_offloading_size=1.0)
    vllm_config.kv_events_config = None
    vllm_config.kv_transfer_config = SimpleNamespace(
        kv_connector=None,
        kv_connector_module_path=None,
        kv_role=None,
        kv_connector_extra_config={
            "secondary_tiers": [
                {"type": "fs", "root_dir": "/tmp/kv", "enable_kv_events": True}
            ]
        },
    )
    with pytest.raises(NotImplementedError, match="kv-events-config"):
        MetalPlatform.check_and_update_config(vllm_config)
