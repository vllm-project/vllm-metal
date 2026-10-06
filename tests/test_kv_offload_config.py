# SPDX-License-Identifier: Apache-2.0
"""Config-time tests for the KV offloading platform wiring.

MetalPlatform.check_and_update_config performs the --kv-offloading-size ->
kv_transfer_config translation itself (it runs before vLLM's own translation,
which would force-set a connector this platform cannot serve) and routes the
connector/spec lookups to the Metal classes.
"""

import os
from types import SimpleNamespace

import pytest
from vllm.config import AuxOutputConfig

from vllm_metal.config import reset_config
from vllm_metal.platform import MetalPlatform

_CONFIG_LOGGER = "vllm_metal.v1.kv_offload.config"


def _base_config(**cache_overrides) -> SimpleNamespace:
    cache_config = SimpleNamespace(
        block_size=None,
        enable_prefix_caching=False,
        kv_offloading_size=None,
        kv_offloading_backend="native",
        prefix_caching_hash_algo="sha256",
        cache_dtype="auto",
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
            revision=None,
            disable_cascade_attn=False,
            tokenizer=None,
            max_model_len=4096,
            multimodal_config=None,
            hf_config=SimpleNamespace(model_type="qwen3"),
            is_hybrid=False,
            runner_type="generate",
            use_mla=False,
            quantization=None,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=2048,
            max_num_scheduled_tokens=None,
        ),
        lora_config=None,
        aux_output_config=AuxOutputConfig(),
        kv_transfer_config=None,
        # Real VllmConfig always carries this; the events path reads it.
        kv_events_config=None,
        # The platform hook reads this directly (vllm-metal #604 dropped the
        # getattr fallback in favour of trusting vLLM's typed config).
        additional_config=None,
    )


@pytest.fixture(autouse=True)
def _offline_platform(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "vllm_metal.utils.get_model_download_path",
        lambda model, *, revision: model,
    )
    monkeypatch.setattr(
        "vllm_metal.stt.detection.is_stt_model",
        lambda _model, *, revision: False,
    )
    # Below the MLX_MAX_MB_PER_BUFFER threshold, so the hook leaves the
    # environment alone on any host.
    monkeypatch.setattr(
        "vllm_metal.platform.psutil.virtual_memory",
        lambda: SimpleNamespace(total=64 * (1 << 30), available=48 * (1 << 30)),
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


def test_offloading_refuses_data_parallelism() -> None:
    """Each DP engine would cap the same disk root and evict the others' files."""
    vllm_config = _base_config(kv_offloading_size=1.0)
    # The dense DP over Ray shape Metal otherwise admits.
    vllm_config.model_config.is_moe = False
    vars(vllm_config.parallel_config).update(
        data_parallel_size=2,
        data_parallel_backend="ray",
        data_parallel_size_local=1,
        data_parallel_external_lb=False,
        data_parallel_hybrid_lb=False,
    )
    with pytest.raises(NotImplementedError, match="data parallelism"):
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


def _nixl_config() -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector="NixlConnector",
        kv_connector_module_path=None,
        kv_role="kv_both",
        kv_connector_extra_config={},
    )


def test_other_connector_without_offloading_is_left_alone() -> None:
    vllm_config = _base_config()
    vllm_config.kv_transfer_config = _nixl_config()
    before = vars(vllm_config.kv_transfer_config).copy()
    MetalPlatform.check_and_update_config(vllm_config)
    assert vars(vllm_config.kv_transfer_config) == before


def test_other_connector_with_offloading_rejected() -> None:
    vllm_config = _base_config()
    vllm_config.kv_transfer_config = _nixl_config()
    vllm_config.cache_config.kv_offloading_size = 4
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


def test_lmcache_backend_rejected() -> None:
    vllm_config = _base_config(kv_offloading_size=1.0, kv_offloading_backend="lmcache")
    with pytest.raises(NotImplementedError, match="lmcache"):
        MetalPlatform.check_and_update_config(vllm_config)


def test_validate_metal_support_guards() -> None:
    """Only a single full-attention group reaches the offload path."""
    import torch
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MLAAttentionSpec,
        SlidingWindowSpec,
    )

    from vllm_metal.v1.kv_offload.connector import validate_metal_support

    dims = {
        "block_size": 16,
        "num_kv_heads": 2,
        "head_size": 64,
        "dtype": torch.float16,
    }
    full = FullAttentionSpec(**dims)

    def check(spec, layer_names=("layer0",)):
        group = SimpleNamespace(kv_cache_spec=spec, layer_names=layer_names)
        validate_metal_support(SimpleNamespace(kv_cache_groups=[group]))

    with pytest.raises(NotImplementedError, match="2 groups"):
        group = SimpleNamespace(kv_cache_spec=full, layer_names=("layer0",))
        validate_metal_support(SimpleNamespace(kv_cache_groups=[group, group]))
    with pytest.raises(NotImplementedError, match="SlidingWindowSpec"):
        check(SlidingWindowSpec(**dims, sliding_window=128))
    with pytest.raises(NotImplementedError, match="MLAAttentionSpec"):
        check(MLAAttentionSpec(**dims))
    # Draft-model spec decode shares the full-attention group; point the user
    # at the flag.
    with pytest.raises(NotImplementedError, match="--speculative-config"):
        check(full, ("layer0", "draft_layers.0.self_attn"))

    check(full)  # no raise


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


def test_tier_events_without_global_events_are_accepted() -> None:
    """The reverse combination emits nothing. Upstream's tier warns; the
    config is accepted and left as given."""
    vllm_config = _base_config(kv_offloading_size=1.0)
    vllm_config.kv_events_config = None
    tier = {"type": "fs", "root_dir": "/tmp/kv", "enable_kv_events": True}
    vllm_config.kv_transfer_config = SimpleNamespace(
        kv_connector=None,
        kv_connector_module_path=None,
        kv_role=None,
        kv_connector_extra_config={"secondary_tiers": [tier]},
    )
    MetalPlatform.check_and_update_config(vllm_config)
    assert tier["enable_kv_events"] is True


def test_pooling_and_mla_rejected_at_config_time() -> None:
    """Shapes the offload path does not serve fail before the weights load,
    not at KV cache registration after them."""
    vllm_config = _base_config(kv_offloading_size=1.0)
    vllm_config.model_config.runner_type = "pooling"
    with pytest.raises(NotImplementedError, match="pooling"):
        MetalPlatform.check_and_update_config(vllm_config)

    vllm_config = _base_config(kv_offloading_size=1.0)
    vllm_config.model_config.use_mla = True
    with pytest.raises(NotImplementedError, match="MLA"):
        MetalPlatform.check_and_update_config(vllm_config)


def _tiered(kv_offloading_size: float = 1.0) -> SimpleNamespace:
    vllm_config = _base_config(kv_offloading_size=kv_offloading_size)
    vllm_config.kv_transfer_config = SimpleNamespace(
        kv_connector=None,
        kv_connector_module_path=None,
        kv_role=None,
        kv_connector_extra_config={
            "secondary_tiers": [{"type": "fs", "root_dir": "/tmp/kv"}]
        },
    )
    return vllm_config


@pytest.mark.parametrize("hash_algo", ["sha256", "sha256_cbor"])
def test_fixed_seed_hash_algo_without_a_seed_is_accepted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, hash_algo: str
) -> None:
    """vLLM gives the sha256 family a fixed seed, so a restarted Metal
    server finds the disk tier's blocks. The hook must not set PYTHONHASHSEED:
    that would override the fixed seed."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    env_before = dict(os.environ)
    vllm_config = _tiered()
    vllm_config.cache_config.prefix_caching_hash_algo = hash_algo
    with caplog.at_level("WARNING", logger=_CONFIG_LOGGER):
        MetalPlatform.check_and_update_config(vllm_config)
    assert "PYTHONHASHSEED" not in caplog.text
    assert "PYTHONHASHSEED" not in os.environ
    assert dict(os.environ) == env_before


@pytest.mark.parametrize("hash_algo", ["xxhash", "xxhash_cbor"])
def test_random_seed_hash_algo_without_a_seed_is_accepted(
    monkeypatch: pytest.MonkeyPatch, hash_algo: str
) -> None:
    """Upstream warns that the hashes are not reproducible. The config is
    accepted and no seed is set for the user."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    env_before = dict(os.environ)
    vllm_config = _tiered()
    vllm_config.cache_config.prefix_caching_hash_algo = hash_algo
    MetalPlatform.check_and_update_config(vllm_config)
    assert dict(os.environ) == env_before


def test_random_seed_hash_algo_with_a_seed_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    vllm_config = _tiered()
    vllm_config.cache_config.prefix_caching_hash_algo = "xxhash"
    MetalPlatform.check_and_update_config(vllm_config)
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_random_seed_hash_algo_without_a_disk_tier_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host pool lives and dies with the process, so the seed is moot."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    vllm_config = _base_config(
        kv_offloading_size=1.0, prefix_caching_hash_algo="xxhash"
    )
    MetalPlatform.check_and_update_config(vllm_config)
    assert "PYTHONHASHSEED" not in os.environ


def test_no_offloading_leaves_the_config_as_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without offloading the hook must do exactly what it did before the
    offload wiring. Settings the offload path would refuse are set here to
    prove it is not consulted."""
    import vllm_metal.v1.kv_offload.config as config_module

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr("vllm.envs.VLLM_USE_SIMPLE_KV_OFFLOAD", True, raising=False)

    def build() -> SimpleNamespace:
        reset_config()
        vllm_config = _base_config(
            prefix_caching_hash_algo="xxhash", kv_offloading_backend="lmcache"
        )
        vllm_config.parallel_config.distributed_executor_backend = "mp"
        vllm_config.model_config.use_mla = True
        return vllm_config

    env_before = dict(os.environ)
    with_offload_hook = build()
    MetalPlatform.check_and_update_config(with_offload_hook)
    assert dict(os.environ) == env_before

    monkeypatch.setattr(config_module, "configure_kv_offloading", lambda _: None)
    without_offload_hook = build()
    MetalPlatform.check_and_update_config(without_offload_hook)

    assert with_offload_hook == without_offload_hook
    assert with_offload_hook.kv_transfer_config is None
