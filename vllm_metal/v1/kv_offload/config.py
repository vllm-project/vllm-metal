# SPDX-License-Identifier: Apache-2.0
"""Config hook for KV offloading on Metal.

No mlx import here: the engine-core process loads this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vllm.envs as vllm_envs
from vllm.logger import init_logger

from vllm_metal.compat import effective_num_loops

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _configure_kv_events(vllm_config: VllmConfig, extra: dict[str, Any]) -> None:
    """Set each tier's ``enable_kv_events`` when KV cache events are on.

    A tier emits events only when its own switch and the global
    ``enable_kv_cache_events`` are both set. Upstream's tier warns when only
    its own switch is set.
    """
    events_config = vllm_config.kv_events_config
    globally_on = bool(events_config and events_config.enable_kv_cache_events)
    for tier in extra.get("secondary_tiers") or []:
        if not isinstance(tier, dict):
            continue
        if globally_on:
            tier.setdefault("enable_kv_events", True)


def _kv_dtype_bytes(cache_dtype: str, model_dtype: Any) -> int:
    """Bytes per KV element for ``--kv-cache-dtype``; ``auto`` follows the model."""
    if cache_dtype == "auto":
        return int(model_dtype.itemsize)
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

    return int(STR_DTYPE_TO_TORCH_DTYPE[cache_dtype].itemsize)


# Measured in #1037: two full-length requests is the smallest pool with no
# store retries on Mistral-7B and Qwen3-32B at concurrency 4. A store that
# does not fit is retried on the next step. In a pool smaller than one request
# every retry can fail, so the tiering spec warns at startup.
_DEFAULT_POOL_REQUESTS = 2

# Copy of MLA_DEFAULT_QK_ROPE_HEAD_DIM. Its module imports mlx.
_MLA_DEFAULT_QK_ROPE_HEAD_DIM = 64

# Internal: set when this hook chose the pool size, so the memory planner may
# shrink it to fit. The hook runs again in the engine core, where the size
# alone would look user-set. Users must not set it.
AUTO_POOL_KEY = "_metal_auto_pool"


def default_host_pool_bytes(vllm_config: VllmConfig) -> int:
    """Host pool bytes for two ``max_model_len`` requests of KV.

    The pool stages blocks for the disk tier. A store that does not fit is
    retried on the next step. Two full-length requests is the smallest pool
    with no retries at concurrency 4 in the #1037 measurements, where every
    size from half a request to four stored the same bytes. Prompts there
    were shorter than each pool. Offloading serves only uniform
    full-attention models, whose KV is ``2 * kv_heads * head_size`` elements
    per token and layer. MLA keeps one latent row of
    ``kv_lora_rank + qk_rope_head_dim``. Weight-tied looped models cache each
    loop iteration as its own KV layer, so physical layers are scaled by
    ``effective_num_loops``. A compressed layout only lets the same bytes hold
    more blocks. The token count rounds up to whole blocks once the block size
    is known.
    """
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config
    text_config = model_config.hf_text_config
    # Metal runs MLA whenever the checkpoint has kv_lora_rank
    # (MetalModelRunner.is_mla). vLLM's use_mla can be False for those, e.g.
    # MiniCPM3, and then reports the per-head K/V shape.
    kv_lora_rank = getattr(text_config, "kv_lora_rank", None)
    if kv_lora_rank is not None:
        per_layer = int(kv_lora_rank) + int(
            getattr(text_config, "qk_rope_head_dim", _MLA_DEFAULT_QK_ROPE_HEAD_DIM)
        )
    else:
        per_layer = (
            2
            * model_config.get_num_kv_heads(parallel_config)
            * model_config.get_head_size()
        )
    num_layers = model_config.get_num_layers_by_block_type(
        parallel_config
    ) * effective_num_loops(text_config)
    per_token = (
        per_layer
        * num_layers
        * _kv_dtype_bytes(cache_config.cache_dtype, model_config.dtype)
    )
    tokens = int(model_config.max_model_len)
    block_size = cache_config.block_size
    if block_size:
        tokens = -(-tokens // block_size) * block_size
    return per_token * tokens * _DEFAULT_POOL_REQUESTS


def configure_kv_offloading(vllm_config: VllmConfig) -> None:
    """Translate --kv-offloading-size and validate the KV connector.

    vLLM's own translation (``VllmConfig._post_init_kv_transfer_config``) runs
    after this hook and would set a connector Metal cannot serve. So translate
    here and clear ``kv_offloading_size``, which disarms the upstream one.
    Without a size or a connector this returns without touching anything.
    With the connector but no size, the host pool defaults to two
    ``max_model_len`` requests of KV.
    """
    cache_config = vllm_config.cache_config
    kv_transfer_config = vllm_config.kv_transfer_config
    kv_offloading_size = cache_config.kv_offloading_size
    explicit_connector = (
        kv_transfer_config.kv_connector if kv_transfer_config is not None else None
    )
    if explicit_connector not in (
        None,
        "OffloadingConnector",
        "MetalOffloadingConnector",
    ):
        # Not an offloading config: leave it to upstream.
        if kv_offloading_size is None:
            return
        raise NotImplementedError(
            f"--kv-offloading-size cannot be combined with KV connector "
            f"'{explicit_connector}' on Metal."
        )
    if kv_offloading_size is None and explicit_connector is None:
        return

    if (
        kv_offloading_size is not None
        and cache_config.kv_offloading_backend != "native"
    ):
        raise NotImplementedError(
            "Metal supports only --kv-offloading-backend native; "
            f"'{cache_config.kv_offloading_backend}' is not supported."
        )
    if vllm_envs.VLLM_USE_SIMPLE_KV_OFFLOAD:
        raise NotImplementedError(
            "VLLM_USE_SIMPLE_KV_OFFLOAD is not supported on Metal; "
            "unset it to use the native KV offloading connector."
        )
    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError(
            "KV offloading on Metal does not support pipeline "
            "parallelism yet; run with pipeline_parallel_size=1."
        )
    # Each DP engine would cap the same disk root and evict the others' files.
    if parallel_config.data_parallel_size > 1:
        raise NotImplementedError(
            "KV offloading on Metal does not support data parallelism; "
            "run with data_parallel_size=1."
        )
    # Refused before the weights load. The connector checks the KV cache spec
    # later, before the KV cache is allocated.
    model_config = vllm_config.model_config
    if model_config.runner_type == "pooling":
        raise NotImplementedError(
            "KV offloading on Metal does not support pooling models."
        )
    # Hybrid models keep recurrent state the connector cannot restore. The
    # connector refuses their extra cache groups too, but after the weights.
    if model_config.is_hybrid:
        raise NotImplementedError(
            "KV offloading on Metal supports a single full-attention KV cache "
            "group; hybrid models are not supported."
        )
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None and speculative_config.method in (
        "dflash",
        "dspark",
    ):
        # Block drafters need both target and draft cache groups restored;
        # the Metal connector only supports one full-attention group.
        raise NotImplementedError(
            "KV offloading on Metal does not support "
            f"{speculative_config.method} target/draft cache groups; remove "
            "--kv-offloading-size and any offloading connector from "
            "--kv-transfer-config."
        )
    # Read as upstream's build_offloading_config does. Metal skips the
    # per-layer canonical check, so the disk tier would file uncertified
    # pages under a canonical namespace.
    if kv_transfer_config is not None and bool(
        kv_transfer_config.kv_connector_extra_config.get("canonical_layout", False)
    ):
        raise NotImplementedError(
            "KV offloading on Metal does not support canonical_layout; remove "
            "it from kv_connector_extra_config."
        )

    if kv_transfer_config is None:
        from vllm.config import KVTransferConfig

        kv_transfer_config = KVTransferConfig()
        vllm_config.kv_transfer_config = kv_transfer_config
    kv_transfer_config.kv_connector = "MetalOffloadingConnector"
    kv_transfer_config.kv_connector_module_path = "vllm_metal.v1.kv_offload.connector"
    # Upstream forces kv_both for offloading. kv_producer would disable the
    # scheduler's defer_block_free protection.
    if kv_transfer_config.kv_role not in (None, "kv_both"):
        logger.warning(
            "KV offloading on Metal overrides kv_role=%r to 'kv_both'.",
            kv_transfer_config.kv_role,
        )
    kv_transfer_config.kv_role = "kv_both"
    extra = kv_transfer_config.kv_connector_extra_config
    if kv_offloading_size is not None:
        extra["cpu_bytes_to_use"] = int(kv_offloading_size * (1 << 30))
        extra.pop(AUTO_POOL_KEY, None)  # a user-set size is never capped
        cache_config.kv_offloading_size = None
    elif "cpu_bytes_to_use" not in extra:
        extra["cpu_bytes_to_use"] = default_host_pool_bytes(vllm_config)
        extra[AUTO_POOL_KEY] = True
        logger.info_once(
            "KV offloading on Metal: no --kv-offloading-size given, so the host "
            "pool defaults to %.2f GiB, two --max-model-len requests (%d tokens "
            "each) of KV; it may be capped to fit the budget.",
            extra["cpu_bytes_to_use"] / 2**30,
            model_config.max_model_len,
        )
    # Only "fs" works here; "obj" needs NIXL, which has no macOS build.
    # The spec renames "fs" to MetalFileSystemTierManager. This hook runs again
    # in the engine core and sees that name.
    tiers = extra.get("secondary_tiers") or []
    for tier in tiers:
        tier_type = tier.get("type") if isinstance(tier, dict) else None
        if tier_type not in ("fs", "MetalFileSystemTierManager"):
            raise NotImplementedError(
                f"Secondary KV tier type '{tier_type}' is not "
                "supported on Metal; only 'fs' (filesystem) is "
                "available (the 'obj' tier requires NIXL, which has "
                "no macOS build)."
            )
    _configure_kv_events(vllm_config, extra)
    # One Metal spec serves the host pool and secondary tiers. Unknown names
    # fail here, not as a CUDA-bound spec deep inside engine start.
    spec_name = extra.get("spec_name")
    if spec_name not in (
        None,
        "CPUOffloadingSpec",
        "TieringOffloadingSpec",
        "MetalTieringOffloadingSpec",
    ):
        raise NotImplementedError(
            f"Offloading spec '{spec_name}' is not supported on Metal."
        )
    extra["spec_name"] = "MetalTieringOffloadingSpec"
    extra["spec_module_path"] = "vllm_metal.v1.kv_offload.spec"
    # The host pool is shared in-process. See shared_region.py.
    if parallel_config.distributed_executor_backend != "uni":
        raise NotImplementedError(
            "KV offloading on Metal requires the single-process executor "
            "(--distributed-executor-backend uni); got "
            f"'{parallel_config.distributed_executor_backend}'."
        )
