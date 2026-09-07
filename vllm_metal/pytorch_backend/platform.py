# SPDX-License-Identifier: Apache-2.0
"""Experimental upstream vLLM models with a CPU control plane and MPS compute."""

import platform
from typing import TYPE_CHECKING

import psutil
import torch
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = init_logger(__name__)


class TorchPlatform(Platform):
    _enum = PlatformEnum.OOT
    # The upstream CPU runner owns scheduling tensors and sampling. Model
    # parameters, activations, and KV cache are allocated separately on MPS.
    device_name = "cpu"
    device_type = "cpu"
    dispatch_key = "CPU"
    dist_backend = "gloo"
    simple_compile_backend = "eager"

    def __init__(self) -> None:
        from vllm_metal import envs

        if envs.VLLM_METAL_MODEL_BACKEND != "torch":
            raise ValueError("VLLM_METAL_MODEL_BACKEND must be 'mlx' or 'torch'.")
        if not self.is_available():
            raise RuntimeError(
                "The torch Metal backend requires PyTorch MPS on Apple Silicon."
            )

    @classmethod
    def is_available(cls) -> bool:
        return platform.machine() == "arm64" and torch.backends.mps.is_available()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "Apple Silicon (PyTorch MPS)"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.mps.manual_seed(seed)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        if device.index not in (None, 0):
            raise ValueError("The torch Metal backend supports one device.")

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.config import CompilationMode, CUDAGraphMode

        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        unsupported = {
            "distributed execution": (
                parallel.world_size != 1 or parallel.data_parallel_size != 1
            ),
            "context parallelism": (
                parallel.decode_context_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
            ),
            "speculative decoding": vllm_config.speculative_config is not None,
            "LoRA": vllm_config.lora_config is not None,
            "KV transfer": vllm_config.kv_transfer_config is not None,
            "KV offloading": vllm_config.cache_config.kv_offloading_size is not None,
            "weight offloading": (
                vllm_config.offload_config.uva.cpu_offload_gb > 0
                or vllm_config.offload_config.prefetch.offload_group_size > 0
            ),
            "weight transfer": vllm_config.weight_transfer_config is not None,
            "profiling": vllm_config.profiler_config.profiler is not None,
            "quantized KV cache": vllm_config.cache_config.cache_dtype != "auto",
            "dual batch overlap": parallel.enable_dbo,
            "V2 model runner": vllm_config.use_v2_model_runner,
        }
        if model is not None:
            unsupported.update(
                {
                    "pooling": model.runner_type != "generate",
                    "quantization": model.quantization is not None,
                    "hybrid/SSM models": model.is_hybrid or model.has_inner_state,
                    "multimodal models": model.is_multimodal_model,
                    "encoder-decoder models": model.is_encoder_decoder,
                    "MoE models": model.is_moe,
                }
            )
        for feature, enabled in unsupported.items():
            if enabled:
                raise NotImplementedError(
                    f"The experimental torch Metal backend does not support {feature}."
                )

        if parallel.distributed_executor_backend in (None, "auto"):
            parallel.distributed_executor_backend = "uni"
        if parallel.distributed_executor_backend not in ("uni", "mp"):
            raise NotImplementedError(
                "The torch Metal backend requires a local 'uni' or 'mp' executor."
            )
        if parallel.worker_cls == "auto":
            parallel.worker_cls = "vllm_metal.pytorch_backend.worker.TorchWorker"
        parallel.disable_custom_all_reduce = True
        if model is not None:
            model.enforce_eager = True
            model.disable_cascade_attn = True
        compilation = vllm_config.compilation_config
        compilation.mode = CompilationMode.NONE
        compilation.cudagraph_mode = CUDAGraphMode.NONE
        compilation.cudagraph_capture_sizes = []
        compilation.backend = "eager"
        compilation.custom_ops = ["none"]
        compilation.ir_enable_torch_wrap = False
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.load_config.device = "mps"
        logger.warning_once(
            "Using experimental upstream vLLM models on MPS. "
            "Model compatibility depends on PyTorch-native operator support."
        )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )

        if selected_backend not in (None, AttentionBackendEnum.CUSTOM):
            raise NotImplementedError(
                "The torch Metal backend requires its custom SDPA attention backend."
            )

        config = attn_selector_config
        if (
            config.use_mla
            or config.use_sparse
            or config.has_sink
            or config.use_mm_prefix
            or config.use_per_head_quant_scales
            or config.attn_type != AttentionType.DECODER
            or config.use_non_causal
        ):
            raise NotImplementedError(
                "The torch Metal backend supports causal decoder SDPA only."
            )
        backend_path = "vllm_metal.pytorch_backend.attention.TorchAttentionBackend"
        register_backend(AttentionBackendEnum.CUSTOM, backend_path)
        return backend_path
