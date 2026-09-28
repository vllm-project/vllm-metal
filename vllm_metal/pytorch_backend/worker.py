# SPDX-License-Identifier: Apache-2.0
"""Opt-in vLLM worker for the MPS proof of concept."""

from functools import wraps

import psutil
import torch
from vllm.utils.torch_utils import set_random_seed

from vllm_metal.pytorch_backend.runner import MPSModelRunner
from vllm_metal.v1.worker import MetalWorker, init_worker_distributed_environment


def _patch_mps_mrv2_validation():
    from vllm.config import VllmConfig

    from vllm_metal import envs

    original = VllmConfig._validate_v2_model_runner
    if getattr(original, "_metal_mrv2", False):
        return

    @wraps(original)
    def validate(config):
        if envs.VLLM_METAL_BACKEND != "mps":
            return original(config)
        # Keep upstream feature validation; MPS replaces the Triton operations.
        unsupported = config._get_v2_model_runner_unsupported_features()
        if unsupported:
            raise ValueError(f"MPS MRV2 does not support: {', '.join(unsupported)}")

    validate._metal_mrv2 = True
    VllmConfig._validate_v2_model_runner = validate


def configure_mps(config):
    from vllm import envs as vllm_envs
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    from vllm_metal import envs

    if envs.VLLM_METAL_BACKEND == "mps":
        if vllm_envs.VLLM_USE_V2_MODEL_RUNNER is not True:
            raise ValueError("Experimental MPS requires VLLM_USE_V2_MODEL_RUNNER=1")

    model = config.model_config
    hf = model.hf_text_config
    if (
        hf.model_type != "qwen3"
        or hf.head_dim != 128
        or model.quantization is not None
        or model.runner_type != "generate"
        or model.dtype not in (torch.float16, torch.bfloat16)
        or config.parallel_config.world_size != 1
        or config.parallel_config.data_parallel_size != 1
        or config.speculative_config is not None
        or config.lora_config is not None
        or config.kv_transfer_config is not None
        or config.additional_config.get("turboquant", False)
    ):
        raise NotImplementedError(
            "Experimental MPS requires unquantized Qwen3, fp16/bf16, "
            "one GPU, and no LoRA/speculative decoding/KV transfer."
        )
    if config.cache_config.block_size not in (None, 16):
        raise ValueError("Experimental MPS requires --block-size 16")
    unsupported = [
        name
        for name in ("use_fp64_gumbel", "enable_trace_replay", "return_sampling_mask")
        if getattr(model, name, False)
    ]
    if unsupported:
        raise NotImplementedError(
            f"Experimental MPS does not support {', '.join(unsupported)}"
        )
    _patch_mps_mrv2_validation()
    config.cache_config.block_size = 16
    config.scheduler_config.async_scheduling = False
    config.compilation_config.mode = CompilationMode.NONE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    model.enforce_eager = True


class MPSWorker(MetalWorker):
    def init_device(self):
        if not torch.backends.mps.is_available():
            raise RuntimeError("The experimental backend requires PyTorch MPS")
        configure_mps(self.vllm_config)
        self.device = torch.device("mps")
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )
        set_random_seed(self.model_config.seed)
        self.model_runner = MPSModelRunner(self.vllm_config)

    def determine_available_memory(self):
        torch.mps.synchronize()
        torch.mps.empty_cache()
        if self.cache_config.kv_cache_memory_bytes is not None:
            return self.cache_config.kv_cache_memory_bytes
        hf = self.model_config.hf_text_config
        # PoC estimate only, not a measured peak. See the MPS roadmap for
        # profiling and unified-memory budgeting before broadening this path.
        reserve = self.scheduler_config.max_num_batched_tokens * (
            hf.intermediate_size * 2 + hf.hidden_size * 8
        ) * 4 + (512 << 20)
        budget = int(
            psutil.virtual_memory().total * self.cache_config.gpu_memory_utilization
        )
        available = budget - torch.mps.driver_allocated_memory() - reserve
        if available <= 0:
            raise ValueError(
                "MPS model and activation allowance exceed the memory budget"
            )
        return available

    def synchronize_device(self):
        torch.mps.synchronize()

    def check_health(self):
        torch.mps.synchronize()

    def get_supported_tasks(self):
        return self.model_runner.get_supported_tasks()

    def shutdown(self):
        torch.mps.synchronize()
        self.vllm_config.compilation_config.static_forward_context.clear()
        super().shutdown()
        torch.mps.empty_cache()
