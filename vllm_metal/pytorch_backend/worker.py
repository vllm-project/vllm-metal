# SPDX-License-Identifier: Apache-2.0
"""Opt-in vLLM worker for the MPS proof of concept."""

import torch
from vllm.config.utils import getattr_iter
from vllm.utils.torch_utils import set_random_seed

from vllm_metal.pytorch_backend.attention import MPSAttentionBackend
from vllm_metal.pytorch_backend.mps_ops import _load_mps_module
from vllm_metal.pytorch_backend.runner import MPSModelRunner
from vllm_metal.v1.worker import MetalWorker, init_worker_distributed_environment

# PyTorch's MPS allocator rounds large heaps up to this size (kRoundLarge).
_MPS_HEAP_ALIGNMENT = 2 << 20


def configure_mps(config):
    from vllm import envs as vllm_envs
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    from vllm_metal import envs

    if envs.VLLM_METAL_BACKEND == "mps":
        if vllm_envs.VLLM_USE_V2_MODEL_RUNNER is not True:
            raise ValueError("Experimental MPS requires VLLM_USE_V2_MODEL_RUNNER=1")

    model = config.model_config
    state_layers = set(getattr(model.hf_text_config, "layer_types", None) or ())
    if (
        model.get_head_size() not in MPSAttentionBackend.get_supported_head_sizes()
        or model.is_encoder_decoder
        or model.is_diffusion
        or model.is_multimodal_model
        or model.is_moe
        or (model.has_inner_state and state_layers != {"conv", "full_attention"})
        or model.use_mla
        or model.quantization is not None
        or model.runner_type != "generate"
        or model.dtype not in (torch.float16, torch.bfloat16)
        or config.parallel_config.world_size != 1
        or config.parallel_config.data_parallel_size != 1
        or config.lora_config is not None
        or config.kv_transfer_config is not None
        or config.additional_config.get("turboquant", False)
    ):
        raise NotImplementedError(
            "Experimental MPS requires an unquantized dense autoregressive "
            "text decoder with a supported attention head size, optional "
            "ShortConv state, fp16/bf16, "
            "one GPU, and no LoRA/KV transfer."
        )
    spec = config.speculative_config
    if spec is not None and (
        spec.method != "eagle3"
        or spec.draft_sample_method != "greedy"
        or spec.rejection_sample_method != "standard"
        or spec.enable_adaptive_verification
        or spec.parallel_drafting
        or model.has_inner_state
        or config.watermark_config is not None
    ):
        raise NotImplementedError("MPS speculation currently supports greedy EAGLE3")
    if config.cache_config.block_size is None:
        config.cache_config.block_size = 16
    if not MPSAttentionBackend.supports_block_size(config.cache_config.block_size):
        raise ValueError("MPS requires a cache block size divisible by 8")
    if getattr(model, "use_fp64_gumbel", False):
        raise NotImplementedError("Experimental MPS does not support use_fp64_gumbel")
    if config.scheduler_config.async_scheduling is None:
        config.scheduler_config.async_scheduling = False
    config.compilation_config.mode = CompilationMode.NONE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    model.enforce_eager = True


class MPSWorker(MetalWorker):
    def init_device(self):
        import vllm_metal.pytorch_backend.rotary  # noqa: F401

        if not torch.backends.mps.is_available():
            raise RuntimeError("The experimental backend requires PyTorch MPS")
        configure_mps(self.vllm_config)
        self.device = torch.device("mps:0")
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
        # Upstream backs all KV layers with one MPS buffer. PyTorch requires
        # size < maxBufferLength, after rounding up to the heap alignment.
        # TODO: total KV capacity should not be capped by one buffer; this is a
        # current limitation of the Apple Silicon MPS path.
        limit = _load_mps_module().max_buffer_length()
        max_buffer = (limit - 1) // _MPS_HEAP_ALIGNMENT * _MPS_HEAP_ALIGNMENT
        requested = self.cache_config.kv_cache_memory_bytes
        if requested is not None:
            if requested > max_buffer:
                raise ValueError(
                    f"--kv-cache-memory-bytes={requested} does not fit in one "
                    f"MPS buffer; use at most {max_buffer} bytes on this device."
                )
            return requested
        hf = self.model_config.hf_text_config
        hidden_size = self.model_config.get_hidden_size()
        intermediate_size = getattr_iter(
            hf, ("intermediate_size", "ffn_dim", "n_inner")
        )
        if intermediate_size is None and hasattr(hf, "n_inner"):
            # Upstream GPT-2/GPT-J use 4 * hidden_size when n_inner is None.
            intermediate_size = 4 * hidden_size
        if intermediate_size is None:
            raise ValueError(
                "Cannot estimate MPS activation memory from this model config; "
                "set --kv-cache-memory-bytes explicitly."
            )
        # PoC estimate only; --kv-cache-memory-bytes bypasses this heuristic.
        # TODO: measure execution overhead at the configured prefill/concurrency
        # limits instead of estimating it from model dimensions.
        reserve = self.scheduler_config.max_num_batched_tokens * (
            intermediate_size * 2 + hidden_size * 8
        ) * 4 + (512 << 20)
        # Like the MLX backend, budget against the GPU working set, not RAM.
        budget = int(
            torch.mps.recommended_max_memory()
            * self.cache_config.gpu_memory_utilization
        )
        available = budget - torch.mps.driver_allocated_memory() - reserve
        if available <= 0:
            raise ValueError(
                "MPS model and activation allowance exceed the memory budget"
            )
        return min(available, max_buffer)

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
