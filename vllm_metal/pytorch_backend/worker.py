# SPDX-License-Identifier: Apache-2.0
"""Single-device worker for upstream PyTorch models on Apple MPS."""

from collections.abc import Callable
from typing import cast

import psutil
import torch
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from vllm_metal.pytorch_backend.model_runner import TorchModelRunner

logger = init_logger(__name__)
_DEFAULT_KV_CACHE_BYTES = 512 * 1024**2


class TorchWorker(WorkerBase):
    """Use the upstream CPU control plane with MPS model and cache storage."""

    model_runner: TorchModelRunner  # type: ignore[assignment]

    def init_device(self) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("The PyTorch Metal backend requires Apple MPS.")
        self.device = torch.device("cpu")
        self.parallel_config.disable_custom_all_reduce = True
        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
        )
        set_random_seed(self.model_config.seed)
        self.model_runner = TorchModelRunner(self.vllm_config)

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model(load_dummy_weights)

    def determine_available_memory(self) -> int:
        torch.mps.synchronize()
        torch.mps.empty_cache()
        available = int(psutil.virtual_memory().available * 0.8)
        device_budget = (
            int(
                torch.mps.recommended_max_memory()
                * self.cache_config.gpu_memory_utilization
            )
            - torch.mps.driver_allocated_memory()
        )
        budget = max(0, min(available, device_budget))
        requested = self.cache_config.kv_cache_memory_bytes
        if requested is not None:
            if requested > budget:
                raise ValueError(
                    f"Requested KV cache ({requested} bytes) exceeds the available "
                    f"MPS memory budget ({budget} bytes)."
                )
            cache_bytes = requested
        else:
            cache_bytes = min(_DEFAULT_KV_CACHE_BYTES, budget)
        if cache_bytes <= 0:
            raise ValueError("Insufficient MPS memory for the KV cache.")
        logger.info("PyTorch Metal KV cache budget: %d bytes", cache_bytes)
        return cache_bytes

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.initialize_kv_cache(kv_cache_config)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # Execution is eager; there are no graphs or compiled kernels to prepare.
        torch.mps.synchronize()
        set_random_seed(self.model_config.seed)
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        with set_current_vllm_config(self.vllm_config):
            output = self.model_runner.execute_model(scheduler_output)
        assert not isinstance(output, IntermediateTensors), "Expected a single stage."
        return output

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        with set_current_vllm_config(self.vllm_config):
            sample = cast(
                Callable[
                    [GrammarOutput | None],
                    ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors,
                ],
                self.model_runner.sample_tokens,
            )
            output = sample(grammar_output)
        assert not isinstance(output, IntermediateTensors), "Expected a single stage."
        return output

    def take_draft_token_ids(self) -> None:
        return None

    def update_max_model_len(self, max_model_len: int) -> None:
        self.model_config.max_model_len = max_model_len
        self.model_runner.update_max_model_len(max_model_len)

    def shutdown(self) -> None:
        if getattr(self, "model_runner", None) is not None:
            torch.mps.synchronize()
            self.model_runner.shutdown()
            del self.model_runner
            torch.mps.empty_cache()
