# SPDX-License-Identifier: Apache-2.0
"""vLLM v1 worker for the ggml backend (``VLLM_METAL_BACKEND=ggml``)."""

from __future__ import annotations

import gc
import time
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from vllm_metal.ggml.model_runner import GGMLModelRunner

logger = init_logger(__name__)

# The engine owns its KV layout; vLLM only needs a stable label.
_KV_CACHE_LAYOUT = "NHD"


class GGMLWorker(WorkerBase):
    model_runner: GGMLModelRunner  # type: ignore[assignment]

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.parallel_config.disable_custom_all_reduce = True

    def init_device(self) -> None:
        self.device = torch.device("cpu")
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
        self.model_runner = GGMLModelRunner(self.vllm_config)

    def load_model(self) -> None:
        self.model_runner.load_model()

    def determine_available_memory(self) -> int:
        return self.model_runner.determine_available_memory()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def get_supported_kv_cache_layouts(self) -> list[str]:
        return [_KV_CACHE_LAYOUT]

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        set_random_seed(self.model_config.seed)
        start = time.perf_counter()
        self.model_runner.warm_up()
        return CompilationTimes(language_model=time.perf_counter() - start, encoder=0.0)

    def synchronize_device(self) -> None:
        # Engine steps are synchronous.
        pass

    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | None:
        return self.model_runner.sample_tokens(grammar_output)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def get_model(self) -> Any:
        return self.model_runner.model

    def update_max_model_len(self, max_model_len: int) -> None:
        self.model_config.max_model_len = max_model_len

    def get_cache_block_size_bytes(self) -> int:
        return self.model_runner.get_cache_block_size_bytes()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("LoRA is not supported by the ggml backend")

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.supported_worker_tasks()  # type: ignore[return-value]

    def sleep(self, level: int = 1) -> None:
        logger.warning("Sleep mode is not supported on Metal, ignoring")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("Sleep mode is not supported on Metal, ignoring")

    def check_health(self) -> None:
        return

    def shutdown(self) -> None:
        if getattr(self, "model_runner", None) is not None:
            self.model_runner.engine = None
            self.model_runner = None  # type: ignore[assignment]
        gc.collect()
