# SPDX-License-Identifier: Apache-2.0
from typing import Any, List, Optional, Set, Tuple
import torch
import mlx.core as mx

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase
from vllm.logger import init_logger
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_metal.v1.worker import MetalWorker
from vllm_metal.v1.spec_decode.proposer import MetalDraftProposer
from vllm_metal.v1.spec_decode.rejection_sampler import MetalRejectionSampler

logger = init_logger(__name__)

class MetalSpecDecodeWorker(WorkerBase):
    """Orchestrator for speculative decoding on Metal.
    
    Wraps a target MetalWorker and manages a draft model.
    Following the vLLM V1 pattern, this worker handles the 
    Draft -> Scorer -> Sampler loop.
    """
    
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
        
        # Initialize the actual target worker (Don't reinvent the wheel)
        self.target_worker = MetalWorker(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
            **kwargs
        )
        
        self.speculative_config = vllm_config.speculative_config
        self.proposer: Optional[MetalDraftProposer] = None
        self.sampler: Optional[MetalRejectionSampler] = None
        
        logger.info("MetalSpecDecodeWorker initialized.")

    def init_device(self) -> None:
        self.target_worker.init_device()
        # Initialize speculative components after device is ready
        self.proposer = MetalDraftProposer(self.vllm_config, self.speculative_config)
        self.sampler = MetalRejectionSampler(self.vllm_config.model_config.get_vocab_size())

    def load_model(self) -> None:
        self.target_worker.load_model()
        # TODO: Load the draft model via proposer.load_model()

    def execute_model(
        self, 
        scheduler_output: SchedulerOutput
    ) -> Optional[ModelRunnerOutput]:
        """Intercepts execution to perform speculative decoding steps."""
        
        # 1. Check if we have any speculative work to do
        # If there are no scheduled speculative tokens, fall back to target worker
        if not scheduler_output.scheduled_spec_decode_tokens:
             return self.target_worker.execute_model(scheduler_output)

        # 2. Draft Phase
        # req_id_to_draft_tokens = self.proposer.propose(scheduler_output)
        
        # 3. Create SpecDecodeMetadata
        # spec_metadata = SpecDecodeMetadata.make_dummy(...) 
        spec_metadata: Optional[SpecDecodeMetadata] = None
        
        # 4. Verification Phase (Call target worker with widened logit extraction)
        output = self.target_worker.execute_model(scheduler_output, spec_metadata)
        
        # 5. Rejection Phase
        # m_accepted = self.sampler.sample(target_logits, draft_logits, draft_tokens)
        
        return output

    # --- Delegated Methods (Reusing MetalWorker implementation) ---
    
    def determine_available_memory(self) -> int:
        return self.target_worker.determine_available_memory()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.target_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def initialize_from_config(self, kv_cache_config) -> None:
        self.target_worker.initialize_from_config(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        self.target_worker.compile_or_warm_up_model()

    def sample_tokens(self, grammar_output: Optional[GrammarOutput]) -> Optional[ModelRunnerOutput]:
        return self.target_worker.sample_tokens(grammar_output)

    def get_model(self) -> Any:
        return self.target_worker.get_model()

    def get_cache_block_size_bytes(self) -> int:
        return self.target_worker.get_cache_block_size_bytes()

    def get_kv_cache_spec(self) -> Any:
        return self.target_worker.get_kv_cache_spec()

    def update_max_model_len(self, max_model_len: int) -> None:
        self.target_worker.update_max_model_len(max_model_len)

    def shutdown(self) -> None:
        self.target_worker.shutdown()
