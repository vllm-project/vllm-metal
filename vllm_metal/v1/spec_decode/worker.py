# SPDX-License-Identifier: Apache-2.0
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.worker_base import WorkerBase

from vllm_metal.v1.spec_decode.proposer import MetalDraftProposer
from vllm_metal.v1.spec_decode.rejection_sampler import MetalRejectionSampler
from vllm_metal.v1.worker import MetalWorker

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
            **kwargs,
        )

        self.proposer: MetalDraftProposer | None = None
        self.sampler: MetalRejectionSampler | None = None

        # Stashed state for asynchronous speculative verification
        self._pending_spec_data: dict[str, Any] | None = None

        logger.info("MetalSpecDecodeWorker initialized.")

    def init_device(self) -> None:
        self.target_worker.init_device()
        # Initialize speculative components after device is ready
        self.proposer = MetalDraftProposer(self.vllm_config)
        self.sampler = MetalRejectionSampler(
            self.vllm_config.model_config.get_vocab_size()
        )

    def load_model(self) -> None:
        self.target_worker.load_model()
        if self.proposer:
            self.proposer.load_model()

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        spec_metadata: SpecDecodeMetadata | None = None,
    ) -> ModelRunnerOutput | None:
        """Intercepts execution to perform speculative decoding steps."""

        # 1. Check if we have any speculative work to do
        # If there are no scheduled speculative tokens, fall back to target worker
        if not scheduler_output.scheduled_spec_decode_tokens:
            return self.target_worker.execute_model(scheduler_output)

        # 2. Draft Phase
        # generate guesses using proposer
        assert self.vllm_config.speculative_config is not None
        k = self.vllm_config.speculative_config.num_speculative_tokens
        input_ids = self.target_worker._get_input_ids(scheduler_output)

        assert self.proposer is not None
        draft_tokens_batch, draft_logits_batch = self.proposer.propose(input_ids, k=k)
        self._pending_spec_data = {
            "draft_tokens_batch": draft_tokens_batch,
            "draft_logits_batch": draft_logits_batch,
            "k": k,
        }

        # widen the pipe
        if scheduler_output.scheduled_new_reqs:
            first_req = scheduler_output.scheduled_new_reqs[0]
            req_id = first_req.req_id
            draft_list = draft_tokens_batch[0].tolist()

            # extend the prompt so the Scorer actually see the guesses
            if first_req.prompt_token_ids is not None:
                first_req.prompt_token_ids.extend(draft_list)

            # update the accounting so the runner knows the sequence grow
            scheduler_output.num_scheduled_tokens[req_id] += len(draft_list)
            scheduler_output.total_num_scheduled_tokens += len(draft_list)

        # 3. Create SpecDecodeMetadata
        if spec_metadata is None:
            spec_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids=draft_tokens_batch.tolist(),
                device=self.target_worker.device,
            )

        # 4. Verification Phase (Call target worker with widened logit extraction)
        output = self.target_worker.execute_model(scheduler_output, spec_metadata)

        # 5. Handle Synchronous Path (Non-Paged)
        if output is not None:
            output = self._verify_and_correct(output)

        return output

    def _verify_and_correct(self, output: ModelRunnerOutput) -> ModelRunnerOutput:
        """Apply rejection sampling and KV cache rewind to the model output."""
        logger.debug(
            "_verify_and_correct called. pending_spec_data: %s",
            self._pending_spec_data is not None,
        )
        if self._pending_spec_data is None:
            return output

        spec_data = self._pending_spec_data
        self._pending_spec_data = None

        logger.debug(
            "output has verification_logits: %s", hasattr(output, "verification_logits")
        )
        if hasattr(output, "verification_logits"):
            # TODO: handle 1st request in the batch for simplicity for now
            req_id = output.req_ids[0]
            logger.debug("req_id: %s", req_id)
            if req_id in output.verification_logits:
                target_logits = output.verification_logits[req_id]  # (k+1, vocab)
                logger.debug("target_logits shape: %s", target_logits.shape)

                assert self.sampler is not None
                # vectorized parallel sampling with residual fix
                num_accepted, bonus_token = self.sampler.sample(
                    target_logits=target_logits,
                    draft_logits=spec_data["draft_logits_batch"][0],
                    draft_tokens=spec_data["draft_tokens_batch"][0],
                )
                logger.debug(
                    "Sampler returned num_accepted: %d, bonus_token: %d",
                    num_accepted,
                    bonus_token,
                )

                # 6. Rewind phase: truncate kv cache and commit the bonus token
                # We must remove:
                # - The rejected draft tokens (k - num_accepted)
                # - The extra token produced by the Scorer (+ 1)
                num_to_remove = spec_data["k"] - num_accepted + 1
                logger.debug(
                    "num_to_remove: %d, bonus_token: %d", num_to_remove, bonus_token
                )

                # tell runner to forget the invalid tokens and add the bonus token
                self.target_worker.model_runner.truncate_cache(
                    req_id, num_to_remove, bonus_token=bonus_token
                )

                # 7. Update output with the verified next token
                output.sampled_token_ids[0] = [bonus_token]
                logger.debug(
                    "output.sampled_token_ids[0] updated to: %s",
                    output.sampled_token_ids[0],
                )
            else:
                logger.debug(
                    "req_id %s not in verification_logits keys: %s",
                    req_id,
                    list(output.verification_logits.keys()),
                )

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

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | None:
        output = self.target_worker.sample_tokens(grammar_output)
        if output is not None:
            output = self._verify_and_correct(output)
        return output

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
