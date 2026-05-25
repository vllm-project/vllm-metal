# SPDX-License-Identifier: Apache-2.0
"""One-shot speech-to-text (STT) model runner for the vLLM v1 engine.

STT checkpoints (Whisper, Qwen3-ASR) run a black-box transcribe in a single
``execute_model`` call: the runtime adapter owns audio-feature extraction and
the full greedy decode, returning the transcript tokens in one shot. That
execution model shares nothing with token generation — no paged-attention KV
cache, no sampler, no iterative decode loop — so it lives in a dedicated runner
instead of branching inside :class:`MetalModelRunner`.

The worker selects this runner for STT models (see ``MetalWorker.init_device``).
Everything here implements the worker-facing runner contract for that one-shot
path only; the heavy generation/pooling machinery is intentionally absent.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from vllm_metal.stt.policy import STT_SCHED_BLOCK_BYTES, STT_SCHED_NOMINAL_HEAD_SIZE
from vllm_metal.stt.runtime import STTRuntimeAdapter
from vllm_metal.stt.serve import VLLMSTTRequestAdapter
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.model_lifecycle import load_stt_model

logger = init_logger(__name__)


class STTModelRunner:
    """Model runner for one-shot speech-to-text transcription on Metal."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device

        self.model: Any = None
        self.tokenizer: Any = None
        self._stt_runtime_adapter: STTRuntimeAdapter | None = None

        # execute_model stashes the output here; sample_tokens returns it,
        # matching the engine's execute -> sample handoff.
        self._pending_output: ModelRunnerOutput | None = None

    # ------------------------------------------------------------------
    # Load / warm-up
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the STT model and build its per-model runtime adapter."""
        model_name = get_model_download_path(self.model_config.model)
        model = load_stt_model(model_name)
        self.model = model
        self.tokenizer = None
        self._stt_runtime_adapter = model.create_runtime_adapter(model_name)

    def warm_up(self) -> None:
        """Run a dummy encode so the model JIT-compiles before serving."""
        if self.model is None:
            logger.warning("Model not loaded, skipping warm-up")
            return
        assert self._stt_runtime_adapter is not None
        logger.info("Warming up STT model...")
        self._stt_runtime_adapter.warm_up()
        logger.info("STT model warm-up complete")

    # ------------------------------------------------------------------
    # Worker task + memory-reporting contract
    # ------------------------------------------------------------------

    def supported_worker_tasks(self) -> tuple[SupportedTask, ...]:
        """STT models only serve transcription."""
        return ("transcription",)

    def validate_paged_attention_support(self) -> None:
        """No-op: STT does not run on the paged-attention path."""
        return None

    def scheduler_memory_reporting_mode(
        self, *, paged_attention_enabled: bool
    ) -> Literal["stt_nominal"]:
        """STT allocates no KV cache, so report nominal scheduler memory."""
        return "stt_nominal"

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Return a minimal placeholder spec.

        STT allocates no real KV cache; this only satisfies vLLM scheduler
        initialization, which expects at least one attention layer spec.
        """
        return {
            "layers.0.self_attn": FullAttentionSpec(
                block_size=self.cache_config.block_size,
                num_kv_heads=1,
                head_size=STT_SCHED_NOMINAL_HEAD_SIZE,
                dtype=torch.float16,
            ),
        }

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Accept the engine KV cache config for API compatibility (no cache)."""
        logger.info(
            "KV cache config received: %d blocks (STT uses no KV cache)",
            kv_cache_config.num_blocks,
        )

    def get_cache_block_size_bytes(self) -> int:
        """Minimal block size; STT holds no real KV cache."""
        return STT_SCHED_BLOCK_BYTES

    # ------------------------------------------------------------------
    # Multimodal / draft hooks (STT has neither)
    # ------------------------------------------------------------------

    def reset_mm_cache(self) -> None:
        """No-op: STT keeps no profiling-time multimodal cache."""
        return None

    def reset_encoder_cache(self) -> None:
        """No-op: STT keeps no cached encoder outputs."""
        return None

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """STT does not run speculative decoding."""
        return None

    # ------------------------------------------------------------------
    # Execution (one-shot transcribe)
    # ------------------------------------------------------------------

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """Transcribe every new request in the batch in a single shot."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self._execute_stt(scheduler_output)

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | None:
        """Return the transcript built by ``execute_model``.

        STT produces all tokens during ``execute_model`` and stashes them in
        ``_pending_output``; this simply hands them back on the engine's
        follow-up call.
        """
        output = self._pending_output
        self._pending_output = None
        return output

    def _execute_stt(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """Execute STT inference for all new requests in the batch.

        Raises:
            ValueError: If a request uses non-greedy sampling params.
        """
        assert self._stt_runtime_adapter is not None

        req_ids: list[str] = []
        req_id_to_index: dict[str, int] = {}
        sampled_tokens: list[list[int]] = []

        eot_token = self._stt_runtime_adapter.eot_token

        for new_req in scheduler_output.scheduled_new_reqs:
            stt_request = VLLMSTTRequestAdapter.from_vllm_request(new_req)
            sampling_params = new_req.sampling_params or SamplingParams()

            # Only greedy decoding is supported for STT
            if sampling_params.temperature > 0:
                raise ValueError(
                    "STT models only support greedy decoding (temperature=0). "
                    f"Got temperature={sampling_params.temperature}"
                )

            audio_features = self._stt_runtime_adapter.extract_audio_features(
                stt_request.input_features
            )
            tokens = self._stt_runtime_adapter.decode_tokens(
                audio_features, list(stt_request.prompt_token_ids)
            )

            req_ids.append(stt_request.req_id)
            req_id_to_index[stt_request.req_id] = len(req_ids) - 1
            sampled_tokens.append(tokens)

        # Handle cached requests: STT processes everything in one shot,
        # so any "cached" (decode-phase) request just gets an EOT to finish.
        cached_req_ids = list(scheduler_output.scheduled_cached_reqs.req_ids)
        for req_id in cached_req_ids:
            req_ids.append(req_id)
            req_id_to_index[req_id] = len(req_ids) - 1
            sampled_tokens.append([eot_token])

        if not req_ids:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        self._pending_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_tokens,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
        )
        return None
