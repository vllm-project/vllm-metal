# SPDX-License-Identifier: Apache-2.0
"""One-shot speech-to-text (STT) model runner for the vLLM v1 engine.

STT checkpoints (Whisper, Qwen3-ASR) run a black-box transcribe in a single
``execute_model`` call: the runtime adapter owns audio-feature extraction and
the whole decode, returning the transcript tokens in one shot. That execution
model shares nothing with token generation — no paged-attention KV cache, no
scheduled decode loop — so it lives in a dedicated runner instead of branching
inside :class:`MetalModelRunner`.

The worker selects this runner for STT models (see ``MetalWorker.init_device``).
Everything here implements the worker-facing runner contract for that one-shot
path only; the heavy generation/pooling machinery is intentionally absent.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.outputs import DraftTokenIds, LogprobsLists, ModelRunnerOutput
from vllm.v1.sample.sampler import Sampler

from vllm_metal.stt.loader import resolve_model_path
from vllm_metal.stt.policy import STT_SCHED_BLOCK_BYTES, STT_SCHED_NOMINAL_HEAD_SIZE
from vllm_metal.stt.runtime import STTRuntimeAdapter
from vllm_metal.stt.sampling import STTSampling, TokenLogprob
from vllm_metal.stt.serve import VLLMSTTRequestAdapter
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
        self._stt_runtime_adapter: STTRuntimeAdapter | None = None
        self._sampler = Sampler()

        # execute_model stashes the output here; sample_tokens returns it,
        # matching the engine's execute -> sample handoff.
        self._pending_output: ModelRunnerOutput | None = None

    # ------------------------------------------------------------------
    # Load / warm-up
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the STT model and build its per-model runtime adapter."""
        model_name = str(
            resolve_model_path(
                self.model_config.model, revision=self.model_config.revision
            )
        )
        model = load_stt_model(model_name, self.model_config.dtype)
        self.model = model
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

    def scheduler_memory_reporting_mode(self) -> Literal["stt_nominal"]:
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

    def get_draft_model_stats(self) -> dict[str, int] | None:
        """STT has no draft model, so there are no stats to report."""
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
        """Execute STT inference for all new requests in the batch."""
        assert self._stt_runtime_adapter is not None

        req_ids: list[str] = []
        req_id_to_index: dict[str, int] = {}
        sampled_tokens: list[list[int]] = []
        # Per request: the scored tokens, or None when it asked for no logprobs.
        sampled_logprobs: list[list[TokenLogprob] | None] = []

        eot_token = self._stt_runtime_adapter.eot_token

        for new_req in scheduler_output.scheduled_new_reqs:
            stt_request = VLLMSTTRequestAdapter.from_vllm_request(new_req)
            params = stt_request.sampling_params
            # vLLM's verbose_json path sets logprobs to build segments.
            logprobs: list[TokenLogprob] | None = (
                [] if params is not None and params.logprobs is not None else None
            )

            audio_features = self._stt_runtime_adapter.extract_audio_features(
                stt_request.input_features
            )
            tokens = self._stt_runtime_adapter.decode_tokens(
                audio_features,
                list(stt_request.prompt_token_ids),
                STTSampling.from_request(params, self._sampler),
                logprobs_out=logprobs,
            )

            req_ids.append(stt_request.req_id)
            req_id_to_index[stt_request.req_id] = len(req_ids) - 1
            sampled_tokens.append(tokens)
            sampled_logprobs.append(logprobs)

        # Handle cached requests: STT processes everything in one shot,
        # so any "cached" (decode-phase) request just gets an EOT to finish.
        cached_req_ids = list(scheduler_output.scheduled_cached_reqs.req_ids)
        for req_id in cached_req_ids:
            req_ids.append(req_id)
            req_id_to_index[req_id] = len(req_ids) - 1
            sampled_tokens.append([eot_token])
            sampled_logprobs.append(None)

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
            logprobs=_pack_sample_logprobs(sampled_tokens, sampled_logprobs),
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
        )
        return None


def _pack_sample_logprobs(
    sampled_tokens: list[list[int]],
    sampled_logprobs: list[list[TokenLogprob] | None],
) -> LogprobsLists | None:
    """Pack per-request token logprobs into vLLM's flat sample-logprob rows.

    One row per sampled token, sampled token in column 0, as vLLM's output
    processor reads it; ``cu_num_generated_tokens`` marks where each request's
    rows start. Tokens left unscored, and every token of a request that asked
    for no logprobs (the scheduler never reads those rows), get 0.0 at rank 1.
    """
    if all(logprobs is None for logprobs in sampled_logprobs):
        return None
    token_ids: list[int] = []
    values: list[float] = []
    ranks: list[int] = []
    starts: list[int] = []
    for tokens, logprobs in zip(sampled_tokens, sampled_logprobs, strict=True):
        starts.append(len(token_ids))
        scored = logprobs or []
        for pos, token in enumerate(tokens):
            logprob, rank = scored[pos] if pos < len(scored) else (0.0, 1)
            token_ids.append(token)
            values.append(logprob)
            ranks.append(rank)
    return LogprobsLists(
        logprob_token_ids=np.array(token_ids, dtype=np.int32).reshape(-1, 1),
        logprobs=np.array(values, dtype=np.float32).reshape(-1, 1),
        sampled_token_ranks=np.array(ranks, dtype=np.int32),
        cu_num_generated_tokens=starts,
    )
