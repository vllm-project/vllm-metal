# SPDX-License-Identifier: Apache-2.0
"""Generic encoder pooling backend.

Encoder pooling runs full prompts with bidirectional attention. It does not use
decoder KV cache, paged attention, or chunked request state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import torch
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.v1.pooling.contract import (
    EMBED_TASK,
    EncoderPoolingOutput,
    PoolingCapabilities,
)
from vllm_metal.v1.pooling.validation import (
    PoolingConfigView,
    validate_pooling_request,
)

_MIN_NORM = 1e-12
_ENCODER_POOLING_TYPES = (None, "CLS", "LAST")


@dataclass(frozen=True, slots=True)
class EncoderPoolingRequest:
    req_id: str
    token_ids: tuple[int, ...]


class MetalEncoderPoolingBackend:
    """Dense encoder embedding backend for full-prompt pooling."""

    capabilities = PoolingCapabilities(
        execution_kind="encoder",
        requires_paged_attention=False,
        uses_kv_cache=False,
        supports_chunked_requests=False,
    )

    def __init__(
        self,
        config: PoolingConfigView,
        forward: Callable[[mx.array, mx.array], mx.array],
    ) -> None:
        self.config = config
        self._forward = forward

    def supported_tasks(self) -> tuple[PoolingTask, ...]:
        if self._supports_embed():
            return (EMBED_TASK,)
        return ()

    def validate_params(self, pooling_params: PoolingParams) -> None:
        if not self._supports_embed() or pooling_params.task not in (None, EMBED_TASK):
            raise NotImplementedError(
                "Metal encoder pooling supports only task='embed' "
                f"for model={self.config.label}."
            )

    def profile_forward(self, input_ids: mx.array) -> mx.array:
        attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        return self.forward_padded(input_ids, attention_mask)

    def forward_padded(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        hidden_states = self._forward(input_ids, attention_mask)
        if hidden_states.ndim != 3:
            raise ValueError(
                "Metal encoder pooling expected hidden states with shape "
                f"[batch, tokens, hidden], got {hidden_states.shape} "
                f"for model={self.config.label}."
            )
        return hidden_states

    def pool_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
        model_config: object,
    ) -> tuple[EncoderPoolingOutput, ...]:
        requests = self._requests_from_scheduler_output(
            scheduler_output,
            model_config,
        )
        return tuple(
            EncoderPoolingOutput(request.req_id, self._pool_request(request))
            for request in requests
        )

    def _pool_request(self, request: EncoderPoolingRequest) -> torch.Tensor:
        input_ids = mx.array([request.token_ids], dtype=mx.int32)
        attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        hidden_states = self.forward_padded(input_ids, attention_mask)
        return self._pool_one(hidden_states, 0, len(request.token_ids))

    def _requests_from_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
        model_config: object,
    ) -> tuple[EncoderPoolingRequest, ...]:
        self._reject_scheduler_state(scheduler_output)
        requests: list[EncoderPoolingRequest] = []
        for new_req in scheduler_output.scheduled_new_reqs:
            validate_pooling_request(
                new_req,
                model_config,
                pooling_backend=self,
                paged_attention_enabled=False,
            )
            if new_req.pooling_params is None:
                raise RuntimeError(
                    "Encoder pooling received a scheduled non-pooling request."
                )
            requests.append(
                EncoderPoolingRequest(
                    req_id=new_req.req_id,
                    token_ids=tuple(new_req.prompt_token_ids or ()),
                )
            )
        return tuple(requests)

    def _reject_scheduler_state(self, scheduler_output: SchedulerOutput) -> None:
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if cached_reqs.req_ids:
            raise NotImplementedError(
                "Metal encoder pooling does not support cached, resumed, "
                "or chunked requests yet."
            )
        if scheduler_output.preempted_req_ids:
            raise NotImplementedError(
                "Metal encoder pooling does not support preempted requests yet."
            )
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            scheduled_tokens = scheduler_output.num_scheduled_tokens[new_req.req_id]
            if new_req.num_computed_tokens != 0 or scheduled_tokens != len(token_ids):
                raise NotImplementedError(
                    "Metal encoder pooling requires full-prompt scheduling; "
                    "partial or chunked pooling requests are not supported yet."
                )

    def _supports_embed(self) -> bool:
        return (
            self.config.is_text_only
            and self.config.task in (None, EMBED_TASK)
            and self.config.sequence_pooling_type in _ENCODER_POOLING_TYPES
            and self.config.embed_activation_allowed
            and not self.config.chunked_processing_enabled
        )

    def _pool_one(
        self,
        hidden_states: mx.array,
        row: int,
        token_count: int,
    ) -> torch.Tensor:
        if token_count <= 0:
            raise ValueError("Metal encoder pooling requires at least one token.")
        token_index = (
            0 if self.config.sequence_pooling_type != "LAST" else token_count - 1
        )
        vector = hidden_states[row, token_index, :].astype(mx.float32)
        norm = mx.sqrt(mx.sum(vector * vector))
        norm = mx.maximum(norm, mx.array(_MIN_NORM, dtype=mx.float32))
        tensor = mlx_to_torch(mx.contiguous(vector / norm), device="cpu")
        return tensor.detach().clone()
