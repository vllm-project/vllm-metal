# SPDX-License-Identifier: Apache-2.0
"""Typed pooling backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import mlx.core as mx
import torch
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_metal.attention.context import OffsetCache

EMBED_TASK: PoolingTask = "embed"
CLASSIFY_TASK: PoolingTask = "classify"
PoolingExecutionKind: TypeAlias = Literal["decoder", "encoder"]


@dataclass(frozen=True, slots=True)
class PoolingCapabilities:
    execution_kind: PoolingExecutionKind
    requires_paged_attention: bool
    uses_kv_cache: bool
    supports_chunked_requests: bool


@dataclass(frozen=True, slots=True)
class DecoderPoolingSpan:
    start_row: int
    num_tokens: int
    is_complete: bool
    pooling_params: PoolingParams


@dataclass(frozen=True, slots=True)
class DecoderPoolingBatch:
    spans: tuple[DecoderPoolingSpan, ...]


@dataclass(frozen=True, slots=True)
class EncoderPoolingOutput:
    req_id: str
    pooler_output: torch.Tensor


class PoolingBackend(Protocol):
    capabilities: PoolingCapabilities

    def profile_forward(self, input_ids: mx.array) -> mx.array: ...

    def supported_tasks(self) -> tuple[PoolingTask, ...]: ...

    def validate_params(self, pooling_params: PoolingParams) -> None: ...


class DecoderPooler(Protocol):
    """Task-specific strategy used by a decoder pooling backend."""

    task: PoolingTask

    def is_supported(self) -> bool: ...

    def pool_one(
        self,
        hidden_states: mx.array,
        span: DecoderPoolingSpan,
    ) -> torch.Tensor: ...


class DecoderPoolingBackend(PoolingBackend, Protocol):
    def forward_packed(
        self,
        input_ids: mx.array,
        offset_caches: list[OffsetCache] | None,
    ) -> mx.array: ...

    def pool_packed(
        self,
        hidden_states: mx.array,
        batch: DecoderPoolingBatch,
    ) -> tuple[torch.Tensor | None, ...]: ...


class EncoderPoolingBackend(PoolingBackend, Protocol):
    def forward_padded(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> mx.array: ...

    def pool_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
        model_config: object,
    ) -> tuple[EncoderPoolingOutput, ...]: ...


ExecutablePoolingBackend: TypeAlias = DecoderPoolingBackend | EncoderPoolingBackend
