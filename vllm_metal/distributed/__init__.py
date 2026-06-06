# SPDX-License-Identifier: Apache-2.0
"""Distributed primitives for vLLM Metal (pipeline parallelism)."""

from vllm_metal.distributed.pipeline import (
    PipelineGroup,
    apply_pipeline_split,
    pipeline_recv,
    pipeline_send,
)

__all__ = [
    "PipelineGroup",
    "apply_pipeline_split",
    "pipeline_recv",
    "pipeline_send",
]
