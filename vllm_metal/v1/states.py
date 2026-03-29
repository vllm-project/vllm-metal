# SPDX-License-Identifier: Apache-2.0
"""Request state and cache type aliases for Metal model runner."""

from dataclasses import dataclass, field
from typing import TypeAlias

import torch
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from vllm.sampling_params import SamplingParams

# Type alias for any per-layer cache type supported by the model.
#
# Notes:
# - Some models (e.g. gpt_oss) use `RotatingKVCache` for sliding-window attention.
# - Hybrid models use `ArraysCache` for non-attention state.
AnyCache: TypeAlias = KVCache | RotatingKVCache | ArraysCache


@dataclass
class RequestState:
    """State for an ongoing request with KV cache."""

    token_ids: list[int]
    # Length of the original prompt (prefix) within `token_ids`.
    # vLLM applies repetition penalties to both prompt+output tokens, but applies
    # presence/frequency penalties only to generated (output) tokens.
    prompt_len: int
    cache: list[AnyCache]  # Per-layer caches (KVCache, RotatingKVCache, or ArraysCache)
    sampling_params: SamplingParams  # Sampling parameters for this request
    generator: torch.Generator | None = None
    generated_tokens: int = 0
    block_ids: list[int] = field(
        default_factory=list
    )  # Scheduler-assigned paged KV blocks
