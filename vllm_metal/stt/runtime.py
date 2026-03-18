# SPDX-License-Identifier: Apache-2.0
"""STT runtime adapter contract used by the vLLM runner.

The vLLM runner delegates STT execution to model-owned runtime adapters under
`stt/<model>/adapter.py` so shared code does not accumulate per-model branches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx
import torch


class STTRuntimeAdapter(ABC):
    """Model-owned bridge between vLLM STT inputs and per-model STT execution.

    Concrete implementations live under `stt/<model>/adapter.py` and own:
    - input_features normalization to the model's expected encoder input shape
    - decoding strategy (prompt handling, token extraction, EOT selection)
    """

    def __init__(self, model: Any, model_path: str) -> None:
        self.model = model
        self._model_path = model_path

    @staticmethod
    def _to_mx_float16(value: Any) -> mx.array:
        """Convert common multimodal payload types into ``mx.float16``."""
        if isinstance(value, torch.Tensor):
            from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx

            return torch_to_mlx(value).astype(mx.float16)

        if not isinstance(value, mx.array):
            return mx.array(value, dtype=mx.float16)

        if value.dtype != mx.float16:
            return value.astype(mx.float16)

        return value

    @property
    @abstractmethod
    def transcriber(self) -> Any: ...

    @property
    @abstractmethod
    def eot_token(self) -> int: ...

    @abstractmethod
    def extract_audio_features(self, input_features: Any) -> mx.array: ...

    @abstractmethod
    def decode_tokens(
        self,
        audio_features: mx.array,
        prompt_token_ids: list[int],
    ) -> list[int]: ...

    @abstractmethod
    def warm_up(self) -> None:
        """Run a dummy encode to JIT-compile the model at startup."""
        ...
