# SPDX-License-Identifier: Apache-2.0
"""Decision 3.0 (d3) code-readout ``classify`` pooling behavior.

A d3 checkpoint is a Qwen3.5 backbone plus ``readout.safetensors``: one row per
answer code. A prompt is scored by reading the last prompt token and projecting
it with the readout, which yields one logit per code. ``/v1/systemone`` turns
those logits into probabilities over the codes of a single question, so it asks
for them unnormalized; a request that sets ``use_activation`` gets upstream's
``PoolerClassify`` activation over the whole code space instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import torch
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask

from vllm_metal.tensor_bridge import mlx_to_torch
from vllm_metal.v1.pooling.contract import CLASSIFY_TASK, DecoderPoolingSpan
from vllm_metal.v1.pooling.validation import PoolingConfigView

D3_ARCH = "D3ForSequenceClassification"
D3_READOUT_FILE = "readout.safetensors"
D3_READOUT_KEY = "weight"
D3_TASKS: tuple[PoolingTask | None, ...] = (None, CLASSIFY_TASK)


@dataclass(frozen=True, slots=True)
class D3ConfigView:
    config: PoolingConfigView

    @property
    def decision_config(self) -> dict[str, Any]:
        value = getattr(self.config.hf_config, "decision_config", None)
        return value if isinstance(value, dict) else {}

    @property
    def attention_mode(self) -> str | None:
        mode = self.decision_config.get("attention_mode")
        return str(mode) if mode is not None else None

    @property
    def is_supported(self) -> bool:
        return (
            self.config.is_text_only
            and self.config.task in D3_TASKS
            and self.config.uses_last_pooling
            and not self.config.chunked_processing_enabled
            and D3_ARCH in self.config.architectures
        )


class D3ReadoutPooler:
    """Pool d3 hidden states as one logit per answer code."""

    task: PoolingTask = CLASSIFY_TASK

    def __init__(self, model_config: Any, checkpoint_dir: Path | None = None) -> None:
        self.config = PoolingConfigView(model_config)
        self.d3_config = D3ConfigView(self.config)
        # Only a d3 checkpoint reads the readout; every other decoder pooling
        # model must not touch the filesystem for it.
        self.readout = (
            self._readout(checkpoint_dir) if self.d3_config.is_supported else None
        )

    def is_supported(self) -> bool:
        return self.d3_config.is_supported and self.readout is not None

    def pool_one(
        self,
        hidden_states: mx.array,
        span: DecoderPoolingSpan,
    ) -> torch.Tensor:
        assert self.readout is not None

        token_index = span.start_row + span.num_tokens - 1
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError(
                "Metal d3 pooling expected hidden states with shape "
                f"[1, tokens, hidden], got {hidden_states.shape} for model="
                f"{self.config.label}."
            )
        if token_index < 0 or token_index >= hidden_states.shape[1]:
            raise ValueError(
                f"Metal d3 pooling token index {token_index} is outside hidden "
                f"state shape {hidden_states.shape} for model={self.config.label}."
            )

        vector = hidden_states[0, token_index, :].astype(mx.float32)
        logits = vector @ self.readout.T
        codes = self.d3_config.decision_config.get("codes")
        if len(logits.shape) != 1 or (codes and logits.shape[0] != len(codes)):
            raise ValueError(
                "Metal d3 pooling expected one logit per answer code, got "
                f"{logits.shape} for {len(codes or [])} codes and model="
                f"{self.config.label}."
            )

        tensor = mlx_to_torch(self._activate(logits, span.pooling_params), device="cpu")
        return tensor.detach().clone()

    def _activate(self, logits: mx.array, pooling_params: PoolingParams) -> mx.array:
        """Apply upstream ``PoolerClassify`` when the request asks for it."""
        use_activation = pooling_params.use_activation
        if use_activation is None:
            use_activation = self.config.pooler_config.use_activation is not False
        if not use_activation:
            return logits
        if logits.shape[0] < 2:
            return mx.sigmoid(logits)
        return mx.softmax(logits, axis=-1)

    def _readout(self, checkpoint_dir: Path | None = None) -> mx.array | None:
        from vllm_metal.v1.model_adapter import _local_checkpoint_dir

        model_path = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else _local_checkpoint_dir(self.config.model_config)
        )
        if model_path is None:
            return None
        readout_path = model_path / D3_READOUT_FILE
        if not readout_path.exists():
            return None

        weights = cast(dict[str, mx.array], mx.load(str(readout_path)))
        if D3_READOUT_KEY not in weights:
            raise ValueError(
                f"Metal d3 readout {readout_path} has no {D3_READOUT_KEY!r} tensor; "
                f"got {sorted(weights)} for model={self.config.label}."
            )
        readout = weights[D3_READOUT_KEY].astype(mx.float32)
        codes = self.d3_config.decision_config.get("codes") or []
        if readout.ndim != 2 or (codes and readout.shape[0] != len(codes)):
            raise ValueError(
                "Metal d3 readout expected one row per answer code, got "
                f"{readout.shape} for {len(codes)} codes and model="
                f"{self.config.label}."
            )
        return readout
