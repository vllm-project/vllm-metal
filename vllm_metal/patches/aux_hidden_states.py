# SPDX-License-Identifier: Apache-2.0
"""Temporary MLX-LM bridge for model-owned auxiliary hidden-state outputs.

Keep this bridge until MLX-LM exposes native capture. It observes known decoder
return contracts while the model retains its own embedding, mask and cache flow.
Layer 0 is the first decoder's input; layer i > 0 is decoder i-1's output before
the backbone's final norm. Captures are returned as graph outputs, never retained
on the model between calls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.attention.patching import find_layers

_Output = TypeVar("_Output")


def _tensor_output(output: mx.array) -> mx.array:
    return output


def _gemma_output(output: tuple[mx.array, Any, Any]) -> mx.array:
    # Gemma4's decoder returns (completed hidden state, shared KV, offset).
    hidden, _, _ = output
    return hidden


class _CaptureLayer(nn.Module):
    def __init__(self, inner: nn.Module, observe: Callable) -> None:
        super().__init__()
        # Mirror the original entries without introducing an "inner." prefix
        # or copying weights. The wrapper is read-only and call-scoped; loading,
        # quantization and module replacement always see the original model.
        dict.update(self, inner)
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_observe", observe)
        self._no_grad = inner._no_grad
        self._training = inner.training

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, x: mx.array, *args: Any, **kwargs: Any) -> Any:
        output = self._inner(x, *args, **kwargs)
        self._observe(x, output)
        return output


class AuxHiddenStateCapture:
    """Observe selected layers of one model instance through its native forward."""

    def __init__(self, model: nn.Module, layer_ids: tuple[int, ...]) -> None:
        from mlx_lm.models.gemma4_text import DecoderLayer as GemmaLayer
        from mlx_lm.models.llama import TransformerBlock as LlamaLayer
        from mlx_lm.models.qwen3 import TransformerBlock as QwenLayer

        contracts = {
            LlamaLayer: _tensor_output,
            QwenLayer: _tensor_output,
            GemmaLayer: _gemma_output,
        }
        layers = find_layers(model)
        if not layer_ids or any(i < 0 or i > len(layers) for i in layer_ids):
            raise ValueError(f"Invalid auxiliary layer IDs: {layer_ids}")
        self.layer_ids = layer_ids
        self._layers = layers
        self._extractors = {}
        for index in sorted({max(i - 1, 0) for i in layer_ids}):
            layer_type = type(layers[index])
            if layer_type not in contracts:
                raise NotImplementedError(
                    f"Auxiliary capture is not implemented for {layer_type.__module__}."
                    f"{layer_type.__name__}"
                )
            self._extractors[index] = contracts[layer_type]

    def run(
        self, forward: Callable[..., _Output], *args: Any, **kwargs: Any
    ) -> tuple[_Output, tuple[mx.array, ...]]:
        # This entire entry point may be compiled: the returned auxiliary
        # arrays then belong to the compiled outputs and remain fresh on replay.
        captured: dict[int, mx.array] = {}
        originals = {i: self._layers[i] for i in self._extractors}

        def observer(index: int, extract: Callable) -> Callable:
            def observe(x: mx.array, output: Any) -> None:
                if index == 0 and 0 in self.layer_ids:
                    captured[0] = x
                if index + 1 in self.layer_ids:
                    captured[index + 1] = extract(output)

            return observe

        try:
            for index, extract in self._extractors.items():
                self._layers[index] = _CaptureLayer(
                    originals[index], observer(index, extract)
                )
            output = forward(*args, **kwargs)
        finally:
            for index, original in originals.items():
                self._layers[index] = original
        if any(i not in captured for i in self.layer_ids):
            raise RuntimeError(
                "Native forward bypassed auxiliary capture; compile the capture "
                "entry point rather than an already-compiled target"
            )
        return output, tuple(captured[i] for i in self.layer_ids)
