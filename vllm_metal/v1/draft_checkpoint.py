# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Z Lab
# Copyright (c) 2026 vLLM Metal contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Shared checkpoint validation and loading for DFlash and DSpark.

Extracted from the MIT-licensed loader in dflash.py.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from safetensors import safe_open

# Full-attention Qwen3 semantics shared by DFlash and DSpark checkpoints.
COMMON_DRAFT_OPTIONS: Mapping[str, Any] = {
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "use_sliding_window": False,
    "sliding_window": None,
    "rope_scaling": None,
    "quantization": None,
    "quantization_config": None,
    "is_causal": False,
    "input_embedding_scale": 1.0,
    "output_multiplier": 1.0,
    "final_logit_softcapping": None,
}


def load_draft_weights(
    model: nn.Module,
    path: Path,
    *,
    model_name: str,
    strip_prefix: str = "",
) -> None:
    """Load one unsharded checkpoint, preserving its uniform floating precision.

    Config and target compatibility must be validated by the caller first.
    ``strip_prefix`` removes a model-side namespace when deriving checkpoint names;
    DSpark's shared backbone is nested under ``backbone.`` only in the model.
    """
    files = sorted(path.glob("*.safetensors"))
    if len(files) != 1 or (path / "model.safetensors.index.json").exists():
        raise ValueError(
            f"{model_name} qualification requires one unsharded safetensors file"
        )
    parameters = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    names = {name.removeprefix(strip_prefix): name for name, _ in parameters}
    if len(names) != len(parameters):
        raise ValueError(f"{model_name} checkpoint tensor names collide after renaming")
    expected = {
        name.removeprefix(strip_prefix): tuple(t.shape) for name, t in parameters
    }
    # Check headers before evaluating any checkpoint or randomly initialized weight.
    with safe_open(files[0], framework="numpy") as stream:
        if set(stream.keys()) != expected.keys():
            raise ValueError(
                f"{model_name} checkpoint tensor names do not match the model"
            )
        dtypes = set()
        for name, shape in expected.items():
            tensor = stream.get_slice(name)
            dtypes.add(tensor.get_dtype())
            if tuple(tensor.get_shape()) != shape or tensor.get_dtype() not in {
                "F16",
                "BF16",
                "F32",
            }:
                raise ValueError(f"Invalid {model_name} tensor shape or dtype: {name}")
        if len(dtypes) != 1:
            raise ValueError(
                f"{model_name} qualification requires uniform checkpoint precision"
            )
    weights = cast(dict[str, mx.array], mx.load(files[0]))
    model.load_weights(
        [(names[name], value) for name, value in weights.items()], strict=True
    )
    model.eval()
    mx.eval(model.parameters())
    parameters = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    if not all(bool(mx.all(mx.isfinite(t))) for _, t in parameters):
        raise ValueError(f"{model_name} checkpoint contains non-finite weights")
