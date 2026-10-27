# SPDX-License-Identifier: Apache-2.0
"""Weight dtype policy shared by native text and draft model loading."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path


def cast_model_weights(model: nn.Module, dtype: mx.Dtype) -> None:
    """Cast floating weights lazily, respecting mlx-lm's precision exceptions."""
    cast_predicate = getattr(model, "cast_predicate", lambda _: True)

    def cast(path: str, array: mx.array) -> mx.array:
        if cast_predicate(path) and mx.issubdtype(array.dtype, mx.floating):
            return array.astype(dtype)
        return array

    model.update(tree_map_with_path(cast, model.parameters()))
