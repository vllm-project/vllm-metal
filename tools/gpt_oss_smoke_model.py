# SPDX-License-Identifier: Apache-2.0
"""Download-free GPT-OSS fixture for real JACCL pipeline validation."""

import mlx.core as mx
from mlx_lm.models.gpt_oss import Model, ModelArgs

SLIDING_WINDOW = 8
NUM_LAYERS = 8


def tiny_model() -> Model:
    """Keep the actual alternating attention and routed experts, at tiny widths."""
    mx.random.seed(20260917)
    model = Model(
        ModelArgs(
            hidden_size=64,
            num_hidden_layers=NUM_LAYERS,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_local_experts=4,
            num_experts_per_tok=2,
            vocab_size=97,
            sliding_window=SLIDING_WINDOW,
        )
    )
    mx.eval(model.parameters())
    return model


def parity_batches() -> list[tuple[str, mx.array]]:
    """Cross a window during chunked prefill and again during cached decode."""
    batches = [
        ("prefill_1", mx.array([[1, 7, 11, 23, 42, 3]], dtype=mx.int32)),
        ("prefill_2", mx.array([[9, 17, 19, 31, 5]], dtype=mx.int32)),
    ]
    batches.extend(
        (f"cached_decode_{step}", mx.array([[token]], dtype=mx.int32))
        for step, token in enumerate((8, 13, 29, 41, 53, 67, 71, 83, 89), 1)
    )
    return batches
