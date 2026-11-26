# SPDX-License-Identifier: Apache-2.0
"""Tiny checkpoint configs for tests that build real engine configs offline."""

TINY_QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 256,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 256,
    "max_position_embeddings": 4096,
}
