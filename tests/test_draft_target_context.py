# SPDX-License-Identifier: Apache-2.0
"""A shorter draft model keeps the target context on Metal."""

import json

from vllm.engine.arg_utils import EngineArgs

from vllm_metal.config import reset_config
from vllm_metal.patches.draft_context_limit import register

_TINY_QWEN3_CONFIG = {
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


def test_shorter_draft_model_keeps_target_context(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_TINY_QWEN3_CONFIG))
    register()
    reset_config()
    try:
        vllm_config = EngineArgs(
            model=str(tmp_path),
            max_model_len=128,
            skip_tokenizer_init=True,
            speculative_config={
                "method": "draft_model",
                "model": str(tmp_path),
                "max_model_len": 64,
                "num_speculative_tokens": 2,
            },
        ).create_engine_config()
    finally:
        reset_config()

    assert vllm_config.model_config.max_model_len == 128
    assert vllm_config.speculative_config.draft_model_config.max_model_len == 64
