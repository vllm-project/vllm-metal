# SPDX-License-Identifier: Apache-2.0
"""Nanbeige config compatibility and loop-count metadata; mlx-lm runs the model.

Kept free of ``mlx`` imports so config-time callers (KV offloading, draft-model
planning, and EngineCore multiprocessing unpickling) can import it directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from transformers.configuration_utils import PretrainedConfig

_UNSUPPORTED_NANBEIGE_FLAGS = (
    "enable_double_loop_split",
    "loop_share_kv",
    "enable_depth_attention",
    "enable_hyper_connection",
    "enable_mhc",
    "qk_layernorm",
)


def _config_field(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def effective_num_loops(config: Any) -> int:
    """Return executed weight-tied loop passes for a Nanbeige config (1 otherwise).

    Non-Nanbeige configs return 1 without inspecting model-specific flags (e.g.
    StableLM 2's ``qk_layernorm`` or ESMFold's ``num_loops``). For
    ``model_type == "nanbeige"``, matches
    ``mlx_lm.models.nanbeige.ModelArgs.effective_num_loops`` and fails fast on
    Nanbeige variants that neither ``mlx_lm`` nor paged attention implements.
    """
    if config is None or _config_field(config, "model_type") != "nanbeige":
        return 1

    for flag in _UNSUPPORTED_NANBEIGE_FLAGS:
        val = _config_field(config, flag)
        if isinstance(val, (bool, int)) and bool(val):
            raise NotImplementedError(f"Nanbeige option {flag}=True is not supported.")
    emb_neighbor_num = _config_field(config, "emb_neighbor_num")
    if emb_neighbor_num is not None and not callable(emb_neighbor_num):
        raise NotImplementedError(
            "Nanbeige N-gram embeddings (emb_neighbor_num) are not supported."
        )

    loop_loss_weights = _config_field(config, "loop_loss_weights")
    if (
        isinstance(loop_loss_weights, Sequence)
        and not isinstance(loop_loss_weights, (str, bytes))
        and len(loop_loss_weights) > 0
    ):
        num_loops = len(loop_loss_weights) + 1
    else:
        raw = _config_field(config, "num_loops")
        if raw is None or callable(raw):
            return 1
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"num_loops must be an integer >= 1, got {raw!r}")
        num_loops = raw

    if num_loops < 1:
        raise ValueError(f"num_loops must be >= 1, got {num_loops}")
    return num_loops


# TODO: Remove Nanbeige fallbacks once the pinned Transformers knows it.
class NanbeigeCompatConfig(PretrainedConfig):
    """Fallback HF config for ``model_type == "nanbeige"``.

    Defined at module top level so ``vllm.LLM`` can pickle ``VllmConfig`` across
    spawned ``EngineCore`` worker processes.
    """

    model_type = "nanbeige"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        num_loops: int = 1,
        skip_loop_final_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        if (
            isinstance(num_loops, bool)
            or not isinstance(num_loops, int)
            or num_loops < 1
        ):
            raise ValueError(f"num_loops must be an integer >= 1, got {num_loops!r}")
        num_attention_heads = kwargs.get("num_attention_heads")
        if (
            kwargs.get("num_key_value_heads") is None
            and num_attention_heads is not None
        ):
            kwargs["num_key_value_heads"] = num_attention_heads
        hidden_size = kwargs.get("hidden_size")
        if (
            kwargs.get("head_dim") is None
            and hidden_size is not None
            and num_attention_heads
        ):
            kwargs["head_dim"] = hidden_size // num_attention_heads
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.num_loops = num_loops
        self.skip_loop_final_norm = skip_loop_final_norm
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
