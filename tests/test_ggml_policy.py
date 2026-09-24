# SPDX-License-Identifier: Apache-2.0
"""Config-time policy for VLLM_METAL_BACKEND=ggml."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_metal.ggml.policy import (
    DEFAULT_BLOCK_SIZE,
    WORKER_CLS,
    apply_ggml_config_policy,
    update_block_size,
)


def make_config(model_type: str = "qwen3_5", *, hybrid: bool = True, **overrides):
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
            quantization=None,
            multimodal_config=object(),
            is_hybrid=hybrid,
            max_model_len=4096,
        ),
        parallel_config=SimpleNamespace(
            worker_cls="auto", pipeline_parallel_size=1, data_parallel_size=1
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            mamba_block_size=16,
            block_size=528,
            user_specified_block_size=False,
        ),
        speculative_config=None,
        lora_config=None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_selects_worker_text_only_and_disables_hybrid_prefix_caching() -> None:
    cfg = make_config()
    apply_ggml_config_policy(cfg)
    assert cfg.parallel_config.worker_cls == WORKER_CLS
    assert cfg.model_config.multimodal_config is None
    assert cfg.cache_config.enable_prefix_caching is False
    assert cfg.cache_config.mamba_cache_mode == "none"
    assert cfg.cache_config.mamba_block_size == 4096


def test_non_hybrid_keeps_prefix_caching() -> None:
    cfg = make_config("gemma4", hybrid=False)
    apply_ggml_config_policy(cfg)
    assert cfg.cache_config.enable_prefix_caching is True


def test_explicit_worker_cls_is_respected() -> None:
    cfg = make_config()
    cfg.parallel_config.worker_cls = "my.Worker"
    apply_ggml_config_policy(cfg)
    assert cfg.parallel_config.worker_cls == "my.Worker"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c.model_config.hf_config, "model_type", "llama"),
        lambda c: setattr(c.model_config, "quantization", "awq"),
        lambda c: setattr(c, "speculative_config", object()),
        lambda c: setattr(c, "lora_config", object()),
        lambda c: setattr(c.parallel_config, "pipeline_parallel_size", 2),
    ],
)
def test_rejects_unsupported(mutate) -> None:
    cfg = make_config()
    mutate(cfg)
    with pytest.raises(NotImplementedError):
        apply_ggml_config_policy(cfg)


def test_block_size_defaults_but_respects_user() -> None:
    cfg = make_config()
    update_block_size(cfg)
    assert cfg.cache_config.block_size == DEFAULT_BLOCK_SIZE
    cfg.cache_config.block_size = 32
    cfg.cache_config.user_specified_block_size = True
    update_block_size(cfg)
    assert cfg.cache_config.block_size == 32


def test_platform_routes_to_ggml_policy(monkeypatch) -> None:
    import vllm_metal.envs as envs

    monkeypatch.setenv("VLLM_METAL_BACKEND", "ggml")
    assert envs.VLLM_METAL_BACKEND == "ggml"
    monkeypatch.setenv("VLLM_METAL_BACKEND", "MLX")
    assert envs.VLLM_METAL_BACKEND == "mlx"
