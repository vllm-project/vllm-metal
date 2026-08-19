# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from vllm_metal.v1.cache_policy import ModelCachePolicy


def test_qwen_mtp_metadata_unwraps_language_model() -> None:
    args = SimpleNamespace(
        num_key_value_heads=2,
        head_dim=64,
        hidden_size=1024,
    )
    text_model = SimpleNamespace(
        mtp=SimpleNamespace(layers=[object()]),
        args=args,
    )
    wrapper = SimpleNamespace(
        supports_mtp=True,
        language_model=text_model,
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(method="mtp")),
        _forward_model=wrapper,
    )

    policy = ModelCachePolicy(runner, object())
    metadata = policy._qwen_mtp_metadata()

    assert metadata == (text_model, 1, 2, 64, 1024)


def test_qwen_mtp_metadata_keeps_direct_text_model() -> None:
    args = SimpleNamespace(
        num_key_value_heads=4,
        head_dim=128,
        hidden_size=2048,
    )
    model = SimpleNamespace(
        supports_mtp=True,
        mtp=SimpleNamespace(layers=[object(), object()]),
        args=args,
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(method="mtp")),
        _forward_model=model,
    )

    policy = ModelCachePolicy(runner, object())
    metadata = policy._qwen_mtp_metadata()

    assert metadata == (model, 2, 4, 128, 2048)
