# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma4 MTP assistant loading contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_metal.v1.gemma4_mtp import (
    GEMMA4_MTP_DRAFT_ARCHITECTURES,
    GEMMA4_MTP_DRAFT_MODEL_TYPES,
    Gemma4MTPAssistantLoader,
    Gemma4MTPAssistantMetadata,
    Gemma4MTPTargetMetadata,
)


@pytest.fixture(autouse=True)
def _reset_assistant_cache():
    Gemma4MTPAssistantLoader.clear_cache()
    yield
    Gemma4MTPAssistantLoader.clear_cache()


def _assistant_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_type": "gemma4_assistant",
        "architectures": ["Gemma4AssistantForCausalLM"],
        "backbone_hidden_size": 1536,
        "tie_word_embeddings": True,
        "use_ordered_embeddings": True,
        "num_centroids": 2048,
        "centroid_intermediate_top_k": 32,
        "text_config": {
            "model_type": "gemma4_text",
            "vocab_size": 262144,
            "hidden_size": 256,
            "intermediate_size": 2048,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_hidden_layers": 4,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        },
    }
    config.update(overrides)
    return config


def _target_args(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model_type": "gemma4_text",
        "vocab_size": 262144,
        "hidden_size": 1536,
        "num_hidden_layers": 4,
        "num_kv_shared_layers": 0,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    }
    values.update(overrides)
    return values


def _validate_assistant_config(
    assistant_config: Any,
    *,
    target_hf_config: Any | None,
    target_model_args: Mapping[str, Any],
) -> Gemma4MTPAssistantMetadata:
    target_metadata = Gemma4MTPTargetMetadata.from_configs(
        target_hf_config=target_hf_config,
        target_model_args=target_model_args,
    )
    if isinstance(assistant_config, Gemma4MTPAssistantMetadata):
        metadata = assistant_config
    else:
        metadata = Gemma4MTPAssistantMetadata.from_config(assistant_config)
    metadata.validate_compatible_with(target_metadata)
    return metadata


def _speculative_config(*, hf_config: object | None = None) -> SimpleNamespace:
    draft_config = SimpleNamespace(
        model="/assistant",
        hf_config=hf_config or SimpleNamespace(model_type="gemma4_mtp"),
    )
    return SimpleNamespace(
        method="mtp",
        model="/assistant",
        revision=None,
        draft_model_config=draft_config,
    )


class _ConfigWithTextAccessor:
    def __init__(self, text_config: object) -> None:
        self._text_config = text_config

    def get_text_config(self) -> object:
        return self._text_config


@pytest.mark.parametrize("model_type", sorted(GEMMA4_MTP_DRAFT_MODEL_TYPES))
def test_detects_raw_and_mapped_model_types(model_type: str) -> None:
    assert Gemma4MTPAssistantMetadata.is_assistant_config(
        SimpleNamespace(model_type=model_type)
    )


@pytest.mark.parametrize("architecture", sorted(GEMMA4_MTP_DRAFT_ARCHITECTURES))
def test_detects_raw_and_mapped_architectures(architecture: str) -> None:
    config = SimpleNamespace(model_type="unknown", architectures=[architecture])

    assert Gemma4MTPAssistantMetadata.is_assistant_config(config)


def test_validate_rejects_unknown_assistant_model_type_with_known_architecture() -> (
    None
):
    with pytest.raises(ValueError, match="model_type='unknown'"):
        _validate_assistant_config(
            _assistant_config(model_type="unknown"),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_detection_ignores_malformed_non_gemma_architectures() -> None:
    config = SimpleNamespace(model_type="deepseek_mtp", architectures=1)

    assert not Gemma4MTPAssistantMetadata.is_assistant_config(config)


def test_validate_accepts_matching_assistant_config() -> None:
    metadata = _validate_assistant_config(
        _assistant_config(),
        target_hf_config=None,
        target_model_args=_target_args(),
    )

    assert metadata.vocab_size == 262144
    assert metadata.backbone_hidden_size == 1536
    assert metadata.tie_word_embeddings is True
    assert metadata.layer_types[-1] == "full_attention"


def test_validate_records_tie_word_embeddings_contract() -> None:
    metadata = _validate_assistant_config(
        _assistant_config(tie_word_embeddings=False),
        target_hf_config=None,
        target_model_args=_target_args(),
    )

    assert metadata.tie_word_embeddings is False


def test_validate_accepts_target_metadata_from_hf_text_config() -> None:
    target_hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            vocab_size=262144,
            hidden_size=1536,
            num_hidden_layers=4,
            num_kv_shared_layers=0,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
    )

    metadata = _validate_assistant_config(
        _assistant_config(),
        target_hf_config=target_hf_config,
        target_model_args={
            "vocab_size": 262144,
            "hidden_size": 1536,
        },
    )

    assert metadata.backbone_hidden_size == 1536


def test_validate_rejects_mismatched_target_vocab_sources() -> None:
    target_hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            vocab_size=32000,
            hidden_size=1536,
            num_hidden_layers=4,
            num_kv_shared_layers=0,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
    )

    with pytest.raises(ValueError, match="vocab_size metadata mismatch"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=target_hf_config,
            target_model_args=_target_args(),
        )


def test_validate_rejects_mismatched_target_layer_type_sources() -> None:
    target_hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            vocab_size=262144,
            hidden_size=1536,
            num_hidden_layers=4,
            num_kv_shared_layers=0,
            layer_types=[
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
    )

    with pytest.raises(ValueError, match="layer_types metadata mismatch"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=target_hf_config,
            target_model_args=_target_args(),
        )


def test_validate_rejects_mismatched_target_kv_shared_sources() -> None:
    target_hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            vocab_size=262144,
            hidden_size=1536,
            num_hidden_layers=4,
            num_kv_shared_layers=1,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
    )

    with pytest.raises(ValueError, match="num_kv_shared_layers metadata mismatch"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=target_hf_config,
            target_model_args=_target_args(),
        )


def test_validate_accepts_top_level_gemma4_target_model_type() -> None:
    metadata = _validate_assistant_config(
        _assistant_config(),
        target_hf_config=None,
        target_model_args=_target_args(model_type="gemma4"),
    )

    assert metadata.backbone_hidden_size == 1536


def test_validate_accepts_target_metadata_from_get_text_config() -> None:
    target_hf_config = _ConfigWithTextAccessor(
        SimpleNamespace(
            model_type="gemma4_text",
            num_hidden_layers=4,
            num_kv_shared_layers=0,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
    )

    metadata = _validate_assistant_config(
        _assistant_config(),
        target_hf_config=target_hf_config,
        target_model_args={
            "vocab_size": 262144,
            "hidden_size": 1536,
        },
    )

    assert metadata.backbone_hidden_size == 1536


def test_validate_rejects_vocab_mismatch() -> None:
    with pytest.raises(ValueError, match="vocab size must match"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(vocab_size=32000),
        )


def test_validate_rejects_assistant_top_level_vocab_mismatch() -> None:
    with pytest.raises(ValueError, match="vocab_size metadata mismatch"):
        _validate_assistant_config(
            _assistant_config(vocab_size=32000),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_non_positive_target_size() -> None:
    with pytest.raises(ValueError, match="target model hidden_size must be positive"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(hidden_size=0),
        )


def test_validate_rejects_backbone_hidden_size_mismatch() -> None:
    with pytest.raises(ValueError, match="backbone hidden size must match"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(hidden_size=2048),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vocab_size", 0),
        ("hidden_size", -1),
        ("num_hidden_layers", 0),
    ],
)
def test_validate_rejects_non_positive_assistant_text_sizes(
    field: str,
    value: int,
) -> None:
    text_config = _assistant_config()["text_config"]
    assert isinstance(text_config, dict)
    with pytest.raises(ValueError, match=f"assistant {field} must be positive"):
        _validate_assistant_config(
            _assistant_config(text_config={**text_config, field: value}),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_non_positive_assistant_backbone_hidden_size() -> None:
    with pytest.raises(
        ValueError,
        match="assistant backbone_hidden_size must be positive",
    ):
        _validate_assistant_config(
            _assistant_config(backbone_hidden_size=0),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


@pytest.mark.parametrize("value", ["1", "bad", 1.5])
def test_validate_rejects_non_integer_assistant_config_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="assistant n_predict must be an integer"):
        _validate_assistant_config(
            _assistant_config(n_predict=value),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


@pytest.mark.parametrize("field", ["num_centroids", "centroid_intermediate_top_k"])
@pytest.mark.parametrize("value", ["1", True, 1.5])
def test_validate_rejects_non_integer_assistant_mask_config_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=f"assistant {field} must be an integer"):
        _validate_assistant_config(
            _assistant_config(**{field: value}),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


@pytest.mark.parametrize("field", ["num_centroids", "centroid_intermediate_top_k"])
def test_validate_rejects_non_positive_assistant_mask_config_values(
    field: str,
) -> None:
    with pytest.raises(ValueError, match=f"assistant {field} must be positive"):
        _validate_assistant_config(
            _assistant_config(**{field: 0}),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_ordered_embedding_vocab_centroid_mismatch() -> None:
    text_config = _assistant_config()["text_config"]
    assert isinstance(text_config, dict)
    with pytest.raises(ValueError, match="vocab_size must be divisible"):
        _validate_assistant_config(
            _assistant_config(
                num_centroids=64,
                text_config={**text_config, "vocab_size": 262143},
            ),
            target_hf_config=None,
            target_model_args=_target_args(vocab_size=262143),
        )


def test_validate_rejects_ordered_embedding_top_k_above_centroids() -> None:
    with pytest.raises(ValueError, match="centroid_intermediate_top_k must be <="):
        _validate_assistant_config(
            _assistant_config(num_centroids=8, centroid_intermediate_top_k=9),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_bool_assistant_config_values() -> None:
    text_config = _assistant_config()["text_config"]
    assert isinstance(text_config, dict)
    with pytest.raises(ValueError, match="assistant hidden_size must be an integer"):
        _validate_assistant_config(
            _assistant_config(text_config={**text_config, "hidden_size": True}),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


@pytest.mark.parametrize("field", ["tie_word_embeddings", "use_ordered_embeddings"])
def test_validate_rejects_non_bool_assistant_config_values(field: str) -> None:
    with pytest.raises(ValueError, match=f"assistant {field} must be a boolean"):
        _validate_assistant_config(
            _assistant_config(**{field: "false"}),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_non_gemma4_target_model_type() -> None:
    with pytest.raises(ValueError, match="Gemma4 target"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(model_type="llama"),
        )


def test_validate_rejects_mismatched_target_model_type_sources() -> None:
    target_hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            num_hidden_layers=4,
            num_kv_shared_layers=0,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
    )

    with pytest.raises(ValueError, match="model_type='llama'"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=target_hf_config,
            target_model_args=_target_args(model_type="llama"),
        )


def test_validate_rejects_missing_target_model_type() -> None:
    with pytest.raises(ValueError, match="model_type=None"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args={
                "vocab_size": 262144,
                "hidden_size": 1536,
                "num_kv_shared_layers": 0,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
            },
        )


def test_validate_accepts_assistant_layer_types_tail_matching_target_non_shared_layers() -> (
    None
):
    metadata = _validate_assistant_config(
        _assistant_config(),
        target_hf_config=None,
        target_model_args=_target_args(
            num_hidden_layers=6,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        ),
    )

    assert metadata.layer_types == (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    )


def test_validate_rejects_layer_types_not_tail_matching_target_non_shared_layers() -> (
    None
):
    with pytest.raises(ValueError, match="tail-match"):
        _validate_assistant_config(
            _assistant_config(
                text_config={
                    **_assistant_config()["text_config"],
                    "num_hidden_layers": 1,
                    "layer_types": ["full_attention"],
                }
            ),
            target_hf_config=None,
            target_model_args=_target_args(
                layer_types=[
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                ],
            ),
        )


def test_validate_rejects_assistant_with_more_layers_than_target_non_shared() -> None:
    with pytest.raises(ValueError, match="more layers"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(num_kv_shared_layers=3),
        )


def test_validate_rejects_target_layer_types_length_mismatch() -> None:
    with pytest.raises(ValueError, match="layer_types must match num_hidden_layers"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(
                num_hidden_layers=5,
                layer_types=[
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
            ),
        )


def test_validate_rejects_mismatched_target_num_hidden_layer_sources() -> None:
    target_hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            vocab_size=262144,
            hidden_size=1536,
            num_hidden_layers=5,
            num_kv_shared_layers=0,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
    )

    with pytest.raises(ValueError, match="num_hidden_layers metadata mismatch"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=target_hf_config,
            target_model_args=_target_args(),
        )


def test_validate_rejects_target_with_no_non_shared_kv_layers() -> None:
    with pytest.raises(ValueError, match="leave at least one non-shared KV layer"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(num_kv_shared_layers=4),
        )


def test_validate_rejects_string_target_layer_types() -> None:
    with pytest.raises(ValueError, match="layer_types must be a non-string sequence"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(layer_types="full_attention"),
        )


def test_validate_rejects_non_string_target_layer_type_entries() -> None:
    with pytest.raises(ValueError, match="layer_types entries must be strings"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(
                layer_types=[
                    "sliding_attention",
                    1,
                    "sliding_attention",
                    "full_attention",
                ],
            ),
        )


def test_validate_rejects_unknown_target_layer_types() -> None:
    with pytest.raises(ValueError, match="Unsupported Gemma4 MTP target layer types"):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(
                layer_types=[
                    "sliding_attention",
                    "unknown_attention",
                    "sliding_attention",
                    "full_attention",
                ],
            ),
        )


def test_validate_rejects_non_integer_target_config_values() -> None:
    with pytest.raises(
        ValueError,
        match="target model num_kv_shared_layers must be an integer",
    ):
        _validate_assistant_config(
            _assistant_config(),
            target_hf_config=None,
            target_model_args=_target_args(num_kv_shared_layers="0"),
        )


def test_validate_rejects_malformed_assistant_layer_types() -> None:
    text_config = _assistant_config()["text_config"]
    assert isinstance(text_config, dict)
    config = _assistant_config(
        text_config={
            **text_config,
            "layer_types": ["sliding_attention"],
        }
    )

    with pytest.raises(ValueError, match="layer_types must match num_hidden_layers"):
        _validate_assistant_config(
            config,
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_string_assistant_layer_types() -> None:
    text_config = _assistant_config()["text_config"]
    assert isinstance(text_config, dict)
    with pytest.raises(ValueError, match="layer_types must be a non-string sequence"):
        _validate_assistant_config(
            _assistant_config(
                text_config={**text_config, "layer_types": "full_attention"}
            ),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_non_string_assistant_layer_type_entries() -> None:
    text_config = _assistant_config()["text_config"]
    assert isinstance(text_config, dict)
    with pytest.raises(ValueError, match="layer_types entries must be strings"):
        _validate_assistant_config(
            _assistant_config(
                text_config={
                    **text_config,
                    "layer_types": [
                        "sliding_attention",
                        1,
                        "sliding_attention",
                        "full_attention",
                    ],
                }
            ),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_mapped_config_with_unsupported_n_predict() -> None:
    with pytest.raises(ValueError, match="n_predict=1"):
        _validate_assistant_config(
            _assistant_config(model_type="gemma4_mtp", n_predict=2),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_malformed_assistant_architectures() -> None:
    with pytest.raises(ValueError, match="architectures must be a sequence"):
        _validate_assistant_config(
            _assistant_config(architectures=1),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_string_assistant_architectures() -> None:
    with pytest.raises(
        ValueError,
        match="architectures must be a non-string sequence",
    ):
        _validate_assistant_config(
            _assistant_config(architectures="Gemma4AssistantForCausalLM"),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_non_string_assistant_architecture_entries() -> None:
    with pytest.raises(ValueError, match="architectures entries must be strings"):
        _validate_assistant_config(
            _assistant_config(architectures=[1]),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_validate_rejects_missing_gemma4_mtp_architecture() -> None:
    with pytest.raises(ValueError, match="requires a Gemma4 MTP architecture"):
        _validate_assistant_config(
            _assistant_config(architectures=["OtherModel"]),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_loader_uses_custom_mlx_model_classes_and_validates_runtime() -> None:
    load_model_calls: list[dict[str, object]] = []
    download_calls: list[tuple[str, str | None]] = []
    assistant_model = object()

    def _load_model(
        *args: object, **kwargs: object
    ) -> tuple[object, dict[str, object]]:
        load_model_calls.append({"args": args, "kwargs": kwargs})
        get_model_classes = kwargs["get_model_classes"]
        assert callable(get_model_classes)
        assert get_model_classes.__name__ == "_get_model_classes"
        return assistant_model, _assistant_config()

    loader = Gemma4MTPAssistantLoader(
        load_model_fn=_load_model,
        download_fn=lambda model_name, revision: (
            download_calls.append((model_name, revision)) or Path(model_name)
        ),
    )

    runtime = loader.load_if_needed(
        speculative_config=_speculative_config(),
        target_hf_config=None,
        target_model_args=_target_args(),
    )

    assert runtime is not None
    assert runtime.model is assistant_model
    assert runtime.metadata.hidden_size == 256
    assert runtime.forward_ready is False
    assert download_calls == [("/assistant", None)]
    assert load_model_calls[0]["args"] == (Path("/assistant"),)
    assert load_model_calls[0]["kwargs"]["strict"] is True


def test_loader_wraps_local_custom_shards_for_mlx_lm(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_assistant_config()))
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "layers-0.safetensors").write_text("", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": "layers-0.safetensors"}}),
        encoding="utf-8",
    )
    load_model_paths: list[Path] = []

    def _load_model(
        model_path: Path,
        *args: object,
        **kwargs: object,
    ) -> tuple[object, dict[str, object]]:
        load_model_paths.append(model_path)
        assert model_path != tmp_path
        assert (model_path / "config.json").is_symlink()
        assert (model_path / "model.safetensors").is_symlink()
        return object(), _assistant_config()

    loader = Gemma4MTPAssistantLoader(
        load_model_fn=_load_model,
        download_fn=lambda model_name, revision: tmp_path,
    )

    loader.load_if_needed(
        speculative_config=_speculative_config(),
        target_hf_config=None,
        target_model_args=_target_args(),
    )

    assert len(load_model_paths) == 1


def test_default_loader_downloads_custom_named_safetensors_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    (tmp_path / "config.json").write_text(json.dumps(_assistant_config()))
    (tmp_path / "layers-0.safetensors").write_text("", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": "layers-0.safetensors"}}),
        encoding="utf-8",
    )
    snapshot_calls: list[dict[str, object]] = []
    load_model_paths: list[Path] = []

    def _snapshot_download(*args: object, **kwargs: object) -> str:
        snapshot_calls.append({"args": args, "kwargs": kwargs})
        return str(tmp_path)

    def _load_model(
        model_path: Path,
        *args: object,
        **kwargs: object,
    ) -> tuple[object, dict[str, object]]:
        load_model_paths.append(model_path)
        assert model_path != tmp_path
        assert (model_path / "model.safetensors").is_symlink()
        return object(), _assistant_config()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot_download)
    spec_config = _speculative_config()
    spec_config.draft_model_config.model = "google/gemma-4-assistant"
    loader = Gemma4MTPAssistantLoader(load_model_fn=_load_model)

    loader.load_if_needed(
        speculative_config=spec_config,
        target_hf_config=None,
        target_model_args=_target_args(),
    )

    assert len(load_model_paths) == 1
    assert snapshot_calls == [
        {
            "args": ("google/gemma-4-assistant",),
            "kwargs": {
                "revision": None,
                "allow_patterns": [
                    "config.json",
                    "generation_config.json",
                    "model*.safetensors.index.json",
                    "*.safetensors",
                ],
            },
        }
    ]


def test_loader_revalidates_cached_runtime_against_target() -> None:
    load_model_calls = 0

    def _load_model(
        *args: object, **kwargs: object
    ) -> tuple[object, dict[str, object]]:
        nonlocal load_model_calls
        load_model_calls += 1
        return object(), _assistant_config()

    loader = Gemma4MTPAssistantLoader(
        load_model_fn=_load_model,
        download_fn=lambda model_name, revision: Path(model_name),
        model_path_resolver=lambda model_name: model_name,
    )

    runtime = loader.load_if_needed(
        speculative_config=_speculative_config(),
        target_hf_config=None,
        target_model_args=_target_args(),
    )
    assert runtime is not None

    with pytest.raises(ValueError, match="backbone hidden size must match"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_hf_config=None,
            target_model_args=_target_args(hidden_size=2048),
        )
    assert load_model_calls == 1


def test_loader_skips_non_gemma4_mtp_speculative_config() -> None:
    loader = Gemma4MTPAssistantLoader(
        load_model_fn=lambda *args, **kwargs: pytest.fail("load_model called"),
        download_fn=lambda model_name, revision: Path(model_name),
    )

    runtime = loader.load_if_needed(
        speculative_config=SimpleNamespace(method="draft_model"),
        target_hf_config=None,
        target_model_args=_target_args(),
    )

    assert runtime is None


def test_loader_skips_non_gemma4_mtp_without_resolving_model() -> None:
    loader = Gemma4MTPAssistantLoader(
        load_model_fn=lambda *args, **kwargs: pytest.fail("load_model called"),
        download_fn=lambda model_name, revision: pytest.fail("download called"),
        model_path_resolver=lambda model_name: pytest.fail("resolver called"),
    )

    runtime = loader.load_if_needed(
        speculative_config=_speculative_config(
            hf_config=SimpleNamespace(model_type="deepseek_mtp")
        ),
        target_hf_config=None,
        target_model_args=_target_args(),
    )

    assert runtime is None


@pytest.mark.parametrize("model_file", ["modeling_gemma4.py", "", 0, None])
def test_loader_rejects_custom_model_file_before_load(
    tmp_path: Path,
    model_file: object,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(_assistant_config(model_file=model_file))
    )
    loader = Gemma4MTPAssistantLoader(
        load_model_fn=lambda *args, **kwargs: pytest.fail("load_model called"),
        download_fn=lambda model_name, revision: tmp_path,
        model_path_resolver=lambda model_name: model_name,
    )

    with pytest.raises(ValueError, match="custom model_file"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_loader_rejects_postload_custom_model_file() -> None:
    loader = Gemma4MTPAssistantLoader(
        load_model_fn=lambda *args, **kwargs: (
            object(),
            _assistant_config(model_file=None),
        ),
        download_fn=lambda model_name, revision: Path(model_name),
    )

    with pytest.raises(ValueError, match="custom model_file"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_loader_prevalidates_config_file_before_load(tmp_path: Path) -> None:
    text_config = _assistant_config()["text_config"]
    assert isinstance(text_config, dict)
    (tmp_path / "config.json").write_text(
        json.dumps(_assistant_config(text_config={**text_config, "hidden_size": "256"}))
    )
    loader = Gemma4MTPAssistantLoader(
        load_model_fn=lambda *args, **kwargs: pytest.fail("load_model called"),
        download_fn=lambda model_name, revision: tmp_path,
        model_path_resolver=lambda model_name: model_name,
    )

    with pytest.raises(ValueError, match="assistant hidden_size must be an integer"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_default_loader_rejects_missing_config_before_load(tmp_path: Path) -> None:
    loader = Gemma4MTPAssistantLoader(
        download_fn=lambda model_name, revision: tmp_path,
        model_path_resolver=lambda model_name: model_name,
    )

    with pytest.raises(ValueError, match="must contain config.json"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_loader_rejects_invalid_json_config_before_load(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{", encoding="utf-8")
    loader = Gemma4MTPAssistantLoader(
        load_model_fn=lambda *args, **kwargs: pytest.fail("load_model called"),
        download_fn=lambda model_name, revision: tmp_path,
        model_path_resolver=lambda model_name: model_name,
    )

    with pytest.raises(ValueError, match="config.json is not valid JSON"):
        loader.load_if_needed(
            speculative_config=_speculative_config(),
            target_hf_config=None,
            target_model_args=_target_args(),
        )


def test_loader_passes_revision_and_keeps_cache_keys_separate() -> None:
    load_model_calls = 0
    download_calls: list[tuple[str, str | None]] = []

    def _load_model(
        *args: object, **kwargs: object
    ) -> tuple[object, dict[str, object]]:
        nonlocal load_model_calls
        load_model_calls += 1
        return object(), _assistant_config()

    loader = Gemma4MTPAssistantLoader(
        load_model_fn=_load_model,
        download_fn=lambda model_name, revision: (
            download_calls.append((model_name, revision)) or Path(model_name)
        ),
        model_path_resolver=lambda model_name: model_name,
    )
    spec_a = _speculative_config()
    spec_a.revision = "rev-a"
    spec_b = _speculative_config()
    spec_b.revision = "rev-b"

    first = loader.load_if_needed(
        speculative_config=spec_a,
        target_hf_config=None,
        target_model_args=_target_args(),
    )
    second = loader.load_if_needed(
        speculative_config=spec_a,
        target_hf_config=None,
        target_model_args=_target_args(),
    )
    third = loader.load_if_needed(
        speculative_config=spec_b,
        target_hf_config=None,
        target_model_args=_target_args(),
    )

    assert first is second
    assert first is not third
    assert load_model_calls == 2
    assert download_calls == [("/assistant", "rev-a"), ("/assistant", "rev-b")]


def test_loader_prefers_resolved_draft_model_revision() -> None:
    download_calls: list[tuple[str, str | None]] = []

    loader = Gemma4MTPAssistantLoader(
        load_model_fn=lambda *args, **kwargs: (object(), _assistant_config()),
        download_fn=lambda model_name, revision: (
            download_calls.append((model_name, revision)) or Path(model_name)
        ),
        model_path_resolver=lambda model_name: model_name,
    )
    spec = _speculative_config()
    spec.revision = "outer-rev"
    spec.draft_model_config.revision = "draft-rev"

    loader.load_if_needed(
        speculative_config=spec,
        target_hf_config=None,
        target_model_args=_target_args(),
    )

    assert download_calls == [("/assistant", "draft-rev")]


def test_model_args_preserve_text_config_vocab_size() -> None:
    from vllm_metal.v1.gemma4_mtp_model import Gemma4MTPAssistantModelArgs

    args = Gemma4MTPAssistantModelArgs(
        vocab_size=262144,
        text_config={"vocab_size": 128},
    )

    assert args.vocab_size == 128
    assert args.text_config is not None
    assert args.text_config["vocab_size"] == 128


def test_model_args_do_not_mutate_source_text_config() -> None:
    from vllm_metal.v1.gemma4_mtp_model import Gemma4MTPAssistantModelArgs

    text_config: dict[str, object] = {"hidden_size": 256}

    args = Gemma4MTPAssistantModelArgs(
        vocab_size=262144,
        text_config=text_config,
    )

    assert text_config == {"hidden_size": 256}
    assert args.text_config is not text_config
    assert args.text_config is not None
    assert args.text_config["vocab_size"] == 262144


def test_model_rejects_missing_num_hidden_layers() -> None:
    from vllm_metal.v1.gemma4_mtp_model import (
        Gemma4MTPAssistantModel,
        Gemma4MTPAssistantModelArgs,
    )

    with pytest.raises(ValueError, match="requires num_hidden_layers"):
        Gemma4MTPAssistantModel(
            Gemma4MTPAssistantModelArgs(
                backbone_hidden_size=1536,
                text_config={"vocab_size": 262144, "hidden_size": 256},
            )
        )


def test_model_rejects_conflicting_num_kv_shared_layers() -> None:
    from vllm_metal.v1.gemma4_mtp_model import (
        Gemma4MTPAssistantModel,
        Gemma4MTPAssistantModelArgs,
    )

    text_config = _assistant_config()["text_config"]
    assert isinstance(text_config, dict)

    with pytest.raises(ValueError, match="num_kv_shared_layers must equal"):
        Gemma4MTPAssistantModel(
            Gemma4MTPAssistantModelArgs(
                backbone_hidden_size=1536,
                text_config={**text_config, "num_kv_shared_layers": 1},
            )
        )
