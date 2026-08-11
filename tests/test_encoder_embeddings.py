# SPDX-License-Identifier: Apache-2.0
"""Tests for the encoder embedding adapter (#589 PR1)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from vllm_metal.v1.encoder_embeddings import (
    EncoderEmbeddingAdapter,
    MlxEmbeddingsEncoderModel,
    is_encoder_embedding_config,
    load_mlx_embeddings_model,
    requires_mlx_embeddings_load,
)
from vllm_metal.v1.pooling import (
    forward_sequence_hidden_states,
    supports_embed_pooling,
)


def _encoder_model_config(**overrides):
    values = {
        "runner_type": "pooling",
        "served_model_name": "mlx-community/bge-m3-mlx-8bit",
        "model": "mlx-community/bge-m3-mlx-8bit",
        "hf_config": SimpleNamespace(
            architectures=["XLMRobertaModel"],
            model_type="xlm-roberta",
        ),
        "pooler_config": SimpleNamespace(
            task="embed",
            pooling_type=None,
            seq_pooling_type="CLS",
            enable_chunked_processing=False,
            use_activation=None,
            dimensions=None,
        ),
        "multimodal_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeEmbeddingsOutput:
    def __init__(self, hidden_states: mx.array) -> None:
        self.last_hidden_state = hidden_states


class _FakeEmbeddingsModule:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_type="xlm-roberta",
            hidden_size=4,
            vocab_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
        )
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def __call__(self, input_ids, attention_mask=None):
        shape = tuple(int(x) for x in input_ids.shape)
        mask_shape = tuple(int(x) for x in attention_mask.shape)
        self.calls.append((shape, mask_shape))
        tokens = int(input_ids.shape[-1])
        rows = [[float(i), float(i + 1), 0.0, 1.0] for i in range(tokens)]
        return _FakeEmbeddingsOutput(mx.array([rows], dtype=mx.float32))


class TestEncoderEmbeddingDetection:
    def test_detects_xlm_roberta_and_bge_architectures(self) -> None:
        assert EncoderEmbeddingAdapter.matches_config(_encoder_model_config())
        assert EncoderEmbeddingAdapter.requires_load(_encoder_model_config())
        # Compatibility aliases keep older imports working.
        assert is_encoder_embedding_config(_encoder_model_config())
        assert requires_mlx_embeddings_load(_encoder_model_config())
        assert EncoderEmbeddingAdapter.matches_config(
            _encoder_model_config(
                hf_config=SimpleNamespace(
                    architectures=["BgeM3EmbeddingModel"],
                    model_type="xlm-roberta",
                )
            )
        )

    def test_rejects_decoder_embedding_configs(self) -> None:
        config = _encoder_model_config(
            hf_config=SimpleNamespace(
                architectures=["Qwen3ForCausalLM"],
                model_type="qwen3",
            )
        )
        assert not EncoderEmbeddingAdapter.matches_config(config)
        assert not EncoderEmbeddingAdapter.requires_load(config)

    def test_owns_cls_pooling_defaults(self) -> None:
        assert EncoderEmbeddingAdapter.default_sequence_pooling_type == "CLS"
        assert EncoderEmbeddingAdapter.allowed_sequence_pooling_types == (
            None,
            "CLS",
            "LAST",
        )
        assert EncoderEmbeddingAdapter.skip_paged_attention_patch is True


class TestMlxEmbeddingsLoad:
    def test_missing_extra_raises_install_hint(self) -> None:
        with (
            patch(
                "vllm_metal.v1.encoder_embeddings._import_mlx_embeddings_load",
                side_effect=ImportError(
                    "Loading encoder embedding models such as XLM-RoBERTa / BGE-M3 "
                    "requires the optional 'mlx-embeddings' package. Install it with: "
                    'pip install "vllm-metal[embeddings]"'
                ),
            ),
            pytest.raises(ImportError, match=r"vllm-metal\[embeddings\]"),
        ):
            EncoderEmbeddingAdapter.load("mlx-community/bge-m3-mlx-8bit")

    def test_wraps_loaded_model_with_adapter(self) -> None:
        fake = _FakeEmbeddingsModule()
        tokenizer = object()
        load_mock = MagicMock(return_value=(fake, tokenizer))
        with patch(
            "vllm_metal.v1.encoder_embeddings._import_mlx_embeddings_load",
            return_value=load_mock,
        ):
            model, loaded_tokenizer, adapter = EncoderEmbeddingAdapter.load(
                "mlx-community/bge-m3-mlx-8bit",
                tokenizer_config={"trust_remote_code": True},
                lazy=True,
            )

        load_mock.assert_called_once_with(
            "mlx-community/bge-m3-mlx-8bit",
            tokenizer_config={"trust_remote_code": True},
            lazy=True,
        )
        assert loaded_tokenizer is tokenizer
        assert isinstance(model, MlxEmbeddingsEncoderModel)
        assert isinstance(adapter, EncoderEmbeddingAdapter)
        assert adapter.model is model
        assert EncoderEmbeddingAdapter.from_loaded_model(model) is not None
        assert EncoderEmbeddingAdapter.from_loaded_model(object()) is None
        hidden = model.model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=MagicMock())
        assert hidden.shape == (1, 3, 4)
        assert fake.calls == [((1, 3), (1, 3))]

    def test_compatibility_load_helper_still_works(self) -> None:
        fake = _FakeEmbeddingsModule()
        tokenizer = object()
        load_mock = MagicMock(return_value=(fake, tokenizer))
        with patch(
            "vllm_metal.v1.encoder_embeddings._import_mlx_embeddings_load",
            return_value=load_mock,
        ):
            model, loaded_tokenizer = load_mlx_embeddings_model(
                "mlx-community/bge-m3-mlx-8bit"
            )
        assert isinstance(model, MlxEmbeddingsEncoderModel)
        assert loaded_tokenizer is tokenizer


class TestEncoderPagedSetup:
    def test_setup_paged_attention_skips_patching_via_adapter(self) -> None:
        from vllm_metal.v1.cache_policy import WorkerCachePlanner

        fake = _FakeEmbeddingsModule()
        model = MlxEmbeddingsEncoderModel(fake)
        adapter = EncoderEmbeddingAdapter(model)
        runner = SimpleNamespace(
            model=model,
            is_mla=False,
            _encoder_embedding_adapter=adapter,
            validate_paged_attention_support=MagicMock(),
            build_paged_attention_runtime=MagicMock(),
            install_gemma4_mtp_kv_sharing=MagicMock(),
            install_drafter=MagicMock(),
            install_paged_attention_runtime=MagicMock(),
            model_args={
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 2,
                "hidden_size": 4,
            },
            kv_cache_dtype=mx.float16,
            metal_config=SimpleNamespace(use_paged_attention=True),
        )
        backend = MagicMock()
        backend.patch_model = MagicMock(return_value=2)
        backend.initialize = MagicMock()
        runner.build_paged_attention_runtime.return_value = backend

        worker = SimpleNamespace(model_runner=runner)
        planner = WorkerCachePlanner(worker)

        plan = SimpleNamespace(
            block_size=16,
            num_blocks=4,
            per_block_bytes=1024,
            format_breakdown=lambda: "stub",
        )
        with (
            patch.object(planner, "_paged_attention_plan", return_value=plan),
            patch(
                "vllm_metal.v1.cache_policy.get_config",
                return_value=SimpleNamespace(turboquant=False, k_quant=None),
            ),
            patch(
                "vllm_metal.v1.cache_policy.try_enable_gemma4_yoco_fast_prefill",
                return_value=None,
            ),
        ):
            planner.setup_paged_attention(overhead=0)

        backend.patch_model.assert_not_called()
        runner.install_paged_attention_runtime.assert_called_once()


class TestEncoderPoolingForward:
    def test_supports_embed_for_encoder_adapter(self) -> None:
        fake = _FakeEmbeddingsModule()
        model = MlxEmbeddingsEncoderModel(fake)
        assert supports_embed_pooling(model, _encoder_model_config())

    def test_forwards_packed_segments_independently(self) -> None:
        fake = _FakeEmbeddingsModule()
        model = MlxEmbeddingsEncoderModel(fake)
        adapter = EncoderEmbeddingAdapter(model)
        hidden = forward_sequence_hidden_states(
            model,
            mx.array([[10, 11, 20, 21, 22]], dtype=mx.int32),
            cache=[MagicMock()],
            model_config=_encoder_model_config(),
            segment_lengths=[2, 3],
            encoder_adapter=adapter,
        )
        assert hidden.shape == (1, 5, 4)
        assert fake.calls == [((1, 2), (1, 2)), ((1, 3), (1, 3))]
        # First token of each independent segment starts its own row index 0.
        assert float(hidden[0, 0, 0]) == 0.0
        assert float(hidden[0, 2, 0]) == 0.0
