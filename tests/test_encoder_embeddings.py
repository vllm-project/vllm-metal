# SPDX-License-Identifier: Apache-2.0
"""Tests for the encoder embedding adapter (#589 PR1)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
import torch

from vllm_metal.v1.encoder_embeddings import (
    BgeM3SparseHead,
    EncoderEmbeddingAdapter,
    MlxEmbeddingsEncoderModel,
    _load_bge_m3_sparse_head,
    is_encoder_embedding_config,
    load_mlx_embeddings_model,
    requires_mlx_embeddings_load,
)
from vllm_metal.v1.pooling import (
    forward_sequence_hidden_states,
    supports_embed_pooling,
)

_BGE_M3_MODEL_ID = "mlx-community/bge-m3-mlx-8bit"
_BGE_M3_MODEL_REVISION = "7eca4a1c6ea1a0c5efc37598b369012f3985910f"
_BGE_M3_HI_TOKEN_ID = 2673
_BGE_M3_HI_REFERENCE_WEIGHT = 0.26710861921310425
_BGE_M3_8BIT_REFERENCE_REL_TOLERANCE = 0.02
_BGE_M3_REQUIRED_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _encoder_model_config(**overrides):
    values = {
        "runner_type": "pooling",
        "served_model_name": "mlx-community/bge-m3-mlx-8bit",
        "model": "mlx-community/bge-m3-mlx-8bit",
        "hf_config": SimpleNamespace(
            architectures=["XLMRobertaModel"],
            model_type="xlm-roberta",
            hidden_size=4,
            bos_token_id=0,
            eos_token_id=2,
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
            bos_token_id=0,
            eos_token_id=2,
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

    def test_loads_sparse_head_only_for_explicit_bge_m3_token_classify(
        self,
    ) -> None:
        fake = _FakeEmbeddingsModule()
        tokenizer = object()
        sparse_head = BgeM3SparseHead(
            weight=mx.ones((1, 4), dtype=mx.float32),
            bias=mx.zeros((1,), dtype=mx.float32),
            bos_token_id=0,
            eos_token_id=2,
        )
        load_mock = MagicMock(return_value=(fake, tokenizer))
        sparse_load_mock = MagicMock(return_value=sparse_head)
        config = _encoder_model_config(
            pooler_config=SimpleNamespace(task="token_classify")
        )

        with (
            patch(
                "vllm_metal.v1.encoder_embeddings._import_mlx_embeddings_load",
                return_value=load_mock,
            ),
            patch(
                "vllm_metal.v1.encoder_embeddings._load_bge_m3_sparse_head",
                sparse_load_mock,
            ),
        ):
            _model, _tokenizer, adapter = EncoderEmbeddingAdapter.load(
                "cached/model/path",
                model_config=config,
            )

        sparse_load_mock.assert_called_once_with(config.hf_config)
        assert adapter.supports_token_classify

    def test_dense_bge_m3_load_does_not_fetch_sparse_head(self) -> None:
        fake = _FakeEmbeddingsModule()
        load_mock = MagicMock(return_value=(fake, object()))
        with (
            patch(
                "vllm_metal.v1.encoder_embeddings._import_mlx_embeddings_load",
                return_value=load_mock,
            ),
            patch(
                "vllm_metal.v1.encoder_embeddings._load_bge_m3_sparse_head"
            ) as sparse_load_mock,
        ):
            _model, _tokenizer, adapter = EncoderEmbeddingAdapter.load(
                "cached/model/path",
                model_config=_encoder_model_config(),
            )

        sparse_load_mock.assert_not_called()
        assert not adapter.supports_token_classify

    def test_token_classify_rejects_unrelated_encoder_model(self) -> None:
        config = _encoder_model_config(
            model="other/xlm-roberta",
            served_model_name="other/xlm-roberta",
            pooler_config=SimpleNamespace(task="token_classify"),
        )
        with pytest.raises(NotImplementedError, match="mlx-community/bge-m3"):
            EncoderEmbeddingAdapter.load(
                "cached/model/path",
                model_config=config,
            )


class TestBgeM3SparseHead:
    def test_applies_bias_relu_and_filters_bos_eos(self) -> None:
        head = BgeM3SparseHead(
            weight=mx.array([[1.0, 0.0, 0.0]], dtype=mx.float32),
            bias=mx.array([-1.5], dtype=mx.float32),
            bos_token_id=0,
            eos_token_id=2,
        )
        hidden_states = mx.array(
            [
                [100.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=mx.float32,
        )

        token_logits = head.project_token_logits(hidden_states)
        activated = head.filter_token_weights(
            token_logits,
            token_ids=[0, 41, 0, 2, 42, 2],
            use_activation=True,
        )
        raw = head.filter_token_weights(
            token_logits,
            token_ids=[0, 41, 0, 2, 42, 2],
            use_activation=False,
        )

        assert mx.allclose(
            activated,
            mx.array([1.5, 0.0, 2.5, 0.0], dtype=mx.float32),
        )
        assert mx.allclose(
            raw,
            mx.array([1.5, -0.5, 2.5, -0.5], dtype=mx.float32),
        )

    def test_rejects_invalid_weight_shape(self) -> None:
        with pytest.raises(ValueError, match=r"\[1, hidden\]"):
            BgeM3SparseHead(
                weight=mx.ones((2, 4), dtype=mx.float32),
                bias=mx.zeros((1,), dtype=mx.float32),
                bos_token_id=0,
                eos_token_id=2,
            )

    def test_official_head_loader_is_revision_pinned(self) -> None:
        state = {
            "weight": torch.ones((1, 4), dtype=torch.float16),
            "bias": torch.zeros((1,), dtype=torch.float16),
        }
        config = SimpleNamespace(hidden_size=4, bos_token_id=0, eos_token_id=2)

        with (
            patch(
                "vllm_metal.v1.encoder_embeddings.hf_hub_download",
                return_value="/tmp/sparse_linear.pt",
            ) as download_mock,
            patch(
                "vllm_metal.v1.encoder_embeddings.torch.load",
                return_value=state,
            ) as torch_load_mock,
        ):
            head = _load_bge_m3_sparse_head(config)

        download_mock.assert_called_once_with(
            repo_id="BAAI/bge-m3",
            filename="sparse_linear.pt",
            revision="5617a9f61b028005a4858fdac845db406aefb181",
        )
        torch_load_mock.assert_called_once_with(
            "/tmp/sparse_linear.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert head.supports_hidden_size(4)

    def test_official_head_loader_rejects_nonfinite_weights(self) -> None:
        state = {
            "weight": torch.tensor([[float("nan"), 0.0]], dtype=torch.float16),
            "bias": torch.zeros((1,), dtype=torch.float16),
        }
        config = SimpleNamespace(hidden_size=2, bos_token_id=0, eos_token_id=2)

        with (
            patch(
                "vllm_metal.v1.encoder_embeddings.hf_hub_download",
                return_value="/tmp/sparse_linear.pt",
            ),
            patch(
                "vllm_metal.v1.encoder_embeddings.torch.load",
                return_value=state,
            ),
            pytest.raises(ValueError, match="must be finite"),
        ):
            _load_bge_m3_sparse_head(config)

    @pytest.mark.slow
    def test_cached_real_bge_m3_hi_matches_upstream_sparse_reference(self) -> None:
        from huggingface_hub import try_to_load_from_cache

        cached_model_files = {
            filename: try_to_load_from_cache(
                _BGE_M3_MODEL_ID,
                filename,
                revision=_BGE_M3_MODEL_REVISION,
            )
            for filename in _BGE_M3_REQUIRED_FILES
        }
        head_path = try_to_load_from_cache(
            "BAAI/bge-m3",
            "sparse_linear.pt",
            revision="5617a9f61b028005a4858fdac845db406aefb181",
        )
        missing_files = [
            filename
            for filename, path in cached_model_files.items()
            if not isinstance(path, str)
        ]
        if not isinstance(head_path, str):
            missing_files.append("BAAI/bge-m3/sparse_linear.pt")
        if missing_files:
            pytest.skip(
                "Pinned BGE-M3 files not in the Hugging Face cache: "
                f"{missing_files}; "
                f"pre-pull with `hf download {_BGE_M3_MODEL_ID} "
                f"--revision {_BGE_M3_MODEL_REVISION}`"
            )

        config_path = cached_model_files["config.json"]
        assert isinstance(config_path, str)
        assert isinstance(head_path, str)
        model_config = _encoder_model_config(
            pooler_config=SimpleNamespace(task="token_classify"),
            hf_config=SimpleNamespace(
                architectures=["XLMRobertaModel"],
                model_type="xlm-roberta",
                hidden_size=1024,
                bos_token_id=0,
                eos_token_id=2,
            ),
        )
        with patch(
            "vllm_metal.v1.encoder_embeddings.hf_hub_download",
            return_value=head_path,
        ):
            _model, tokenizer, adapter = EncoderEmbeddingAdapter.load(
                str(Path(config_path).parent),
                model_config=model_config,
            )
        token_ids = tokenizer.encode("Hi", add_special_tokens=True)
        hidden_states = adapter.forward_sequence_hidden_states(
            mx.array(token_ids, dtype=mx.int32),
            segment_lengths=[len(token_ids)],
        )
        weights = adapter.filter_sparse_token_weights(
            adapter.sparse_token_logits(hidden_states[0]),
            token_ids=token_ids,
            use_activation=True,
        )
        mx.eval(weights)

        assert token_ids == [0, _BGE_M3_HI_TOKEN_ID, 2]
        assert tuple(weights.shape) == (1,)
        assert float(weights.item()) == pytest.approx(
            _BGE_M3_HI_REFERENCE_WEIGHT,
            rel=_BGE_M3_8BIT_REFERENCE_REL_TOLERANCE,
        )


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
