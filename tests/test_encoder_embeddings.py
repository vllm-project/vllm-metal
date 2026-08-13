# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed encoder embedding backend (#589 restart)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from vllm_metal.v1.encoder.backend import EncoderSequenceModel
from vllm_metal.v1.encoder.registry import (
    backend_from_loaded_model,
    encoder_family_for_config,
    encoder_pooling_policy,
    load_encoder_backend,
)
from vllm_metal.v1.encoder.xlm_roberta import XLMRobertaArgs, XLMRobertaModel
from vllm_metal.v1.encoder.xlm_roberta_family import (
    XLMRobertaEmbeddingBackend,
    XLMRobertaEmbeddingFamily,
    XLMRobertaSequenceModel,
)
from vllm_metal.v1.pooling import (
    forward_sequence_hidden_states,
    pool_sequence_embedding,
    supports_embed_pooling,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SHARED_POOLING_PATHS = (
    "vllm_metal/v1/pooling.py",
    "vllm_metal/v1/model_lifecycle.py",
    "vllm_metal/v1/model_runner.py",
    "vllm_metal/v1/cache_policy.py",
)
_FAMILY_MARKERS = (
    "xlm-roberta",
    "xlm_roberta",
    "XLMRoberta",
    "RobertaModel",
    "BgeM3",
    "bge-m3",
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


def _tiny_args() -> XLMRobertaArgs:
    return XLMRobertaArgs(
        model_type="xlm-roberta",
        hidden_size=8,
        num_hidden_layers=1,
        intermediate_size=16,
        num_attention_heads=2,
        max_position_embeddings=32,
        vocab_size=32,
        type_vocab_size=1,
        add_pooling_layer=False,
        pad_token_id=1,
    )


class _FakeEncoderModule(XLMRobertaModel):
    """Minimal stand-in that records calls like the real encoder body."""

    def __init__(self) -> None:
        self.args = XLMRobertaArgs(hidden_size=4, num_hidden_layers=1)
        self.config = self.args
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...] | None]] = []

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
        shape = tuple(int(x) for x in input_ids.shape)
        mask_shape = (
            None
            if attention_mask is None
            else tuple(int(x) for x in attention_mask.shape)
        )
        self.calls.append((shape, mask_shape))
        tokens = int(input_ids.shape[-1])
        rows = [[float(i), float(i + 1), 0.0, 1.0] for i in range(tokens)]
        return mx.array([rows], dtype=mx.float32)


class TestEncoderBackendIsolation:
    def test_shared_pooling_path_has_no_family_names(self) -> None:
        for relative in _SHARED_POOLING_PATHS:
            source = (_REPO_ROOT / relative).read_text()
            for marker in _FAMILY_MARKERS:
                assert marker not in source, f"{relative} mentions {marker!r}"


class TestEncoderFamilyRegistry:
    def test_resolves_xlm_roberta_and_bge_architectures(self) -> None:
        config = _encoder_model_config()
        assert encoder_family_for_config(config) is XLMRobertaEmbeddingFamily
        assert encoder_pooling_policy(config) is not None
        assert encoder_family_for_config(
            _encoder_model_config(
                hf_config=SimpleNamespace(
                    architectures=["BgeM3EmbeddingModel"],
                    model_type="xlm-roberta",
                )
            )
        ) is XLMRobertaEmbeddingFamily

    def test_rejects_decoder_embedding_configs(self) -> None:
        config = _encoder_model_config(
            hf_config=SimpleNamespace(
                architectures=["Qwen3ForCausalLM"],
                model_type="qwen3",
            )
        )
        assert encoder_family_for_config(config) is None
        assert encoder_pooling_policy(config) is None

    def test_owns_cls_pooling_defaults(self) -> None:
        policy = XLMRobertaEmbeddingFamily.pooling_policy()
        assert policy.default_sequence_pooling_type == "CLS"
        assert policy.allowed_sequence_pooling_types == (None, "CLS", "LAST")
        assert policy.skip_paged_attention_patch is True
        assert encoder_pooling_policy(_encoder_model_config()) == policy


class TestNativeEncoderLoad:
    def test_wraps_loaded_model_with_backend(self) -> None:
        fake = _FakeEncoderModule()
        tokenizer = object()
        with patch(
            "vllm_metal.v1.encoder.xlm_roberta_family.load_encoder_model",
            return_value=(fake, tokenizer),
        ) as load_mock:
            model, loaded_tokenizer, backend = XLMRobertaEmbeddingFamily.load(
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
        assert isinstance(model, XLMRobertaSequenceModel)
        assert isinstance(backend, XLMRobertaEmbeddingBackend)
        assert backend.model is model
        assert XLMRobertaEmbeddingFamily.from_loaded_model(model) is not None
        assert XLMRobertaEmbeddingFamily.from_loaded_model(object()) is None
        assert backend_from_loaded_model(model) is not None
        hidden = model.model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=MagicMock())
        assert hidden.shape == (1, 3, 4)
        assert fake.calls == [((1, 3), (1, 3))]


class TestEncoderPagedSetup:
    def test_setup_paged_attention_skips_patching_via_backend(self) -> None:
        from vllm_metal.v1.cache_policy import WorkerCachePlanner

        fake = _FakeEncoderModule()
        model = XLMRobertaSequenceModel(fake)
        backend = XLMRobertaEmbeddingBackend(model)
        runner = SimpleNamespace(
            model=model,
            is_mla=False,
            _encoder_embedding_backend=backend,
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
        paged_backend = MagicMock()
        paged_backend.patch_model = MagicMock(return_value=2)
        paged_backend.initialize = MagicMock()
        runner.build_paged_attention_runtime.return_value = paged_backend

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

        paged_backend.patch_model.assert_not_called()
        runner.install_paged_attention_runtime.assert_called_once()


class TestEncoderPoolingForward:
    def test_supports_embed_for_encoder_backend(self) -> None:
        model = XLMRobertaSequenceModel(_FakeEncoderModule())
        assert supports_embed_pooling(model, _encoder_model_config())

    def test_forwards_packed_segments_independently(self) -> None:
        fake = _FakeEncoderModule()
        model = XLMRobertaSequenceModel(fake)
        backend = XLMRobertaEmbeddingBackend(model)
        hidden = forward_sequence_hidden_states(
            model,
            mx.array([[10, 11, 20, 21, 22]], dtype=mx.int32),
            cache=[MagicMock()],
            model_config=_encoder_model_config(),
            segment_lengths=[2, 3],
            encoder_backend=backend,
        )
        assert hidden.shape == (1, 5, 4)
        assert fake.calls == [((1, 2), (1, 2)), ((1, 3), (1, 3))]
        assert float(hidden[0, 0, 0]) == 0.0
        assert float(hidden[0, 2, 0]) == 0.0


def _write_tiny_checkpoint(tmp_path: Path) -> tuple[Path, XLMRobertaModel]:
    args = _tiny_args()
    source = XLMRobertaModel(args)
    mx.eval(source.parameters())
    weights = dict(tree_flatten(source.parameters()))
    weight_path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(weight_path), weights)
    config = {
        "model_type": "xlm-roberta",
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "intermediate_size": args.intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "max_position_embeddings": args.max_position_embeddings,
        "vocab_size": args.vocab_size,
        "type_vocab_size": args.type_vocab_size,
        "add_pooling_layer": False,
        "pad_token_id": args.pad_token_id,
        "architectures": ["XLMRobertaModel"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path, source


class TestEncoderLoadAndClsParity:
    def test_loads_tiny_checkpoint_and_matches_cls_pooling(self, tmp_path: Path) -> None:
        ckpt_dir, source = _write_tiny_checkpoint(tmp_path)
        tokenizer = object()
        config = _encoder_model_config()
        with patch(
            "vllm_metal.v1.encoder.loader._load_tokenizer",
            return_value=tokenizer,
        ):
            model, loaded_tokenizer, backend = load_encoder_backend(
                str(ckpt_dir),
                config,
                lazy=True,
            )

        assert loaded_tokenizer is tokenizer
        assert isinstance(model, EncoderSequenceModel)
        assert isinstance(backend, XLMRobertaEmbeddingBackend)

        input_ids = mx.array([[0, 5, 6, 7, 2]], dtype=mx.int32)
        loaded_hidden = backend.forward_sequence_hidden_states(input_ids)
        source_hidden = source(
            input_ids,
            attention_mask=mx.ones(input_ids.shape, dtype=mx.int32),
        )
        mx.eval(loaded_hidden, source_hidden)
        np.testing.assert_allclose(
            np.array(loaded_hidden),
            np.array(source_hidden),
            rtol=1e-4,
            atol=1e-4,
        )

        pooled = pool_sequence_embedding(
            loaded_hidden,
            token_index=0,
            model_config=config,
        )
        cls = np.array(source_hidden[0, 0], dtype=np.float32)
        expected = cls / max(float(np.linalg.norm(cls)), 1e-12)
        np.testing.assert_allclose(
            pooled.detach().numpy(),
            expected,
            rtol=1e-4,
            atol=1e-4,
        )
        assert pooled.shape == (8,)
