# SPDX-License-Identifier: Apache-2.0
"""High-value contract tests for Metal V1 text pooling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
import torch
from torch.nn.functional import normalize
from transformers import XLMRobertaConfig as TorchXLMRobertaConfig
from transformers import XLMRobertaModel as TorchXLMRobertaModel

pytest.importorskip("vllm", reason="vllm not installed")

from vllm.pooling_params import LateInteractionParams, PoolingParams  # noqa: E402
from vllm.v1.core.sched.output import NewRequestData  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: E402

from tests.stub_runner import make_stub_runner  # noqa: E402
from vllm_metal.attention.runtime.mha import MHAPagedAttentionRuntime  # noqa: E402
from vllm_metal.multimodal import MultiModalFeatureSpec, PlaceholderRange  # noqa: E402
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch  # noqa: E402
from vllm_metal.v1 import model_runner as mr  # noqa: E402
from vllm_metal.v1.model_lifecycle import (  # noqa: E402
    LoadedEncoderPoolingModel,
    LoadedGenerationModel,
    ModelLifecycle,
)
from vllm_metal.v1.pooling.backends.decoder.models.qwen3 import (  # noqa: E402
    Qwen3RerankerPooler,
)
from vllm_metal.v1.pooling.backends.decoder.runtime import (  # noqa: E402
    DecoderModelView,
    MetalDecoderPoolingBackend,
)
from vllm_metal.v1.pooling.backends.encoder.models.xlm_roberta import (  # noqa: E402
    load_xlm_roberta_backend,
)
from vllm_metal.v1.pooling.backends.encoder.runtime import (  # noqa: E402
    MetalEncoderPoolingBackend,
)
from vllm_metal.v1.pooling.contract import (  # noqa: E402
    DecoderPoolingSpan,
    PoolingCapabilities,
)
from vllm_metal.v1.pooling.validation import PoolingConfigView  # noqa: E402


class _SequenceModel:
    def __init__(self, *, bad_shape: bool = False) -> None:
        self.bad_shape = bad_shape

    def __call__(self, input_ids, cache=None):
        if self.bad_shape:
            return mx.array([[1.0, 2.0]], dtype=mx.float32)

        token_ids = np.array(input_ids).reshape(-1).tolist()
        rows = [[float(tok), float(tok + 1), 1.0] for tok in token_ids]
        return mx.array([rows], dtype=mx.float32)


class _TiedEmbedding:
    def as_linear(self, vector):
        logits = mx.zeros((8,), dtype=mx.float32)
        return mx.concatenate(
            [
                mx.stack([vector[0], vector[0] * 2.0]),
                logits[2:],
            ]
        )


class _UntiedLmHead:
    def __call__(self, vector):
        logits = mx.zeros((8,), dtype=mx.float32)
        return mx.concatenate(
            [
                mx.stack([vector[0] * 0.0, vector[0] + 10.0]),
                logits[2:],
            ]
        )


class _BadTiedEmbedding:
    def as_linear(self, vector):
        return mx.zeros((2, 2), dtype=mx.float32)


class _ClassifierSequenceModel(_SequenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = _TiedEmbedding()


class _BadClassifierSequenceModel(_SequenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = _BadTiedEmbedding()


class _RecordingSequenceModel(_SequenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, input_ids, cache=None):
        self.calls += 1
        return super().__call__(input_ids, cache=cache)


class _NonArraySequenceModel:
    def __call__(self, input_ids, cache=None):
        del input_ids, cache
        return object()


class _EncoderModel:
    def __call__(self, input_ids, attention_mask):
        del attention_mask
        rows = []
        for row in np.array(input_ids).tolist():
            rows.append([[float(tok), float(tok + 1), 1.0] for tok in row])
        return mx.array(rows, dtype=mx.float32)


class _PoolingModel:
    def __init__(self, sequence_model: object | None = None) -> None:
        self.model = sequence_model or _SequenceModel()


class _UntiedClassifierModel(_PoolingModel):
    def __init__(self) -> None:
        super().__init__(_ClassifierSequenceModel())
        self.args = SimpleNamespace(tie_word_embeddings=False)
        self.lm_head = _UntiedLmHead()


class _ClassifierTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int | None:
        return {"no": 0, "yes": 1}.get(token)

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        token_id = self.convert_tokens_to_ids(token)
        return [] if token_id is None else [token_id]


class _HFConfig(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()


def _hf_config(**overrides):
    values = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
    }
    values.update(overrides)
    return _HFConfig(**values)


def _qwen3_reranker_hf_config(**overrides):
    values = {
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
        "num_labels": 1,
        "tie_word_embeddings": True,
    }
    values.update(overrides)
    return _hf_config(**values)


def _pooler_config(**overrides):
    values = {
        "task": None,
        "pooling_type": None,
        "seq_pooling_type": "LAST",
        "tok_pooling_type": "ALL",
        "enable_chunked_processing": False,
        "logit_mean": None,
        "logit_sigma": None,
        "use_activation": None,
        "dimensions": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _pooling_model_config(**overrides):
    values = {
        "runner_type": "pooling",
        "multimodal_config": None,
        "served_model_name": "stub-pooling-model",
        "model": "stub-pooling-model",
        "dtype": torch.float16,
        "hf_config": _hf_config(),
        "is_multimodal_model": False,
        "pooler_config": _pooler_config(),
        "quantization": None,
        "trust_remote_code": False,
        "revision": None,
        "tokenizer": "stub-pooling-model",
        "tokenizer_revision": None,
        "get_head_size": lambda: 128,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _classification_model_config(**overrides):
    values = {
        "hf_config": _qwen3_reranker_hf_config(),
        "pooler_config": _pooler_config(),
    }
    values.update(overrides)
    return _pooling_model_config(**values)


def _encoder_model_config(**overrides):
    values = {
        "hf_config": _hf_config(
            architectures=["XLMRobertaModel"],
            model_type="xlm-roberta",
        ),
        "pooler_config": _pooler_config(seq_pooling_type="CLS"),
    }
    values.update(overrides)
    return _pooling_model_config(**values)


def _make_runner(
    *,
    paged: bool = True,
    model: object | None = None,
    model_config: object | None = None,
    tokenizer: object | None = None,
):
    return make_stub_runner(
        model=model or _PoolingModel(),
        model_config=model_config or _pooling_model_config(),
        tokenizer=tokenizer,
        _paged_attention_runtime=(
            MHAPagedAttentionRuntime(
                num_layers=1,
                num_kv_heads=1,
                head_dim=4,
                block_size=4,
                dtype=mx.float32,
            )
            if paged
            else None
        ),
        _paged_block_size=4,
        num_layers=1,
    )


def _pooling_params(task: str | None = None, **overrides) -> PoolingParams:
    return PoolingParams(task=task, **overrides)


def _new_req(
    req_id: str,
    token_ids: list[int],
    *,
    task: str | None = None,
    num_computed_tokens: int = 0,
    block_ids: list[int] | None = None,
    pooling_params: PoolingParams | None = None,
) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=token_ids,
        mm_features=[],
        sampling_params=None,
        pooling_params=pooling_params or _pooling_params(task),
        block_ids=(block_ids or [0, 1],),
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
        prompt_embeds=None,
    )


def _cached_req_data(req_ids: list[str], num_computed_tokens: list[int]):
    return SimpleNamespace(
        req_ids=req_ids,
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None] * len(req_ids),
        num_computed_tokens=num_computed_tokens,
        num_output_tokens=[0] * len(req_ids),
    )


def _scheduler_output(
    *,
    new_reqs: list[NewRequestData] | None = None,
    cached_req_ids: list[str] | None = None,
    cached_num_computed_tokens: list[int] | None = None,
    num_scheduled_tokens: dict[str, int] | None = None,
):
    new_reqs = new_reqs or []
    cached_req_ids = cached_req_ids or []
    if num_scheduled_tokens is None:
        num_scheduled_tokens = {
            req.req_id: len(req.prompt_token_ids or []) - req.num_computed_tokens
            for req in new_reqs
        }
        num_scheduled_tokens.update(dict.fromkeys(cached_req_ids, 1))

    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=_cached_req_data(
            cached_req_ids,
            cached_num_computed_tokens or [0] * len(cached_req_ids),
        ),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=None,
        kv_cache_block_copies=None,
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=set(),
        has_structured_output_requests=False,
    )


def _expected_embedding(token_id: int) -> torch.Tensor:
    vector = torch.tensor([float(token_id), float(token_id + 1), 1.0])
    return vector / vector.norm()


def _assert_embedding(tensor: torch.Tensor | None, token_id: int) -> None:
    assert tensor is not None
    assert tensor.device.type == "cpu"
    assert tensor.shape == (3,)
    assert torch.allclose(tensor, _expected_embedding(token_id), atol=1e-6)


def _expected_score(token_id: int, *, activated: bool = True) -> torch.Tensor:
    raw = torch.tensor([float(token_id)])
    return torch.sigmoid(raw) if activated else raw


def _assert_score(
    tensor: torch.Tensor | None,
    token_id: int,
    *,
    activated: bool = True,
) -> None:
    assert tensor is not None
    assert tensor.device.type == "cpu"
    assert tensor.shape == (1,)
    assert torch.allclose(tensor, _expected_score(token_id, activated=activated))


def _execute_pooling(runner, sched):
    out = runner.execute_model(sched)
    assert out is not None
    return out


class TestMetalPoolingCapabilities:
    def test_supported_worker_tasks_for_last_text_embedding_model(self) -> None:
        runner = _make_runner()

        assert runner.supported_worker_tasks() == ("embed",)

    def test_supported_worker_tasks_rejects_incompatible_pooling_model(self) -> None:
        runner = make_stub_runner(
            model=object(),
            model_config=_pooling_model_config(),
        )

        assert runner.supported_worker_tasks() == ()

    def test_supported_worker_tasks_for_qwen3_reranker_classify_model(self) -> None:
        runner = _make_runner(
            model=_PoolingModel(_ClassifierSequenceModel()),
            model_config=_classification_model_config(),
            tokenizer=_ClassifierTokenizer(),
        )

        assert runner.supported_worker_tasks() == ("classify",)

    def test_supported_worker_tasks_for_encoder_embedding_model_without_paged_attention(
        self,
    ) -> None:
        runner = _make_runner(
            paged=False,
            model_config=_encoder_model_config(),
        )
        runner._pooling_backend = MetalEncoderPoolingBackend(
            PoolingConfigView(runner.model_config),
            _EncoderModel(),
        )

        assert runner.supported_worker_tasks() == ("embed",)

    def test_supported_worker_tasks_for_untied_qwen3_reranker_model(self) -> None:
        runner = _make_runner(
            model=_UntiedClassifierModel(),
            model_config=_classification_model_config(
                hf_config=_qwen3_reranker_hf_config(tie_word_embeddings=False)
            ),
            tokenizer=_ClassifierTokenizer(),
        )

        assert runner.supported_worker_tasks() == ("classify",)

    @pytest.mark.parametrize(
        ("model", "model_config", "tokenizer"),
        [
            (
                _PoolingModel(_ClassifierSequenceModel()),
                _classification_model_config(
                    hf_config=_qwen3_reranker_hf_config(
                        classifier_from_token=["no", "maybe"],
                    )
                ),
                _ClassifierTokenizer(),
            ),
            (
                _PoolingModel(_ClassifierSequenceModel()),
                _classification_model_config(
                    hf_config=_qwen3_reranker_hf_config(
                        is_original_qwen3_reranker=False,
                    )
                ),
                _ClassifierTokenizer(),
            ),
            (
                _PoolingModel(_ClassifierSequenceModel()),
                _classification_model_config(
                    hf_config=_qwen3_reranker_hf_config(tie_word_embeddings=False)
                ),
                _ClassifierTokenizer(),
            ),
            (
                _PoolingModel(_ClassifierSequenceModel()),
                _classification_model_config(
                    hf_config=_qwen3_reranker_hf_config(tie_word_embeddings=None)
                ),
                _ClassifierTokenizer(),
            ),
            (
                SimpleNamespace(
                    model=_ClassifierSequenceModel(),
                    lm_head=_UntiedLmHead(),
                ),
                _classification_model_config(
                    hf_config=_qwen3_reranker_hf_config(tie_word_embeddings=None)
                ),
                _ClassifierTokenizer(),
            ),
            (
                _PoolingModel(_SequenceModel()),
                _classification_model_config(),
                _ClassifierTokenizer(),
            ),
            (
                _PoolingModel(_ClassifierSequenceModel()),
                _classification_model_config(),
                object(),
            ),
        ],
    )
    def test_supported_worker_tasks_rejects_incomplete_qwen3_reranker_contract(
        self,
        model: object,
        model_config: object,
        tokenizer: object,
    ) -> None:
        runner = _make_runner(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
        )

        assert runner.supported_worker_tasks() == ()

    def test_supported_worker_tasks_rejects_non_paged_pooling(self) -> None:
        runner = _make_runner(paged=False)

        assert runner.supported_worker_tasks() == ()

    def test_supported_worker_tasks_uses_backend_paged_capability(self) -> None:
        class _NoPagedBackend:
            capabilities = PoolingCapabilities(
                execution_kind="encoder",
                requires_paged_attention=False,
                uses_kv_cache=False,
                supports_chunked_requests=False,
            )

            def supported_tasks(self):
                return ("embed",)

            def validate_params(self, pooling_params):
                del pooling_params

            def profile_forward(self, input_ids):
                return input_ids

        runner = _make_runner(paged=False)
        runner._pooling_backend = _NoPagedBackend()

        assert runner.supported_worker_tasks() == ("embed",)

    def test_supported_worker_tasks_preserves_generation(self) -> None:
        gen_runner = make_stub_runner(
            model_config=SimpleNamespace(runner_type="generate")
        )

        assert gen_runner.supported_worker_tasks() == ("generate",)

    def test_encoder_pooling_reports_empty_kv_spec(self) -> None:
        runner = _make_runner(
            paged=False,
            model_config=_encoder_model_config(),
        )
        runner._pooling_backend = MetalEncoderPoolingBackend(
            PoolingConfigView(runner.model_config),
            _EncoderModel(),
        )

        assert runner.scheduler_memory_reporting_mode(
            paged_attention_enabled=False
        ) == ("pooling_no_kv")
        assert runner.get_kv_cache_spec() == {}
        runner.initialize_kv_cache(
            KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[])
        )

    def test_load_model_installs_pooling_backend_after_lora_setup(self) -> None:
        events: list[str] = []
        runner = _make_runner()
        runner._pooling_backend = None
        lifecycle = ModelLifecycle(runner, runner._model_adapter)
        runner._model_lifecycle = lifecycle
        runner.metal_config = SimpleNamespace(use_paged_attention=True)
        runner.scheduler_config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=1,
        )
        runner.kv_cache_dtype = None
        loaded = LoadedGenerationModel(
            model=runner.model,
            tokenizer=runner.tokenizer,
            model_args={},
        )

        def setup_lora(**kwargs) -> None:
            del kwargs
            events.append("lora")
            assert runner._pooling_backend is None

        with (
            patch.object(lifecycle, "_load_generation", return_value=loaded),
            patch.object(lifecycle, "_install_generation_model"),
            patch.object(lifecycle, "resolve_model_dims"),
            patch.object(lifecycle, "_install_runtime_extensions"),
        ):
            runner._lora = SimpleNamespace(setup=setup_lora)
            runner.load_model()

        assert events == ["lora"]
        assert runner._pooling_backend is not None
        assert runner._pooling_backend.supported_tasks() == ("embed",)

    def test_load_model_installs_encoder_backend_without_generation_extensions(
        self,
    ) -> None:
        runner = _make_runner(
            paged=False,
            model_config=_encoder_model_config(),
        )
        runner._pooling_backend = None
        pooling_backend = MetalEncoderPoolingBackend(
            PoolingConfigView(runner.model_config),
            _EncoderModel(),
        )
        lifecycle = ModelLifecycle(runner, runner._model_adapter)
        runner._model_lifecycle = lifecycle
        loaded = LoadedEncoderPoolingModel(
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=16)),
            tokenizer=object(),
            model_args={"vocab_size": 16},
            pooling_backend=pooling_backend,
        )

        with (
            patch.object(lifecycle, "_load_encoder_pooling", return_value=loaded),
            patch.object(lifecycle, "resolve_model_dims") as resolve_dims,
            patch.object(lifecycle, "_install_runtime_extensions") as extensions,
        ):
            runner.load_model()

        assert runner._pooling_backend is pooling_backend
        resolve_dims.assert_not_called()
        extensions.assert_not_called()

    def test_duplicate_supported_pooler_tasks_fail_at_backend_construction(
        self,
    ) -> None:
        class _SupportedEmbedPooler:
            task = "embed"

            def is_supported(self) -> bool:
                return True

            def pool_one(self, hidden_states, span):
                del hidden_states, span
                return torch.zeros((1,), dtype=torch.float32)

        config = PoolingConfigView(_pooling_model_config())
        model_view = DecoderModelView(_PoolingModel())

        with pytest.raises(RuntimeError, match="multiple supported poolers"):
            MetalDecoderPoolingBackend(
                config,
                model_view,
                (_SupportedEmbedPooler(), _SupportedEmbedPooler()),
            )

    def test_xlm_roberta_checkpoint_matches_transformers(
        self,
        tmp_path,
    ) -> None:
        torch.manual_seed(0)
        torch_config = TorchXLMRobertaConfig(
            vocab_size=17,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=16,
            type_vocab_size=1,
            layer_norm_eps=1e-5,
            pad_token_id=1,
            hidden_act="gelu",
            position_embedding_type="absolute",
        )
        transformers_model = TorchXLMRobertaModel(torch_config).eval()
        transformers_model.save_pretrained(tmp_path, safe_serialization=True)

        model_config = _encoder_model_config(
            model=str(tmp_path),
            tokenizer="custom-tokenizer",
            tokenizer_revision="tokenizer-revision",
            hf_config=torch_config,
            dtype=torch.float32,
        )

        with patch(
            "vllm_metal.v1.pooling.backends.encoder.models.xlm_roberta."
            "AutoTokenizer.from_pretrained",
            return_value=object(),
        ) as load_tokenizer:
            mlx_model, _, model_args, pooling_backend = load_xlm_roberta_backend(
                model_config
            )

        input_ids = torch.tensor(
            [
                [0, 5, 6, 2, 1],
                [0, 7, 2, 1, 1],
            ],
            dtype=torch.long,
        )
        attention_mask = (input_ids != torch_config.pad_token_id).to(torch.long)
        with torch.no_grad():
            expected_hidden = transformers_model(
                input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        actual_hidden = mlx_model(
            mx.array(input_ids.numpy(), dtype=mx.int32),
            mx.array(attention_mask.numpy(), dtype=mx.int32),
        )
        actual_hidden_torch = mlx_to_torch(
            actual_hidden.astype(mx.float32),
            device="cpu",
        )

        assert model_args["hidden_size"] == torch_config.hidden_size
        assert torch.allclose(
            actual_hidden_torch,
            expected_hidden,
            atol=1e-5,
            rtol=1e-5,
        )

        request = _new_req("req-0", [0, 5, 6, 2], task="embed")
        outputs = pooling_backend.pool_scheduler_output(
            _scheduler_output(new_reqs=[request]),
            model_config,
        )
        expected_cls = normalize(expected_hidden[0, 0].float(), dim=0)

        assert len(outputs) == 1
        assert torch.allclose(
            outputs[0].pooler_output,
            expected_cls,
            atol=1e-5,
            rtol=1e-5,
        )
        load_tokenizer.assert_called_once_with(
            "custom-tokenizer",
            revision="tokenizer-revision",
            trust_remote_code=False,
        )

    def test_xlm_roberta_loader_rejects_quantization_before_download(self) -> None:
        model_config = _encoder_model_config(
            model="missing-xlm-roberta-model",
            hf_config=TorchXLMRobertaConfig(),
            quantization="awq",
        )

        with pytest.raises(NotImplementedError, match="quantization"):
            load_xlm_roberta_backend(model_config)


class TestMetalPoolingRunnerOutput:
    def test_paged_embed_preserves_request_order(self) -> None:
        runner = _make_runner()
        req_b = _new_req("req-b", [4, 5])
        req_a = _new_req("req-a", [7, 8, 9])
        sched = _scheduler_output(new_reqs=[req_b, req_a])

        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            out = _execute_pooling(runner, sched)

        assert out.req_ids == ["req-b", "req-a"]
        assert out.sampled_token_ids == [[], []]
        assert out.pooler_output is not None
        _assert_embedding(out.pooler_output[0], 5)
        _assert_embedding(out.pooler_output[1], 9)

    def test_encoder_embed_preserves_request_order_without_paged_attention(
        self,
    ) -> None:
        runner = _make_runner(
            paged=False,
            model_config=_encoder_model_config(),
        )
        runner._pooling_backend = MetalEncoderPoolingBackend(
            PoolingConfigView(runner.model_config),
            _EncoderModel(),
        )
        req_b = _new_req("req-b", [4, 5])
        req_a = _new_req("req-a", [7, 8, 9])

        with patch("vllm_metal.v1.model_runner.prepare_grouped") as prepare:
            out = _execute_pooling(runner, _scheduler_output(new_reqs=[req_b, req_a]))

        prepare.assert_not_called()
        assert out.req_ids == ["req-b", "req-a"]
        assert out.sampled_token_ids == [[], []]
        assert out.pooler_output is not None
        _assert_embedding(out.pooler_output[0], 4)
        _assert_embedding(out.pooler_output[1], 7)

    def test_chunked_prefill_returns_pooler_output_only_on_final_chunk(self) -> None:
        runner = _make_runner()
        req = _new_req("req-0", [1, 2, 3, 4])
        first = _scheduler_output(
            new_reqs=[req],
            num_scheduled_tokens={"req-0": 2},
        )

        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            partial = _execute_pooling(runner, first)

        assert partial.sampled_token_ids == [[]]
        assert partial.pooler_output == [None]
        assert runner._request_states["req-0"].pooling_params is not None

        second = _scheduler_output(
            cached_req_ids=["req-0"],
            cached_num_computed_tokens=[2],
            num_scheduled_tokens={"req-0": 2},
        )
        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            final = _execute_pooling(runner, second)

        assert final.sampled_token_ids == [[]]
        assert final.pooler_output is not None
        _assert_embedding(final.pooler_output[0], 4)

    def test_paged_classify_returns_qwen3_reranker_scores(self) -> None:
        runner = _make_runner(
            model=_PoolingModel(_ClassifierSequenceModel()),
            model_config=_classification_model_config(),
            tokenizer=_ClassifierTokenizer(),
        )
        req_b = _new_req("req-b", [4, 5], task="classify")
        req_a = _new_req("req-a", [7, 8, 9], task="classify")
        sched = _scheduler_output(new_reqs=[req_b, req_a])

        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            out = _execute_pooling(runner, sched)

        assert out.req_ids == ["req-b", "req-a"]
        assert out.sampled_token_ids == [[], []]
        assert out.pooler_output is not None
        _assert_score(out.pooler_output[0], 5)
        _assert_score(out.pooler_output[1], 9)

    def test_paged_classify_can_return_raw_qwen3_reranker_scores(self) -> None:
        runner = _make_runner(
            model=_PoolingModel(_ClassifierSequenceModel()),
            model_config=_classification_model_config(),
            tokenizer=_ClassifierTokenizer(),
        )
        req = _new_req(
            "req-0",
            [2, 3],
            pooling_params=_pooling_params(task="classify", use_activation=False),
        )
        sched = _scheduler_output(new_reqs=[req])

        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            out = _execute_pooling(runner, sched)

        assert out.pooler_output is not None
        _assert_score(out.pooler_output[0], 3, activated=False)

    def test_paged_classify_uses_lm_head_for_untied_qwen3_reranker(self) -> None:
        runner = _make_runner(
            model=_UntiedClassifierModel(),
            model_config=_classification_model_config(
                hf_config=_qwen3_reranker_hf_config(tie_word_embeddings=False)
            ),
            tokenizer=_ClassifierTokenizer(),
        )
        req = _new_req(
            "req-0",
            [2, 3],
            pooling_params=_pooling_params(task="classify", use_activation=False),
        )
        sched = _scheduler_output(new_reqs=[req])

        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            out = _execute_pooling(runner, sched)

        assert out.pooler_output is not None
        assert torch.allclose(out.pooler_output[0], torch.tensor([13.0]))

    def test_paged_classify_applies_logit_calibration(self) -> None:
        runner = _make_runner(
            model=_PoolingModel(_ClassifierSequenceModel()),
            model_config=_classification_model_config(
                pooler_config=_pooler_config(logit_mean=1.0, logit_sigma=2.0)
            ),
            tokenizer=_ClassifierTokenizer(),
        )
        req = _new_req("req-0", [2, 3], task="classify")
        sched = _scheduler_output(new_reqs=[req])

        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            out = _execute_pooling(runner, sched)

        assert out.pooler_output is not None
        assert torch.allclose(
            out.pooler_output[0],
            torch.sigmoid(torch.tensor([(3.0 - 1.0) / 2.0])),
        )


class TestMetalPoolingFailFast:
    def test_non_decoder_pooling_models_fail_fast_on_execute(
        self,
    ) -> None:
        runner = _make_runner(
            model_config=_pooling_model_config(
                hf_config=_hf_config(architectures=["Qwen3ForSequenceClassification"])
            )
        )
        req = _new_req("req-0", [1, 2], task="embed")

        with pytest.raises(NotImplementedError, match="task='embed'"):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_pooling_requires_paged_attention(self) -> None:
        runner = _make_runner(paged=False)
        req = _new_req("req-0", [1, 2], task="embed")

        with pytest.raises(NotImplementedError, match="paged attention"):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_encoder_pooling_rejects_chunked_requests(self) -> None:
        runner = _make_runner(
            paged=False,
            model_config=_encoder_model_config(),
        )
        runner._pooling_backend = MetalEncoderPoolingBackend(
            PoolingConfigView(runner.model_config),
            _EncoderModel(),
        )
        req = _new_req("req-0", [1, 2, 3], num_computed_tokens=1)

        with pytest.raises(NotImplementedError, match="full-prompt"):
            runner.execute_model(
                _scheduler_output(
                    new_reqs=[req],
                    num_scheduled_tokens={"req-0": 2},
                )
            )

    @pytest.mark.parametrize(
        "task",
        ["token_embed", "token_classify", "plugin"],
    )
    def test_unsupported_pooling_tasks_fail_fast(self, task: str) -> None:
        runner = _make_runner()
        req = _new_req("req-0", [1, 2], task=task)

        with pytest.raises(NotImplementedError, match="task"):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_incomplete_classify_contract_fails_before_forward(self) -> None:
        runner = _make_runner(
            model=_PoolingModel(_SequenceModel()),
            model_config=_classification_model_config(),
            tokenizer=_ClassifierTokenizer(),
        )
        req = _new_req("req-0", [1, 2], task="classify")

        assert runner._pooling_backend is not None
        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped") as prepare,
            patch.object(runner._pooling_backend, "forward_packed") as forward,
            pytest.raises(NotImplementedError, match="task='classify'"),
        ):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

        prepare.assert_not_called()
        forward.assert_not_called()

    def test_body_output_contract_fails_fast(self) -> None:
        runner = _make_runner(model=_PoolingModel(_NonArraySequenceModel()))
        req = _new_req("req-0", [1, 2])

        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped"),
            patch("vllm_metal.v1.model_runner.clear_context"),
            pytest.raises(ValueError, match="expected MLX hidden states"),
        ):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_classify_hidden_state_shape_fails_fast(self) -> None:
        model = _PoolingModel(_ClassifierSequenceModel())
        span = DecoderPoolingSpan(
            start_row=0,
            num_tokens=1,
            is_complete=True,
            pooling_params=_pooling_params(task="classify"),
        )
        with pytest.raises(ValueError, match="hidden states with shape"):
            Qwen3RerankerPooler(
                model,
                model.model,
                _classification_model_config(),
                _ClassifierTokenizer(),
            ).pool_one(
                mx.array([[1.0, 2.0]], dtype=mx.float32),
                span,
            )

    def test_classify_logits_shape_fails_fast(self) -> None:
        model = _PoolingModel(_BadClassifierSequenceModel())
        span = DecoderPoolingSpan(
            start_row=0,
            num_tokens=1,
            is_complete=True,
            pooling_params=_pooling_params(task="classify"),
        )
        with pytest.raises(ValueError, match="classifier logits with shape"):
            Qwen3RerankerPooler(
                model,
                model.model,
                _classification_model_config(),
                _ClassifierTokenizer(),
            ).pool_one(
                mx.array([[[1.0, 2.0, 3.0]]], dtype=mx.float32),
                span,
            )

    @pytest.mark.parametrize(
        ("attr", "pooling_type"),
        [
            ("seq_pooling_type", "MEAN"),
            ("pooling_type", "CLS"),
        ],
    )
    def test_unsupported_pooling_strategies_fail_fast(
        self,
        attr: str,
        pooling_type: str,
    ) -> None:
        runner = _make_runner(
            model_config=_pooling_model_config(
                pooler_config=_pooler_config(**{attr: pooling_type}),
            )
        )
        req = _new_req("req-0", [1, 2])

        with pytest.raises(NotImplementedError, match="LAST"):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_multimodal_pooling_fails_fast(self) -> None:
        runner = _make_runner()
        req = _new_req("req-0", [1, 2])
        req.mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="image-0",
                mm_position=PlaceholderRange(offset=0, length=1),
            )
        ]

        assert runner._pooling_backend is not None
        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped") as prepare,
            patch.object(runner._pooling_backend, "forward_packed") as forward,
            pytest.raises(NotImplementedError, match="Multimodal pooling"),
        ):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

        prepare.assert_not_called()
        forward.assert_not_called()
        assert "req-0" not in runner._request_states

    def test_prompt_embeds_pooling_fails_fast_before_forward(self) -> None:
        runner = _make_runner()
        req = _new_req("req-0", [1, 2])
        req.prompt_embeds = torch.zeros((1, 2, 3), dtype=torch.float32)

        assert runner._pooling_backend is not None
        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped") as prepare,
            patch.object(runner._pooling_backend, "forward_packed") as forward,
            pytest.raises(NotImplementedError, match="Prompt-embedding pooling"),
        ):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

        prepare.assert_not_called()
        forward.assert_not_called()
        assert "req-0" not in runner._request_states

    @pytest.mark.parametrize(
        ("pooling_params", "message"),
        [
            (
                _pooling_params(
                    late_interaction_params=LateInteractionParams(
                        mode="cache_query",
                        query_key="query-0",
                    )
                ),
                "late-interaction",
            ),
            (_pooling_params(requires_token_ids=True), "token-level ALL"),
            (_pooling_params(step_tag_id=1), "STEP"),
            (_pooling_params(returned_token_ids=[1]), "returned_token_ids"),
            (_pooling_params(extra_kwargs={"foo": True}), "extra pooling kwargs"),
            (_pooling_params(use_activation=False), "use_activation=False"),
            (_pooling_params(dimensions=2), "dimension"),
            (
                _pooling_params(requires_token_ids=True, dimensions=2),
                "token-level ALL",
            ),
        ],
    )
    def test_unsupported_pooling_options_fail_fast(
        self,
        pooling_params: PoolingParams,
        message: str,
    ) -> None:
        runner = _make_runner()
        req = _new_req("req-0", [1, 2], pooling_params=pooling_params)

        with pytest.raises(NotImplementedError, match=message):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_unknown_hidden_state_shape_fails_fast(self) -> None:
        runner = _make_runner(model=_PoolingModel(_SequenceModel(bad_shape=True)))
        req = _new_req("req-0", [1, 2])
        sched = _scheduler_output(new_reqs=[req])

        with (
            patch("vllm_metal.v1.model_runner.prepare_grouped"),
            patch("vllm_metal.v1.model_runner.clear_context"),
            pytest.raises(ValueError, match="hidden states"),
        ):
            runner.execute_model(sched)


class TestMetalPoolingProfileWarmup:
    def test_profile_run_uses_pooling_forward_without_logits(self) -> None:
        sequence_model = _RecordingSequenceModel()
        runner = _make_runner(model=_PoolingModel(sequence_model))
        runner.scheduler_config = SimpleNamespace(max_num_batched_tokens=3)
        runner._extract_logits = MagicMock(side_effect=AssertionError("logits path"))

        with (
            patch.object(mr.mx, "clear_cache"),
            patch.object(mr.mx, "get_cache_memory", side_effect=[100, 180]),
            patch.object(mr.mx, "set_cache_limit") as set_cache_limit,
        ):
            overhead = runner.profile_run()

        assert overhead == 80
        assert sequence_model.calls == 1
        runner._extract_logits.assert_not_called()
        set_cache_limit.assert_called_once_with(80)
