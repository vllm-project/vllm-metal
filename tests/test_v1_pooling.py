# SPDX-License-Identifier: Apache-2.0
"""High-value contract tests for Metal V1 text embedding pooling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
import torch

pytest.importorskip("vllm", reason="vllm not installed")

from vllm.pooling_params import LateInteractionParams, PoolingParams  # noqa: E402

from tests.stub_runner import make_stub_runner  # noqa: E402
from vllm_metal.v1 import model_runner as mr  # noqa: E402


class _SequenceModel:
    def __init__(self, *, bad_shape: bool = False) -> None:
        self.bad_shape = bad_shape

    def __call__(self, input_ids, cache=None):
        if self.bad_shape:
            return mx.array([[1.0, 2.0]], dtype=mx.float32)

        token_ids = np.array(input_ids).reshape(-1).tolist()
        rows = [[float(tok), float(tok + 1), 1.0] for tok in token_ids]
        return mx.array([rows], dtype=mx.float32)


class _RecordingSequenceModel(_SequenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, input_ids, cache=None):
        self.calls += 1
        return super().__call__(input_ids, cache=cache)


class _PoolingModel:
    def __init__(self, sequence_model: object | None = None) -> None:
        self.model = sequence_model or _SequenceModel()


def _hf_config(**overrides):
    values = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _pooler_config(**overrides):
    values = {
        "task": None,
        "pooling_type": None,
        "seq_pooling_type": "LAST",
        "tok_pooling_type": "ALL",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _pooling_model_config(**overrides):
    values = {
        "runner_type": "pooling",
        "multimodal_config": None,
        "served_model_name": "stub-pooling-model",
        "model": "stub-pooling-model",
        "hf_config": _hf_config(),
        "pooler_config": _pooler_config(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_runner(
    *,
    paged: bool = True,
    model: object | None = None,
    model_config: object | None = None,
):
    return make_stub_runner(
        model=model or _PoolingModel(),
        model_config=model_config or _pooling_model_config(),
        _paged_attention_backend=object() if paged else None,
        _paged_block_size=4,
        num_layers=1,
    )


def _pooling_params(task: str | None = None, **overrides) -> PoolingParams:
    params = PoolingParams(task=task)
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def _new_req(
    req_id: str,
    token_ids: list[int],
    *,
    task: str | None = None,
    num_computed_tokens: int = 0,
    block_ids: list[int] | None = None,
    pooling_params: PoolingParams | None = None,
):
    return SimpleNamespace(
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
    new_reqs: list[object] | None = None,
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

    def test_supported_worker_tasks_rejects_non_paged_pooling(self) -> None:
        runner = _make_runner(paged=False)

        assert runner.supported_worker_tasks() == ()

    def test_supported_worker_tasks_preserves_generation_and_stt(self) -> None:
        gen_runner = make_stub_runner(
            model_config=SimpleNamespace(runner_type="generate")
        )
        stt_runner = make_stub_runner(
            _is_stt=True,
            model_config=SimpleNamespace(runner_type="generate"),
        )

        assert gen_runner.supported_worker_tasks() == ("generate",)
        assert stt_runner.supported_worker_tasks() == ("transcription",)


class TestMetalPoolingRunnerOutput:
    def test_paged_embed_preserves_request_order(self) -> None:
        runner = _make_runner()
        req_b = _new_req("req-b", [4, 5])
        req_a = _new_req("req-a", [7, 8, 9])
        sched = _scheduler_output(new_reqs=[req_b, req_a])

        with (
            patch("vllm_metal.v1.model_runner.prepare_unified"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            out = _execute_pooling(runner, sched)

        assert out.req_ids == ["req-b", "req-a"]
        assert out.sampled_token_ids == [[], []]
        assert out.pooler_output is not None
        _assert_embedding(out.pooler_output[0], 5)
        _assert_embedding(out.pooler_output[1], 9)

    def test_chunked_prefill_returns_pooler_output_only_on_final_chunk(self) -> None:
        runner = _make_runner()
        req = _new_req("req-0", [1, 2, 3, 4])
        first = _scheduler_output(
            new_reqs=[req],
            num_scheduled_tokens={"req-0": 2},
        )

        with (
            patch("vllm_metal.v1.model_runner.prepare_unified"),
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
            patch("vllm_metal.v1.model_runner.prepare_unified"),
            patch("vllm_metal.v1.model_runner.clear_context"),
        ):
            final = _execute_pooling(runner, second)

        assert final.sampled_token_ids == [[]]
        assert final.pooler_output is not None
        _assert_embedding(final.pooler_output[0], 4)


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

        with pytest.raises(NotImplementedError, match="decoder-style checkpoint"):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_pooling_requires_paged_attention(self) -> None:
        runner = _make_runner(paged=False)
        req = _new_req("req-0", [1, 2], task="embed")

        with pytest.raises(NotImplementedError, match="paged attention"):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    @pytest.mark.parametrize(
        "task",
        ["classify", "score"],
    )
    def test_unsupported_pooling_tasks_fail_fast(self, task: str) -> None:
        runner = _make_runner()
        req = _new_req("req-0", [1, 2], task=task)

        with pytest.raises(NotImplementedError, match="task"):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

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

    def test_late_interaction_pooling_fails_fast(self) -> None:
        runner = _make_runner()
        params = _pooling_params(
            late_interaction_params=LateInteractionParams(
                mode="cache_query",
                query_key="query-0",
            )
        )
        req = _new_req("req-0", [1, 2], pooling_params=params)

        with pytest.raises(NotImplementedError, match="late-interaction"):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_multimodal_pooling_fails_fast(self) -> None:
        runner = _make_runner()
        req = _new_req("req-0", [1, 2])
        req.mm_features = [object()]

        with pytest.raises(NotImplementedError, match="Multimodal pooling"):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_prompt_embeds_pooling_fails_fast_before_forward(self) -> None:
        runner = _make_runner()
        req = _new_req("req-0", [1, 2])
        req.prompt_embeds = mx.zeros((1, 2, 3), dtype=mx.float32)

        with (
            patch("vllm_metal.v1.model_runner.prepare_unified") as prepare,
            patch(
                "vllm_metal.v1.model_runner.forward_sequence_hidden_states"
            ) as forward,
            pytest.raises(NotImplementedError, match="Prompt-embedding pooling"),
        ):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

        prepare.assert_not_called()
        forward.assert_not_called()
        assert "req-0" not in runner._request_states

    @pytest.mark.parametrize(
        ("attr", "value", "message"),
        [
            ("requires_token_ids", True, "token-level ALL"),
            ("use_activation", False, "use_activation=False"),
        ],
    )
    def test_unsupported_pooling_options_fail_fast(
        self,
        attr: str,
        value: object,
        message: str,
    ) -> None:
        runner = _make_runner()
        params = _pooling_params()
        setattr(params, attr, value)
        req = _new_req("req-0", [1, 2], pooling_params=params)

        with pytest.raises(NotImplementedError, match=message):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_embedding_dimensions_are_rejected(self) -> None:
        runner = _make_runner()
        req = _new_req(
            "req-0",
            [1, 2],
            pooling_params=_pooling_params(dimensions=2),
        )

        with pytest.raises(NotImplementedError, match="dimension"):
            runner.execute_model(_scheduler_output(new_reqs=[req]))

    def test_unknown_hidden_state_shape_fails_fast(self) -> None:
        runner = _make_runner(model=_PoolingModel(_SequenceModel(bad_shape=True)))
        req = _new_req("req-0", [1, 2])
        sched = _scheduler_output(new_reqs=[req])

        with (
            patch("vllm_metal.v1.model_runner.prepare_unified"),
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
