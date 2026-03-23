# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.outputs import ModelRunnerOutput

import vllm_metal.v1.model_runner as mr


class TestV1MetalModelRunnerGenerate:
    def _make_runner(self) -> mr.MetalModelRunner:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model = object()
        runner.tokenizer = object()
        return runner

    def test_accumulates_streamed_segments(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
            captured["prompt"] = prompt
            captured["max_tokens"] = max_tokens
            captured["kwargs"] = kwargs
            yield SimpleNamespace(text="hello")
            yield SimpleNamespace(text=" ")
            yield SimpleNamespace(text="world")

        monkeypatch.setattr(mr, "stream_generate", fake_stream_generate)

        runner = self._make_runner()
        out = runner.generate("p", max_tokens=3, temperature=0.0)

        assert out == "hello world"
        assert captured["prompt"] == "p"
        assert captured["max_tokens"] == 3
        kwargs = captured.get("kwargs")
        assert isinstance(kwargs, dict)
        # mlx_lm 0.29+ uses sampler parameter instead of temp
        assert "sampler" in kwargs
        assert callable(kwargs["sampler"])

    def test_passes_sampler_for_temperature_sampling(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
            captured["kwargs"] = kwargs
            assert "sampler" in kwargs
            assert callable(kwargs["sampler"])
            yield SimpleNamespace(text="a")
            yield SimpleNamespace(text="b")

        monkeypatch.setattr(mr, "stream_generate", fake_stream_generate)

        runner = self._make_runner()
        out = runner.generate("p", max_tokens=2, temperature=0.5)

        assert out == "ab"
        kwargs = captured.get("kwargs")
        assert isinstance(kwargs, dict)
        assert "sampler" in kwargs


class TestV1MetalModelRunnerSampleTokens:
    """Tests for `MetalModelRunner.sample_tokens`.

    vLLM v1 may call `sample_tokens()` even if `execute_model()` failed before
    producing output. In that case, `sample_tokens()` must return `None` so vLLM
    can surface the original `execute_model()` exception (instead of raising a
    misleading error from `sample_tokens()` itself).
    """

    def _make_runner(self) -> mr.MetalModelRunner:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner._pending_output = None
        runner.use_async_scheduling = True
        return runner

    def test_returns_pending_output_and_clears_state(self) -> None:
        runner = self._make_runner()
        pending = ModelRunnerOutput(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            sampled_token_ids=[[123]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
        )
        runner._pending_output = pending

        out = runner.sample_tokens(grammar_output=None)

        assert out is pending
        assert runner._pending_output is None

    def test_returns_none_when_no_pending_output(self) -> None:
        runner = self._make_runner()
        out = runner.sample_tokens(grammar_output=None)

        assert out is None

    def test_raises_when_no_pending_output_and_not_async(self) -> None:
        runner = self._make_runner()
        runner.use_async_scheduling = False

        with pytest.raises(
            RuntimeError, match="sample_tokens called without pending output"
        ):
            runner.sample_tokens(grammar_output=None)


class TestResolveModelDims:
    def _make_runner(self, args: dict) -> mr.MetalModelRunner:
        r = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        r.model_args = args
        return r

    def test_standard_mha(self) -> None:
        r = self._make_runner({"num_hidden_layers": 32, "num_attention_heads": 32,
                               "num_key_value_heads": 8, "hidden_size": 4096})
        r._resolve_model_dims()
        assert r.num_layers == 32
        assert r.num_kv_heads == 8
        assert r.head_dim == 128  # 4096 // 32

    def test_mla_overrides_kv_heads_and_head_dim(self) -> None:
        # GLM-4.7-Flash: kv_lora_rank=512, qk_rope_head_dim=64
        r = self._make_runner({"num_hidden_layers": 47, "num_attention_heads": 20,
                               "num_key_value_heads": 20, "hidden_size": 2048,
                               "kv_lora_rank": 512, "qk_rope_head_dim": 64})
        r._resolve_model_dims()
        assert r.num_kv_heads == 1
        assert r.head_dim == 576  # 512 + 64

    def test_mla_default_rope_head_dim(self) -> None:
        r = self._make_runner({"num_hidden_layers": 28, "num_attention_heads": 16,
                               "hidden_size": 2048, "kv_lora_rank": 256})
        r._resolve_model_dims()
        assert r.head_dim == 320  # 256 + default 64

    def test_missing_dims_raise(self) -> None:
        r = self._make_runner({"num_hidden_layers": 32})
        with pytest.raises(ValueError, match="Cannot resolve model dimensions"):
            r._resolve_model_dims()

    def test_is_mla_true_when_kv_lora_rank_present(self) -> None:
        r = self._make_runner({"kv_lora_rank": 512})
        assert r.is_mla is True

    def test_is_mla_false_for_standard_mha(self) -> None:
        r = self._make_runner({"num_hidden_layers": 32, "num_attention_heads": 32,
                               "hidden_size": 4096})
        assert r.is_mla is False
