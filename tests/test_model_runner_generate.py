# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_metal.model_runner as mr


class TestMetalModelRunnerGenerate:
    def _make_runner(self) -> mr.MetalModelRunner:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model = object()
        runner.tokenizer = object()
        return runner

    def test_accumulates_streamed_segments(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_make_sampler(*, temp: float):
            captured["temp"] = temp
            return object()

        def fake_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
            captured["prompt"] = prompt
            captured["max_tokens"] = max_tokens
            captured["kwargs"] = kwargs
            yield SimpleNamespace(text="hello")
            yield SimpleNamespace(text=" ")
            yield SimpleNamespace(text="world")

        monkeypatch.setattr(mr, "make_sampler", fake_make_sampler)
        monkeypatch.setattr(mr, "stream_generate", fake_stream_generate)

        runner = self._make_runner()
        out = runner.generate("p", max_tokens=3, temperature=0.7)

        assert out == "hello world"
        assert captured["prompt"] == "p"
        assert captured["max_tokens"] == 3
        assert captured["temp"] == 0.7
        assert "sampler" in captured["kwargs"]  # type: ignore

    def test_execute_model_batches_by_sequence_length(self) -> None:
        mx = pytest.importorskip("mlx.core")

        class DummySeqData:
            def __init__(self, tokens: list[int]) -> None:
                self._tokens = tokens

            def get_token_ids(self) -> list[int]:
                return self._tokens

        class DummySeqGroup:
            def __init__(self, tokens: list[int], is_prompt: bool) -> None:
                self.seq_data = {0: DummySeqData(tokens)}
                self.is_prompt = is_prompt

        class DummyModel:
            def __init__(self, vocab_size: int) -> None:
                self._vocab_size = vocab_size

            def __call__(self, input_ids: mx.array) -> mx.array:
                eye = mx.eye(self._vocab_size)
                return eye[input_ids] * 10.0

        runner = self._make_runner()
        runner.model = DummyModel(vocab_size=16)

        seq_groups = [
            DummySeqGroup(tokens=[1, 2, 3], is_prompt=True),
            DummySeqGroup(tokens=[7, 8], is_prompt=False),
        ]

        outputs = runner.execute_model(seq_groups)

        assert len(outputs) == len(seq_groups)
        assert outputs[0]["seq_group"] is seq_groups[0]
        assert outputs[1]["seq_group"] is seq_groups[1]
        assert [output["token_id"] for output in outputs] == [3, 8]
