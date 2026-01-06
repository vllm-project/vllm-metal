# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

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
        assert "sampler" in captured["kwargs"]
