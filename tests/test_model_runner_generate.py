# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import vllm_metal.model_runner as mr


class TestMetalModelRunnerGenerate:
    def _make_runner(self) -> mr.MetalModelRunner:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model = object()
        runner.tokenizer = object()
        runner.processor = None
        runner.vlm_config = None
        runner._is_vlm = False
        return runner

    def _make_vlm_runner(self) -> mr.MetalModelRunner:
        """Create a VLM-enabled runner for testing."""
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model = object()
        runner.tokenizer = object()
        runner.processor = object()
        runner.vlm_config = object()
        runner._is_vlm = True
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

    def test_vlm_generate_with_images(self, monkeypatch) -> None:
        """Test VLM generation with images uses mlx-vlm path."""
        captured: dict[str, object] = {}

        def fake_apply_chat_template(processor, config, prompt, num_images):
            captured["num_images"] = num_images
            return f"<image>{prompt}"

        def fake_mlx_vlm_generate(model, processor, prompt, images, **kwargs):
            captured["prompt"] = prompt
            captured["images"] = images
            captured["kwargs"] = kwargs
            return "A cat sitting on a mat"

        monkeypatch.setattr(mr, "apply_chat_template", fake_apply_chat_template)
        monkeypatch.setattr(mr, "mlx_vlm_generate", fake_mlx_vlm_generate)

        runner = self._make_vlm_runner()
        out = runner.generate(
            "Describe this image",
            images=["test.jpg", "test2.jpg"],
            max_tokens=50,
            temperature=0.5,
        )

        assert out == "A cat sitting on a mat"
        assert captured["num_images"] == 2
        assert captured["images"] == ["test.jpg", "test2.jpg"]
        assert captured["kwargs"]["max_tokens"] == 50
        assert captured["kwargs"]["temp"] == 0.5

    def test_vlm_without_images_uses_text_path(self, monkeypatch) -> None:
        """Test VLM without images falls back to text generation."""
        captured: dict[str, object] = {}

        def fake_make_sampler(*, temp: float):
            captured["temp"] = temp
            return object()

        def fake_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
            captured["prompt"] = prompt
            yield SimpleNamespace(text="text response")

        monkeypatch.setattr(mr, "make_sampler", fake_make_sampler)
        monkeypatch.setattr(mr, "stream_generate", fake_stream_generate)

        runner = self._make_vlm_runner()
        out = runner.generate("Hello", max_tokens=10)

        assert out == "text response"
        assert captured["prompt"] == "Hello"
