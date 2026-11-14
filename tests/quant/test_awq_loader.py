# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``AWQQuantLoader``.

``TestConfigDetection`` covers ``for_model``: AWQ accept, non-AWQ silent
decline, GPTQ explicit reject, ``text_config`` fallback, alias
normalization, Hub-fetch-error propagation.

Pure helpers; no model load, no downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import mlx.nn as nn
import pytest
from huggingface_hub.utils import HFValidationError

from vllm_metal.quant.awq_config import UnsupportedQuantizationConfigError
from vllm_metal.quant.awq_loader import AWQQuantLoader

_AWQ_INNER = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "version": "gemm",
}


def _write_config(tmp_path: Path, config: dict) -> Path:
    """Drop a config.json into ``tmp_path`` and return the directory."""
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


class TestConfigDetection:
    """``AWQQuantLoader.for_model``: detection, accept, decline, reject."""

    @pytest.mark.parametrize("revision", [None, "release-tag", "a" * 40])
    def test_revision_reaches_config_and_weight_loaders(
        self, monkeypatch, tmp_path, revision
    ):
        config_dir = _write_config(tmp_path, {"quantization_config": _AWQ_INNER})
        download = Mock(return_value=str(config_dir / "config.json"))
        load = Mock(return_value=(nn.Module(), None))
        monkeypatch.setattr("vllm_metal.quant.awq_loader.hf_hub_download", download)
        resolve = Mock(return_value=config_dir)
        monkeypatch.setattr(
            "vllm_metal.quant.awq_loader.resolve_mlx_lm_model_path", resolve
        )
        monkeypatch.setattr("vllm_metal.quant.awq_loader.mlx_lm_load_model", load)

        loader = AWQQuantLoader.for_model("org/model", revision=revision)
        assert loader is not None
        loader.load("org/model", revision=revision)

        download.assert_called_once_with("org/model", "config.json", revision=revision)
        load.assert_called_once()
        resolve.assert_called_once_with("org/model", revision=revision)
        assert load.call_args.args == (config_dir,)
        assert load.call_args.kwargs["model_config"] == {
            "quantization_config": _AWQ_INNER
        }

    def test_returns_none_for_non_awq(self, tmp_path):
        """Non-AWQ checkpoints get None back, no exception."""
        model_dir = _write_config(
            tmp_path,
            {
                "model_type": "qwen2",
                "quantization_config": {"quant_method": "fp8", "bits": 8},
            },
        )
        assert AWQQuantLoader.for_model(str(model_dir)) is None

    def test_returns_none_for_no_quant_config(self, tmp_path):
        model_dir = _write_config(tmp_path, {"model_type": "qwen2"})
        assert AWQQuantLoader.for_model(str(model_dir)) is None

    def test_propagates_hub_fetch_errors(self):
        """A malformed repo id / unreachable model_name surfaces the
        underlying Hub error to the caller rather than silently demoting
        to the generic loader path. Silent fall-through on a transient
        Hub failure would let an AWQ checkpoint bypass the preflight.

        ``Path("/nonexistent/path/zzz").is_dir()`` is False so we fall
        through to ``hf_hub_download``, which raises
        ``HFValidationError`` on a path with multiple slashes (no
        network call needed). A legitimate local snapshot without a
        ``config.json`` is handled by the directory branch and still
        returns ``None`` silently.
        """
        with pytest.raises(HFValidationError):
            AWQQuantLoader.for_model("/nonexistent/path/zzz")

    def test_picks_up_nested_text_config(self, tmp_path):
        """Multimodal wrapper configs nest the quant config under
        ``text_config``. ``mlx_lm.utils.load_model`` falls back to it;
        the AWQ owner mirrors that fallback so the alias normalization /
        reject logic still runs against the nested config.
        """
        model_dir = _write_config(
            tmp_path,
            {
                "model_type": "wrapper_vlm",
                "text_config": {
                    "model_type": "qwen2",
                    "quantization_config": _AWQ_INNER,
                },
            },
        )
        assert AWQQuantLoader.for_model(str(model_dir)) is not None

    def test_top_level_quant_config_wins_over_text_config(self, tmp_path):
        """If both are present, top-level takes precedence (matches
        mlx-lm). Pinned by giving ``text_config`` an invalid config
        (``bits=8``); if the owner picked text_config instead of
        top-level, ``normalize_quant_config`` would raise on the
        invalid bits.
        """
        nested = {**_AWQ_INNER, "bits": 8}
        model_dir = _write_config(
            tmp_path,
            {
                "model_type": "wrapper",
                "quantization_config": _AWQ_INNER,
                "text_config": {"quantization_config": nested},
            },
        )
        assert AWQQuantLoader.for_model(str(model_dir)) is not None

    def test_raises_for_gptq(self, tmp_path):
        """GPTQ is rejected loudly at the loader entry point: the loader
        must not silently take ownership of a quant_method it has not
        yet validated.
        """
        model_dir = _write_config(
            tmp_path,
            {
                "model_type": "qwen2",
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                    "zero_point": True,
                    "version": "gemm",
                },
            },
        )
        with pytest.raises(UnsupportedQuantizationConfigError) as excinfo:
            AWQQuantLoader.for_model(str(model_dir))
        assert "GPTQ" in str(excinfo.value)

    def test_returns_loader_for_awq(self, tmp_path):
        """AWQ checkpoint: ``for_model`` returns a configured loader."""
        model_dir = _write_config(
            tmp_path,
            {"model_type": "qwen2", "quantization_config": _AWQ_INNER},
        )
        assert AWQQuantLoader.for_model(str(model_dir)) is not None

    def test_propagates_reject_via_text_config(self, tmp_path):
        """The text_config fallback must surface reject errors too;
        nesting must not be a hole that lets unsupported configs
        through.
        """
        model_dir = _write_config(
            tmp_path,
            {
                "model_type": "wrapper",
                "text_config": {
                    "quantization_config": {
                        "quant_method": "awq",
                        "bits": 8,  # reject
                        "group_size": 128,
                    },
                },
            },
        )
        with pytest.raises(UnsupportedQuantizationConfigError):
            AWQQuantLoader.for_model(str(model_dir))

    def test_accepts_aliased_config(self, tmp_path):
        """``for_model`` runs ``normalize_quant_config`` on detection,
        so an AutoAWQ-style aliased config (``w_bit``, ``q_group_size``,
        uppercase ``GEMM``) yields a valid loader rather than raising.
        """
        aliased = {
            "quant_method": "awq",
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "version": "GEMM",
        }
        model_dir = _write_config(
            tmp_path,
            {"model_type": "qwen2", "quantization_config": aliased},
        )
        assert AWQQuantLoader.for_model(str(model_dir)) is not None
