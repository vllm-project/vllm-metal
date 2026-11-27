# SPDX-License-Identifier: Apache-2.0
"""Tests for block-FP8 preflight rejection before mlx-lm weight loading."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_metal.quant.awq_config import UnsupportedQuantizationConfigError
from vllm_metal.quant.fp8_config import (
    resolve_mlx_lm_model_type,
    validate_mlx_lm_block_fp8_support,
)

_BLOCK_FP8_QC = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


@pytest.mark.parametrize(
    ("model_type", "architectures", "expected_in_message"),
    [
        ("qwen3", ["Qwen3ForCausalLM"], "'qwen3'"),
        ("qwen3_moe", ["Qwen3MoeForCausalLM"], "'qwen3_moe'"),
        ("llama", ["LlamaForCausalLM"], "'llama'"),
        ("mistral", ["MistralForCausalLM"], "mapped to 'llama'"),
        ("ministral3", ["Mistral3ForConditionalGeneration"], "'ministral3'"),
        ("qwen3_next", ["Qwen3NextForCausalLM"], "'qwen3_next'"),
        ("glm4", ["Glm4ForCausalLM"], "'glm4'"),
    ],
)
def test_rejects_unsupported_block_fp8_architectures(
    model_type: str,
    architectures: list[str],
    expected_in_message: str,
) -> None:
    hf_config = SimpleNamespace(
        model_type=model_type,
        architectures=architectures,
        quantization_config=dict(_BLOCK_FP8_QC),
    )

    with pytest.raises(UnsupportedQuantizationConfigError) as excinfo:
        validate_mlx_lm_block_fp8_support(hf_config)

    assert isinstance(excinfo.value, ValueError)
    msg = str(excinfo.value)
    assert expected_in_message in msg
    assert "[128, 128]" in msg
    assert "MLX-quantized" in msg
    assert "bf16" in msg


def test_rejects_nested_text_config_block_fp8() -> None:
    hf_config = SimpleNamespace(
        model_type="qwen3_vl",
        architectures=["Qwen3VLForConditionalGeneration"],
        text_config=SimpleNamespace(
            model_type="qwen3",
            quantization_config=dict(_BLOCK_FP8_QC),
        ),
    )

    with pytest.raises(UnsupportedQuantizationConfigError, match="qwen3_vl"):
        validate_mlx_lm_block_fp8_support(hf_config)


def test_rejects_local_config_json_without_hf_config_and_marks_draft(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "quantization_config": _BLOCK_FP8_QC,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedQuantizationConfigError) as excinfo:
        validate_mlx_lm_block_fp8_support(model_path=tmp_path, is_draft=True)

    msg = str(excinfo.value)
    assert "draft model" in msg
    assert "qwen3_moe" in msg


@pytest.mark.parametrize(
    ("model_type", "architectures"),
    [
        ("deepseek_v3", None),
        ("kimi_k2", None),
        ("joyai_llm_flash", None),
        ("deepseek_v32", None),
        ("mimo_v2_flash", None),
        ("minimax", None),
        ("minimax_m2", None),
        ("mistral4", None),
        ("qwen3_5", ["Qwen3_5ForConditionalGeneration"]),
        ("qwen3_5_moe", ["Qwen3_5MoeForConditionalGeneration"]),
    ],
)
def test_allows_supported_block_fp8_architectures(
    model_type: str,
    architectures: list[str] | None,
) -> None:
    hf_config = SimpleNamespace(
        model_type=model_type,
        architectures=architectures,
        quantization_config=dict(_BLOCK_FP8_QC),
    )

    validate_mlx_lm_block_fp8_support(hf_config)


def test_allows_mxfp8_ue8m0_32x32_on_any_architecture() -> None:
    hf_config = SimpleNamespace(
        model_type="qwen3",
        quantization_config={
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [32, 32],
        },
    )

    validate_mlx_lm_block_fp8_support(hf_config)


@pytest.mark.parametrize(
    "quantization_config",
    [
        None,
        {"quant_method": "awq", "bits": 4, "group_size": 128},
        {"quant_method": "fp8"},
        {"quant_method": "fp8", "activation_scheme": "dynamic"},
        {"quant_method": "fp8", "weight_block_size": []},
    ],
)
def test_allows_non_block_fp8_and_other_quant_methods(
    quantization_config: dict[str, object] | None,
) -> None:
    hf_config = SimpleNamespace(
        model_type="qwen3",
        quantization_config=quantization_config,
    )

    validate_mlx_lm_block_fp8_support(hf_config)


def test_allows_custom_model_file_with_block_fp8() -> None:
    hf_config = SimpleNamespace(
        model_type="qwen3",
        model_file="custom_model.py",
        quantization_config=dict(_BLOCK_FP8_QC),
    )

    validate_mlx_lm_block_fp8_support(hf_config)


def test_skips_missing_hf_config_and_non_dir_model_path() -> None:
    validate_mlx_lm_block_fp8_support(None, model_path="Qwen/Qwen3-0.6B-FP8")


def test_resolve_mlx_lm_model_type_applies_arch_and_model_remapping() -> None:
    assert resolve_mlx_lm_model_type("kimi_k2") == "deepseek_v3"
    assert (
        resolve_mlx_lm_model_type("bailing_hybrid", ["BailingMoeV3ForCausalLM"])
        == "bailing_moe_v3"
    )
