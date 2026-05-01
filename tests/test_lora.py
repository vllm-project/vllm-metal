# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal LoRA adapter loading helpers."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_LORA_PATH = Path(__file__).resolve().parents[1] / "vllm_metal" / "v1" / "lora.py"
_LORA_SPEC = importlib.util.spec_from_file_location("metal_lora_under_test", _LORA_PATH)
assert _LORA_SPEC is not None and _LORA_SPEC.loader is not None
metal_lora = importlib.util.module_from_spec(_LORA_SPEC)
sys.modules[_LORA_SPEC.name] = metal_lora
_LORA_SPEC.loader.exec_module(metal_lora)


def test_resolve_module_name_strips_unsloth_peft_prefix() -> None:
    module_names = {
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
    }

    resolved = metal_lora._resolve_module_name(
        "base_model.model.model.layers.0.self_attn.q_proj",
        module_names,
    )

    assert resolved == "layers.0.self_attn.q_proj"


def test_load_peft_adapter_transposes_unsloth_weights(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 8,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj"],
            }
        )
    )
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(10, dtype=np.float32).reshape(5, 2)
    monkeypatch.setattr(
        metal_lora,
        "_read_safetensors",
        lambda _path: {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
        },
    )

    [weights] = metal_lora._load_adapter_weights(
        tmp_path,
        {"layers.0.self_attn.q_proj"},
    )

    assert weights.module_name == "layers.0.self_attn.q_proj"
    assert weights.rank == 2
    assert weights.scale == 4.0
    assert weights.dropout == 0.05
    np.testing.assert_array_equal(weights.lora_a, a.T)
    np.testing.assert_array_equal(weights.lora_b, b.T)


def test_load_peft_adapter_uses_rslora_scaling(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 4,
                "lora_alpha": 16,
                "use_rslora": True,
            }
        )
    )
    monkeypatch.setattr(
        metal_lora,
        "_read_safetensors",
        lambda _path: {
            "base_model.model.layers.0.mlp.up_proj.lora_A.weight": np.zeros(
                (4, 3), dtype=np.float32
            ),
            "base_model.model.layers.0.mlp.up_proj.lora_B.weight": np.zeros(
                (5, 4), dtype=np.float32
            ),
        },
    )

    [weights] = metal_lora._load_adapter_weights(
        tmp_path,
        {"layers.0.mlp.up_proj"},
    )

    assert weights.scale == 16 / math.sqrt(4)


def test_manager_rejects_mixed_lora_batch() -> None:
    manager = metal_lora.MetalLoRAManager(SimpleNamespace())

    with pytest.raises(NotImplementedError, match="one LoRA adapter per scheduled"):
        manager.validate_uniform_lora({None, 1})
