# SPDX-License-Identifier: Apache-2.0
"""Exercise the public checkpoint configuration path without remote code."""

import json

import pytest
from transformers import AutoConfig
from vllm.transformers_utils.config import get_config

from vllm_metal.multimodal.nemotron_nano_vl.config import (
    NemotronNanoVLConfig,
    register_config,
)


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_checkpoint_without_remote_python_loads_locally(tmp_path, trust_remote_code):
    raw = {
        "model_type": "NemotronH_Nano_VL_V2",
        "architectures": ["NemotronH_Nano_VL_V2"],
        "auto_map": {"AutoConfig": "configuration.NemotronH_Nano_VL_V2_Config"},
        "llm_config": {
            "model_type": "nemotron_h",
            "hidden_size": 5120,
            "num_hidden_layers": 3,
            "hybrid_override_pattern": "M*-",
            "mamba_head_dim": 80,
        },
        "vision_config": {"patch_size": 16, "args": {"model": "vit_huge_patch16_224"}},
        "force_image_size": 512,
        "img_context_token_id": 131072,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    register_config()
    config = get_config(tmp_path, trust_remote_code=trust_remote_code)
    assert isinstance(config, NemotronNanoVLConfig)
    assert config.get_text_config().hidden_size == 5120
    assert config.text_config.layers_block_type == ["mamba", "attention", "mlp"]
    assert config.text_config.mamba_head_dim == 80
    assert config.vision_config.patch_size == 16
    assert config.vision_config.args == raw["vision_config"]["args"]
    assert config.force_image_size == 512
    assert config.img_context_token_id == 131072
    config.save_pretrained(tmp_path / "roundtrip")
    reloaded = AutoConfig.from_pretrained(
        tmp_path / "roundtrip", trust_remote_code=False
    )
    assert reloaded.text_config.to_dict() == config.text_config.to_dict()
    assert json.loads((tmp_path / "config.json").read_text()) == raw


@pytest.mark.parametrize("pattern,count", [("MME", 3), ("M*", 3), ("", 0)])
def test_invalid_vl_v2_hybrid_layout_rejected(pattern, count):
    with pytest.raises(ValueError, match="Nano VL V2"):
        NemotronNanoVLConfig(
            llm_config={"hybrid_override_pattern": pattern, "num_hidden_layers": count}
        )
