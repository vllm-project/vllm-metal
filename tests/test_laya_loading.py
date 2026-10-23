# SPDX-License-Identifier: Apache-2.0
"""Native Laya configuration and tokenizer loading without real weights."""

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoTokenizer, ModernBertConfig, PreTrainedTokenizerFast
from vllm.engine.arg_utils import EngineArgs
from vllm.renderers.registry import renderer_from_config
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.config import get_config

from vllm_metal.v1.pooling.backends.encoder.models.laya import register


@pytest.fixture
def source_checkpoint(tmp_path):
    source = tmp_path / "original"
    (source / "encoder").mkdir(parents=True)
    (source / "config.json").write_text(
        json.dumps({"model_type": "laya", "architectures": ["LayaTypedDecisions"]})
    )
    (source / "encoder/config.json").write_text(
        ModernBertConfig(
            hidden_size=128,
            intermediate_size=192,
            num_hidden_layers=3,
            num_attention_heads=2,
            vocab_size=32,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=0,
            eos_token_id=0,
            cls_token_id=0,
            sep_token_id=0,
        ).to_json_string()
    )
    (source / "rl_agent_config.json").write_text(
        json.dumps({"head_layers": 2, "max_len": 1024, "temperature": [1, 2, 3]})
    )
    vocabulary = {
        token: i
        for i, token in enumerate(
            ["[UNK]", "[MASK]", "choice", "score", "noul", "question:"]
        )
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", mask_token="[MASK]"
    ).save_pretrained(source / "tokenizer")
    return source


@pytest.mark.parametrize(
    "dtype,expected_dtype",
    [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("auto", torch.float16),
    ],
)
def test_native_layout_works_through_vllm_frontend(
    source_checkpoint, inspect_vllm_models_in_process, dtype, expected_dtype
):
    source = source_checkpoint
    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    register()
    config = EngineArgs(
        model=str(source),
        config_format="laya",
        tokenizer_mode="laya",
        runner="pooling",
        dtype=dtype,
        max_model_len=128,
        enforce_eager=True,
        enable_prefix_caching=False,
    ).create_engine_config()
    hf_config = config.model_config.hf_config
    assert config.model_config.dtype == expected_dtype
    assert hf_config.hidden_size == 128
    assert hf_config.architectures == ["LayaForDecision"]
    assert hf_config.laya_config["temperature"] == [1, 2, 3]
    tokenizer = cached_tokenizer_from_config(config.model_config)
    original = AutoTokenizer.from_pretrained(source / "tokenizer")
    text = "choice question: [MASK] score noul"
    assert tokenizer(text)["input_ids"] == original(text)["input_ids"]
    assert tokenizer.mask_token_id == 1
    assert (
        renderer_from_config(config).tokenizer(text)["input_ids"]
        == original(text)["input_ids"]
    )
    assert before == {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("failure", ["backbone", "agent_config"])
def test_invalid_native_configuration(source_checkpoint, failure):
    source = source_checkpoint
    register()
    if failure == "backbone":
        (source / "encoder/config.json").write_text(json.dumps({"model_type": "bert"}))
        error = NotImplementedError
    else:
        (source / "rl_agent_config.json").unlink()
        error = OSError
    with pytest.raises(error):
        get_config(source, trust_remote_code=False, config_format="laya")


def test_hub_config_and_tokenizer_use_their_own_revisions(
    source_checkpoint, monkeypatch
):
    from vllm_metal.v1.pooling.backends.encoder.models import laya

    register()
    config_calls = []

    def cached_file(model, filename, **kwargs):
        config_calls.append((model, filename, kwargs))
        return str(source_checkpoint / filename)

    monkeypatch.setattr(laya, "cached_file", cached_file)
    parser = laya.LayaConfigParser()
    _, config = parser.parse(
        "owner/laya", False, revision="model-commit", token="hf-test"
    )
    assert config.hidden_size == 128
    assert config_calls == [
        (
            "owner/laya",
            name,
            {"revision": "model-commit", "token": "hf-test", "cache_dir": None},
        )
        for name in ("encoder/config.json", "rl_agent_config.json")
    ]
    calls = []

    def load_tokenizer(model, *args, **kwargs):
        calls.append((model, kwargs))
        return AutoTokenizer.from_pretrained(source_checkpoint / "tokenizer")

    monkeypatch.setattr(laya.CachedHfTokenizer, "from_pretrained", load_tokenizer)
    tokenizer = laya.LayaTokenizer.from_pretrained(
        "owner/tokenizer", revision="tokenizer-commit", trust_remote_code=False
    )
    assert tokenizer.mask_token_id == 1
    assert len(calls) == 1
    model, kwargs = calls[0]
    assert model == "owner/tokenizer"
    assert isinstance(kwargs.pop("config"), laya.LayaMetadataConfig)
    assert kwargs == {
        "subfolder": "tokenizer",
        "revision": "tokenizer-commit",
        "trust_remote_code": False,
    }
