# SPDX-License-Identifier: Apache-2.0
"""Local checkpoint conversion contracts; no downloads or real weights."""

import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tools.convert_laya_checkpoint import convert_checkpoint


def source_checkpoint(tmp_path):
    source = tmp_path / "original"
    (source / "encoder").mkdir(parents=True)
    (source / "encoder/config.json").write_text(
        json.dumps({"model_type": "modernbert", "hidden_size": 128})
    )
    (source / "rl_agent_config.json").write_text(
        json.dumps({"head_layers": 2, "max_len": 1024, "temperature": [1, 2, 3]})
    )
    (source / "model.safetensors").write_bytes(b"unchanged checkpoint bytes")
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


@pytest.mark.parametrize("copy_weights", [False, True])
def test_convert_preserves_weights_and_token_ids(tmp_path, monkeypatch, copy_weights):
    source = source_checkpoint(tmp_path)
    output = tmp_path / "converted"
    if copy_weights:

        def no_hardlink(*args):
            raise OSError("Cross-device link")

        monkeypatch.setattr("tools.convert_laya_checkpoint.os.link", no_hardlink)
    original = AutoTokenizer.from_pretrained(source / "tokenizer")
    convert_checkpoint(source, output)
    converted = AutoTokenizer.from_pretrained(output)
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == ["LayaForDecision"]
    assert config["dtype"] == "float32"
    assert config["laya_config"]["qtype_token_ids"] == [2, 3, 4]
    assert config["laya_config"]["mask_token_id"] == 1
    assert config["laya_config"]["temperature"] == [1, 2, 3]
    assert config["laya_config"]["max_len"] == 1024
    assert (output / "model.safetensors").read_bytes() == (
        source / "model.safetensors"
    ).read_bytes()
    text = "choice question: [MASK] score noul"
    assert original(text)["input_ids"] == converted(text)["input_ids"]
    assert not (source / "config.json").exists()
    with pytest.raises(FileExistsError):
        convert_checkpoint(source, output)


@pytest.mark.parametrize("failure", ["backbone", "weights"])
def test_invalid_source_does_not_create_output(tmp_path, failure):
    source = source_checkpoint(tmp_path)
    output = tmp_path / "converted"
    if failure == "backbone":
        (source / "encoder/config.json").write_text(json.dumps({"model_type": "bert"}))
        error = NotImplementedError
    else:
        (source / "model.safetensors").unlink()
        error = FileNotFoundError
    with pytest.raises(error):
        convert_checkpoint(source, output)
    assert not output.exists()
