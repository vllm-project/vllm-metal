# SPDX-License-Identifier: Apache-2.0
"""Laya encoder parity and token-classification contracts without downloads."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from torch import nn
from transformers import ModernBertConfig, ModernBertModel
from vllm.pooling_params import PoolingParams

from vllm_metal.laya_registration import register
from vllm_metal.v1.pooling.backends.encoder.models.laya import (
    LayaModel,
    LayaPooler,
    clamp_temperature,
    temperature_for,
)
from vllm_metal.v1.pooling.contract import EncoderPoolingRequest
from vllm_metal.v1.pooling.validation import PoolingConfigView


def models():
    torch.manual_seed(71)
    ec = ModernBertConfig(
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=3,
        num_attention_heads=2,
        vocab_size=512,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        local_attention=128,
        reference_compile=False,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        mlp_dropout=0.0,
    )
    ec._attn_implementation = "eager"
    ac = {
        "head_layers": 2,
        "act_costs": {"escalate": 0.5},
        "mask_token_id": 7,
        "qtype_token_ids": [10, 11, 12],
        "temperature": [1.2, 1.4, 1.6],
        "temperature_by_options": {"choice:2": 1.9, "score:3-5": 1.3, "noul:2": 2.0},
    }
    reference = nn.Module()
    reference.encoder = ModernBertModel(ec)
    reference.type_emb = nn.Embedding(3, 128)
    reference.head = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            128, 2, 512, dropout=0.0, batch_first=True, norm_first=True
        ),
        2,
        enable_nested_tensor=False,
    )
    reference.scorer = nn.Sequential(
        nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1)
    )
    reference.act_head = nn.Sequential(
        nn.Linear(132, 256), nn.GELU(), nn.Linear(256, 2)
    )
    reference.eval()
    model = LayaModel(ec.to_dict(), ac)
    weights = model.sanitize(
        {k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()}
    )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    hf = SimpleNamespace(laya_config=ac)
    pooler_config = SimpleNamespace(
        task="token_classify", tok_pooling_type="ALL", enable_chunked_processing=False
    )
    view = PoolingConfigView(
        SimpleNamespace(
            hf_config=hf, pooler_config=pooler_config, multimodal_config=None
        )
    )
    return model, reference, LayaPooler(view, model), ac


@pytest.mark.parametrize(
    "qtype,count", [(0, 1), (0, 2), (0, 6), (0, 11), (1, 3), (2, 2)]
)
def test_torch_parity(qtype, count):
    model, reference, pooler, ac = models()
    ids = torch.randint(20, 512, (1, 150))
    ids[0, 0], ids[0, 1] = 2, ac["qtype_token_ids"][qtype]
    marker_positions = torch.arange(5, 5 + count * 3, 3)
    ids[0, marker_positions] = 7
    mask = torch.ones_like(ids)
    with torch.no_grad():
        h = reference.encoder(ids, attention_mask=mask).last_hidden_state
        h = reference.head(h + reference.type_emb(torch.tensor([qtype]))[:, None, :])
        logits = reference.scorer(h[:, marker_positions]).squeeze(-1).float()
        p = logits.softmax(-1)
        top = torch.nn.functional.pad(p, (0, 1)).topk(2, -1).values
        k = torch.tensor(float(max(count, 2)))
        ent = -(p * p.clamp_min(1e-9).log()).sum(-1) / k.log()
        features = torch.stack(
            [top[:, 0], top[:, 0] - top[:, 1], ent, (k / 255).expand(1)], -1
        )
        actions = reference.act_head(torch.cat([h[:, 0], features], -1))
    mh = model(mx.array(ids.numpy()), mx.array(mask.numpy()))
    np.testing.assert_allclose(np.array(mh), h.numpy(), atol=3e-5, rtol=3e-5)
    for activated in (False, True):
        req = EncoderPoolingRequest(
            "r",
            tuple(ids[0].tolist()),
            PoolingParams(task="token_classify", use_activation=activated),
        )
        actual = pooler.pool_one(mh, req).numpy()
        expected_p = (
            (logits / temperature_for(ac, qtype, count)).softmax(-1)
            if activated
            else logits
        )
        expected_a = actions.softmax(-1) if activated else actions
        expected = torch.cat(
            [expected_p[:, :, None], expected_a[:, None, :].expand(1, count, 2)], -1
        )[0]
        np.testing.assert_allclose(actual, expected.numpy(), atol=2e-5, rtol=3e-5)


def test_padding_and_profiling():
    model, _, pooler, _ = models()
    ids = mx.array([[2, 11, 7, 21, 7, 22]])
    h = model(ids, mx.ones(ids.shape, dtype=mx.int32))
    padded = mx.concatenate([ids, mx.zeros((1, 7), dtype=mx.int32)], -1)
    mask = mx.concatenate([mx.ones_like(ids), mx.zeros((1, 7), dtype=mx.int32)], -1)
    hp = model(padded, mask)
    np.testing.assert_allclose(np.array(h), np.array(hp[:, :6]), atol=2e-5, rtol=2e-5)
    dummy = model(mx.zeros((1, 1), dtype=mx.int32), mx.ones((1, 1), dtype=mx.int32))
    req = EncoderPoolingRequest("dummy", (0,), PoolingParams(task="token_classify"))
    assert pooler.pool_one(dummy, req).shape == (0, 3)
    with pytest.raises(NotImplementedError, match="dimensions"):
        pooler.validate_params(PoolingParams(task="token_classify", dimensions=2))
    with pytest.raises(NotImplementedError, match="token_classify"):
        pooler.validate_params(PoolingParams(task="embed"))


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, 1.0),
        (None, 1.0),
        ("bad", 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (0.1, 0.5),
        (9, 5.0),
        (1.5, 1.5),
    ],
)
def test_temperature(value, expected):
    assert clamp_temperature(value) == expected


def test_registry_metadata():
    from vllm.model_executor.models import ModelRegistry

    register()
    register()
    info = ModelRegistry.models["LayaForDecision"].inspect_model_cls()
    assert info.is_pooling_model
    assert info.attn_type == "encoder_only"
    assert info.default_tok_pooling_type == "ALL"


def checkpoint_config(reference, ac, path, **overrides):
    config = reference.encoder.config
    config.architectures = ["LayaForDecision"]
    config.laya_config = ac
    values = {
        "model": str(path),
        "tokenizer": "laya-tokenizer",
        "tokenizer_revision": "pinned-tokenizer-revision",
        "trust_remote_code": False,
        "hf_config": config,
        "dtype": torch.float32,
        "quantization": None,
        "runner_type": "pooling",
        "multimodal_config": None,
        "pooler_config": SimpleNamespace(
            task="token_classify",
            tok_pooling_type="ALL",
            enable_chunked_processing=False,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_checkpoint_loads_through_factory(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from vllm_metal.v1.pooling.backends.encoder.factory import (
        load_encoder_pooling_backend,
        supports_encoder_pooling_backend,
    )
    from vllm_metal.v1.pooling.backends.encoder.models import laya

    model, reference, _, ac = models()
    # Exercise stored FP16 weights, as shipped by the real checkpoint, rather
    # than testing only the in-memory FP32 mapping used by the math test.
    state = {k: v.half().contiguous() for k, v in reference.state_dict().items()}
    save_file(state, str(tmp_path / "model.safetensors"))
    expected = model.sanitize(
        {k: mx.array(v.float().numpy()) for k, v in state.items()}
    )
    model.load_weights(list(expected.items()), strict=True)
    calls = []
    tokenizer = object()

    def load_tokenizer(name, **kwargs):
        calls.append((name, kwargs))
        return tokenizer

    monkeypatch.setattr(laya.AutoTokenizer, "from_pretrained", load_tokenizer)
    config = checkpoint_config(reference, ac, tmp_path)
    assert supports_encoder_pooling_backend(config)
    loaded = load_encoder_pooling_backend(config)
    assert loaded.tokenizer is tokenizer
    assert calls == [
        (
            "laya-tokenizer",
            {"revision": "pinned-tokenizer-revision", "trust_remote_code": False},
        )
    ]
    ids = mx.array([[2, 10, 7, 21, 7, 22]])
    mask = mx.ones_like(ids)
    np.testing.assert_allclose(
        np.array(loaded.model(ids, mask)), np.array(model(ids, mask)), atol=1e-6
    )
    assert loaded.pooling_backend.supported_tasks() == ("token_classify",)
    assert not supports_encoder_pooling_backend(
        checkpoint_config(reference, ac, tmp_path, runner_type="generate")
    )
    assert not supports_encoder_pooling_backend(
        checkpoint_config(reference, ac, tmp_path, multimodal_config=object())
    )


@pytest.mark.parametrize(
    "overrides,config_field,value,error,message",
    [
        ({"dtype": torch.float16}, None, None, NotImplementedError, "float32"),
        ({"dtype": torch.bfloat16}, None, None, NotImplementedError, "float32"),
        ({"quantization": "awq"}, None, None, NotImplementedError, "quantization"),
        ({}, "quantization_config", {"bits": 4}, NotImplementedError, "quantization"),
        ({}, "laya_config", {"qtype_token_ids": [10, 10, 12]}, ValueError, "distinct"),
    ],
)
def test_loader_rejects_unsupported_before_io(
    tmp_path, monkeypatch, overrides, config_field, value, error, message
):
    from vllm_metal.v1.pooling.backends.encoder.models import laya

    _, reference, _, ac = models()
    config = checkpoint_config(reference, ac, tmp_path / "not-downloaded", **overrides)
    if config_field:
        setattr(config.hf_config, config_field, value)

    def unexpected_io(*args, **kwargs):
        pytest.fail("Rejected model must not download or load weights/tokenizer")

    monkeypatch.setattr(laya, "encoder_model_path", unexpected_io)
    monkeypatch.setattr(laya.AutoTokenizer, "from_pretrained", unexpected_io)
    with pytest.raises(error, match=message):
        laya.load_laya_backend(config)


@pytest.mark.parametrize("preexisting", [False, True])
def test_registry_preserves_existing_architecture(monkeypatch, preexisting):
    import vllm.model_executor.models as vllm_models

    sentinel = object()
    entries = {"LayaForDecision": sentinel} if preexisting else {}
    registry = SimpleNamespace(
        get_supported_archs=lambda: list(entries),
        register_model=lambda name, model: entries.update({name: model}),
    )
    monkeypatch.setattr(vllm_models, "ModelRegistry", registry)
    register()
    register()
    assert entries["LayaForDecision"] == (
        sentinel if preexisting else "vllm_metal.laya_registration:LayaForDecision"
    )
